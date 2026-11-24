"""Tenant isolation at the mounted API boundary; no database, Docker or AI calls."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user, require_owned_workspaces
from app.api.v1.routes import agent_orchestration
from app.core.config import settings
from app.core.workspace_access import owned_workspaces, workspace_access
from app.db.session import get_db
from app.main import app
from app.services import code_patch_apply, local_context
from app.services.workspace import (
    WorkspaceAccessDenied, WorkspaceConfigurationError,
    owned_workspace_roots, read_configured_workspace_file, resolve_allowed_workspace_root,
)
from app.services.workspace_indexing import WorkspaceIndexingService
from app.services.workspace_ingestion import IngestedWorkspaceDocument, WorkspaceIngestionResult
from app.services.vector_store import VectorSearchHit


@pytest.fixture
def tenants(tmp_path, monkeypatch):
    alice, bob, unassigned = uuid4(), uuid4(), uuid4()
    a, b = tmp_path / "alice", tmp_path / "bob"
    a.mkdir()
    b.mkdir()
    (a / "alice.py").write_text("private_alice = True\n", encoding="utf-8")
    (b / "bob.py").write_text("private_bob = True\n", encoding="utf-8")
    monkeypatch.setattr(settings, "workspace_allowed_roots", str(tmp_path))
    monkeypatch.setattr(settings, "workspace_root_owners", {str(a): alice, str(b): bob})
    monkeypatch.setattr(settings, "embedding_provider", None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    return alice, bob, unassigned, a.resolve(), b.resolve()


@pytest.fixture
def client(tenants, monkeypatch):
    from fastapi import Request

    async def current_user(request: Request):
        user_id = dict(zip(("alice", "bob", "unassigned"), tenants[:3])).get(request.headers.get("x-test-user"))
        if user_id is None:
            raise HTTPException(401, "Authentication required")
        return SimpleNamespace(id=user_id, is_active=True)

    async def database():
        yield SimpleNamespace()

    monkeypatch.setitem(app.dependency_overrides, get_current_user, current_user)
    monkeypatch.setitem(app.dependency_overrides, get_db, database)
    # No lifespan: never initialize providers, vectors or the checkpoint database.
    return TestClient(app)


def test_tree_is_owner_scoped_and_default_denies(client, tenants):
    for name, root in (("alice", tenants[3]), ("bob", tenants[4])):
        result = client.get("/api/v1/workspace/tree", headers={"x-test-user": name})
        assert result.status_code == 200
        assert [item["workspace_root"] for item in result.json()["children"]] == [str(root)]
    assert client.get("/api/v1/workspace/tree", headers={"x-test-user": "unassigned"}).json()["children"] == []
    assert client.get("/api/v1/workspace/tree").status_code == 401


def test_every_mounted_legacy_route_has_ownership_dependency():
    prefixes = tuple("/api/v1/" + prefix for prefix in (
        "workspace", "code-review", "code-patch", "local-context",
        "task-planning", "test-execution", "agent/workflows",
    ))
    matched = 0
    for route in app.routes:
        if getattr(route, "path", "").startswith(prefixes):
            matched += 1
            assert any(dep.call is require_owned_workspaces for dep in route.dependant.dependencies), route.path
    assert matched >= 20


@pytest.mark.parametrize("method,path", [
    ("POST", "workspace/context/prepare"), ("POST", "workspace/index"),
    ("POST", "workspace/search"), ("POST", "code-review/file"),
    ("POST", "code-patch"), ("POST", "code-patch/apply"),
    ("POST", "test-execution/run"), ("POST", "task-planning/plan"),
    ("POST", "agent/workflows"), ("PUT", "local-context/instructions"),
    ("GET", "local-context/instructions"), ("GET", "code-patch/backups"),
])
def test_foreign_targets_denied_before_services(client, tenants, method, path):
    target = {"workspace_root": str(tenants[4])}
    response = client.request(method, "/api/v1/" + path, headers={"x-test-user": "alice"},
                              **({"params": target} if method == "GET" else {"json": target}))
    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Workspace access denied."


def test_owned_context_and_instructions_work(client, tenants):
    headers = {"x-test-user": "alice"}
    root = str(tenants[3])
    response = client.post("/api/v1/workspace/context/prepare", headers=headers, json={"workspace_root": root})
    assert response.status_code == 200, response.text
    saved = client.put("/api/v1/local-context/instructions", headers=headers,
                       json={"workspace_root": root, "instructions": "Alice instructions"})
    assert saved.status_code == 200, saved.text
    assert client.get("/api/v1/local-context/instructions", headers=headers,
                      params={"workspace_root": root}).json()["instructions"] == "Alice instructions"


def test_explicit_global_roots_cannot_bypass_request_scope(tenants):
    alice, _, _, a, b = tenants
    with owned_workspaces(alice, (a,)):
        with pytest.raises(WorkspaceAccessDenied):
            resolve_allowed_workspace_root(b, [b])
        with pytest.raises(WorkspaceAccessDenied):
            read_configured_workspace_file(workspace_root=b, relative_path="bob.py")
        with pytest.raises(WorkspaceAccessDenied):
            resolve_allowed_workspace_root(a / ".." / "bob")
        assert resolve_allowed_workspace_root(a) == a


def test_overlap_and_outside_allowlist_fail_closed(tenants, monkeypatch, tmp_path):
    alice, bob, _, a, b = tenants
    monkeypatch.setattr(settings, "workspace_root_owners", {str(tmp_path): alice, str(b): bob})
    with pytest.raises(WorkspaceConfigurationError):
        owned_workspace_roots(alice)
    monkeypatch.setattr(settings, "workspace_allowed_roots", str(a))
    monkeypatch.setattr(settings, "workspace_root_owners", {str(b): bob})
    with pytest.raises(WorkspaceAccessDenied):
        owned_workspace_roots(bob)


def test_request_scope_is_isolated_across_tasks_and_threads(tenants):
    async def scenario():
        async def read(user, root):
            with owned_workspaces(user, (root,)):
                await asyncio.sleep(0)
                return await asyncio.to_thread(lambda: workspace_access.get().user_id)
        assert await asyncio.gather(read(tenants[0], tenants[3]), read(tenants[1], tenants[4])) == list(tenants[:2])
        assert workspace_access.get() is None
    asyncio.run(scenario())


def test_backup_ids_and_instruction_storage_are_owner_scoped(tenants):
    alice, bob, _, a, b = tenants
    with owned_workspaces(alice, (a,)):
        directory_a = code_patch_apply._backup_directory()
        instructions_a = local_context._database_path()
        # Malformed content deliberately proves Bob cannot even read Alice's manifest.
        (directory_a / ("a" * 32 + ".json")).write_text("private-invalid-json", encoding="utf-8")
    with owned_workspaces(bob, (b,)):
        assert code_patch_apply._backup_directory() != directory_a
        assert local_context._database_path() != instructions_a
        with pytest.raises(code_patch_apply.PatchBackupNotFound):
            code_patch_apply._read_rollback_manifest("a" * 32)


def test_legacy_shared_backups_are_not_exposed_to_assigned_users(client, tenants):
    legacy = code_patch_apply._backup_directory()
    backup_id = "b" * 32
    (legacy / f"{backup_id}.json").write_text("unowned-private-metadata", encoding="utf-8")
    headers = {"x-test-user": "alice"}
    assert client.get(f"/api/v1/code-patch/backups/{backup_id}", headers=headers).status_code == 404
    assert client.post("/api/v1/code-patch/rollback", headers=headers,
                       json={"backup_id": backup_id, "approval": "rollback"}).status_code == 404


def test_resume_rechecks_filesystem_ownership(client, tenants, monkeypatch):
    graph = SimpleNamespace(
        aget_state=AsyncMock(return_value=SimpleNamespace(values={
            "owner_user_id": str(tenants[0]), "workspace_root": str(tenants[4]),
            "status": "awaiting_patch_approval",
        })),
        ainvoke=AsyncMock(side_effect=AssertionError("Must not resume a foreign workspace")),
    )
    monkeypatch.setattr(agent_orchestration, "_get_runtime", AsyncMock(return_value=SimpleNamespace(graph=graph)))
    response = client.post(f"/api/v1/agent/workflows/{uuid4()}/resume", headers={"x-test-user": "alice"}, json={"action": "approve"})
    assert response.status_code == 403, response.text
    graph.ainvoke.assert_not_awaited()


def test_semantic_search_uses_owner_namespace_and_rejects_unscoped_queries(tenants, tmp_path):
    alice, bob, _, a, b = tenants
    service = object.__new__(WorkspaceIndexingService)
    service.embedding_service = SimpleNamespace(embed_query=AsyncMock(return_value=[1.0]))
    service.vector_store = SimpleNamespace(persist_directory=tmp_path)
    with owned_workspaces(alice, (a,)):
        namespace_a = service._storage_namespace(str(a))
        async def search():
            with pytest.raises(WorkspaceAccessDenied):
                await service.search("private source")
            with pytest.raises(WorkspaceAccessDenied):
                await service.search("private source", workspace_root=str(b))
        asyncio.run(search())
        service.embedding_service.embed_query.assert_not_awaited()
        service.vector_store.search = lambda *_args, **kwargs: [
            VectorSearchHit("owned", "owned code", {"workspace_root": kwargs["workspace_root"]}, 0.0),
            VectorSearchHit("foreign", "private code", {"workspace_root": str(b)}, 0.0),
        ]
        hits = asyncio.run(service.search("source", workspace_root=str(a)))
        assert [hit.id for hit in hits] == ["owned"]
        assert hits[0].metadata["workspace_root"] == str(a)
    # Even reassignment of the same directory cannot expose a previous owner's vectors.
    with owned_workspaces(bob, (a,)):
        assert service._storage_namespace(str(a)) != namespace_a


def test_index_upserts_and_deletion_share_only_the_owned_namespace(tenants, tmp_path):
    alice, _, _, a, _ = tenants
    service = object.__new__(WorkspaceIndexingService)
    service.embedding_service = SimpleNamespace(embed_documents=AsyncMock(return_value=[[1.0]]))
    store = SimpleNamespace(
        persist_directory=tmp_path,
        get_file_document_sha256=Mock(return_value=None),
        get_file_chunk_ids=Mock(return_value=[]),
        upsert_chunks=Mock(return_value=1),
        get_workspace_file_paths=Mock(return_value={"alice.py", "deleted.py"}),
        delete_file_chunks=Mock(return_value=1),
    )
    service.vector_store = store
    document = IngestedWorkspaceDocument(str(a), "alice.py", "python", "value = 1\n", "a" * 64, 10, 1)
    ingestion = WorkspaceIngestionResult(str(a), (document,), 1, 0, 10, False, ("alice.py",))
    with owned_workspaces(alice, (a,)):
        namespace = service._storage_namespace(str(a))
        result = asyncio.run(service.index_workspace(ingestion))
    assert result.deleted_files == 1
    assert store.upsert_chunks.call_args.args[0][0].workspace_root == namespace
    store.get_workspace_file_paths.assert_called_once_with(workspace_root=namespace)
    store.delete_file_chunks.assert_called_once_with(workspace_root=namespace, relative_path="deleted.py")
