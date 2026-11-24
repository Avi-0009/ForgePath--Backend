"""Offline checks for user-scoped semantic index cleanup."""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.routes import projects
from app.core.config import settings
from app.services.project_semantic_index import (
    remove_project_semantic_indexes, snapshot_vector_namespace,
)
from app.services.project_snapshots import delete_project_snapshot_storage


def _snapshot_dirs(tmp_path, monkeypatch, user, project, *versions):
    root = tmp_path / "snapshots"
    root.mkdir(mode=0o700, exist_ok=True)
    monkeypatch.setattr(settings, "project_snapshot_directory", str(root))
    parent = root / str(user) / str(project)
    for version in versions:
        (parent / str(version)).mkdir(parents=True, mode=0o700)
    return parent


def test_replacement_prunes_old_vectors_and_keeps_current_index(tmp_path, monkeypatch):
    user, project, previous, current = (uuid4() for _ in range(4))
    _snapshot_dirs(tmp_path, monkeypatch, user, project, previous, current)
    deleted = []

    class Store:
        persist_directory = tmp_path / "chroma"
        def delete_workspace(self, namespace):
            deleted.append(namespace)
            return 2

    service = SimpleNamespace(vector_store=Store())
    for version in (previous, current):
        path = Store.persist_directory / "project-index-receipts" / str(user) / str(project) / f"{version}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"snapshot_id": str(version)}), encoding="utf-8")

    summary = remove_project_semantic_indexes(service, user, project, keep_snapshot_id=current)
    assert summary == {"deleted_vectors": 2, "deleted_receipts": 1}
    assert deleted == [snapshot_vector_namespace(service, user, project, previous)]
    assert not (Store.persist_directory / "project-index-receipts" / str(user) / str(project) / f"{previous}.json").exists()
    assert (Store.persist_directory / "project-index-receipts" / str(user) / str(project) / f"{current}.json").exists()


def test_partial_index_without_receipt_is_still_removed(tmp_path, monkeypatch):
    user, project, version = uuid4(), uuid4(), uuid4()
    _snapshot_dirs(tmp_path, monkeypatch, user, project, version)
    deleted = []
    service = SimpleNamespace(vector_store=SimpleNamespace(
        persist_directory=tmp_path / "chroma",
        delete_workspace=lambda namespace: deleted.append(namespace) or 1,
    ))
    summary = remove_project_semantic_indexes(service, user, project)
    assert summary["deleted_vectors"] == 1
    assert deleted == [snapshot_vector_namespace(service, user, project, version)]


def test_source_storage_delete_is_scoped_to_one_project(tmp_path, monkeypatch):
    user, project_a, project_b = uuid4(), uuid4(), uuid4()
    folder_a = _snapshot_dirs(tmp_path, monkeypatch, user, project_a, uuid4())
    folder_b = _snapshot_dirs(tmp_path, monkeypatch, user, project_b, uuid4())
    delete_project_snapshot_storage(user, project_a)
    assert not folder_a.exists()
    assert folder_b.exists()


def test_foreign_project_delete_does_not_purge(monkeypatch):
    class Database:
        async def execute(self, _statement):
            return SimpleNamespace(scalar_one_or_none=lambda: None)

    monkeypatch.setattr(projects, "remove_project_semantic_indexes", lambda *_: pytest.fail("Must not purge foreign project"))
    with pytest.raises(HTTPException) as error:
        asyncio.run(projects.delete_project(
            uuid4(), SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
            SimpleNamespace(id=uuid4()), Database(),
        ))
    assert error.value.status_code == 404


def test_delete_refuses_when_index_store_is_unavailable(tmp_path, monkeypatch):
    from app.services import project_storage_cleanup
    broken_store = tmp_path / "missing-db"
    broken_store.mkdir()
    (broken_store / "orphan-index").write_text("incomplete", encoding="utf-8")
    monkeypatch.setattr(project_storage_cleanup, "get_vector_store_path", lambda: broken_store)
    user, project, version = uuid4(), uuid4(), uuid4()
    folder = _snapshot_dirs(tmp_path, monkeypatch, user, project, version)

    class Database:
        async def execute(self, _statement):
            return SimpleNamespace(scalar_one_or_none=lambda: SimpleNamespace(id=project))
        async def delete(self, _project):
            pytest.fail("DB project must be retained when cleanup is unavailable")

    with pytest.raises(HTTPException) as error:
        asyncio.run(projects.delete_project(
            project, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(workspace_indexing_service=None))),
            SimpleNamespace(id=user), Database(),
        ))
    assert error.value.status_code == 503
    assert folder.exists()
