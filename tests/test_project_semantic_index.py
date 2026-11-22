"""Semantic project indexes are private, version-pinned and opt-in."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.routes import project_semantic_index as routes
from app.services import project_semantic_index as indexing


class FakeDB:
    def __init__(self, owned_project):
        self.owned_project = owned_project

    async def execute(self, _statement):
        value = self.owned_project

        class Result:
            def scalar_one_or_none(self):
                return value

        return Result()


def test_foreign_project_rejected_before_snapshot_or_provider(monkeypatch):
    def forbidden(*_args):
        raise AssertionError("A foreign snapshot must never be read")

    monkeypatch.setattr(routes, "read_snapshot_manifest", forbidden)
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.get_project_semantic_status(
            uuid4(), SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
            SimpleNamespace(id=uuid4()), FakeDB(None),
        ))
    assert error.value.status_code == 404


def test_disabled_status_never_embeds(monkeypatch):
    snapshot = uuid4()
    monkeypatch.setattr(routes, "read_snapshot_manifest", lambda *_: SimpleNamespace(
        snapshot_id=snapshot, file_count=3, sha256="abc",
    ))
    project = uuid4()
    result = asyncio.run(routes.get_project_semantic_status(
        project, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(workspace_indexing_service=None))),
        SimpleNamespace(id=uuid4()), FakeDB(project),
    ))
    assert result["status"] == "disabled"
    assert result["snapshot_id"] == str(snapshot)


def test_confirmation_must_match_current_snapshot(monkeypatch):
    project = uuid4()
    snapshot = uuid4()
    monkeypatch.setattr(routes, "read_snapshot_manifest", lambda *_: SimpleNamespace(
        snapshot_id=snapshot, file_count=1, sha256="abc",
    ))
    monkeypatch.setattr(routes, "read_snapshot_sources", lambda *_: [("app/a.py", "print('ok')")])
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.create_project_semantic_index(
            project, routes.ConfirmIndexRequest(expected_snapshot_id=uuid4(), confirm_indexing=True),
            SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
            SimpleNamespace(id=uuid4()), FakeDB(project),
        ))
    assert error.value.status_code == 409


def test_successful_index_marks_receipt_only_after_upsert(tmp_path):
    user, project, snapshot = uuid4(), uuid4(), uuid4()

    class Store:
        persist_directory = tmp_path
        embedding_model = "fake:local"

        def get_workspace_file_paths(self, *, workspace_root):
            return {"app/a.py"}

    class FakeService:
        vector_store = Store()

        async def index_documents(self, documents):
            assert len(documents) == 1
            return SimpleNamespace(chunks_processed=2, vectors_upserted=2,
                                   unchanged_files_skipped=0)

    service = FakeService()
    result = asyncio.run(indexing.index_project_snapshot(
        service, user, project, snapshot, "digest", [("app/a.py", "print('hi')")],
    ))
    assert result["vectors_upserted"] == 2
    assert indexing.is_snapshot_index_ready(service, user, project, snapshot, "digest", 1)
    assert not indexing.is_snapshot_index_ready(service, user, project, uuid4(), "digest", 1)
    assert not indexing.is_snapshot_index_ready(service, user, project, snapshot, "different", 1)


def test_failed_embedding_never_creates_receipt(tmp_path):
    user, project, snapshot = uuid4(), uuid4(), uuid4()

    class FakeService:
        vector_store = SimpleNamespace(persist_directory=tmp_path, embedding_model="fake:local")

        async def index_documents(self, _documents):
            raise RuntimeError("provider failed")

    service = FakeService()
    with pytest.raises(RuntimeError):
        asyncio.run(indexing.index_project_snapshot(
            service, user, project, snapshot, "digest", [("app/a.py", "print('hi')")],
        ))
    assert not indexing.is_snapshot_index_ready(service, user, project, snapshot, "digest", 1)
