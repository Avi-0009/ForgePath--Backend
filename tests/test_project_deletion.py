"""Real temporary SQLite/source deletion with mocked PostgreSQL and vector clients."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from app.api.v1.routes import projects
from app.core.config import settings
from app.services import project_conversations as conversations
from app.services import project_storage_cleanup as cleanup
from app.services.project_snapshots import SourceSnapshot, store_snapshot
from app.services.vector_store import VectorStoreError


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEPATH_CONVERSATION_DB", str(tmp_path / "conversations.sqlite3"))
    monkeypatch.setattr(settings, "project_snapshot_directory", str(tmp_path / "sources"))
    monkeypatch.setattr(cleanup, "get_vector_store_path", lambda: tmp_path / "vectors")
    return tmp_path


def test_purge_cascades_turns_preserves_other_projects_and_blocks_late_writes(storage):
    owner, project, other, snapshot = (uuid4() for _ in range(4))
    thread = UUID(conversations.create_thread(owner, project, snapshot, "sha")["id"])
    token, _ = conversations.reserve_turn(owner, project, thread, snapshot, "sha", 0)
    conversations.finish_turn(owner, project, thread, token, 0, "private question", {"summary": "private quote"}, snapshot, "sha")
    pending, _ = conversations.reserve_turn(owner, project, thread, snapshot, "sha", 1)
    conversations.create_thread(owner, other, snapshot, "sha")
    foreign_owner = uuid4()
    conversations.create_thread(foreign_owner, project, snapshot, "sha")
    for _ in range(2):
        conversations.purge_project_conversations(owner, project)
    with conversations._connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM forgepath_conversation_turns").fetchone()[0] == 0
    assert conversations.list_threads(owner, project, snapshot, "sha") == []
    assert len(conversations.list_threads(owner, other, snapshot, "sha")) == 1
    assert len(conversations.list_threads(foreign_owner, project, snapshot, "sha")) == 1
    with pytest.raises(conversations.ConversationNotFound):
        conversations.create_thread(owner, project, snapshot, "sha")
    with pytest.raises(conversations.ConversationNotFound):
        conversations.finish_turn(owner, project, thread, pending, 1, "late", {}, snapshot, "sha")


def test_delete_unindexed_snapshot_without_embedding_provider(storage, monkeypatch):
    owner, project = uuid4(), uuid4()
    manifest = store_snapshot(owner, project, SourceSnapshot(files=[{"relative_path": "src/main.py", "content": "print(1)"}]))
    conversations.create_thread(owner, project, manifest.snapshot_id, manifest.sha256)
    class DB:
        deleted = False
        committed = False
        async def execute(self, _):
            return SimpleNamespace(scalar_one_or_none=lambda: SimpleNamespace(id=project))
        async def delete(self, _):
            assert not (storage / "sources" / str(owner) / str(project)).exists()
            assert conversations.list_threads(owner, project, manifest.snapshot_id, manifest.sha256) == []
            self.deleted = True
        async def commit(self):
            self.committed = True
    monkeypatch.setattr(cleanup.chromadb, "PersistentClient", Mock(side_effect=AssertionError("No vector DB should be opened")))
    db = DB()
    asyncio.run(projects.delete_project(project, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())), SimpleNamespace(id=owner), db))
    assert db.deleted and db.committed


def test_disabled_provider_can_delete_existing_vectors(storage, monkeypatch):
    directory = storage / "vectors"
    directory.mkdir()
    (directory / "chroma.sqlite3").touch()
    collection = Mock()
    collection.get.return_value = {"ids": ["owned-vector"]}
    client = Mock()
    client.get_collection.return_value = collection
    factory = Mock(return_value=client)
    monkeypatch.setattr(cleanup.chromadb, "PersistentClient", factory)
    with cleanup.deletion_index_service(None) as service:
        assert service.vector_store.delete_workspace("owned-namespace") == 1
    client.get_collection.assert_called_once_with(cleanup.COLLECTION_NAME, embedding_function=None)
    collection.get.assert_called_once_with(where={"workspace_root": "owned-namespace"}, include=[])
    collection.delete.assert_called_once_with(ids=["owned-vector"])
    client.close.assert_called_once()


def test_sqlite_failure_retains_postgres_row_and_snapshot(storage, monkeypatch):
    owner, project = uuid4(), uuid4()
    store_snapshot(owner, project, SourceSnapshot(files=[{"relative_path": "main.py", "content": "print(1)"}]))
    class DB:
        async def execute(self, _):
            return SimpleNamespace(scalar_one_or_none=lambda: SimpleNamespace(id=project))
        async def delete(self, _):
            pytest.fail("Must retain project until cleanup completes")
    def fail(*_):
        raise conversations.ConversationStorageError("unavailable")
    monkeypatch.setattr(projects, "purge_project_conversations", fail)
    with pytest.raises(HTTPException) as error:
        asyncio.run(projects.delete_project(project, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())), SimpleNamespace(id=owner), DB()))
    assert error.value.status_code == 503
    assert (storage / "sources" / str(owner) / str(project)).exists()


def test_missing_database_with_artifacts_fails_closed(storage):
    directory = storage / "vectors"
    directory.mkdir()
    (directory / "orphan-index").write_text("data", encoding="utf-8")
    with pytest.raises(VectorStoreError):
        with cleanup.deletion_index_service(None):
            pytest.fail("Must not silently replace a missing vector DB")
