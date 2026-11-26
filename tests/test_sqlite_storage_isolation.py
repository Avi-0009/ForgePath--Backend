"""Real temporary SQLite databases preserve integrity and tenant isolation."""

import asyncio
import sqlite3
from uuid import uuid4

from app.core.config import settings
from app.core.workspace_access import owned_workspaces
from app.services import local_context
from app.services import project_conversations as conversations


def _quick_check(path) -> tuple[str, str]:
    with sqlite3.connect(path) as database:
        integrity = database.execute("PRAGMA quick_check").fetchone()[0]
        journal_mode = database.execute("PRAGMA journal_mode").fetchone()[0]
    return integrity, journal_mode


def test_local_instruction_sqlite_is_separate_for_each_owner(tmp_path, monkeypatch):
    data_root = tmp_path / "local-data"
    project = tmp_path / "shared-project"
    project.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(data_root))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_root))
    monkeypatch.setattr(settings, "workspace_allowed_roots", str(project))
    alice, bob = uuid4(), uuid4()

    async def save_and_read(user_id, instructions):
        with owned_workspaces(user_id, (project,)):
            saved = await local_context.save_project_instructions(
                project_root=project,
                instructions=instructions,
            )
            database_path = local_context._database_path()
            loaded = await local_context.get_project_instructions(
                project_root=project,
            )
            return saved, loaded, database_path

    alice_saved, alice_loaded, alice_db = asyncio.run(
        save_and_read(alice, "Alice private instructions")
    )
    bob_saved, bob_loaded, bob_db = asyncio.run(
        save_and_read(bob, "Bob private instructions")
    )

    assert alice_db != bob_db
    assert alice_saved.instructions == alice_loaded.instructions
    assert bob_saved.instructions == bob_loaded.instructions
    assert alice_loaded.instructions == "Alice private instructions"
    assert bob_loaded.instructions == "Bob private instructions"
    assert _quick_check(alice_db) == ("ok", "wal")
    assert _quick_check(bob_db) == ("ok", "wal")

    with owned_workspaces(alice, (project,)):
        reread = asyncio.run(local_context.get_project_instructions(
            project_root=project,
        ))
    assert reread is not None
    assert reread.instructions == "Alice private instructions"


def test_conversation_sqlite_integrity_and_owner_scoped_purge(tmp_path, monkeypatch):
    database_path = tmp_path / "private" / "conversations.sqlite3"
    monkeypatch.setenv("FORGEPATH_CONVERSATION_DB", str(database_path))
    alice, bob, project, snapshot = uuid4(), uuid4(), uuid4(), uuid4()
    snapshot_sha = "snapshot-sha"

    alice_thread = conversations.create_thread(
        alice, project, snapshot, snapshot_sha,
    )
    bob_thread = conversations.create_thread(
        bob, project, snapshot, snapshot_sha,
    )

    assert [item["id"] for item in conversations.list_threads(
        alice, project, snapshot, snapshot_sha,
    )] == [alice_thread["id"]]
    assert [item["id"] for item in conversations.list_threads(
        bob, project, snapshot, snapshot_sha,
    )] == [bob_thread["id"]]
    assert _quick_check(database_path) == ("ok", "wal")

    conversations.purge_project_conversations(alice, project)

    assert conversations.list_threads(
        alice, project, snapshot, snapshot_sha,
    ) == []
    assert [item["id"] for item in conversations.list_threads(
        bob, project, snapshot, snapshot_sha,
    )] == [bob_thread["id"]]
    assert _quick_check(database_path) == ("ok", "wal")
