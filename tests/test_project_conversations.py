"""Durable history, scope boundaries, concurrency and evidence handling."""
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from app.services import project_conversations as store


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEPATH_CONVERSATION_DB", str(tmp_path / "private" / "history.sqlite3"))


def ids():
    return uuid4(), uuid4(), uuid4(), "sha-of-snapshot"


def result():
    return {"summary": "Explanation backed by current evidence", "selected_files": ["src/auth.py"],
            "evidence_links": [{"path": "src/auth.py", "quote": "def login():", "start_line": 1,
                                "end_line": 1, "verification_status": "source_quote_verified"}],
            "findings": [], "gaps": [], "follow_up_questions": []}


def test_create_list_and_history_survive_new_connection():
    owner, project, snap, sha = ids()
    created = store.create_thread(owner, project, snap, sha)
    assert created["revision"] == 0 and not created["stale"]
    thread_id = __import__("uuid").UUID(created["id"])
    reservation, history = store.reserve_turn(owner, project, thread_id, snap, sha, 0)
    assert history == []
    saved = store.finish_turn(owner, project, thread_id, reservation, 0, "Why is login failing?", result(), snap, sha)
    assert saved["revision"] == 1
    assert saved["turns"][0]["report"]["evidence_links"][0]["path"] == "src/auth.py"
    assert store.list_threads(owner, project, snap, sha)[0]["title"] == "Why is login failing?"
    reread = store.get_thread(owner, project, thread_id, snap, sha)
    assert reread["turns"] == saved["turns"]


def test_owner_and_project_boundaries():
    owner, project, snap, sha = ids()
    thread_id = __import__("uuid").UUID(store.create_thread(owner, project, snap, sha)["id"])
    for other_owner, other_project in [(uuid4(), project), (owner, uuid4()), (uuid4(), uuid4())]:
        with pytest.raises(store.ConversationNotFound):
            store.get_thread(other_owner, other_project, thread_id, snap, sha)
        with pytest.raises(store.ConversationNotFound):
            store.reserve_turn(other_owner, other_project, thread_id, snap, sha, 0)
        with pytest.raises(store.ConversationNotFound):
            store.delete_thread(other_owner, other_project, thread_id)
        assert store.list_threads(other_owner, other_project, snap, sha) == []


def test_snapshot_id_and_digest_pin_conversation():
    owner, project, snap, sha = ids()
    thread_id = __import__("uuid").UUID(store.create_thread(owner, project, snap, sha)["id"])
    assert store.get_thread(owner, project, thread_id, uuid4(), sha)["stale"]
    assert store.get_thread(owner, project, thread_id, snap, "different-hash")["stale"]
    with pytest.raises(store.ConversationConflict):
        store.reserve_turn(owner, project, thread_id, snap, "different-hash", 0)
    with pytest.raises(store.ConversationConflict):
        store.reserve_turn(owner, project, thread_id, uuid4(), sha, 0)


def test_concurrent_requests_only_one_reserves_provider_turn():
    owner, project, snap, sha = ids()
    thread_id = __import__("uuid").UUID(store.create_thread(owner, project, snap, sha)["id"])

    def reserve():
        try:
            return store.reserve_turn(owner, project, thread_id, snap, sha, 0)[0]
        except store.ConversationConflict:
            return None

    with ThreadPoolExecutor(max_workers=5) as workers:
        tokens = list(workers.map(lambda _: reserve(), range(5)))
    assert len([token for token in tokens if token]) == 1
    store.cancel_turn(owner, project, thread_id, next(token for token in tokens if token))
    assert store.reserve_turn(owner, project, thread_id, snap, sha, 0)[0]


def test_revision_conflict_no_double_save_and_failed_pending_is_released():
    owner, project, snap, sha = ids()
    thread_id = __import__("uuid").UUID(store.create_thread(owner, project, snap, sha)["id"])
    token, _ = store.reserve_turn(owner, project, thread_id, snap, sha, 0)
    with pytest.raises(store.ConversationConflict):
        store.finish_turn(owner, project, thread_id, "invalid", 0, "Question", result(), snap, sha)
    store.cancel_turn(owner, project, thread_id, token)
    token, _ = store.reserve_turn(owner, project, thread_id, snap, sha, 0)
    store.finish_turn(owner, project, thread_id, token, 0, "Question", result(), snap, sha)
    with pytest.raises(store.ConversationConflict):
        store.reserve_turn(owner, project, thread_id, snap, sha, 0)


def test_history_budget_and_continuity_use_bounded_questions_only():
    turns = [
        {"question": "Find auth refresh", "report": {"summary": "A" * 6000,
          "selected_files": ["auth/routes.py", "auth/session.py"]}},
        {"question": "Where does this get called?", "report": {"summary": "Calls session",
          "selected_files": ["auth/session.py", "tests/test_auth.py"]}},
    ]
    query = store.prior_retrieval_query("What about failure retries?", turns)
    assert "auth refresh" in query and "failure retries" in query
    assert store.continuity_paths(turns) == ["auth/session.py", "tests/test_auth.py", "auth/routes.py"]
    prompt = store.bounded_history_prompt(turns)
    assert "Previous model summary (not verified evidence)" in prompt
    assert len(prompt) < 5601


def test_delete_cascades_turns_and_is_owner_scoped():
    owner, project, snap, sha = ids()
    thread_id = __import__("uuid").UUID(store.create_thread(owner, project, snap, sha)["id"])
    token, _ = store.reserve_turn(owner, project, thread_id, snap, sha, 0)
    store.finish_turn(owner, project, thread_id, token, 0, "Inspect auth", result(), snap, sha)
    store.delete_thread(owner, project, thread_id)
    with pytest.raises(store.ConversationNotFound):
        store.get_thread(owner, project, thread_id, snap, sha)
    assert store.list_threads(owner, project, snap, sha) == []


def test_stored_report_size_limit():
    owner, project, snap, sha = ids()
    thread_id = __import__("uuid").UUID(store.create_thread(owner, project, snap, sha)["id"])
    token, _ = store.reserve_turn(owner, project, thread_id, snap, sha, 0)
    with pytest.raises(store.ConversationLimit):
        store.finish_turn(owner, project, thread_id, token, 0, "Why?", {"summary": "x" * 60_000}, snap, sha)
    store.cancel_turn(owner, project, thread_id, token)
    assert store.get_thread(owner, project, thread_id, snap, sha)["revision"] == 0


def test_absolute_database_path_required(monkeypatch):
    monkeypatch.setenv("FORGEPATH_CONVERSATION_DB", "relative-file.sqlite3")
    with pytest.raises(store.ConversationStorageError):
        store.list_threads(*ids())