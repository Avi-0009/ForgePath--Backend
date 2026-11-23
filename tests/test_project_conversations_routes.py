"""API-level behavior: scope checks, free preview, version pinning, and paid calls."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.routes import project_conversations as routes


@pytest.fixture(autouse=True)
def temp_history(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEPATH_CONVERSATION_DB", str(tmp_path / "private" / "history.sqlite3"))


@pytest.fixture
def ready(monkeypatch):
    owner = SimpleNamespace(id=uuid4())
    project = uuid4()
    manifest = SimpleNamespace(snapshot_id=uuid4(), sha256="snapshot-hash", file_count=2)
    sources = [
        ("src/auth.py", "def login():\n    return validate_token()\n"),
        ("src/session.py", "def validate_token():\n    return True\n"),
    ]

    async def owned(project_id, user, _db):
        if project_id != project or user.id != owner.id:
            raise HTTPException(404, "Project not found")
        return manifest, sources

    monkeypatch.setattr(routes, "_owned_sources", owned)
    monkeypatch.setattr(routes, "read_snapshot_manifest", lambda *_: manifest)
    return owner, project, manifest, sources


def _request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(workspace_indexing_service=None)))


def test_create_list_get_requires_owner_and_no_ai(monkeypatch, ready):
    owner, project, manifest, _ = ready

    async def forbidden(**_kwargs):
        raise AssertionError("No AI call is allowed for history operations")

    monkeypatch.setattr(routes, "generate_completion", forbidden)
    payload = routes.NewConversation(expected_snapshot_id=manifest.snapshot_id)
    created = asyncio.run(routes.conversations_create(project, payload, owner, object()))
    thread_id = __import__("uuid").UUID(created["id"])
    listed = asyncio.run(routes.conversations_list(project, owner, object()))
    assert len(listed["items"]) == 1 and listed["ai_calls"] == 0
    fetched = asyncio.run(routes.conversations_get(project, thread_id, owner, object()))
    assert fetched["turns"] == [] and fetched["revision"] == 0
    with pytest.raises(HTTPException) as denied:
        asyncio.run(routes.conversations_get(project, thread_id, SimpleNamespace(id=uuid4()), object()))
    assert denied.value.status_code == 404


def test_preview_free_and_reuses_previous_question_context(monkeypatch, ready):
    owner, project, manifest, sources = ready
    created = asyncio.run(routes.conversations_create(
        project, routes.NewConversation(expected_snapshot_id=manifest.snapshot_id), owner, object()))
    thread_id = __import__("uuid").UUID(created["id"])

    async def forbidden(**_kwargs):
        raise AssertionError("No AI call is allowed in preview")

    monkeypatch.setattr(routes, "generate_completion", forbidden)
    preview = asyncio.run(routes.conversations_preview(
        project, thread_id, routes.ConversationPreviewRequest(
            question="How is login validated?", max_files=2, expected_revision=0), owner, object()))
    assert preview["ai_calls"] == 0
    assert preview["prior_turns"] == 0
    assert preview["snapshot_id"] == str(manifest.snapshot_id)
    assert preview["selected_files"]
    assert all("excerpt" not in file for file in preview["selected_files"])


def test_one_confirmed_model_call_is_saved_and_history_returned(monkeypatch, ready):
    owner, project, manifest, _ = ready
    created = asyncio.run(routes.conversations_create(
        project, routes.NewConversation(expected_snapshot_id=manifest.snapshot_id), owner, object()))
    thread_id = __import__("uuid").UUID(created["id"])
    prompts = []

    async def completion(**kwargs):
        prompts.append(kwargs["prompt"])
        return SimpleNamespace(text=json.dumps({
            "summary": "Login calls validate_token", "findings": [],
            "evidence_links": [{"path": "src/auth.py", "quote": "return validate_token()", "explanation": "Call site"}],
            "gaps": ["No runtime execution"], "follow_up_questions": ["Where does validate_token get tested?"],
        }))

    monkeypatch.setattr(routes, "generate_completion", completion)
    first = routes.ConversationRunRequest(
        question="How does login validate tokens?", max_files=2,
        expected_revision=0, expected_snapshot_id=manifest.snapshot_id,
    )
    result = asyncio.run(routes.conversations_message(project, thread_id, first, _request(), owner, object()))
    assert len(prompts) == 1 and result["ai_calls"] == 1 and result["revision"] == 1
    assert result["turn"]["report"]["evidence_links"][0]["verification_status"] == "source_quote_verified"
    saved = asyncio.run(routes.conversations_get(project, thread_id, owner, object()))
    assert saved["revision"] == 1 and len(saved["turns"]) == 1
    second = routes.ConversationRunRequest(
        question="Where is that function defined?", max_files=2,
        expected_revision=1, expected_snapshot_id=manifest.snapshot_id,
    )
    asyncio.run(routes.conversations_message(project, thread_id, second, _request(), owner, object()))
    assert "Earlier conversation" in prompts[-1]
    assert "How does login validate tokens?" in prompts[-1]
    assert "LATEST QUESTION: Where is that function defined?" in prompts[-1]


def test_stale_thread_rejected_before_model_and_kept_readable(monkeypatch, ready):
    owner, project, manifest, _ = ready
    created = asyncio.run(routes.conversations_create(
        project, routes.NewConversation(expected_snapshot_id=manifest.snapshot_id), owner, object()))
    thread_id = __import__("uuid").UUID(created["id"])
    manifest.snapshot_id = uuid4()

    async def forbidden(**_kwargs):
        raise AssertionError("Do not call provider for a stale snapshot")

    monkeypatch.setattr(routes, "generate_completion", forbidden)
    fetched = asyncio.run(routes.conversations_get(project, thread_id, owner, object()))
    assert fetched["stale"] is True
    payload = routes.ConversationRunRequest(
        question="Inspect token validation", expected_snapshot_id=manifest.snapshot_id,
        expected_revision=0,
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.conversations_message(project, thread_id, payload, _request(), owner, object()))
    assert error.value.status_code == 409


def test_invalid_ai_result_does_not_save_turn_and_reservation_clears(monkeypatch, ready):
    owner, project, manifest, _ = ready
    created = asyncio.run(routes.conversations_create(
        project, routes.NewConversation(expected_snapshot_id=manifest.snapshot_id), owner, object()))
    thread_id = __import__("uuid").UUID(created["id"])

    async def invalid(**_kwargs):
        return SimpleNamespace(text="not valid json")

    monkeypatch.setattr(routes, "generate_completion", invalid)
    payload = routes.ConversationRunRequest(
        question="Inspect token validation", expected_snapshot_id=manifest.snapshot_id,
        expected_revision=0, max_files=2,
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.conversations_message(project, thread_id, payload, _request(), owner, object()))
    assert error.value.status_code == 502
    fetched = asyncio.run(routes.conversations_get(project, thread_id, owner, object()))
    assert fetched["turns"] == [] and fetched["revision"] == 0
    # A failed provider response must release the reservation.
    reservation, _ = routes.reserve_turn(owner.id, project, thread_id, manifest.snapshot_id, manifest.sha256, 0)
    routes.cancel_turn(owner.id, project, thread_id, reservation)


def test_existing_revision_conflict_skips_paid_completion(monkeypatch, ready):
    owner, project, manifest, _ = ready
    created = asyncio.run(routes.conversations_create(
        project, routes.NewConversation(expected_snapshot_id=manifest.snapshot_id), owner, object()))
    thread_id = __import__("uuid").UUID(created["id"])
    lock, _ = routes.reserve_turn(owner.id, project, thread_id, manifest.snapshot_id, manifest.sha256, 0)

    async def forbidden(**_kwargs):
        raise AssertionError("Do not call provider when another request is active")

    monkeypatch.setattr(routes, "generate_completion", forbidden)
    payload = routes.ConversationRunRequest(
        question="Inspect auth handling", expected_snapshot_id=manifest.snapshot_id,
        expected_revision=0,
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.conversations_message(project, thread_id, payload, _request(), owner, object()))
    assert error.value.status_code == 409
    routes.cancel_turn(owner.id, project, thread_id, lock)