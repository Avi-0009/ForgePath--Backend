"""Contract tests for the version-pinned, owner-checked context investigation."""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.routes import project_context_investigation as routes


SOURCES = [
    ("app/api/routes/auth.py", "from app.auth.session import verify_token\ndef login():\n    return verify_token()\n"),
    ("app/auth/session.py", "def verify_token():\n    return 'verified'\n"),
    ("tests/test_auth.py", "from app.auth.session import verify_token\ndef test_auth():\n    assert verify_token()\n"),
]


def _setup(monkeypatch):
    snapshot_id, project_id, user_id = uuid4(), uuid4(), uuid4()
    manifest = SimpleNamespace(snapshot_id=snapshot_id, sha256="sha", file_count=len(SOURCES))
    requested = []

    async def owned(project, user, _db):
        requested.append((project, user.id))
        if project != project_id or user.id != user_id:
            raise HTTPException(404, "Project not found")
        return manifest, SOURCES

    monkeypatch.setattr(routes, "_owned_sources", owned)
    monkeypatch.setattr(routes, "read_snapshot_manifest", lambda *_: manifest)
    return project_id, SimpleNamespace(id=user_id), manifest, requested


def _request(service=None):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(workspace_indexing_service=service)))


def test_preview_is_free_owned_and_does_not_disclose_source(monkeypatch):
    project, user, _, requested = _setup(monkeypatch)

    async def forbidden(**_kwargs):
        raise AssertionError("Preview must not invoke the model")

    monkeypatch.setattr(routes, "generate_completion", forbidden)
    payload = routes.ContextPreviewRequest(question="How does the auth session work?", max_files=3)
    result = asyncio.run(routes.preview_investigation_context(project, payload, user, object()))
    assert result["ai_calls"] == 0
    assert result["snapshot_id"]
    assert result["selected_files"]
    assert all("excerpt" not in item for item in result["selected_files"])
    assert requested == [(project, user.id)]
    with pytest.raises(HTTPException) as exception:
        asyncio.run(routes.preview_investigation_context(uuid4(), payload, user, object()))
    assert exception.value.status_code == 404


def test_run_rejects_snapshot_mismatch_before_ai(monkeypatch):
    project, user, _, _ = _setup(monkeypatch)

    async def forbidden(**_kwargs):
        raise AssertionError("Stale snapshot must not use AI")

    monkeypatch.setattr(routes, "generate_completion", forbidden)
    payload = routes.ContextRunRequest(
        question="Where do we verify the auth session?", max_files=3,
        expected_snapshot_id=uuid4(),
    )
    with pytest.raises(HTTPException) as exception:
        asyncio.run(routes.run_context_investigation(project, payload, _request(), user, object()))
    assert exception.value.status_code == 409


def test_run_accepts_only_evidenced_citations_and_one_completion(monkeypatch):
    project, user, manifest, _ = _setup(monkeypatch)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({
            "summary": "Token checking is reached through the login route.",
            "findings": [
                {"path": "app/auth/session.py", "title": "Inspect stub", "severity": "low",
                 "evidence": "return 'verified'", "explanation": "The helper is only a stub."},
                {"path": "private/secrets.py", "title": "Invented", "severity": "high",
                 "evidence": "secret", "explanation": "Not in context"},
            ],
            "evidence_links": [
                {"path": "app/auth/session.py", "quote": "return 'verified'", "explanation": "Source stub"},
                {"path": "private/secrets.py", "quote": "secret", "explanation": "Invented"},
            ],
            "gaps": ["No runtime request was executed."],
            "follow_up_questions": ["What calls login?"],
        }))

    monkeypatch.setattr(routes, "generate_completion", completion)
    payload = routes.ContextRunRequest(
        question="Where do we verify the auth session?", max_files=3,
        expected_snapshot_id=manifest.snapshot_id,
    )
    result = asyncio.run(routes.run_context_investigation(project, payload, _request(), user, object()))
    assert len(calls) == 1
    assert result["ai_calls"] == result["completion_calls"] == 1
    assert result["embedding_query_calls"] == 0
    assert len(result["findings"]) == 1
    assert len(result["evidence_links"]) == 1
    assert result["evidence_links"][0]["verification_status"] == "source_quote_verified"
    assert result["context_files"] and result["gaps"] and result["follow_up_questions"]
    assert "BEGIN_SOURCE" in calls[0]["prompt"]


def test_semantic_opt_in_requires_current_owner_scoped_index(monkeypatch):
    project, user, manifest, _ = _setup(monkeypatch)
    monkeypatch.setattr(routes, "is_snapshot_index_ready", lambda *_: False)
    payload = routes.ContextRunRequest(
        question="Check auth session handling", max_files=3,
        expected_snapshot_id=manifest.snapshot_id, use_semantic_index=True,
    )
    service = SimpleNamespace()
    with pytest.raises(HTTPException) as exception:
        asyncio.run(routes.run_context_investigation(project, payload, _request(service), user, object()))
    assert exception.value.status_code == 409


def test_semantic_results_must_match_namespace_and_exact_source(monkeypatch):
    project, user, manifest, _ = _setup(monkeypatch)
    monkeypatch.setattr(routes, "is_snapshot_index_ready", lambda *_: True)
    monkeypatch.setattr(routes, "snapshot_vector_namespace", lambda *_: "owner/project/snapshot")

    class Service:
        async def search(self, query, *, top_k, workspace_root):
            assert query == "Find token verification"
            assert top_k == 8 and workspace_root == "owner/project/snapshot"
            return [
                SimpleNamespace(metadata={"workspace_root": workspace_root, "relative_path": "app/auth/session.py"}, document="return 'verified'"),
                SimpleNamespace(metadata={"workspace_root": "other/project", "relative_path": "app/auth/session.py"}, document="return 'verified'"),
                SimpleNamespace(metadata={"workspace_root": workspace_root, "relative_path": "app/auth/session.py"}, document="SECRET_DOES_NOT_EXIST"),
            ]

    hits = asyncio.run(routes._verified_semantic_hits(
        _request(Service()), user, project, manifest, SOURCES, "Find token verification",
    ))
    assert hits == [{"relative_path": "app/auth/session.py", "content": "return 'verified'"}]


def test_invalid_provider_json_does_not_create_report(monkeypatch):
    project, user, manifest, _ = _setup(monkeypatch)

    async def invalid(**_kwargs):
        return SimpleNamespace(text="I found three issues, trust me!")

    monkeypatch.setattr(routes, "generate_completion", invalid)
    payload = routes.ContextRunRequest(
        question="Check auth session handling", expected_snapshot_id=manifest.snapshot_id,
    )
    with pytest.raises(HTTPException) as exception:
        asyncio.run(routes.run_context_investigation(project, payload, _request(), user, object()))
    assert exception.value.status_code == 502


def test_snapshot_replaced_while_model_runs_does_not_return_stale_report(monkeypatch):
    project, user, manifest, _ = _setup(monkeypatch)

    async def completion(**_kwargs):
        return SimpleNamespace(text=json.dumps({"summary": "Checked source", "findings": []}))

    monkeypatch.setattr(routes, "generate_completion", completion)
    monkeypatch.setattr(routes, "read_snapshot_manifest", lambda *_: SimpleNamespace(
        snapshot_id=uuid4(), sha256="new-version",
    ))
    payload = routes.ContextRunRequest(
        question="Check auth session handling", expected_snapshot_id=manifest.snapshot_id,
    )
    with pytest.raises(HTTPException) as exception:
        asyncio.run(routes.run_context_investigation(project, payload, _request(), user, object()))
    assert exception.value.status_code == 409