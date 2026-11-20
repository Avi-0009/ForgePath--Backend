"""Question-first investigation must be bounded, owned and explicit about AI use."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.routes import project_investigations as routes
from app.services.project_investigation_context import select_investigation_files


class FakeDB:
    def __init__(self, result):
        self.result = result

    async def execute(self, _statement):
        result = self.result

        class Response:
            def scalar_one_or_none(self):
                return result

        return Response()


def _sources(monkeypatch):
    snapshot_id = uuid4()
    monkeypatch.setattr(routes, "read_snapshot_manifest", lambda *_: SimpleNamespace(snapshot_id=snapshot_id))
    monkeypatch.setattr(routes, "read_snapshot_sources", lambda *_: [
        ("app/auth.py", "def refresh_token():\n    return False"),
        ("app/tasks.py", "def create_task():\n    return refresh_token()"),
        ("docs/notes.md", "Documentation unrelated to login"),
    ])


def test_ranker_is_deterministic_bounded_and_can_select_multiple_files():
    files = [(f"app/file{i}.py", "refresh_token()" if i % 2 else "print('ok')") for i in range(20)]
    one = select_investigation_files(files, "refresh_token", 10)
    two = select_investigation_files(files, "refresh_token", 10)
    assert one == two
    assert 1 <= len(one) <= 10
    assert all(len(item["excerpt"]) <= 3600 for item in one)
    assert all(item["path"].endswith(".py") for item in one)


def test_authentication_question_prefers_real_auth_implementations():
    sources = [
        ("alembic/versions/add_auth_tokens.py", "# auth token session " * 200),
        ("app/api/v1/routes/auth.py", "from app.api.dependencies import get_current_user\n"
         "def login():\n    return create_access_token()"),
        ("app/api/dependencies.py", "def get_current_user():\n    return verify_jwt()"),
        ("app/services/auth_session.py", "def refresh_session():\n    return rotate_refresh_token()"),
        ("app/services/other.py", "def unrelated():\n    return True"),
    ]
    found = select_investigation_files(
        sources, "Find authentication validating access tokens refreshing expired sessions", 3,
    )
    paths = [item["path"] for item in found]
    assert "app/api/v1/routes/auth.py" in paths
    assert "app/services/auth_session.py" in paths
    assert "alembic/versions/add_auth_tokens.py" not in paths


def test_preview_requires_ownership_and_never_calls_ai(monkeypatch):
    async def forbidden(**_kwargs):
        raise AssertionError("Preview must not call AI")

    monkeypatch.setattr(routes, "generate_completion", forbidden)
    _sources(monkeypatch)
    project_id = uuid4()
    payload = routes.QuestionInvestigationRequest(question="refresh token", max_files=3)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes.preview_project_investigation(project_id, payload, SimpleNamespace(id=uuid4()), FakeDB(None)))
    assert exc.value.status_code == 404

    result = asyncio.run(routes.preview_project_investigation(
        project_id, payload, SimpleNamespace(id=uuid4()), FakeDB(project_id)))
    assert result["ai_calls"] == 0
    assert result["total_files"] == 3
    assert len(result["selected_files"]) >= 2


def test_question_review_uses_one_ai_call_and_rejects_unsupported_findings(monkeypatch):
    _sources(monkeypatch)
    calls = []

    async def fake_generate_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=(
            '{"summary":"Checked selected excerpts.","findings":['
            '{"path":"app/auth.py","title":"Check refresh",'
            '"severity":"medium","evidence":"return False",'
            '"explanation":"Refresh always fails."},'
            '{"path":"not-real.py","title":"Invented",'
            '"severity":"high","evidence":"fake",'
            '"explanation":"Not in source."}]}'
        ))

    monkeypatch.setattr(routes, "generate_completion", fake_generate_completion)
    project_id = uuid4()
    result = asyncio.run(routes.investigate_project_question(
        project_id, routes.QuestionInvestigationRequest(question="refresh token", max_files=3),
        SimpleNamespace(id=uuid4()), FakeDB(project_id)))
    assert len(calls) == 1
    assert result["ai_calls"] == 1
    assert len(result["findings"]) == 1
    assert result["findings"][0]["path"] == "app/auth.py"
    assert result["scope"] == "selected_excerpts_only"


def test_invalid_json_fails_explicitly(monkeypatch):
    _sources(monkeypatch)

    async def fake_generate_completion(**_kwargs):
        return SimpleNamespace(text="Not JSON")

    monkeypatch.setattr(routes, "generate_completion", fake_generate_completion)
    project_id = uuid4()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes.investigate_project_question(
            project_id, routes.QuestionInvestigationRequest(question="refresh token"),
            SimpleNamespace(id=uuid4()), FakeDB(project_id)))
    assert exc.value.status_code == 502