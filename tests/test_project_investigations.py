"""Project investigation reads only authenticated snapshot sources."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.routes import project_investigations as routes
from app.schemas.code_review import CodeReviewResponse


class FakeDB:
    def __init__(self, project_id):
        self.project_id = project_id

    async def execute(self, statement):
        class Result:
            def scalar_one_or_none(inner):
                return self.project_id
        return Result()


def test_foreign_project_is_hidden_before_snapshot_access(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Must not read a foreign snapshot")

    monkeypatch.setattr(routes, "read_snapshot_manifest", forbidden)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes.list_investigation_files(
            uuid4(), SimpleNamespace(id=uuid4()), FakeDB(None),
        ))
    assert exc.value.status_code == 404


def test_lists_only_snapshot_files(monkeypatch):
    user_id, project_id, snapshot_id = uuid4(), uuid4(), uuid4()
    monkeypatch.setattr(routes, "read_snapshot_manifest", lambda u, p: SimpleNamespace(snapshot_id=snapshot_id))
    monkeypatch.setattr(routes, "read_snapshot_sources", lambda u, p, s: [
        ("app/main.py", "print('ok')"), ("tests/test_app.py", "assert True"),
    ])
    result = asyncio.run(routes.list_investigation_files(
        project_id, SimpleNamespace(id=user_id), FakeDB(project_id),
    ))
    assert result["files"] == ["app/main.py", "tests/test_app.py"]


def test_review_requires_exact_snapshot_path_and_does_not_call_ai(monkeypatch):
    project_id, snapshot_id = uuid4(), uuid4()
    monkeypatch.setattr(routes, "read_snapshot_manifest", lambda u, p: SimpleNamespace(snapshot_id=snapshot_id))
    monkeypatch.setattr(routes, "read_snapshot_sources", lambda u, p, s: [("app/main.py", "print('ok')")])

    async def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected AI call")

    monkeypatch.setattr(routes, "review_code", forbidden)
    for path, status in [("../app/main.py", 400), ("app/no.py", 404)]:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(routes.investigate_project_file(
                project_id, routes.ProjectInvestigationRequest(relative_path=path),
                SimpleNamespace(id=uuid4()), FakeDB(project_id),
            ))
        assert exc.value.status_code == status


def test_review_uses_selected_source_and_bounded_related_context(monkeypatch):
    project_id, snapshot_id = uuid4(), uuid4()
    monkeypatch.setattr(routes, "read_snapshot_manifest", lambda u, p: SimpleNamespace(snapshot_id=snapshot_id))
    monkeypatch.setattr(routes, "read_snapshot_sources", lambda u, p, s: [
        ("app/main.py", "def run():\n    return True"),
        ("tests/test_main.py", "assert run()"),
    ])
    captured = []

    async def fake_review(request, **kwargs):
        captured.append((request, kwargs))
        return CodeReviewResponse(summary="Okay", findings=[])

    monkeypatch.setattr(routes, "review_code", fake_review)
    result = asyncio.run(routes.investigate_project_file(
        project_id, routes.ProjectInvestigationRequest(relative_path="app/main.py"),
        SimpleNamespace(id=uuid4()), FakeDB(project_id),
    ))
    assert result.relative_path == "app/main.py"
    assert captured[0][0].code == "def run():\n    return True"
    assert result.review.summary == "Okay"