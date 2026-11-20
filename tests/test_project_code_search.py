"""Bounded project-source search requires no external model or embedding calls."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.routes import project_code_search as routes
from app.services.project_code_search import search_snapshot_sources


def test_search_returns_exact_paths_and_line_numbers():
    hits = search_snapshot_sources([
        ("src/auth.py", "def login():\n    token = refresh_session()\n    return token"),
        ("src/other.py", "print('unrelated')"),
    ], "refresh_session", 4)
    assert len(hits) == 1
    assert hits[0]["relative_path"] == "src/auth.py"
    assert hits[0]["start_line"] == 1
    assert hits[0]["end_line"] == 3
    assert "refresh_session()" in hits[0]["content"]
    assert hits[0]["distance"] is None


def test_search_enforces_hard_limit_and_bounded_evidence():
    content = "\n".join("session = refresh_token()" for _ in range(500))
    hits = search_snapshot_sources([("src/auth.py", content)], "refresh_token", 3)
    assert len(hits) == 3
    assert all(len(hit["content"]) <= 1200 for hit in hits)
    with pytest.raises(ValueError):
        search_snapshot_sources([("src/a.py", "x")], "x", 21)


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeDB:
    def __init__(self, result):
        self.result = result
        self.queries = []

    async def execute(self, query):
        self.queries.append(str(query))
        return FakeResult(self.result)


def test_foreign_project_is_hidden_before_snapshot_access(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    db = FakeDB(None)

    def forbidden(*args):
        raise AssertionError("A foreign snapshot must never be accessed")

    monkeypatch.setattr(routes, "read_snapshot_manifest", forbidden)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes.search_project_code(
            uuid4(), routes.ProjectCodeSearchRequest(query="auth"), user, db
        ))
    assert exc.value.status_code == 404
    assert "projects.user_id" in db.queries[0]


def test_owned_project_reads_only_owned_snapshot(monkeypatch):
    user_id, project_id, snapshot_id = uuid4(), uuid4(), uuid4()
    read_arguments = []

    def manifest_reader(user, project):
        read_arguments.append((user, project))
        return SimpleNamespace(snapshot_id=snapshot_id)

    def source_reader(user, project, snapshot):
        read_arguments.append((user, project, snapshot))
        return [("app/main.py", "def run():\n    return True")]

    monkeypatch.setattr(routes, "read_snapshot_manifest", manifest_reader)
    monkeypatch.setattr(routes, "read_snapshot_sources", source_reader)
    result = asyncio.run(routes.search_project_code(
        project_id, routes.ProjectCodeSearchRequest(query="run"),
        SimpleNamespace(id=user_id), FakeDB(project_id)
    ))
    assert result["result_count"] == 1
    assert result["snapshot_id"] == str(snapshot_id)
    assert read_arguments == [(user_id, project_id), (user_id, project_id, snapshot_id)]