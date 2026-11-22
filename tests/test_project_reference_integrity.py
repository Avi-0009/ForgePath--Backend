"""Ownership and snapshot isolation for the read-only references endpoint."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.routes import project_references as routes


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeDB:
    def __init__(self, value):
        self.value = value

    async def execute(self, _query):
        return FakeResult(self.value)


def test_foreign_project_is_hidden_before_any_snapshot_is_read(monkeypatch):
    def forbidden(*_args):
        raise AssertionError("Foreign project snapshot must not be read")

    monkeypatch.setattr(routes, "read_snapshot_manifest", forbidden)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(routes.check_project_references(
            uuid4(), SimpleNamespace(id=uuid4()), FakeDB(None),
        ))
    assert caught.value.status_code == 404


def test_owned_project_reads_only_its_current_snapshot(monkeypatch):
    user_id, project_id, snapshot_id = uuid4(), uuid4(), uuid4()
    calls = []

    def manifest(user, project):
        calls.append(("manifest", user, project))
        return SimpleNamespace(snapshot_id=snapshot_id, asset_paths=("src/icon.svg",))

    def sources(user, project, version):
        calls.append(("source", user, project, version))
        return [("src/main.ts", 'import icon from "./icon.svg";')]

    monkeypatch.setattr(routes, "read_snapshot_manifest", manifest)
    monkeypatch.setattr(routes, "read_snapshot_sources", sources)
    result = asyncio.run(routes.check_project_references(
        project_id, SimpleNamespace(id=user_id), FakeDB(project_id),
    ))
    assert result["references_checked"] == 1
    assert result["references_resolved"] == 1
    assert result["references_unresolved"] == 0
    assert result["snapshot_id"] == str(snapshot_id)
    assert calls == [
        ("manifest", user_id, project_id),
        ("source", user_id, project_id, snapshot_id),
    ]


def test_missing_snapshot_gives_conflict(monkeypatch):
    monkeypatch.setattr(routes, "read_snapshot_manifest", lambda *_: None)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(routes.check_project_references(
            uuid4(), SimpleNamespace(id=uuid4()), FakeDB(uuid4()),
        ))
    assert caught.value.status_code == 409