"""Uploading a snapshot reconciles exact approved patch proposals."""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.routes import project_snapshots as routes


class Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return self

    def all(self):
        return self.value


class Database:
    def __init__(self, project, tasks):
        self.results = iter((project, tasks))
        self.commits = 0

    async def execute(self, _query):
        return Result(next(self.results))

    async def commit(self):
        self.commits += 1


class Request:
    def __init__(self, body: bytes, expected_snapshot_id=None):
        self.headers = {"content-length": str(len(body))}
        if expected_snapshot_id is not None:
            self.headers["x-forgepath-expected-snapshot-id"] = str(expected_snapshot_id)
        self.app = SimpleNamespace(state=SimpleNamespace(workspace_indexing_service=None))
        self._body = body

    async def stream(self):
        yield self._body


def test_upload_marks_exact_approved_patch_as_applied(monkeypatch):
    user_id, project_id, snapshot_id, task_id = (uuid4() for _ in range(4))
    proposed = "def main():\n    return 'fixed'\n"
    proposed_hash = hashlib.sha256(proposed.encode("utf-8")).hexdigest()
    project = SimpleNamespace(id=project_id, status="ready")
    task = SimpleNamespace(
        id=task_id,
        path="app/main.py",
        patch_status="approved",
        patch={
            "relative_path": "app/main.py",
            "proposed_sha256": proposed_hash,
        },
        applied_snapshot_id=None,
        applied_at=None,
    )
    db = Database(project, [task])
    manifest = SimpleNamespace(
        snapshot_id=snapshot_id,
        file_count=1,
        total_bytes=len(proposed.encode("utf-8")),
        sha256="snapshot-digest",
    )

    monkeypatch.setattr(routes, "store_snapshot", lambda *_: manifest)
    monkeypatch.setattr(routes, "list_project_snapshot_ids", lambda *_: {snapshot_id})

    body = json.dumps({
        "files": [{"relative_path": "app/main.py", "content": proposed}],
        "asset_paths": [],
    }).encode("utf-8")

    response = asyncio.run(
        routes.upload_project_snapshot(
            project_id,
            Request(body),
            SimpleNamespace(id=user_id),
            db,
        )
    )

    assert response["applied_patch_count"] == 1
    assert response["applied_patch_task_ids"] == [str(task_id)]
    assert task.patch_status == "applied"
    assert task.applied_snapshot_id == snapshot_id
    assert task.applied_at is not None
    assert db.commits == 1


def test_upload_rejects_changed_expected_base_snapshot(monkeypatch):
    user_id, project_id, expected_snapshot_id = (uuid4() for _ in range(3))
    project = SimpleNamespace(id=project_id, status="ready")
    db = Database(project, [])
    body = json.dumps({
        "files": [{"relative_path": "app/main.py", "content": "print('changed')\n"}],
        "asset_paths": [],
    }).encode("utf-8")
    monkeypatch.setattr(
        routes,
        "read_snapshot_manifest",
        lambda *_: SimpleNamespace(snapshot_id=uuid4()),
    )
    monkeypatch.setattr(
        routes,
        "store_snapshot",
        lambda *_: (_ for _ in ()).throw(AssertionError("stale upload must not be stored")),
    )

    with pytest.raises(HTTPException) as exception:
        asyncio.run(routes.upload_project_snapshot(
            project_id,
            Request(body, expected_snapshot_id),
            SimpleNamespace(id=user_id),
            db,
        ))

    assert exception.value.status_code == 409
    assert db.commits == 0
