"""Uploading a snapshot reconciles exact approved patch proposals."""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

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
    def __init__(self, body: bytes):
        self.headers = {"content-length": str(len(body))}
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
    assert task.patch_status == "applied"
    assert task.applied_snapshot_id == snapshot_id
    assert task.applied_at is not None
    assert db.commits == 1
