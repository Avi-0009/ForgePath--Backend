"""Offline background indexing tests: no embedding requests or network calls."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services import project_index_jobs as jobs_module
from app.services.project_index_jobs import ProjectIndexJobManager


def _manager(tmp_path):
    return ProjectIndexJobManager(SimpleNamespace(
        vector_store=SimpleNamespace(persist_directory=tmp_path),
    ))


def test_background_job_persists_real_progress_and_completion(tmp_path, monkeypatch):
    user, project, snapshot = uuid4(), uuid4(), uuid4()
    manager = _manager(tmp_path)
    monkeypatch.setattr(jobs_module, "read_snapshot_manifest", lambda *_: SimpleNamespace(
        snapshot_id=snapshot, sha256="digest",
    ))

    async def fake_index(_service, _user, _project, _snapshot, _sha, _sources, *, progress):
        await progress(2, 3, 4)
        await progress(3, 3, 6)
        return {"vectors_upserted": 6}

    monkeypatch.setattr(jobs_module, "index_project_snapshot", fake_index)

    async def run():
        first = manager.start(user, project, snapshot, "digest", [("a.py", "a"), ("b.py", "b"), ("c.py", "c")], 6)
        await manager._tasks[uuid4_from(first["job_id"])]
        return first

    first = asyncio.run(run())
    finished = manager.get(user, project, uuid4_from(first["job_id"]))
    assert finished["status"] == "completed"
    assert finished["files_done"] == 3
    assert finished["chunks_embedded"] == 6
    restored = _manager(tmp_path).get(user, project, uuid4_from(first["job_id"]))
    assert restored["status"] == "completed"
    assert manager.get(uuid4(), project, uuid4_from(first["job_id"])) is None


def uuid4_from(value):
    from uuid import UUID
    return UUID(value)


def test_running_job_is_marked_interrupted_after_restart(tmp_path):
    user, project, snapshot = uuid4(), uuid4(), uuid4()
    manager = _manager(tmp_path)
    job_id = uuid4()
    manager._save({
        "job_id": str(job_id), "user_id": str(user), "project_id": str(project),
        "snapshot_id": str(snapshot), "status": "running", "created_at": "2026-11-22T00:00:00+00:00",
        "updated_at": "2026-11-22T00:00:00+00:00", "message": "Embedding source batches",
    })
    restored = _manager(tmp_path).get(user, project, job_id)
    assert restored["status"] == "interrupted"


def test_stale_snapshot_is_refused_before_embedding(tmp_path, monkeypatch):
    user, project, snapshot = uuid4(), uuid4(), uuid4()
    manager = _manager(tmp_path)
    monkeypatch.setattr(jobs_module, "read_snapshot_manifest", lambda *_: SimpleNamespace(
        snapshot_id=uuid4(), sha256="other",
    ))

    async def forbidden(*_args, **_kwargs):
        pytest.fail("A stale snapshot must not incur embedding calls")

    monkeypatch.setattr(jobs_module, "index_project_snapshot", forbidden)

    async def run():
        first = manager.start(user, project, snapshot, "digest", [("a.py", "a")], 1)
        await manager._tasks[uuid4_from(first["job_id"])]
        return first

    first = asyncio.run(run())
    assert manager.get(user, project, uuid4_from(first["job_id"]))["status"] == "interrupted"


def test_failed_provider_job_is_saved_without_leaking_exception(tmp_path, monkeypatch):
    user, project, snapshot = uuid4(), uuid4(), uuid4()
    manager = _manager(tmp_path)
    monkeypatch.setattr(jobs_module, "read_snapshot_manifest", lambda *_: SimpleNamespace(
        snapshot_id=snapshot, sha256="digest",
    ))

    async def failed_index(*_args, **_kwargs):
        raise RuntimeError("private provider credentials must not appear in job status")

    monkeypatch.setattr(jobs_module, "index_project_snapshot", failed_index)

    async def run():
        first = manager.start(user, project, snapshot, "digest", [("a.py", "a")], 1)
        await manager._tasks[uuid4_from(first["job_id"])]
        return first

    first = asyncio.run(run())
    finished = manager.get(user, project, uuid4_from(first["job_id"]))
    assert finished["status"] == "failed"
    assert "credentials" not in finished["message"]
