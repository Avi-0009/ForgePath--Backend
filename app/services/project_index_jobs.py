"""Single-process, durable-status indexing jobs for private project snapshots.

Job metadata is persisted locally. Interrupted jobs are never resumed without
another explicit, owner-authenticated confirmation (which may incur charges).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from app.core.ai_usage_context import ai_usage_context
from app.services.project_semantic_index import index_lifecycle_lock, index_project_snapshot
from app.services.project_snapshots import read_snapshot_manifest
from app.services.workspace_indexing import WorkspaceIndexingService

logger = logging.getLogger(__name__)
ACTIVE_STATES = {"queued", "running"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectIndexJobManager:
    def __init__(self, service: WorkspaceIndexingService) -> None:
        self.service = service
        self._jobs: dict[UUID, dict] = {}
        self._tasks: dict[UUID, asyncio.Task] = {}

    def _directory(self, user_id: UUID, project_id: UUID) -> Path:
        # Compact UUID names keep nested pytest/Windows paths below MAX_PATH.
        return (self.service.vector_store.persist_directory / "jobs"
                / user_id.hex / project_id.hex)

    def _save(self, job: dict) -> None:
        directory = self._directory(UUID(job["user_id"]), UUID(job["project_id"]))
        if directory.is_symlink():
            raise OSError("Unsafe project indexing job directory")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = directory / f"{UUID(job['job_id']).hex}.json"
        if target.is_symlink():
            raise OSError("Unsafe project indexing job file")
        temporary = directory / f".{uuid4().hex[:12]}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as output:
                os.chmod(temporary, 0o600)
                json.dump(job, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _update(self, job: dict, **changes) -> None:
        job.update(changes)
        job["updated_at"] = _now()
        self._save(job)

    def get(self, user_id: UUID, project_id: UUID, job_id: UUID) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None:
            path = self._directory(user_id, project_id) / f"{job_id.hex}.json"
            if not path.is_file() or path.is_symlink():
                return None
            job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("user_id") != str(user_id) or job.get("project_id") != str(project_id):
            return None
        if job["status"] in ACTIVE_STATES and job_id not in self._tasks:
            # The process that owned this task was interrupted. Never resume
            # embedding calls automatically after a restart.
            self._update(job, status="interrupted", message="Server restarted. Confirm indexing again to retry.")
        return self._public(job)

    def latest(self, user_id: UUID, project_id: UUID, snapshot_id: UUID) -> dict | None:
        directory = self._directory(user_id, project_id)
        if not directory.is_dir() or directory.is_symlink():
            return None
        latest_job = None
        for path in directory.glob("*.json"):
            if path.is_symlink():
                continue
            try:
                job_id = UUID(path.stem)
                job = self.get(user_id, project_id, job_id)
            except (ValueError, KeyError, OSError, json.JSONDecodeError):
                continue
            if job and job.get("snapshot_id") == str(snapshot_id):
                if latest_job is None or job["created_at"] > latest_job["created_at"]:
                    latest_job = job
        return latest_job

    @staticmethod
    def _public(job: dict) -> dict:
        return {key: value for key, value in job.items() if key != "user_id"}

    def start(
        self, user_id: UUID, project_id: UUID, snapshot_id: UUID,
        snapshot_sha256: str, sources: list[tuple[str, str]], total_chunks: int,
    ) -> dict:
        if any(job["status"] in ACTIVE_STATES for job in self._jobs.values()):
            raise ValueError("An indexing job is already running")
        job_id = uuid4()
        job = {
            "job_id": str(job_id), "user_id": str(user_id), "project_id": str(project_id),
            "snapshot_id": str(snapshot_id), "status": "queued",
            "files_done": 0, "total_files": sum(bool(content.strip()) for _, content in sources),
            "chunks_embedded": 0, "estimated_chunks": total_chunks,
            "message": "Waiting to start", "created_at": _now(), "updated_at": _now(),
        }
        self._save(job)
        self._jobs[job_id] = job
        task = asyncio.create_task(self._run(job, snapshot_sha256, sources))
        self._tasks[job_id] = task
        task.add_done_callback(lambda _done: self._tasks.pop(job_id, None))
        return self._public(job)

    async def _run(self, job: dict, digest: str, sources: list[tuple[str, str]]) -> None:
        user_id, project_id = UUID(job["user_id"]), UUID(job["project_id"])
        snapshot_id = UUID(job["snapshot_id"])
        try:
            async with index_lifecycle_lock:
                current = await asyncio.to_thread(read_snapshot_manifest, user_id, project_id)
                if current is None or current.snapshot_id != snapshot_id or current.sha256 != digest:
                    self._update(job, status="interrupted", message="Snapshot changed. Preview and confirm again.")
                    return
                self._update(job, status="running", message="Embedding source batches")

                async def report(done: int, total: int, chunks: int) -> None:
                    self._update(job, files_done=done, total_files=total, chunks_embedded=chunks)

                with ai_usage_context(user_id=user_id, operation="project_embedding"):
                    result = await index_project_snapshot(
                        self.service, user_id, project_id, snapshot_id, digest, sources,
                        progress=report,
                    )
                self._update(job, status="completed", message="Index ready", result=result)
        except asyncio.CancelledError:
            self._update(job, status="interrupted", message="Indexing was interrupted. Confirm to retry.")
            raise
        except Exception:
            logger.exception("Background project indexing failed")
            self._update(job, status="failed", message="Indexing failed. Confirm to retry.")

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def prune(self, user_id: UUID, project_id: UUID, *, keep_snapshot_id: UUID | None = None) -> int:
        """Erase only owned job records; preserve current snapshot history on replacement."""
        directory = self._directory(user_id, project_id)
        if not directory.exists():
            return 0
        if directory.is_symlink() or not directory.is_dir():
            raise OSError("Unsafe project index job directory")
        removed = 0
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file():
                raise OSError("Unsafe project index job entry")
            if path.suffix != ".json":
                continue
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("user_id") != str(user_id) or job.get("project_id") != str(project_id):
                raise OSError("Unexpected project indexing job ownership")
            if keep_snapshot_id is not None and job.get("snapshot_id") == str(keep_snapshot_id):
                continue
            job_id = UUID(job["job_id"])
            if (job_id in self._tasks and not self._tasks[job_id].done()
                    and job.get("status") in ACTIVE_STATES):
                raise OSError("Cannot delete metadata for an active indexing job")
            path.unlink()
            self._jobs.pop(job_id, None)
            removed += 1
        if keep_snapshot_id is None and not list(directory.iterdir()):
            directory.rmdir()
        return removed
