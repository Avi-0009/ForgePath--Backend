"""Execute owned, pinned browser snapshots using the development Docker adapter."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import UUID, uuid4

from app.services.project_snapshots import (
    SnapshotConfigurationError,
    SnapshotValidationError,
    read_snapshot_manifest,
    read_snapshot_sources,
)
from app.services.test_execution import sandbox
from app.services.test_execution.runner import TestRunner


class ProjectSnapshotChanged(Exception):
    """The requested snapshot is not the current version."""


class ProjectTestConfigurationError(Exception):
    """The snapshot lacks required test metadata."""


class ProjectTestUnavailable(Exception):
    """Snapshot storage or the execution runtime is unavailable."""


# Development-process limit. A hosted worker needs independent admission control.
_slots = threading.BoundedSemaphore(2)


@dataclass(frozen=True, slots=True)
class ProjectTestResult:
    execution_id: UUID
    project_id: UUID
    snapshot_id: UUID
    runner: TestRunner
    status: Literal["passed", "failed", "timed_out", "error"]
    command: list[str]
    exit_code: int | None
    duration_ms: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    started_at: datetime
    finished_at: datetime


def _validate_runner_metadata(files: dict[str, str], runner: TestRunner) -> None:
    if runner == "npm":
        try:
            data = json.loads(files["package.json"])
            script = data["scripts"]["test"]
            if not isinstance(script, str) or not script.strip():
                raise ValueError("Empty test script")
        except (KeyError, TypeError, ValueError):
            raise ProjectTestConfigurationError(
                "The snapshot needs a root package.json with a valid test script."
            ) from None
    elif runner == "go":
        if not files.get("go.mod", "").strip():
            raise ProjectTestConfigurationError(
                "The snapshot needs a root go.mod file. Refresh the project snapshot."
            )
    elif runner != "pytest":
        raise ProjectTestConfigurationError("Unsupported test runner.")


def _stage_sources(files: dict[str, str], target: Path) -> None:
    """Stage only verified text files into a new disposable directory."""
    root = target.resolve()
    for relative, content in files.items():
        path = PurePosixPath(relative)
        if (
            not path.parts or path.is_absolute() or "\\" in relative
            or ":" in relative or "\x00" in relative
            or any(part in {".", ".."} for part in path.parts)
            or (os.name == "nt" and any(os.path.isreserved(part) for part in path.parts))
        ):
            raise ProjectTestUnavailable("Unsafe snapshot source path.")
        destination = root.joinpath(*path.parts)
        if not destination.resolve().is_relative_to(root):
            raise ProjectTestUnavailable("Unsafe snapshot source path.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as file:
            file.write(content.encode("utf-8"))


def execute_project_tests(
    *,
    user_id: UUID,
    project_id: UUID,
    expected_snapshot_id: UUID,
    runner: TestRunner,
    timeout_seconds: int,
) -> ProjectTestResult:
    """Execute a snapshot after the caller has verified project ownership."""
    if not 1 <= timeout_seconds <= 600:
        raise ProjectTestConfigurationError("Invalid test timeout.")
    if runner not in sandbox.SANDBOX_RUNNERS:
        raise ProjectTestConfigurationError("Unsupported test runner.")
    if not _slots.acquire(blocking=False):
        raise ProjectTestUnavailable("Test execution capacity is busy. Try again later.")
    try:
        started_at = datetime.now(timezone.utc)
        start = time.monotonic()
        try:
            manifest = read_snapshot_manifest(user_id, project_id)
            if manifest is None or manifest.snapshot_id != expected_snapshot_id:
                raise ProjectSnapshotChanged("Project snapshot changed. Refresh and approve again.")
            sources = read_snapshot_sources(user_id, project_id, expected_snapshot_id)
            files = dict(sources)
            if not files or len(files) != len(sources):
                raise ProjectTestUnavailable("Project snapshot is unavailable.")
            _validate_runner_metadata(files, runner)
            # Keep the copy private and alive until Docker cleanup finishes.
            with tempfile.TemporaryDirectory(prefix="forgepath-project-test-") as tmp:
                snapshot = Path(tmp) / "src"
                snapshot.mkdir()
                _stage_sources(files, snapshot)
                outcome = sandbox._run_prepared_snapshot(
                    snapshot=snapshot,
                    runner=runner,
                    timeout_seconds=timeout_seconds,
                )
        except (SnapshotConfigurationError, SnapshotValidationError, OSError, ValueError, KeyError) as exc:
            raise ProjectTestUnavailable("Project snapshot is unavailable or invalid.") from exc
        except sandbox.SandboxError as exc:
            # Do not disclose host paths or Docker internals to callers.
            raise ProjectTestUnavailable("Isolated test runtime is unavailable.") from exc

        status = "timed_out" if outcome.timed_out else (
            "passed" if outcome.exit_code == 0 else "failed"
        )
        commands = {
            "pytest": ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            "npm": ["npm", "test"],
            "go": ["go", "test", "./..."],
        }
        return ProjectTestResult(
            execution_id=uuid4(), project_id=project_id,
            snapshot_id=expected_snapshot_id, runner=runner, status=status,
            command=commands[runner], exit_code=outcome.exit_code,
            duration_ms=max(0, int((time.monotonic() - start) * 1000)),
            stdout=outcome.stdout, stderr=outcome.stderr,
            stdout_truncated=outcome.stdout_truncated,
            stderr_truncated=outcome.stderr_truncated,
            started_at=started_at, finished_at=datetime.now(timezone.utc),
        )
    finally:
        _slots.release()
