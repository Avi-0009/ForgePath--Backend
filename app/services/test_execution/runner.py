import os
import signal
import subprocess
import sys
import threading
import time
import shutil

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4
import json

from app.core.config import settings
from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceConfigurationError,
    WorkspaceReadError,
    read_configured_workspace_file,
    resolve_allowed_workspace_root,
)


TestRunner = Literal["pytest", "npm", "go"]

MAX_OUTPUT_BYTES = 64 * 1024
READ_CHUNK_SIZE = 8192


class TestExecutionError(Exception):
    """Base exception for local test execution failures."""


class TestRunnerUnavailable(TestExecutionError):
    """The requested test runner is unavailable."""


class TestProjectConfigurationError(TestExecutionError):
    """The selected project is not configured for the requested runner."""


@dataclass(frozen=True, slots=True)
class TestExecutionResult:
    """Structured result of one local test execution."""

    execution_id: UUID
    workspace_root: str
    runner: TestRunner

    status: Literal[
        "passed",
        "failed",
        "timed_out",
        "error",
    ]

    command: list[str]
    exit_code: int | None
    duration_ms: int

    stdout: str
    stderr: str

    stdout_truncated: bool
    stderr_truncated: bool

    started_at: datetime
    finished_at: datetime


class _BoundedOutput:
    """Capture output without allowing unbounded memory growth."""

    def __init__(self, max_bytes: int = MAX_OUTPUT_BYTES) -> None:
        self.max_bytes = max_bytes
        self.data = bytearray()
        self.truncated = False
        self._lock = threading.Lock()

    def read_stream(self, stream) -> None:
        """Continuously drain a process stream into a bounded buffer."""

        try:
            while True:
                chunk = stream.read(READ_CHUNK_SIZE)

                if not chunk:
                    break

                with self._lock:
                    available = self.max_bytes - len(self.data)

                    if available > 0:
                        self.data.extend(chunk[:available])

                    if len(chunk) > available:
                        self.truncated = True

        except (OSError, ValueError):
            # The stream may be closed during process termination.
            pass

    def text(self) -> str:
        """Return captured output as readable UTF-8 text."""

        with self._lock:
            return self.data.decode("utf-8", errors="replace")


def _resolve_workspace(workspace_root: str | Path) -> Path:
    """Resolve a project directory using the existing workspace policy."""

    try:
        return resolve_allowed_workspace_root(
            workspace_root=workspace_root,
            allowed_workspace_roots=settings.workspace_root_list,
        )

    except (
        WorkspaceAccessDenied,
        WorkspaceConfigurationError,
    ):
        raise
    
    except Exception as exc:
        raise TestExecutionError(
            "The selected workspace is unavailable or not allowlisted."
        ) from exc


def _validate_npm_project(workspace_root: Path) -> None:
    """Ensure the project defines an npm test script."""

    try:
        package_content = _read_project_metadata_file(
            workspace_root,
            "package.json",
        )
        package_data = json.loads(package_content)

    except TestProjectConfigurationError:
        raise

    except (ValueError, UnicodeError):
        raise TestProjectConfigurationError(
            "The project's package.json contains invalid JSON."
        ) from None

    if not isinstance(package_data, dict):
        raise TestProjectConfigurationError(
            "The project's package.json must contain a JSON object."
        )

    scripts = package_data.get("scripts", {})

    test_script = (
        scripts.get("test")
        if isinstance(scripts, dict)
        else None
    )

    if not isinstance(test_script, str) or not test_script.strip():
        raise TestProjectConfigurationError(
            "The project's package.json does not define a valid test script."
        )


def _build_command(
    workspace_root: Path,
    runner: TestRunner,
) -> list[str]:
    """Build a predefined command without accepting arbitrary arguments."""

    if runner == "pytest":
        python_executable = sys.executable

        if os.name == "nt":
            candidates = [
                workspace_root / ".venv" / "Scripts" / "python.exe",
                workspace_root / "venv" / "Scripts" / "python.exe",
                workspace_root / "env" / "Scripts" / "python.exe",
            ]
        else:
            candidates = [
                workspace_root / ".venv" / "bin" / "python",
                workspace_root / "venv" / "bin" / "python",
                workspace_root / "env" / "bin" / "python",
            ]

        for candidate in candidates:
            validated_candidate = _resolve_project_python_executable(
                workspace_root,
                candidate,
            )

            if validated_candidate is not None:
                python_executable = str(validated_candidate)
                break

        return [
            python_executable,
            "-m",
            "pytest",
            "-q",
        ]

    if runner == "npm":
        _validate_npm_project(workspace_root)

        if os.name == "nt":
            return [
                "cmd.exe",
                "/d",
                "/c",
                "npm test",
            ]

        return [
            "npm",
            "test",
        ]

    if runner == "go":
        _read_project_metadata_file(
            workspace_root,
            "go.mod",
        )

        return [
            "go",
            "test",
            "./...",
        ]

    raise TestProjectConfigurationError(
        "The requested test runner is not supported."
    )


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True

    junction_check = getattr(path, "is_junction", None)
    return bool(junction_check and junction_check())


def _resolve_project_python_executable(
    workspace_root: Path,
    candidate: Path,
) -> Path | None:
    """Accept only a real Python executable inside the workspace."""

    root = workspace_root.resolve(strict=True)
    current = workspace_root

    try:
        relative_parts = candidate.relative_to(workspace_root).parts

        for part in relative_parts:
            current = current / part

            if _is_link_or_junction(current):
                raise TestProjectConfigurationError(
                    "Project Python executables cannot use symbolic links "
                    "or directory junctions."
                )

            if not current.exists():
                return None

        resolved = candidate.resolve(strict=True)

    except TestProjectConfigurationError:
        raise

    except (OSError, RuntimeError, ValueError):
        raise TestProjectConfigurationError(
            "The project Python executable could not be safely validated."
        ) from None

    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise TestProjectConfigurationError(
            "The project Python executable must be a regular file "
            "inside the selected workspace."
        )

    return resolved


def _read_project_metadata_file(
    workspace_root: Path,
    filename: str,
) -> str:
    """Read project metadata using the secure workspace reader."""

    metadata_path = workspace_root / filename

    try:
        if _is_link_or_junction(metadata_path) or not metadata_path.is_file():
            raise TestProjectConfigurationError(
                f"The project does not contain a valid {filename}."
            )

        resolved_root = workspace_root.resolve(strict=True)
        resolved_path = metadata_path.resolve(strict=True)

        if (
            not resolved_path.is_relative_to(resolved_root)
            or not resolved_path.is_file()
        ):
            raise TestProjectConfigurationError(
                f"The {filename} file is outside the selected workspace."
            )

        source = read_configured_workspace_file(
            workspace_root=workspace_root,
            relative_path=filename,
        )

        if (
            _is_link_or_junction(metadata_path)
            or metadata_path.resolve(strict=True) != resolved_path
        ):
            raise TestProjectConfigurationError(
                f"The {filename} path changed during validation."
            )

        return source.content

    except TestProjectConfigurationError:
        raise

    except WorkspaceReadError:
        raise TestProjectConfigurationError(
            f"The {filename} file could not be safely read."
        ) from None

    except (OSError, RuntimeError, ValueError):
        raise TestProjectConfigurationError(
            f"The {filename} file could not be safely validated."
        ) from None


def _build_test_environment() -> dict[str, str]:
    """Pass only the basic OS environment needed by test runners."""

    allowed_variables = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
    }

    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed_variables
    }

    environment.update({
        "CI": "true",
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONUNBUFFERED": "1",
    })

    return environment


def _stop_process_tree(process: subprocess.Popen) -> None:
    """Terminate the test process and its descendants where supported."""

    try:
        if os.name == "nt":
            subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )

        else:
            os.killpg(process.pid, signal.SIGKILL)

    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def execute_tests(
    *,
    workspace_root: str | Path,
    runner: TestRunner,
    timeout_seconds: int = 120,
) -> TestExecutionResult:
    """Execute tests only inside the disposable sandbox; never on the API host."""
    from app.services.test_execution.sandbox import SandboxError, run_in_sandbox

    if not 1 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be between 1 and 600.")
    resolved_root = _resolve_workspace(workspace_root)
    # Preserve existing metadata checks; no host executable is launched.
    if runner == "npm":
        _validate_npm_project(resolved_root)
    elif runner == "go":
        _read_project_metadata_file(resolved_root, "go.mod")
    elif runner != "pytest":
        raise TestProjectConfigurationError("Unsupported test runner.")
    started_at = datetime.now(timezone.utc)
    start = time.monotonic()
    try:
        outcome = run_in_sandbox(
            workspace_root=resolved_root, runner=runner,
            timeout_seconds=timeout_seconds,
        )
    except SandboxError as exc:
        raise TestExecutionError(str(exc)) from exc
    finished_at = datetime.now(timezone.utc)
    status = "timed_out" if outcome.timed_out else (
        "passed" if outcome.exit_code == 0 else "failed"
    )
    return TestExecutionResult(
        execution_id=uuid4(), workspace_root=str(resolved_root),
        runner=runner, status=status,
        command={"pytest": ["python", "-m", "pytest", "-q"],
                 "npm": ["npm", "test"],
                 "go": ["go", "test", "./..."]}[runner],
        exit_code=outcome.exit_code,
        duration_ms=max(0, int((time.monotonic() - start) * 1000)),
        stdout=outcome.stdout, stderr=outcome.stderr,
        stdout_truncated=outcome.stdout_truncated,
        stderr_truncated=outcome.stderr_truncated,
        started_at=started_at, finished_at=finished_at,
    )
