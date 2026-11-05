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

    package_file = workspace_root / "package.json"

    if package_file.is_symlink() or not package_file.is_file():
        raise TestProjectConfigurationError(
            "The selected workspace does not contain a valid package.json."
        )

    try:
        resolved_package_file = package_file.resolve(strict=True)

        if not resolved_package_file.is_relative_to(workspace_root):
            raise TestProjectConfigurationError(
                "The package.json file is outside the selected workspace."
            )

        package_data = json.loads(
            resolved_package_file.read_text(encoding="utf-8")
        )

    except TestProjectConfigurationError:
        raise

    except (OSError, ValueError, UnicodeError):
        raise TestProjectConfigurationError(
            "The project's package.json could not be read."
        ) from None

    scripts = package_data.get("scripts", {})

    if not isinstance(scripts, dict) or not scripts.get("test"):
        raise TestProjectConfigurationError(
            "The project's package.json does not define a test script."
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
            if candidate.is_file():
                python_executable = str(candidate)
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
        go_mod = workspace_root / "go.mod"

        if go_mod.is_symlink() or not go_mod.is_file():
            raise TestProjectConfigurationError(
                "The selected workspace does not contain a valid go.mod."
            )

        return [
            "go",
            "test",
            "./...",
        ]

    raise TestProjectConfigurationError(
        "The requested test runner is not supported."
    )


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
    """
    Execute a predefined project test command locally.

    Execution takes place under the backend process's OS permissions.
    This is controlled process execution, not an OS-level sandbox.
    """

    if not 1 <= timeout_seconds <= 600:
        raise ValueError(
            "timeout_seconds must be between 1 and 600."
        )

    resolved_root = _resolve_workspace(workspace_root)

    command = _build_command(
        resolved_root,
        runner,
    )

    execution_id = uuid4()

    started_at = datetime.now(timezone.utc)
    start_time = time.monotonic()

    stdout_capture = _BoundedOutput()
    stderr_capture = _BoundedOutput()

    environment = _build_test_environment()

    # Encourage test runners to exit instead of entering watch mode.
    environment["CI"] = "true"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"

    creation_options = {}

    if os.name == "nt":
        creation_options["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        creation_options["start_new_session"] = True

    try:
        process = subprocess.Popen(
            command,
            cwd=str(resolved_root),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            **creation_options,
        )

    except FileNotFoundError:
        raise TestRunnerUnavailable(
            f"The {runner} test runner is not installed or not available in PATH."
        ) from None

    except OSError:
        raise TestExecutionError(
            "The test process could not be started."
        ) from None

    stdout_thread = threading.Thread(
        target=stdout_capture.read_stream,
        args=(process.stdout,),
        daemon=True,
    )

    stderr_thread = threading.Thread(
        target=stderr_capture.read_stream,
        args=(process.stderr,),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    timed_out = False

    try:
        process.wait(timeout=timeout_seconds)

    except subprocess.TimeoutExpired:
        timed_out = True

        _stop_process_tree(process)

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    finally:
        # On POSIX, also clean up child processes that outlive the runner.
        if os.name != "nt" and not timed_out:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                pass

        stdout_thread.join(timeout=3)
        stderr_thread.join(timeout=3)

        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

    finished_at = datetime.now(timezone.utc)

    duration_ms = max(
        0,
        int((time.monotonic() - start_time) * 1000),
    )

    exit_code = None if timed_out else process.returncode

    if timed_out:
        execution_status = "timed_out"
    elif exit_code == 0:
        execution_status = "passed"
    else:
        execution_status = "failed"

    return TestExecutionResult(
        execution_id=execution_id,
        workspace_root=str(resolved_root),
        runner=runner,
        status=execution_status,
        command=command,
        exit_code=exit_code,
        duration_ms=duration_ms,
        stdout=stdout_capture.text(),
        stderr=stderr_capture.text(),
        stdout_truncated=stdout_capture.truncated,
        stderr_truncated=stderr_capture.truncated,
        started_at=started_at,
        finished_at=finished_at,
    )