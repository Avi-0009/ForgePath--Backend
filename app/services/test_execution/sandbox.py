
"""Development-only Docker isolation adapter.

Do not run Docker directly from a public API host.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.services.workspace import (
    EXCLUDED_DIRECTORIES,
    WorkspaceAccessDenied,
    WorkspaceBinaryFile,
    WorkspaceFileTooLarge,
    read_configured_workspace_file,
)

MAX_SNAPSHOT_FILES = 1000
MAX_SNAPSHOT_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
DOCKER_IMAGE = "forgepath-pytest-sandbox:dev"


class SandboxError(Exception):
    """Sandbox creation or execution failed; no host execution is permitted."""


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool


class _Capture:
    def __init__(self):
        self.data = bytearray()
        self.truncated = False

    def consume(self, stream):
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break

                remaining = MAX_OUTPUT_BYTES - len(self.data)

                if remaining > 0:
                    self.data.extend(chunk[:remaining])

                if len(chunk) > remaining:
                    self.truncated = True

        except (OSError, ValueError):
            pass

    def text(self):
        return self.data.decode("utf-8", errors="replace")


def _snapshot(source: Path, target: Path) -> None:
    """Copy only securely read, eligible text files. Reject links/junctions."""
    count = total = 0

    for current, dirs, files in os.walk(source, followlinks=False):
        current_path = Path(current)
        safe_dirs = []

        for name in dirs:
            candidate = current_path / name
            is_junction = getattr(candidate, "is_junction", None)

            if (
                name.lower() not in EXCLUDED_DIRECTORIES
                and not candidate.is_symlink()
                and not (is_junction and is_junction())
            ):
                safe_dirs.append(name)

        dirs[:] = safe_dirs

        for name in files:
            path = current_path / name
            is_junction = getattr(path, "is_junction", None)

            if path.is_symlink() or (is_junction and is_junction()):
                continue

            relative = path.relative_to(source).as_posix()

            try:
                item = read_configured_workspace_file(
                    workspace_root=source,
                    relative_path=relative,
                )

            except (WorkspaceAccessDenied, WorkspaceBinaryFile):
                continue

            except WorkspaceFileTooLarge as exc:
                raise SandboxError(
                    "A source file exceeds the sandbox snapshot limit."
                ) from exc

            except Exception as exc:
                raise SandboxError(
                    "Unable to safely snapshot the project."
                ) from exc

            count += 1
            total += item.size_bytes

            if (
                count > MAX_SNAPSHOT_FILES
                or total > MAX_SNAPSHOT_BYTES
            ):
                raise SandboxError(
                    "Project exceeds sandbox snapshot limits."
                )

            dest = target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(item.content, encoding="utf-8")

    if count == 0:
        raise SandboxError("No eligible project files were found.")


def _docker_environment() -> dict[str, str]:
    """Pass only a restricted environment to Docker CLI."""
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "PATHEXT",
        "TEMP",
        "TMP",
    }

    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed
    }


def _verify_container_cleanup(name: str) -> None:
    """Remove the sandbox container and verify that it no longer exists."""
    environment = _docker_environment()

    try:
        subprocess.run(
            ["docker", "rm", "-f", name],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )

        result = subprocess.run(
            ["docker", "container", "inspect", name],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )

    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxError(
            "Sandbox cleanup could not be verified."
        ) from exc

    if result.returncode == 0:
        raise SandboxError(
            "Sandbox container still exists after cleanup."
        )

    error = result.stderr.decode(
        "utf-8",
        errors="replace",
    ).lower()

    if (
        "no such object" not in error
        and "no such container" not in error
    ):
        raise SandboxError(
            "Docker could not confirm sandbox removal."
        )


def run_in_sandbox(
    *,
    workspace_root: Path,
    runner: str,
    timeout_seconds: int,
) -> SandboxResult:
    """Run a fixed pytest command in a disposable network-disabled container.

    npm/go intentionally fail closed until their runtime images and
    dependency provisioning are reviewed.

    No user-controlled executable or Docker options are accepted.
    """
    if runner != "pytest":
        raise SandboxError(
            f"Sandboxed {runner} execution is not yet configured."
        )

    name = f"forgepath-test-{uuid4().hex}"

    with tempfile.TemporaryDirectory(
        prefix="forgepath-snapshot-"
    ) as tmp:
        snapshot = Path(tmp) / "src"
        snapshot.mkdir()

        _snapshot(workspace_root, snapshot)

        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--memory-swap",
            "512m",
            "--ulimit",
            "nofile=256:256",
            "--init",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m,mode=1777",
            "--tmpfs",
            "/work:rw,nosuid,size=128m,mode=1777",
            "--mount",
            (
                f"type=bind,source={snapshot.resolve()},"
                "target=/source,readonly"
            ),
            "--workdir",
            "/work",
            "--env",
            "CI=true",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            DOCKER_IMAGE,
            "/bin/sh",
            "-c",
            (
                "cp -R /source/. /work/ && "
                "python -m pytest -q -p no:cacheprovider"
            ),
        ]

        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                env=_docker_environment(),
            )

        except (OSError, FileNotFoundError) as exc:
            raise SandboxError(
                "Docker is unavailable; tests were not executed."
            ) from exc

        out = _Capture()
        err = _Capture()

        threads = [
            threading.Thread(
                target=out.consume,
                args=(proc.stdout,),
                daemon=True,
            ),
            threading.Thread(
                target=err.consume,
                args=(proc.stderr,),
                daemon=True,
            ),
        ]

        for thread in threads:
            thread.start()

        timed_out = False

        try:
            proc.wait(timeout=timeout_seconds)

        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait(timeout=5)

        finally:
            cleanup_error = None

            try:
                _verify_container_cleanup(name)

            except SandboxError as exc:
                cleanup_error = exc

            for thread in threads:
                thread.join(timeout=4)

            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()

            if cleanup_error is not None:
                raise cleanup_error

        if not timed_out and proc.returncode in (125, 126, 127):
            raise SandboxError(
                "Docker failed to start or run the sandbox image."
            )

        return SandboxResult(
            exit_code=None if timed_out else proc.returncode,
            timed_out=timed_out,
            stdout=out.text(),
            stderr=err.text(),
            stdout_truncated=out.truncated,
            stderr_truncated=err.truncated,
        )