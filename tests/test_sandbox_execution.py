import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import subprocess

import pytest

from app.core.config import settings
from app.services.test_execution import runner, sandbox


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    folder = tmp_path / "project"
    folder.mkdir()
    monkeypatch.setattr(settings, "workspace_allowed_roots", str(tmp_path))
    (folder / "test_example.py").write_text("def test_ok(): assert True\n")
    (folder / ".env").write_text("SECRET=never-copy\n")
    return folder


def test_snapshot_excludes_env(workspace, tmp_path):
    output = tmp_path / "snapshot"
    output.mkdir()
    sandbox._snapshot(workspace, output)
    assert (output / "test_example.py").exists()
    assert not (output / ".env").exists()


def test_snapshot_excludes_symlinks(workspace, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    try:
        (workspace / "linked.py").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not permitted")
    output = tmp_path / "snapshot"
    output.mkdir()
    sandbox._snapshot(workspace, output)
    assert not (output / "linked.py").exists()


def test_missing_docker_fails_closed(workspace, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("docker")
    monkeypatch.setattr(sandbox.subprocess, "Popen", missing)
    with pytest.raises(runner.TestExecutionError, match="Docker is unavailable"):
        runner.execute_tests(workspace_root=workspace, runner="pytest")


def test_npm_not_executed_without_configured_sandbox(workspace):
    (workspace / "package.json").write_text('{"scripts":{"test":"echo hello"}}')
    with pytest.raises(runner.TestExecutionError, match="not yet configured"):
        runner.execute_tests(workspace_root=workspace, runner="npm")


def test_no_host_command_fallback(workspace, monkeypatch):
    def broken_sandbox(**kwargs):
        raise sandbox.SandboxError("sandbox unavailable")
    monkeypatch.setattr(sandbox, "run_in_sandbox", broken_sandbox)
    with pytest.raises(runner.TestExecutionError, match="sandbox unavailable"):
        runner.execute_tests(workspace_root=workspace, runner="pytest")


@pytest.mark.skipif(
    os.getenv("FORGEPATH_DOCKER_TESTS") != "1",
    reason="Real Docker tests require explicit opt-in",
)
def test_real_docker_sandbox(workspace):
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI unavailable")

    result = runner.execute_tests(
        workspace_root=workspace,
        runner="pytest",
        timeout_seconds=60,
    )

    assert result.status == "passed", result.stderr
    assert result.exit_code == 0
    assert "1 passed" in result.stdout

    # Verify the original workspace was not modified.
    assert (workspace / "test_example.py").read_text() == (
        "def test_ok(): assert True\n"
    )

    # Verify the sensitive file was not modified.
    assert (workspace / ".env").read_text() == (
        "SECRET=never-copy\n"
    )


def test_container_cleanup_verified(monkeypatch):
    from unittest.mock import Mock

    from app.services.test_execution import sandbox

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

        if command[1:3] == ["container", "inspect"]:
            return Mock(
                returncode=1,
                stderr=b"Error: No such object: test-container",
            )

        return Mock(returncode=0)

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    sandbox._verify_container_cleanup("test-container")

    assert len(calls) == 2
    assert calls[0] == ["docker", "rm", "-f", "test-container"]
    assert calls[1] == [
        "docker", "container", "inspect", "test-container"
    ]


def test_container_cleanup_detects_remaining_container(monkeypatch):
    from unittest.mock import Mock
    import pytest

    from app.services.test_execution import sandbox

    def fake_run(command, **kwargs):
        if command[1:3] == ["container", "inspect"]:
            return Mock(returncode=0, stderr=b"")

        return Mock(returncode=0)

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    with pytest.raises(
        sandbox.SandboxError,
        match="still exists",
    ):
        sandbox._verify_container_cleanup("test-container")


def test_container_cleanup_detects_docker_failure(monkeypatch):
    from unittest.mock import Mock
    import pytest

    from app.services.test_execution import sandbox

    def fake_run(command, **kwargs):
        if command[1:3] == ["container", "inspect"]:
            return Mock(
                returncode=1,
                stderr=b"Cannot connect to the Docker daemon",
            )

        return Mock(returncode=1)

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    with pytest.raises(
        sandbox.SandboxError,
        match="could not confirm",
    ):
        sandbox._verify_container_cleanup("test-container")


@pytest.mark.skipif(
    os.getenv("FORGEPATH_DOCKER_TESTS") != "1",
    reason="Requires real Docker execution",
)
def test_real_docker_execution_timeout(workspace):
    """Long-running tests must time out without modifying the workspace."""

    if shutil.which("docker") is None:
        pytest.skip("Docker CLI unavailable")

    slow_test = workspace / "test_slow.py"

    slow_test.write_text(
        "import time\n"
        "\n"
        "def test_slow_execution():\n"
        "    time.sleep(30)\n",
        encoding="utf-8",
    )

    result = runner.execute_tests(
        workspace_root=workspace,
        runner="pytest",
        timeout_seconds=3,
    )

    assert result.status == "timed_out"
    assert result.exit_code is None

    assert slow_test.read_text(encoding="utf-8") == (
        "import time\n"
        "\n"
        "def test_slow_execution():\n"
        "    time.sleep(30)\n"
    )


@pytest.mark.skipif(
    os.getenv("FORGEPATH_DOCKER_TESTS") != "1",
    reason="Requires real Docker execution",
)
def test_real_docker_resource_limits(workspace):
    """Verify resource limits from inside the actual container."""

    if shutil.which("docker") is None:
        pytest.skip("Docker CLI unavailable")

    resource_test = workspace / "test_resource_limits.py"

    resource_test.write_text(
        """
from pathlib import Path


def test_container_limits():
    cgroup = Path("/sys/fs/cgroup")

    # Docker Desktop uses cgroup v2.
    memory = int((cgroup / "memory.max").read_text().strip())
    pids = int((cgroup / "pids.max").read_text().strip())

    cpu_quota, cpu_period = (
        (cgroup / "cpu.max").read_text().strip().split()
    )

    assert memory <= 512 * 1024 * 1024
    assert pids <= 64
    assert cpu_quota != "max"
    assert int(cpu_quota) <= int(cpu_period)
""",
        encoding="utf-8",
    )

    result = runner.execute_tests(
        workspace_root=workspace,
        runner="pytest",
        timeout_seconds=60,
    )

    assert result.status == "passed", (
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )
    assert result.exit_code == 0
    assert "2 passed" in result.stdout


@pytest.mark.skipif(
    os.getenv("FORGEPATH_DOCKER_TESTS") != "1",
    reason="Requires real Docker execution",
)
def test_real_docker_memory_exhaustion(workspace):
    """Memory exhaustion must not escape sandbox limits."""

    if shutil.which("docker") is None:
        pytest.skip("Docker CLI unavailable")

    (workspace / "test_memory_limit.py").write_text(
        """
def test_memory_exhaustion():
    blocks = []
    while True:
        blocks.append(bytearray(16 * 1024 * 1024))
""",
        encoding="utf-8",
    )

    result = runner.execute_tests(
        workspace_root=workspace,
        runner="pytest",
        timeout_seconds=15,
    )

    assert result.status in {"failed", "timed_out", "execution_error"}
    assert result.status != "passed"


@pytest.mark.skipif(
    os.getenv("FORGEPATH_DOCKER_TESTS") != "1",
    reason="Requires real Docker execution",
)
def test_real_docker_output_limits(workspace):
    """Large test output must be truncated safely."""

    if shutil.which("docker") is None:
        pytest.skip("Docker CLI unavailable")

    (workspace / "test_large_output.py").write_text(
        """
def test_large_output():
    print("FORGEPATH_OUTPUT_" * 20000)
    assert False, "Intentional failure to expose captured output"
""",
        encoding="utf-8",
    )

    result = runner.execute_tests(
        workspace_root=workspace,
        runner="pytest",
        timeout_seconds=60,
    )

    assert result.status == "failed"
    assert result.exit_code != 0

    # Output must stay within the configured 64 KB limit.
    assert len(result.stdout.encode("utf-8")) <= 65536
    assert len(result.stderr.encode("utf-8")) <= 65536

    assert result.stdout_truncated is True



@pytest.mark.skipif(
    os.getenv("FORGEPATH_DOCKER_TESTS") != "1",
    reason="Requires real Docker execution",
)
def test_real_docker_child_process_cleanup(workspace):
    """A timed-out test must not leave its child processes running."""

    if shutil.which("docker") is None:
        pytest.skip("Docker CLI unavailable")

    (workspace / "test_child_process.py").write_text(
        """
import subprocess
import sys
import time


def test_child_process():
    subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"]
    )

    time.sleep(120)
""",
        encoding="utf-8",
    )

    result = runner.execute_tests(
        workspace_root=workspace,
        runner="pytest",
        timeout_seconds=5,
    )

    assert result.status == "timed_out"
    assert result.exit_code is None

    containers = subprocess.run(
        [
            "docker", "ps", "-a",
            "--filter", "name=forgepath-test-",
            "--format", "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )

    assert not containers.stdout.strip(), (
        f"Sandbox containers remain: {containers.stdout}"
    )


@pytest.mark.skipif(
    os.getenv("FORGEPATH_DOCKER_TESTS") != "1",
    reason="Requires real Docker execution",
)
def test_real_docker_temporary_disk_limit(workspace):
    """The sandbox must enforce its temporary filesystem size limit."""

    if shutil.which("docker") is None:
        pytest.skip("Docker CLI unavailable")

    (workspace / "test_disk_limit.py").write_text(
        """
import errno
from pathlib import Path


def test_temporary_storage_limit():
    target = Path("/work/forgepath_disk_limit.bin")

    try:
        with target.open("wb") as output:
            block = b"x" * (8 * 1024 * 1024)

            for _ in range(32):
                output.write(block)

    except OSError as exc:
        assert exc.errno == errno.ENOSPC, str(exc)

    else:
        raise AssertionError(
            "Sandbox temporary storage limit was not enforced"
        )

    finally:
        target.unlink(missing_ok=True)
""",
        encoding="utf-8",
    )

    result = runner.execute_tests(
        workspace_root=workspace,
        runner="pytest",
        timeout_seconds=60,
    )

    assert result.status == "passed", (
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )

    assert result.exit_code == 0
    assert "2 passed" in result.stdout