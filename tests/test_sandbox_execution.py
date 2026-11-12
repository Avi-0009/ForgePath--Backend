import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import subprocess
import tempfile

import pytest

from app.core.config import settings
from app.services.test_execution import runner, sandbox
from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceInvalidPath,
    read_configured_workspace_file,
)


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



@pytest.mark.skipif(
    os.getenv("FORGEPATH_DOCKER_TESTS") != "1",
    reason="Requires real Docker execution",
)
def test_real_docker_security_boundaries(workspace, monkeypatch):
    """Verify isolation through ForgePath's real execution path."""

    if shutil.which("docker") is None:
        pytest.skip("Docker CLI unavailable")

    monkeypatch.setenv(
        "FORGEPATH_TEST_SECRET",
        "must-not-enter-sandbox",
    )

    (workspace / "test_security_boundaries.py").write_text(
        '''
import os
import socket
from pathlib import Path

def test_security_boundaries():
    # Sensitive files must not enter the snapshot.
    assert not Path("/source/.env").exists()
    assert not Path("/work/.env").exists()

    # Backend environment secrets must not be inherited.
    assert "FORGEPATH_TEST_SECRET" not in os.environ

    # The source mount must be read-only.
    try:
        Path("/source/forbidden.txt").write_text("blocked")
    except OSError:
        pass
    else:
        raise AssertionError("Source mount is writable")

    # The container must not have outbound TCP access.
    with socket.socket() as connection:
        connection.settimeout(3)
        result = connection.connect_ex(("1.1.1.1", 443))
        assert result != 0, "Network access was permitted"

    # Sandbox must run without root privileges.
    assert os.getuid() != 0
''',
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

    # Original source must remain unchanged.
    assert not (workspace / "forbidden.txt").exists()
    assert (workspace / ".env").read_text() == (
        "SECRET=never-copy\n"
    )



def test_snapshot_rejects_oversized_file(workspace):
    from app.services.test_execution import sandbox

    large_file = workspace / "oversized.py"
    large_file.write_bytes(b"x" * (sandbox.MAX_SNAPSHOT_BYTES + 1))

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(sandbox.SandboxError):
            sandbox._snapshot(workspace, Path(tmp))


def test_snapshot_never_copies_parent_directory(workspace):
    from app.services.test_execution import sandbox

    outside = workspace.parent / "forgepath_outside_secret.txt"

    try:
        outside.write_text(
            "PRIVATE_DATA",
            encoding="utf-8",
        )

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            sandbox._snapshot(workspace, destination)

            copied = list(destination.rglob("*"))

            assert not any(
                item.is_file()
                and "PRIVATE_DATA" in item.read_text(
                    encoding="utf-8"
                )
                for item in copied
            )

    finally:
        outside.unlink(missing_ok=True)


def test_snapshot_excludes_nested_secrets(workspace, tmp_path):
    """Nested environment files and credential directories must be excluded."""

    nested = workspace / "src"
    nested.mkdir()

    (nested / "module.py").write_text(
        "VALUE = 42\n",
        encoding="utf-8",
    )

    (nested / ".env.production").write_text(
        "PRIVATE_TOKEN=do-not-copy\n",
        encoding="utf-8",
    )

    (nested / "credentials.json").write_text(
        '{"secret": "do-not-copy"}',
        encoding="utf-8",
    )

    ssh = workspace / ".ssh"
    ssh.mkdir()

    (ssh / "id_rsa").write_text(
        "PRIVATE_KEY",
        encoding="utf-8",
    )

    destination = tmp_path / "secure_snapshot"
    destination.mkdir()

    sandbox._snapshot(workspace, destination)

    assert (destination / "src" / "module.py").exists()
    assert not (destination / "src" / ".env.production").exists()
    assert not (destination / "src" / "credentials.json").exists()
    assert not (destination / ".ssh").exists()


def test_snapshot_enforces_file_count_limit(workspace, tmp_path, monkeypatch):
    """Snapshot must reject projects exceeding the configured file count."""

    (workspace / "second.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sandbox,
        "MAX_SNAPSHOT_FILES",
        1,
    )

    destination = tmp_path / "limited_snapshot"
    destination.mkdir()

    with pytest.raises(
        sandbox.SandboxError,
        match="snapshot limits",
    ):
        sandbox._snapshot(workspace, destination)


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.py",
        "../../outside.py",
        r"..\outside.py",
        "/etc/passwd",
        r"C:\Windows\system.ini",
    ],
)
def test_workspace_rejects_path_traversal(workspace, relative_path):
    """File access must reject absolute and parent-directory paths."""

    with pytest.raises(
        (WorkspaceAccessDenied, WorkspaceInvalidPath),
    ):
        read_configured_workspace_file(
            workspace_root=workspace,
            relative_path=relative_path,
        )


def test_docker_environment_excludes_backend_secrets(monkeypatch):
    """Docker CLI must not inherit backend API keys or Docker host settings."""

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "private-openai-key",
    )
    monkeypatch.setenv(
        "NVIDIA_API_KEY",
        "private-nvidia-key",
    )
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://private",
    )
    monkeypatch.setenv(
        "DOCKER_HOST",
        "tcp://untrusted-host:2375",
    )

    environment = sandbox._docker_environment()

    assert "OPENAI_API_KEY" not in environment
    assert "NVIDIA_API_KEY" not in environment
    assert "DATABASE_URL" not in environment
    assert "DOCKER_HOST" not in environment


def test_snapshot_excludes_directory_symlink(workspace, tmp_path):
    """Directory links must not allow snapshotting files outside the project."""

    outside = tmp_path / "private_directory"
    outside.mkdir()

    (outside / "secret.py").write_text(
        "SECRET = 'private'\n",
        encoding="utf-8",
    )

    linked = workspace / "linked_directory"

    try:
        linked.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip(
            "Directory symlink creation is not permitted on this system"
        )

    destination = tmp_path / "link_snapshot"
    destination.mkdir()

    sandbox._snapshot(workspace, destination)

    assert not (destination / "linked_directory").exists()
    assert not (destination / "linked_directory" / "secret.py").exists()