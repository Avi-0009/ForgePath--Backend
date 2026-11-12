import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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