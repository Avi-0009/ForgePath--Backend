import io
import json
import subprocess

import pytest

from app.core.config import settings
from app.services.test_execution import runner
from app.services.workspace import WorkspaceAccessDenied


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Create an isolated, allowlisted test workspace."""

    project = tmp_path / "project"
    project.mkdir()

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(tmp_path),
    )

    return project


def test_pytest_command_uses_predefined_arguments(workspace):
    command = runner._build_command(workspace, "pytest")

    assert command[-3:] == ["-m", "pytest", "-q"]


def test_npm_command_requires_test_script(workspace):
    package_file = workspace / "package.json"

    package_file.write_text(
        json.dumps({
            "scripts": {
                "test": "vitest run",
            }
        }),
        encoding="utf-8",
    )

    command = runner._build_command(workspace, "npm")

    if runner.os.name == "nt":
        assert command == ["cmd.exe", "/d", "/c", "npm test"]
    else:
        assert command == ["npm", "test"]


def test_npm_command_rejects_missing_test_script(workspace):
    package_file = workspace / "package.json"

    package_file.write_text(
        json.dumps({"scripts": {}}),
        encoding="utf-8",
    )

    with pytest.raises(runner.TestProjectConfigurationError):
        runner._build_command(workspace, "npm")


def test_go_command_requires_go_mod(workspace):
    with pytest.raises(runner.TestProjectConfigurationError):
        runner._build_command(workspace, "go")


def test_go_command_uses_predefined_arguments(workspace):
    (workspace / "go.mod").write_text(
        "module example.com/test\n\ngo 1.23\n",
        encoding="utf-8",
    )

    command = runner._build_command(workspace, "go")

    assert command == ["go", "test", "./..."]


def test_output_capture_enforces_limit():
    capture = runner._BoundedOutput(max_bytes=5)

    capture.read_stream(io.BytesIO(b"hello world"))

    assert capture.text() == "hello"
    assert capture.truncated is True


def test_test_environment_does_not_expose_backend_secrets(
    monkeypatch,
):
    monkeypatch.setenv(
        "JWT_SECRET_KEY",
        "test-secret",
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-openai-key",
    )

    environment = runner._build_test_environment()

    assert "JWT_SECRET_KEY" not in environment
    assert "OPENAI_API_KEY" not in environment

    assert environment["CI"] == "true"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["PYTHONUNBUFFERED"] == "1"


def test_execution_rejects_workspace_outside_allowlist(
    workspace,
    tmp_path,
):
    outside_workspace = tmp_path.parent / "outside"
    outside_workspace.mkdir(exist_ok=True)

    with pytest.raises(WorkspaceAccessDenied):
        runner.execute_tests(
            workspace_root=outside_workspace,
            runner="pytest",
        )


def test_execution_rejects_invalid_timeout(workspace):
    with pytest.raises(ValueError):
        runner.execute_tests(
            workspace_root=workspace,
            runner="pytest",
            timeout_seconds=0,
        )


def test_execution_captures_successful_result(workspace):
    test_file = workspace / "test_sample.py"

    test_file.write_text(
        "def test_example():\n"
        "    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )

    result = runner.execute_tests(
        workspace_root=workspace,
        runner="pytest",
        timeout_seconds=30,
    )

    assert result.status == "passed"
    assert result.exit_code == 0
    assert result.runner == "pytest"
    assert result.execution_id is not None
    assert result.duration_ms >= 0
    assert "1 passed" in result.stdout


def test_execution_captures_failed_result(workspace):
    test_file = workspace / "test_sample.py"

    test_file.write_text(
        "def test_example():\n"
        "    assert 2 + 2 == 5\n",
        encoding="utf-8",
    )

    result = runner.execute_tests(
        workspace_root=workspace,
        runner="pytest",
        timeout_seconds=30,
    )

    assert result.status == "failed"
    assert result.exit_code != 0


def test_execution_handles_missing_runner(
    workspace,
    monkeypatch,
):
    (workspace / "go.mod").write_text(
        "module example.com/test\n",
        encoding="utf-8",
    )

    def missing_process(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        missing_process,
    )

    with pytest.raises(runner.TestRunnerUnavailable):
        runner.execute_tests(
            workspace_root=workspace,
            runner="go",
        )


def test_execution_handles_timeout(
    workspace,
    monkeypatch,
):
    (workspace / "go.mod").write_text(
        "module example.com/test\n",
        encoding="utf-8",
    )

    class FakeProcess:
        def __init__(self):
            self.stdout = io.BytesIO(b"partial output")
            self.stderr = io.BytesIO(b"")
            self.returncode = -9
            self.wait_count = 0

        def wait(self, timeout=None):
            self.wait_count += 1

            if self.wait_count == 1:
                raise subprocess.TimeoutExpired(
                    cmd="go test ./...",
                    timeout=1,
                )

            return self.returncode

        def kill(self):
            self.returncode = -9

    fake_process = FakeProcess()

    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *args, **kwargs: fake_process,
    )

    monkeypatch.setattr(
        runner,
        "_stop_process_tree",
        lambda process: None,
    )

    result = runner.execute_tests(
        workspace_root=workspace,
        runner="go",
        timeout_seconds=1,
    )

    assert result.status == "timed_out"
    assert result.exit_code is None
    assert "partial output" in result.stdout