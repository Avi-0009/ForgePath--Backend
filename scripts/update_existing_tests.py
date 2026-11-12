"""Update only four legacy host-execution tests for the new sandbox contract."""
from pathlib import Path
p = Path('tests/test_test_execution.py')
s = p.read_text(encoding='utf-8')
start = s.index('def test_execution_captures_successful_result(')
end = s.index('@pytest.mark.parametrize(\n    "package_content"', start)
replacement = '''def test_execution_captures_successful_result(workspace, monkeypatch):
    from app.services.test_execution import sandbox
    monkeypatch.setattr(
        sandbox, "run_in_sandbox",
        lambda **kwargs: sandbox.SandboxResult(
            exit_code=0, timed_out=False, stdout="1 passed", stderr="",
            stdout_truncated=False, stderr_truncated=False,
        ),
    )
    result = runner.execute_tests(
        workspace_root=workspace, runner="pytest", timeout_seconds=30,
    )
    assert result.status == "passed"
    assert result.exit_code == 0
    assert result.runner == "pytest"
    assert "1 passed" in result.stdout


def test_execution_captures_failed_result(workspace, monkeypatch):
    from app.services.test_execution import sandbox
    monkeypatch.setattr(
        sandbox, "run_in_sandbox",
        lambda **kwargs: sandbox.SandboxResult(
            exit_code=1, timed_out=False, stdout="1 failed", stderr="",
            stdout_truncated=False, stderr_truncated=False,
        ),
    )
    result = runner.execute_tests(workspace_root=workspace, runner="pytest")
    assert result.status == "failed"
    assert result.exit_code == 1


def test_execution_handles_missing_runner(workspace, monkeypatch):
    from app.services.test_execution import sandbox
    def missing(**kwargs):
        raise sandbox.SandboxError("Docker is unavailable; tests were not executed.")
    monkeypatch.setattr(sandbox, "run_in_sandbox", missing)
    with pytest.raises(runner.TestExecutionError, match="Docker is unavailable"):
        runner.execute_tests(workspace_root=workspace, runner="pytest")


def test_execution_handles_timeout(workspace, monkeypatch):
    from app.services.test_execution import sandbox
    monkeypatch.setattr(
        sandbox, "run_in_sandbox",
        lambda **kwargs: sandbox.SandboxResult(
            exit_code=None, timed_out=True, stdout="partial output", stderr="",
            stdout_truncated=False, stderr_truncated=False,
        ),
    )
    result = runner.execute_tests(
        workspace_root=workspace, runner="pytest", timeout_seconds=1,
    )
    assert result.status == "timed_out"
    assert result.exit_code is None
    assert "partial output" in result.stdout


'''
p.write_text(s[:start]+replacement+s[end:], encoding='utf-8')
print('Updated four legacy host-execution tests in', p)