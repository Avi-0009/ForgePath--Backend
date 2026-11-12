"""Run once from repository root: python scripts/apply_runner_patch.py"""
from pathlib import Path

path = Path('app/services/test_execution/runner.py')
src = path.read_text(encoding='utf-8')
start = src.index('def execute_tests(\n')
assert 'Execution takes place under the backend process' in src[start:], 'Unexpected runner version: review manually.'
assert 'def ' not in src[start + len('def execute_tests('):].split('\n', 1)[0], 'Unexpected runner version.'
replacement = '''def execute_tests(
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
'''
path.write_text(src[:start] + replacement, encoding='utf-8')
print('Patched', path)