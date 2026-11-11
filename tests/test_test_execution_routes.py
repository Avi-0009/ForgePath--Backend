from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.main import app
from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceConfigurationError,
)
from app.services.test_execution.runner import (
    TestExecutionError as ExecutionError,
    TestExecutionResult as ExecutionResult,
    TestProjectConfigurationError as ProjectConfigurationError,
    TestRunnerUnavailable as RunnerUnavailable,
)
from app.api.v1.routes import execution as test_execution_route


TEST_URL = "/api/v1/test-execution/run"


@pytest.fixture
def api_client(monkeypatch):
    """Create an authenticated API client without a real database."""

    monkeypatch.setattr(
        settings,
        "embedding_provider",
        None,
    )

    async def override_db():
        yield SimpleNamespace()

    async def override_current_user():
        return SimpleNamespace(
            id=uuid4(),
            is_active=True,
        )

    app.dependency_overrides.clear()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = (
        override_current_user
    )

    try:
        with TestClient(app) as client:
            yield client

    finally:
        app.dependency_overrides.clear()


def make_payload():
    return {
        "workspace_root": "C:/allowed/project",
        "runner": "pytest",
        "timeout_seconds": 30,
        "approval": "run",
    }


def make_execution_result():
    now = datetime.now(timezone.utc)

    return ExecutionResult(
        execution_id=uuid4(),
        workspace_root="C:/allowed/project",
        runner="pytest",
        status="passed",
        command=[
            "python",
            "-m",
            "pytest",
            "-q",
        ],
        exit_code=0,
        duration_ms=1250,
        stdout="2 passed in 0.50s",
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
        started_at=now,
        finished_at=now,
    )


def test_test_execution_requires_authentication(api_client):
    """Requests without an access token must be rejected."""

    app.dependency_overrides.pop(get_current_user)

    response = api_client.post(
        TEST_URL,
        json=make_payload(),
    )

    assert response.status_code == 401


def test_test_execution_requires_explicit_approval(api_client):
    """Only the explicit run approval value is accepted."""

    payload = make_payload()
    payload["approval"] = "yes"

    response = api_client.post(
        TEST_URL,
        json=payload,
    )

    assert response.status_code == 422


def test_test_execution_success(api_client, monkeypatch):
    """A successful local execution returns its structured result."""

    captured = {}

    def fake_execute_tests(**kwargs):
        captured.update(kwargs)
        return make_execution_result()

    monkeypatch.setattr(
        test_execution_route,
        "execute_tests",
        fake_execute_tests,
    )

    response = api_client.post(
        TEST_URL,
        json=make_payload(),
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["status"] == "passed"
    assert data["runner"] == "pytest"
    assert data["exit_code"] == 0
    assert data["duration_ms"] == 1250
    assert data["stdout"] == "2 passed in 0.50s"
    assert data["stderr"] == ""
    assert data["stdout_truncated"] is False
    assert data["stderr_truncated"] is False

    assert captured["workspace_root"] == "C:/allowed/project"
    assert captured["runner"] == "pytest"
    assert captured["timeout_seconds"] == 30


@pytest.mark.parametrize(
    ("execution_error", "expected_status"),
    [
        (
            WorkspaceAccessDenied("Workspace is not allowlisted."),
            403,
        ),
        (
            WorkspaceConfigurationError("Invalid workspace configuration."),
            500,
        ),
        (
            RunnerUnavailable("pytest is unavailable."),
            503,
        ),
        (
            ProjectConfigurationError("Invalid project configuration."),
            400,
        ),
        (
            ExecutionError("Execution failed."),
            500,
        ),
    ],
)
def test_test_execution_error_responses(
    api_client,
    monkeypatch,
    execution_error,
    expected_status,
):
    """Execution errors are translated into the expected HTTP status."""

    def fake_execute_tests(**kwargs):
        raise execution_error

    monkeypatch.setattr(
        test_execution_route,
        "execute_tests",
        fake_execute_tests,
    )

    response = api_client.post(
        TEST_URL,
        json=make_payload(),
    )

    assert response.status_code == expected_status
    assert "detail" in response.json()


def test_test_execution_rejects_extra_request_fields(api_client):
    """Unexpected request fields must not be accepted."""

    payload = make_payload()
    payload["command"] = "some arbitrary command"

    response = api_client.post(
        TEST_URL,
        json=payload,
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "unsupported_runner",
    [
        "powershell",
        "bash",
        "cmd.exe",
        "pytest && whoami",
    ],
)
def test_test_execution_rejects_unsupported_runner(
    api_client,
    unsupported_runner,
):
    payload = make_payload()
    payload["runner"] = unsupported_runner

    response = api_client.post(
        TEST_URL,
        json=payload,
    )

    assert response.status_code == 422