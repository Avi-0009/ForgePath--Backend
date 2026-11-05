from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.ai_provider import AIProviderError
from app.core.config import settings
from app.db.session import get_db
from app.main import app
from app.schemas.task_planning import (
    TaskPlanStep,
    TaskPlanningResponse,
)
from app.services.local_context import LocalContextError
from app.services.task_planning import planner
from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceConfigurationError,
    WorkspaceReadError,
)
from app.api.v1.routes import task_planning as task_planning_route


PLAN_URL = "/api/v1/task-planning/plan"


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
        "task_description": "Add a user data export endpoint.",
        "planning_mode": "feature",
        "additional_context": "Reuse existing authentication.",
        "max_steps": 10,
    }


def make_plan():
    return TaskPlanningResponse(
        workspace_root="C:/allowed/project",
        task_summary="Add a user data export endpoint.",
        objective="Allow authenticated users to export their data.",
        complexity="medium",
        assumptions=[],
        steps=[
            TaskPlanStep(
                step_number=1,
                title="Inspect user services",
                description="Understand the existing user data access.",
                dependencies=[],
                candidate_files_to_inspect=[
                    "app/services/users.py",
                ],
                acceptance_criteria=[
                    "Existing user data access is understood."
                ],
                verification_actions=[
                    "Review existing service tests."
                ],
            ),
            TaskPlanStep(
                step_number=2,
                title="Implement export endpoint",
                description="Add an endpoint using existing conventions.",
                dependencies=[1],
                candidate_files_to_inspect=[
                    "app/api/v1/routes/users.py",
                ],
                acceptance_criteria=[
                    "Authenticated users can request an export."
                ],
                verification_actions=[
                    "Add endpoint tests."
                ],
            ),
        ],
        risks=[],
        open_questions=[],
        overall_verification_strategy=[
            "Run relevant API tests."
        ],
        generated_at=datetime.now(timezone.utc),
    )


def test_task_planning_requires_authentication(api_client):
    """Requests without an authenticated user must be rejected."""

    app.dependency_overrides.pop(get_current_user)

    response = api_client.post(
        PLAN_URL,
        json=make_payload(),
    )

    assert response.status_code == 401


def test_task_planning_success(api_client, monkeypatch):
    """A valid planning request returns a structured implementation plan."""

    captured = {}

    async def fake_get_project_instructions(**kwargs):
        captured["context_workspace"] = kwargs["project_root"]
        return SimpleNamespace(
            instructions="Use existing project conventions."
        )

    async def fake_generate_task_plan(
        payload,
        *,
        project_instructions=None,
    ):
        captured["payload"] = payload
        captured["project_instructions"] = project_instructions
        return make_plan()

    monkeypatch.setattr(
        task_planning_route,
        "get_project_instructions",
        fake_get_project_instructions,
    )

    monkeypatch.setattr(
        task_planning_route,
        "generate_task_plan",
        fake_generate_task_plan,
    )

    response = api_client.post(
        PLAN_URL,
        json=make_payload(),
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["workspace_root"] == "C:/allowed/project"
    assert data["task_summary"] == "Add a user data export endpoint."
    assert data["complexity"] == "medium"
    assert len(data["steps"]) == 2
    assert data["steps"][1]["dependencies"] == [1]
    assert data["plan_id"]

    assert captured["context_workspace"] == "C:/allowed/project"
    assert (
        captured["project_instructions"]
        == "Use existing project conventions."
    )

    assert captured["payload"].planning_mode == "feature"


def test_task_planning_continues_without_saved_context(
    api_client,
    monkeypatch,
):
    """A failed optional context lookup must not block planning."""

    async def fake_get_project_instructions(**kwargs):
        raise LocalContextError("Context database unavailable.")

    async def fake_generate_task_plan(
        payload,
        *,
        project_instructions=None,
    ):
        assert project_instructions is None
        return make_plan()

    monkeypatch.setattr(
        task_planning_route,
        "get_project_instructions",
        fake_get_project_instructions,
    )

    monkeypatch.setattr(
        task_planning_route,
        "generate_task_plan",
        fake_generate_task_plan,
    )

    response = api_client.post(
        PLAN_URL,
        json=make_payload(),
    )

    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    ("planning_error", "expected_status"),
    [
        (
            WorkspaceAccessDenied("Workspace is not allowlisted."),
            403,
        ),
        (
            WorkspaceConfigurationError(
                "Invalid workspace configuration."
            ),
            500,
        ),
        (
            WorkspaceReadError("Could not read workspace."),
            400,
        ),
        (
            AIProviderError("AI provider unavailable."),
            502,
        ),
        (
            planner.TaskPlanningError("Invalid planning result."),
            502,
        ),
    ],
)
def test_task_planning_error_responses(
    api_client,
    monkeypatch,
    planning_error,
    expected_status,
):
    """Planning errors are mapped to the expected HTTP statuses."""

    async def fake_get_project_instructions(**kwargs):
        return None

    async def fake_generate_task_plan(
        payload,
        *,
        project_instructions=None,
    ):
        raise planning_error

    monkeypatch.setattr(
        task_planning_route,
        "get_project_instructions",
        fake_get_project_instructions,
    )

    monkeypatch.setattr(
        task_planning_route,
        "generate_task_plan",
        fake_generate_task_plan,
    )

    response = api_client.post(
        PLAN_URL,
        json=make_payload(),
    )

    assert response.status_code == expected_status
    assert "detail" in response.json()


def test_task_planning_rejects_invalid_max_steps(api_client):
    """The API must enforce the maximum step limit."""

    payload = make_payload()
    payload["max_steps"] = 25

    response = api_client.post(
        PLAN_URL,
        json=payload,
    )

    assert response.status_code == 422


def test_task_planning_rejects_extra_request_fields(api_client):
    """Unexpected request fields must not be accepted."""

    payload = make_payload()
    payload["execute_plan"] = True

    response = api_client.post(
        PLAN_URL,
        json=payload,
    )

    assert response.status_code == 422


def test_task_planning_rejects_empty_task_description(api_client):
    """A whitespace-only task description must be rejected."""

    payload = make_payload()
    payload["task_description"] = "   "

    response = api_client.post(
        PLAN_URL,
        json=payload,
    )

    assert response.status_code == 422