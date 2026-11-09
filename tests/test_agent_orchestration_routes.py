from types import SimpleNamespace
from uuid import uuid4

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.types import Command

from app.api.dependencies import get_current_user
from app.api.v1.routes.agent_orchestration import router
from app.db.session import get_db


WORKFLOW_URL = "/api/v1/agent/workflows"


class FakeGraph:
    """In-memory graph substitute for API route tests."""

    def __init__(self):
        self.workflows = {}
        self.last_initial_state = None
        self.last_resume = None

    async def ainvoke(self, graph_input, config):
        thread_id = config["configurable"]["thread_id"]

        if isinstance(graph_input, Command):
            self.last_resume = graph_input

            values = self.workflows[thread_id]

            action = graph_input.resume["action"]

            if action == "approve":
                values["status"] = "running"
                values["message"] = "Plan approved."
                values["pending_action"] = None

            elif action == "reject":
                values["status"] = "cancelled"
                values["message"] = "Plan rejected."
                values["pending_action"] = None

            elif action == "cancel":
                values["status"] = "cancelled"
                values["message"] = "Workflow cancelled."
                values["pending_action"] = None

        else:
            self.last_initial_state = graph_input

            values = dict(graph_input)

            values.update({
                "status": "awaiting_plan_approval",
                "message": "Implementation plan generated.",
                "pending_action": {
                    "kind": "plan_approval",
                    "message": "Review and approve the plan.",
                },
            })

            self.workflows[thread_id] = values

        return self.workflows[thread_id]

    async def aget_state(self, config):
        thread_id = config["configurable"]["thread_id"]

        values = self.workflows.get(thread_id, {})

        return SimpleNamespace(
            values=values,
        )


class FakeRuntimeManager:
    def __init__(self, graph):
        self.graph = graph

    async def get_runtime(self):
        return SimpleNamespace(
            graph=self.graph,
        )


@pytest.fixture
def api_client():
    """Create an isolated API application with a fake graph."""

    test_app = FastAPI()

    test_app.include_router(
        router,
        prefix="/api/v1",
    )

    graph = FakeGraph()

    test_app.state.agent_runtime_manager = (
        FakeRuntimeManager(graph)
    )

    test_user_id = uuid4()

    async def override_current_user():
        return SimpleNamespace(
            id=test_app.state.current_user_id,
            is_active=True,
        )

    async def override_db():
        yield SimpleNamespace()

    test_app.state.current_user_id = test_user_id

    test_app.dependency_overrides[get_current_user] = (
        override_current_user
    )

    test_app.dependency_overrides[get_db] = override_db

    with TestClient(test_app) as client:
        yield client, test_app, graph


def make_payload():
    return {
        "workspace_root": "C:/allowed/project",
        "task_description": "Add a user data export endpoint.",
        "planning_mode": "feature",
        "additional_context": "Reuse existing authentication.",
        "max_steps": 10,
        "test_runner": "pytest",
        "test_timeout_seconds": 120,
    }


def start_workflow(client):
    response = client.post(
        WORKFLOW_URL,
        json=make_payload(),
    )

    assert response.status_code == 200, response.text

    return response.json()


def test_agent_workflow_requires_authentication(api_client):
    client, test_app, _graph = api_client

    test_app.dependency_overrides.pop(get_current_user)

    response = client.post(
        WORKFLOW_URL,
        json=make_payload(),
    )

    assert response.status_code == 401


def test_start_agent_workflow(api_client):
    client, test_app, graph = api_client

    response_data = start_workflow(client)

    assert response_data["thread_id"]
    assert response_data["status"] == "awaiting_plan_approval"
    assert response_data["workspace_root"] == "C:/allowed/project"

    assert response_data["pending_action"]["kind"] == "plan_approval"

    assert graph.last_initial_state["owner_user_id"] == (
        str(test_app.state.current_user_id)
    )

    assert graph.last_initial_state["test_runner"] == "pytest"


def test_get_agent_workflow(api_client):
    client, _test_app, _graph = api_client

    workflow = start_workflow(client)

    response = client.get(
        f"{WORKFLOW_URL}/{workflow['thread_id']}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["thread_id"] == workflow["thread_id"]
    assert data["status"] == "awaiting_plan_approval"
    assert data["pending_action"]["kind"] == "plan_approval"


def test_agent_workflow_cannot_be_accessed_by_another_user(api_client):
    client, test_app, _graph = api_client

    workflow = start_workflow(client)

    test_app.state.current_user_id = uuid4()

    response = client.get(
        f"{WORKFLOW_URL}/{workflow['thread_id']}"
    )

    assert response.status_code == 404


def test_resume_agent_workflow_with_plan_approval(api_client):
    client, _test_app, graph = api_client

    workflow = start_workflow(client)

    response = client.post(
        f"{WORKFLOW_URL}/{workflow['thread_id']}/resume",
        json={
            "action": "approve",
        },
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["status"] == "running"
    assert data["pending_action"] is None

    assert isinstance(graph.last_resume, Command)
    assert graph.last_resume.resume["action"] == "approve"


def test_resume_agent_workflow_rejects_invalid_action(api_client):
    client, _test_app, graph = api_client

    workflow = start_workflow(client)

    response = client.post(
        f"{WORKFLOW_URL}/{workflow['thread_id']}/resume",
        json={
            "action": "select_file",
            "selected_file": "app/main.py",
        },
    )

    assert response.status_code == 409
    assert graph.last_resume is None


def test_resume_agent_workflow_requires_file_for_selection(api_client):
    client, _test_app, _graph = api_client

    workflow = start_workflow(client)

    response = client.post(
        f"{WORKFLOW_URL}/{workflow['thread_id']}/resume",
        json={
            "action": "select_file",
        },
    )

    assert response.status_code == 422


def test_resume_agent_workflow_rejects_extra_fields(api_client):
    client, _test_app, _graph = api_client

    workflow = start_workflow(client)

    response = client.post(
        f"{WORKFLOW_URL}/{workflow['thread_id']}/resume",
        json={
            "action": "approve",
            "execute_anything": True,
        },
    )

    assert response.status_code == 422


def test_get_unknown_agent_workflow_returns_404(api_client):
    client, _test_app, _graph = api_client

    response = client.get(
        f"{WORKFLOW_URL}/{uuid4()}"
    )

    assert response.status_code == 404


def test_agent_workflow_rejects_invalid_thread_id(api_client):
    client, _test_app, _graph = api_client

    response = client.get(
        f"{WORKFLOW_URL}/not-a-uuid"
    )

    assert response.status_code == 404