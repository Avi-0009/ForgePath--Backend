from types import SimpleNamespace
from uuid import uuid4
from datetime import datetime, timezone

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.types import Command

from app.api.dependencies import get_current_user
from app.api.v1.routes.agent_orchestration import router
from app.db.session import get_db
from app.api.v1.routes import agent_orchestration as agent_orchestration_routes


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
def api_client(monkeypatch):
    """Create an isolated API application with fake graph and history storage."""

    test_app = FastAPI()

    test_app.include_router(
        router,
        prefix="/api/v1",
    )

    graph = FakeGraph()

    test_app.state.agent_runtime_manager = FakeRuntimeManager(graph)

    test_user_id = uuid4()
    history_records = {}

    async def fake_create_history(_db, **kwargs):
        now = datetime.now(timezone.utc)

        record = SimpleNamespace(
            id=uuid4(),
            created_at=now,
            updated_at=now,
            finished_at=None,
            status="running",
            message=None,
            total_steps=0,
            completed_steps=0,
            repair_attempts=0,
            **kwargs,
        )

        history_records[record.thread_id] = record
        return record

    async def fake_sync_history(_db, record, state):
        now = datetime.now(timezone.utc)

        record.status = state.get("status", "running")
        record.message = (state.get("message") or "")[:2000] or None

        plan = state.get("plan")
        if isinstance(plan, dict):
            record.total_steps = len(plan.get("steps", []))

        record.completed_steps = len(
            set(state.get("completed_steps", []))
        )

        record.repair_attempts = state.get("repair_attempts", 0)
        record.updated_at = now

        terminal_statuses = {
            "completed",
            "completed_without_tests",
            "tests_failed",
            "cancelled",
            "failed",
        }

        if record.status in terminal_statuses:
            if record.finished_at is None:
                record.finished_at = now
        else:
            record.finished_at = None

        return record

    async def fake_get_history(_db, *, user_id, thread_id):
        record = history_records.get(thread_id)

        if record is None or record.user_id != user_id:
            return None

        return record

    async def fake_list_history(
        _db,
        *,
        user_id,
        limit=20,
        offset=0,
        status=None,
    ):
        records = [
            record
            for record in history_records.values()
            if record.user_id == user_id
            and (status is None or record.status == status)
        ]

        records.sort(
            key=lambda record: (
                record.created_at,
                str(record.id),
            ),
            reverse=True,
        )

        total = len(records)

        return records[offset:offset + limit], total

    # Mock the persistence functions used by the API routes.
    monkeypatch.setattr(
        agent_orchestration_routes,
        "create_agent_workflow_history",
        fake_create_history,
    )

    monkeypatch.setattr(
        agent_orchestration_routes,
        "sync_agent_workflow_history",
        fake_sync_history,
    )

    monkeypatch.setattr(
        agent_orchestration_routes,
        "get_agent_workflow_history",
        fake_get_history,
    )

    monkeypatch.setattr(
        agent_orchestration_routes,
        "list_agent_workflow_history",
        fake_list_history,
    )

    test_app.state.history_records = history_records
    test_app.state.current_user_id = test_user_id

    async def override_current_user():
        return SimpleNamespace(
            id=test_app.state.current_user_id,
            is_active=True,
        )

    async def override_db():
        yield SimpleNamespace()

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


def test_list_agent_workflow_history(api_client):
    client, _test_app, _graph = api_client

    workflow = start_workflow(client)

    response = client.get(
        f"{WORKFLOW_URL}/history",
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 1
    assert data["limit"] == 20
    assert data["offset"] == 0
    assert len(data["items"]) == 1

    item = data["items"][0]

    assert item["thread_id"] == workflow["thread_id"]
    assert item["workspace_root"] == "C:/allowed/project"
    assert item["status"] == "awaiting_plan_approval"
    assert item["repair_attempts"] == 0
    assert item["finished_at"] is None


def test_get_agent_workflow_history_record(api_client):
    client, _test_app, _graph = api_client

    workflow = start_workflow(client)

    response = client.get(
        f"{WORKFLOW_URL}/history/{workflow['thread_id']}",
    )

    assert response.status_code == 200

    data = response.json()

    assert data["thread_id"] == workflow["thread_id"]
    assert data["task_description"] == (
        "Add a user data export endpoint."
    )
    assert data["planning_mode"] == "feature"
    assert data["test_runner"] == "pytest"
    assert data["status"] == "awaiting_plan_approval"


def test_history_status_filter_and_sync(api_client):
    client, _test_app, _graph = api_client

    workflow = start_workflow(client)

    # Resume the workflow, changing its status to running.
    resume_response = client.post(
        f"{WORKFLOW_URL}/{workflow['thread_id']}/resume",
        json={"action": "approve"},
    )

    assert resume_response.status_code == 200

    running_response = client.get(
        f"{WORKFLOW_URL}/history",
        params={"status": "running"},
    )

    assert running_response.status_code == 200
    assert running_response.json()["total"] == 1
    assert running_response.json()["items"][0]["status"] == "running"

    waiting_response = client.get(
        f"{WORKFLOW_URL}/history",
        params={"status": "awaiting_plan_approval"},
    )

    assert waiting_response.status_code == 200
    assert waiting_response.json()["total"] == 0


def test_history_is_isolated_by_user(api_client):
    client, test_app, _graph = api_client

    workflow = start_workflow(client)

    # Switch to another authenticated user.
    test_app.state.current_user_id = uuid4()

    list_response = client.get(
        f"{WORKFLOW_URL}/history",
    )

    assert list_response.status_code == 200
    assert list_response.json()["total"] == 0
    assert list_response.json()["items"] == []

    detail_response = client.get(
        f"{WORKFLOW_URL}/history/{workflow['thread_id']}",
    )

    assert detail_response.status_code == 404


def test_history_pagination(api_client):
    client, _test_app, _graph = api_client

    first_workflow = start_workflow(client)
    second_workflow = start_workflow(client)

    response = client.get(
        f"{WORKFLOW_URL}/history",
        params={
            "limit": 1,
            "offset": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2
    assert data["limit"] == 1
    assert data["offset"] == 0
    assert len(data["items"]) == 1

    # The two records must be distinct.
    assert first_workflow["thread_id"] != second_workflow["thread_id"]

    next_page = client.get(
        f"{WORKFLOW_URL}/history",
        params={
            "limit": 1,
            "offset": 1,
        },
    )

    assert next_page.status_code == 200
    assert next_page.json()["total"] == 2
    assert len(next_page.json()["items"]) == 1

    assert (
        data["items"][0]["thread_id"]
        != next_page.json()["items"][0]["thread_id"]
    )