import asyncio
import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.schemas.agent_orchestration import (
    AgentWorkflowResumeRequest,
)
from app.schemas.code_patch import CodePatchResponse
from app.schemas.execution import (
    TestExecutionResponse as ExecutionResponse,
)
from app.schemas.task_planning import (
    TaskPlanStep,
    TaskPlanningResponse,
)
from app.services.agent_orchestration import nodes
from app.services.agent_orchestration.graph import (
    build_agent_graph,
    route_after_test_execution,
    route_after_failure_analysis,
    route_after_repair_file_selection,
    route_after_patch_approval,
)
from app.services.test_execution.runner import (
    TestExecutionResult as ExecutionResult,
)


def make_plan(workspace_root):
    return TaskPlanningResponse(
        workspace_root=workspace_root,
        task_summary="Add a health endpoint.",
        objective="Expose a health-check API endpoint.",
        complexity="low",
        steps=[
            TaskPlanStep(
                step_number=1,
                title="Implement health endpoint",
                description="Add a health-check endpoint.",
                dependencies=[],
                candidate_files_to_inspect=["main.py"],
                acceptance_criteria=[
                    "The health endpoint returns HTTP 200."
                ],
                verification_actions=[
                    "Add an endpoint test."
                ],
            )
        ],
        risks=[],
        open_questions=[],
        overall_verification_strategy=[
            "Run the relevant API tests."
        ],
    )


def make_patch():
    proposed_code = "print('health endpoint implemented')\n"

    return CodePatchResponse(
        relative_path="main.py",
        original_sha256="a" * 64,
        proposed_sha256=hashlib.sha256(
            proposed_code.encode("utf-8")
        ).hexdigest(),
        original_line_count=1,
        proposed_line_count=1,
        summary="Implement the health endpoint.",
        diff="--- a/main.py\n+++ b/main.py\n",
        proposed_code=proposed_code,
        patch_applied=False,
    )


def test_real_agent_graph_completes_approval_workflow(monkeypatch):
    """Exercise all real graph interrupts and resume transitions."""

    workspace_root = "C:/allowed/project"
    user_id = str(uuid4())
    thread_id = str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 200,
    }

    counters = {
        "patch_generation": 0,
        "patch_application": 0,
        "test_execution": 0,
    }

    async def fake_get_project_instructions(**kwargs):
        return None

    async def fake_generate_task_plan(
        request,
        *,
        project_instructions=None,
    ):
        return make_plan(request.workspace_root)

    def fake_read_workspace_file(**kwargs):
        return SimpleNamespace(
            relative_path="main.py",
            sha256="a" * 64,
            content="print('original')\n",
            language="python",
        )

    async def fake_generate_code_patch(request):
        counters["patch_generation"] += 1
        assert request.relative_path == "main.py"
        assert request.expected_sha256 == "a" * 64
        return make_patch()

    async def fake_apply_code_patch(request):
        counters["patch_application"] += 1

        assert request.relative_path == "main.py"
        assert request.approval == "apply"

        return SimpleNamespace(
            relative_path="main.py",
            backup_id="b" * 32,
            original_sha256="a" * 64,
            applied_sha256=make_patch().proposed_sha256,
            patch_applied=True,
        )

    def fake_execute_tests(**kwargs):
        counters["test_execution"] += 1

        assert kwargs["workspace_root"] == workspace_root
        assert kwargs["runner"] == "pytest"

        now = datetime.now(timezone.utc)

        return ExecutionResult(
            execution_id=uuid4(),
            workspace_root=workspace_root,
            runner="pytest",
            status="passed",
            command=["python", "-m", "pytest", "-q"],
            exit_code=0,
            duration_ms=1000,
            stdout="5 passed",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr(
        nodes,
        "get_project_instructions",
        fake_get_project_instructions,
    )

    monkeypatch.setattr(
        nodes,
        "generate_task_plan",
        fake_generate_task_plan,
    )

    monkeypatch.setattr(
        nodes,
        "read_configured_workspace_file",
        fake_read_workspace_file,
    )

    monkeypatch.setattr(
        nodes,
        "generate_code_patch",
        fake_generate_code_patch,
    )

    monkeypatch.setattr(
        nodes,
        "apply_code_patch",
        fake_apply_code_patch,
    )

    monkeypatch.setattr(
        nodes,
        "execute_tests",
        fake_execute_tests,
    )

    graph = build_agent_graph(
        checkpointer=InMemorySaver()
    )

    initial_state = {
        "thread_id": thread_id,
        "owner_user_id": user_id,
        "workspace_root": workspace_root,
        "task_description": "Add a health-check endpoint.",
        "planning_mode": "feature",
        "additional_context": None,
        "max_steps": 5,
        "test_runner": "pytest",
        "test_timeout_seconds": 120,
        "current_step_index": 0,
        "completed_steps": [],
        "manual_steps": [],
        "skipped_steps": [],
        "applied_patches": [],
        "selected_file": None,
        "pending_action": None,
        "patch_proposal": None,
        "test_result": None,
        "error_message": None,
    }

    async def run_workflow():
        # 1. Start workflow and pause for plan approval.
        result = await graph.ainvoke(
            initial_state,
            config=config,
        )

        assert "__interrupt__" in result

        plan_interrupt = result["__interrupt__"][0]

        assert plan_interrupt.value["kind"] == "plan_approval"
        assert (
            plan_interrupt.response_schema
            == AgentWorkflowResumeRequest.model_json_schema()
        )

        # 2. Approve plan and pause for file selection.
        result = await graph.ainvoke(
            Command(
                resume={
                    "action": "approve",
                }
            ),
            config=config,
        )

        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value["kind"] == "file_selection"

        # 3. Select the source file and pause for patch approval.
        result = await graph.ainvoke(
            Command(
                resume={
                    "action": "select_file",
                    "selected_file": "main.py",
                }
            ),
            config=config,
        )

        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value["kind"] == "patch_approval"

        assert counters["patch_generation"] == 1
        assert counters["patch_application"] == 0

        # 4. Approve patch. It is applied, then graph pauses for tests.
        result = await graph.ainvoke(
            Command(
                resume={
                    "action": "approve",
                }
            ),
            config=config,
        )

        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value["kind"] == "test_approval"

        assert counters["patch_application"] == 1
        assert counters["test_execution"] == 0

        # 5. Approve test execution.
        result = await graph.ainvoke(
            Command(
                resume={
                    "action": "approve",
                }
            ),
            config=config,
        )

        assert "__interrupt__" not in result

        # 6. Verify final persisted workflow state.
        snapshot = await graph.aget_state(config)
        values = snapshot.values

        assert values["status"] == "completed"
        assert values["completed_steps"] == [1]
        assert values["manual_steps"] == []
        assert values["skipped_steps"] == []
        assert len(values["applied_patches"]) == 1

        patch_record = values["applied_patches"][0]

        assert patch_record["relative_path"] == "main.py"
        assert patch_record["backup_id"] == "b" * 32

        test_result = ExecutionResponse.model_validate(
            values["test_result"]
        )

        assert test_result.status == "passed"
        assert test_result.exit_code == 0

        assert counters == {
            "patch_generation": 1,
            "patch_application": 1,
            "test_execution": 1,
        }

    asyncio.run(run_workflow())


def test_agent_graph_requires_checkpointer():
    """Approval interrupts must never run without persistence."""

    import pytest

    with pytest.raises(
        ValueError,
        match="checkpointer is required",
    ):
        build_agent_graph(checkpointer=None)


def test_failed_tests_route_to_failure_analysis():
    state = {
        "status": "running",
        "test_result": {"status": "failed"},
    }
    assert route_after_test_execution(state) == "analyze_test_failure"


def test_passed_tests_end_workflow():
    state = {
        "status": "completed",
        "test_result": {"status": "passed"},
    }
    assert route_after_test_execution(state) == "__end__"


def test_failure_analysis_routes_to_file_selection():
    state = {"status": "awaiting_repair_file_selection"}
    assert route_after_failure_analysis(state) == "select_repair_file"


def test_repair_file_selection_routes_to_patch_generation():
    state = {
        "status": "running",
        "selected_file": "main.py",
    }
    assert route_after_repair_file_selection(state) == "generate_repair_patch"


def test_approved_repair_routes_back_to_test_approval():
    state = {
        "status": "awaiting_test_approval",
        "repair_mode": True,
    }
    assert route_after_patch_approval(state) == "approve_and_run_tests"