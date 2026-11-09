import logging
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from langgraph.types import Command

from app.api.dependencies import get_current_user
from app.core.ai_provider import AIProviderError
from app.models.user import User
from app.schemas.agent_orchestration import (
    AgentAppliedPatch,
    AgentPendingAction,
    AgentWorkflowResponse,
    AgentWorkflowResumeRequest,
    AgentWorkflowStartRequest,
)
from app.schemas.code_patch import CodePatchResponse
from app.schemas.execution import TestExecutionResponse
from app.schemas.task_planning import TaskPlanningResponse
from app.services.agent_orchestration.runtime import (
    AgentRuntimeError,
)
from app.schemas.error_analysis import TestFailureAnalysis


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent/workflows",
    tags=["Agent Orchestration"],
)

RECURSION_LIMIT = 200


def _get_config(thread_id: str) -> dict:
    """Build the persistent LangGraph thread configuration."""

    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": RECURSION_LIMIT,
    }


async def _get_runtime(request: Request):
    """Retrieve the application's persistent agent runtime."""

    manager = getattr(
        request.app.state,
        "agent_runtime_manager",
        None,
    )

    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent orchestration is unavailable.",
        )

    try:
        return await manager.get_runtime()

    except AgentRuntimeError:
        logger.exception(
            "Could not initialize the agent runtime."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The agent runtime could not be initialized.",
        ) from None


async def _load_owned_snapshot(
    runtime,
    thread_id: str,
    user_id: str,
):
    """Retrieve a workflow checkpoint and enforce user ownership."""

    try:
        normalized_thread_id = str(UUID(thread_id))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found.",
        ) from None

    try:
        snapshot = await runtime.graph.aget_state(
            _get_config(normalized_thread_id)
        )

    except Exception:
        logger.exception(
            "Could not retrieve agent workflow state."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve workflow state.",
        ) from None

    values = snapshot.values

    if (
        not isinstance(values, dict)
        or not values
        or values.get("owner_user_id") != user_id
    ):
        # Do not reveal whether another user's workflow exists.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found.",
        )

    return normalized_thread_id, snapshot


def _build_workflow_response(
    thread_id: str,
    values: dict,
) -> AgentWorkflowResponse:
    """Convert persisted graph state into a safe API response."""

    plan_data = values.get("plan")
    patch_data = values.get("patch_proposal")
    test_data = values.get("test_result")
    analysis_data = values.get("failure_analysis")
    pending_data = values.get("pending_action")

    plan = (
        TaskPlanningResponse.model_validate(plan_data)
        if isinstance(plan_data, dict)
        else None
    )

    patch = (
        CodePatchResponse.model_validate(patch_data)
        if isinstance(patch_data, dict)
        else None
    )

    test_result = (
        TestExecutionResponse.model_validate(test_data)
        if isinstance(test_data, dict)
        else None
    )

    failure_analysis = (
        TestFailureAnalysis.model_validate(analysis_data)
        if isinstance(analysis_data, dict)
        else None
    )

    pending_action = (
        AgentPendingAction.model_validate(pending_data)
        if isinstance(pending_data, dict)
        else None
    )

    applied_patches = [
        AgentAppliedPatch.model_validate(item)
        for item in values.get("applied_patches", [])
    ]

    current_step_number = None

    if pending_action and pending_action.step_number:
        current_step_number = pending_action.step_number

    elif plan is not None:
        index = values.get("current_step_index", 0)

        if 0 <= index < len(plan.steps):
            current_step_number = plan.steps[index].step_number

    return AgentWorkflowResponse(
        thread_id=thread_id,
        status=values.get("status", "running"),
        workspace_root=values["workspace_root"],
        message=values.get(
            "message",
            "Agent workflow is running.",
        ),
        plan=plan,
        current_step_number=current_step_number,
        pending_action=pending_action,
        patch_proposal=patch,
        applied_patches=applied_patches,
        manual_steps=values.get("manual_steps", []),
        skipped_steps=values.get("skipped_steps", []),
        test_result=test_result,
        failure_analysis=failure_analysis,
        repair_attempts=values.get("repair_attempts", 0),
        max_repair_attempts=values.get("max_repair_attempts", 2),
        error_message=values.get("error_message"),
    )


@router.post(
    "",
    response_model=AgentWorkflowResponse,
    status_code=status.HTTP_200_OK,
)
async def start_agent_workflow(
    payload: AgentWorkflowStartRequest,
    request: Request,
    current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> AgentWorkflowResponse:
    """Start a new resumable agent workflow."""

    runtime = await _get_runtime(request)

    thread_id = str(uuid4())

    initial_state = {
        "thread_id": thread_id,
        "owner_user_id": str(current_user.id),
        "workspace_root": payload.workspace_root,
        "task_description": payload.task_description,
        "planning_mode": payload.planning_mode,
        "additional_context": payload.additional_context,
        "max_steps": payload.max_steps,
        "test_runner": payload.test_runner,
        "test_timeout_seconds": payload.test_timeout_seconds,
        "current_step_index": 0,
        "completed_steps": [],
        "manual_steps": [],
        "skipped_steps": [],
        "applied_patches": [],
        "selected_file": None,
        "pending_action": None,
        "patch_proposal": None,
        "test_result": None,
        "failure_analysis": None,
        "repair_mode": False,
        "repair_attempts": 0,
        "max_repair_attempts": 2,
        "error_message": None,
    }

    try:
        await runtime.graph.ainvoke(
            initial_state,
            config=_get_config(thread_id),
        )

        snapshot = await runtime.graph.aget_state(
            _get_config(thread_id)
        )

        return _build_workflow_response(
            thread_id,
            snapshot.values,
        )

    except Exception:
        logger.exception(
            "Could not start agent workflow %s.",
            thread_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not start the agent workflow.",
        ) from None


@router.get(
    "/{thread_id}",
    response_model=AgentWorkflowResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_workflow(
    thread_id: str,
    request: Request,
    current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> AgentWorkflowResponse:
    """Retrieve the latest state of an owned agent workflow."""

    runtime = await _get_runtime(request)

    normalized_thread_id, snapshot = await _load_owned_snapshot(
        runtime,
        thread_id,
        str(current_user.id),
    )

    return _build_workflow_response(
        normalized_thread_id,
        snapshot.values,
    )


@router.post(
    "/{thread_id}/resume",
    response_model=AgentWorkflowResponse,
    status_code=status.HTTP_200_OK,
)
async def resume_agent_workflow(
    thread_id: str,
    payload: AgentWorkflowResumeRequest,
    request: Request,
    current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> AgentWorkflowResponse:
    """Resume a paused workflow with a validated user decision."""

    runtime = await _get_runtime(request)

    normalized_thread_id, snapshot = await _load_owned_snapshot(
        runtime,
        thread_id,
        str(current_user.id),
    )

    values = snapshot.values
    current_status = values.get("status")

    allowed_actions = {
        "awaiting_plan_approval": {
            "approve",
            "reject",
            "cancel",
        },
        "awaiting_file_selection": {
            "select_file",
            "cancel",
        },
        "awaiting_patch_approval": {
            "approve",
            "reject",
            "cancel",
        },
        "awaiting_test_approval": {
            "approve",
            "reject",
            "cancel",
        },
        "awaiting_repair_file_selection": {
            "select_file",
            "cancel",
        },
    }

    permitted = allowed_actions.get(current_status, set())

    if payload.action not in permitted:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This action is not valid for the workflow's "
                "current state."
            ),
        )

    resume_value = payload.model_dump(
        mode="json",
        exclude_none=True,
    )

    try:
        await runtime.graph.ainvoke(
            Command(resume=resume_value),
            config=_get_config(normalized_thread_id),
        )

        updated_snapshot = await runtime.graph.aget_state(
            _get_config(normalized_thread_id)
        )

        updated_values = updated_snapshot.values

        if (
            not isinstance(updated_values, dict)
            or updated_values.get("owner_user_id")
            != str(current_user.id)
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow not found.",
            )

        return _build_workflow_response(
            normalized_thread_id,
            updated_values,
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Could not resume agent workflow %s.",
            normalized_thread_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not resume the agent workflow.",
        ) from None