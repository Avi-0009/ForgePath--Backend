import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user_with_usage_context
from app.core.ai_provider import AIProviderError
from app.models.user import User
from app.schemas.task_planning import (
    TaskPlanningRequest,
    TaskPlanningResponse,
)
from app.services.local_context import (
    LocalContextError,
    get_project_instructions,
)
from app.services.task_planning.planner import (
    TaskPlanningError,
    generate_task_plan,
)
from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceConfigurationError,
    WorkspaceReadError,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/task-planning",
    tags=["Task Planning"],
)


@router.post(
    "/plan",
    response_model=TaskPlanningResponse,
    status_code=status.HTTP_200_OK,
)
async def create_task_plan(
    payload: TaskPlanningRequest,
    _current_user: Annotated[
        User,
        Depends(get_current_user_with_usage_context),
    ],
) -> TaskPlanningResponse:
    """Generate a structured implementation plan for a local project."""

    project_instructions = None

    # Saved project instructions are optional context.
    # A context lookup failure must not block task planning.
    try:
        project_context = await get_project_instructions(
            project_root=payload.workspace_root,
        )

        if project_context is not None:
            project_instructions = project_context.instructions

    except LocalContextError:
        logger.warning(
            "Could not load saved project instructions. "
            "Continuing without them."
        )

    try:
        return await generate_task_plan(
            payload,
            project_instructions=project_instructions,
        )

    except WorkspaceConfigurationError as exc:
        logger.error(
            "Task-planning workspace configuration error: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workspace configuration is invalid or unavailable.",
        ) from exc

    except WorkspaceAccessDenied as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except WorkspaceReadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except AIProviderError as exc:
        logger.warning(
            "AI provider failed during task planning."
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    except TaskPlanningError as exc:
        logger.warning(
            "Could not generate a valid task plan: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not generate a valid implementation plan.",
        ) from exc

    except Exception:
        logger.exception(
            "Unexpected task-planning failure."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Task planning failed unexpectedly.",
        ) from None