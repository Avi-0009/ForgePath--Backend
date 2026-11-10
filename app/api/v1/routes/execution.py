import asyncio
import logging
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user_with_usage_context
from app.models.user import User
from app.schemas.execution import (
    TestExecutionRequest,
    TestExecutionResponse,
)
from app.services.ai_usage_tracking import record_agent_test_run
from app.services.test_execution.runner import (
    TestExecutionError,
    TestProjectConfigurationError,
    TestRunnerUnavailable,
    execute_tests,
)
from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceConfigurationError,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/test-execution",
    tags=["Test Execution"],
)


@router.post(
    "/run",
    response_model=TestExecutionResponse,
    status_code=status.HTTP_200_OK,
)
async def run_local_tests(
    payload: TestExecutionRequest,
    _current_user: Annotated[
        User,
        Depends(get_current_user_with_usage_context),
    ],
) -> TestExecutionResponse:
    """Execute local tests and persist metadata without storing output."""

    try:
        result = await asyncio.to_thread(
            execute_tests,
            workspace_root=payload.workspace_root,
            runner=payload.runner,
            timeout_seconds=payload.timeout_seconds,
        )

        await record_agent_test_run(result)

        return TestExecutionResponse(**asdict(result))

    except WorkspaceAccessDenied as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except WorkspaceConfigurationError as exc:
        logger.error("Test execution workspace configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workspace configuration is invalid or unavailable.",
        ) from exc

    except TestRunnerUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except TestProjectConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except TestExecutionError:
        logger.exception("Local test execution failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not execute the requested tests.",
        ) from None

    except Exception:
        logger.exception("Unexpected local test execution failure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Test execution failed unexpectedly.",
        ) from None