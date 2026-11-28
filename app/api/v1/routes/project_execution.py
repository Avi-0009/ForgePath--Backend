"""Explicitly approved test execution for authenticated project owners."""

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models.project import Project
from app.models.user import User
from app.services.ai_usage_tracking import record_agent_test_run
from app.services.test_execution.project_runner import (
    ProjectSnapshotChanged,
    ProjectTestConfigurationError,
    ProjectTestUnavailable,
    execute_project_tests,
)

router = APIRouter(prefix="/projects", tags=["Project Test Execution"])
logger = logging.getLogger(__name__)


class ProjectTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_snapshot_id: UUID
    runner: Literal["pytest", "npm", "go"]
    timeout_seconds: int = Field(default=120, ge=1, le=600)
    approval: Literal["run"]


class ProjectTestResponse(BaseModel):
    execution_id: UUID
    project_id: UUID
    snapshot_id: UUID
    runner: Literal["pytest", "npm", "go"]
    status: Literal["passed", "failed", "timed_out", "error"]
    command: list[str]
    exit_code: int | None
    duration_ms: int = Field(ge=0)
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    started_at: datetime
    finished_at: datetime


@router.post("/{project_id}/tests/run", response_model=ProjectTestResponse)
async def run_project_tests(
    project_id: UUID,
    payload: ProjectTestRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectTestResponse:
    user_id = user.id
    found = await db.execute(select(Project.id).where(
        Project.id == project_id,
        Project.user_id == user_id,
        Project.status == "ready",
    ))
    if found.scalar_one_or_none() is None:
        raise HTTPException(404, "Project not found or not ready.")
    # Do not retain a database connection for the full test runtime.
    await db.rollback()
    try:
        result = await asyncio.to_thread(
            execute_project_tests,
            user_id=user_id,
            project_id=project_id,
            expected_snapshot_id=payload.expected_snapshot_id,
            runner=payload.runner,
            timeout_seconds=payload.timeout_seconds,
        )
    except ProjectSnapshotChanged as exc:
        raise HTTPException(409, str(exc)) from None
    except ProjectTestConfigurationError as exc:
        raise HTTPException(400, str(exc)) from None
    except ProjectTestUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    except Exception:
        logger.exception("Project test execution failed.")
        raise HTTPException(500, "Test execution failed.") from None
    await record_agent_test_run(result, user_id=user_id)
    return ProjectTestResponse(**asdict(result))
