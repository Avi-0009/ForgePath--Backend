"""Authenticated project-scoped review tasks.

Task decisions never start work automatically. The explicit planning endpoint
may call the configured AI provider, but it cannot run code or apply patches.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    status,
)
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_current_user,
    get_current_user_with_usage_context,
)
from app.core.ai_provider import AIProviderError
from app.db.session import get_db
from app.models.project import Project
from app.models.project_review import ProjectReview
from app.models.project_review_task import ProjectReviewTask
from app.models.user import User
from app.services.project_review_tasks import get_task_finding
from app.services.project_snapshots import (
    SnapshotConfigurationError,
    SnapshotValidationError,
    read_snapshot_manifest,
    read_snapshot_sources,
)
from app.schemas.task_planning import TaskPlanningRequest, TaskPlanningResponse
from app.services.task_planning.planner import (
    TaskPlanningError,
    generate_snapshot_task_plan,
)


router = APIRouter(
    prefix="/projects",
    tags=["Project Review Tasks"],
)


class ProjectReviewTaskCreate(BaseModel):
    review_id: UUID
    finding_index: int = Field(ge=0, le=100)


class ProjectReviewTaskDecision(BaseModel):
    status: Literal["approved", "dismissed"]


class ProjectReviewTaskPlanCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_snapshot_id: UUID
    additional_context: str | None = Field(default=None, max_length=12_000)
    max_steps: int = Field(default=10, ge=1, le=20)


class ProjectReviewTaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    review_id: UUID
    snapshot_id: UUID
    finding_index: int
    title: str
    path: str
    severity: str
    evidence: str
    explanation: str
    status: Literal["proposed", "approved", "dismissed"]
    planning_status: Literal["not_started", "planning", "ready", "failed"]
    plan: TaskPlanningResponse | None
    planning_error: str | None
    planned_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ProjectReviewTaskListResponse(BaseModel):
    items: list[ProjectReviewTaskResponse]
    limit: int
    offset: int


async def _owned_project(
    db: AsyncSession,
    user_id: UUID,
    project_id: UUID,
    *,
    lock: bool = False,
) -> Project:
    stmt = select(Project).where(
        Project.id == project_id,
        Project.user_id == user_id,
    )

    if lock:
        stmt = stmt.with_for_update()

    project = (
        await db.execute(stmt)
    ).scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=404,
            detail="Project not found",
        )

    return project


@router.post(
    "/{project_id}/tasks",
    response_model=ProjectReviewTaskResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_project_review_task(
    project_id: UUID,
    payload: ProjectReviewTaskCreate,
    response: Response,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectReviewTask:
    """Create at most one task per saved finding."""

    project = await _owned_project(
        db,
        user.id,
        project_id,
        lock=True,
    )

    if project.status != "ready":
        raise HTTPException(
            status_code=409,
            detail="Project is not ready",
        )

    review = (
        await db.execute(
            select(ProjectReview).where(
                ProjectReview.id == payload.review_id,
                ProjectReview.user_id == user.id,
                ProjectReview.project_id == project_id,
            )
        )
    ).scalar_one_or_none()

    if review is None:
        raise HTTPException(
            status_code=404,
            detail="Review not found",
        )

    if review.status not in {"completed", "partial"}:
        raise HTTPException(
            status_code=409,
            detail="Review is not finished",
        )

    finding = get_task_finding(
        review.findings,
        payload.finding_index,
    )

    if finding is None:
        raise HTTPException(
            status_code=422,
            detail="Finding is unavailable",
        )

    existing = (
        await db.execute(
            select(ProjectReviewTask).where(
                ProjectReviewTask.user_id == user.id,
                ProjectReviewTask.project_id == project_id,
                ProjectReviewTask.review_id == review.id,
                ProjectReviewTask.finding_index == payload.finding_index,
            )
        )
    ).scalar_one_or_none()

    if existing is not None:
        response.status_code = status.HTTP_200_OK
        await db.commit()
        return existing

    task = ProjectReviewTask(
        user_id=user.id,
        project_id=project_id,
        review_id=review.id,
        snapshot_id=review.snapshot_id,
        finding_index=payload.finding_index,
        status="proposed",
        **finding,
    )

    db.add(task)

    await db.commit()
    await db.refresh(task)

    return task


def _task_description(task: ProjectReviewTask) -> str:
    return (
        f"Resolve this {task.severity} review finding: {task.title}\n"
        f"Affected snapshot path: {task.path}\n\n"
        f"Finding: {task.explanation[:4_000]}\n\n"
        f"Saved evidence: {task.evidence[:4_000]}"
    )


def _planning_is_active(task: ProjectReviewTask) -> bool:
    if task.planning_status != "planning" or task.planning_started_at is None:
        return False

    started_at = task.planning_started_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)

    return datetime.now(timezone.utc) - started_at < timedelta(minutes=10)


async def _finish_planning_failure(
    db: AsyncSession,
    task_id: UUID,
    user_id: UUID,
    token: UUID,
    message: str,
) -> None:
    task = (
        await db.execute(
            select(ProjectReviewTask)
            .where(
                ProjectReviewTask.id == task_id,
                ProjectReviewTask.user_id == user_id,
                ProjectReviewTask.planning_token == token,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()

    if task is not None:
        task.planning_status = "failed"
        task.planning_token = None
        task.planning_started_at = None
        task.planning_error = message[:300]
        await db.commit()


@router.post(
    "/{project_id}/tasks/{task_id}/plan",
    response_model=ProjectReviewTaskResponse,
)
async def create_project_review_task_plan(
    project_id: UUID,
    task_id: UUID,
    payload: ProjectReviewTaskPlanCreate,
    user: Annotated[User, Depends(get_current_user_with_usage_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectReviewTask:
    """Generate one durable plan from the task's exact private snapshot."""

    await _owned_project(db, user.id, project_id)

    try:
        manifest = await asyncio.to_thread(
            read_snapshot_manifest,
            user.id,
            project_id,
        )
    except SnapshotConfigurationError:
        raise HTTPException(503, "Project snapshot storage is unavailable") from None

    if manifest is None:
        raise HTTPException(409, "Project snapshot is unavailable")

    task = (
        await db.execute(
            select(ProjectReviewTask)
            .where(
                ProjectReviewTask.id == task_id,
                ProjectReviewTask.project_id == project_id,
                ProjectReviewTask.user_id == user.id,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()

    if task is None:
        raise HTTPException(404, "Task not found")
    if task.status != "approved":
        raise HTTPException(409, "Approve this task before creating a plan")
    if (
        payload.expected_snapshot_id != task.snapshot_id
        or manifest.snapshot_id != task.snapshot_id
    ):
        raise HTTPException(
            409,
            "This task belongs to an older project snapshot. Create a new review task.",
        )
    if task.planning_status == "ready" and task.plan is not None:
        return task
    if _planning_is_active(task):
        raise HTTPException(409, "Planning is already in progress")

    planning_token = uuid4()
    task.planning_status = "planning"
    task.planning_token = planning_token
    task.planning_started_at = datetime.now(timezone.utc)
    task.planning_error = None
    await db.commit()

    # Do not disclose internal owner, project, or snapshot identifiers to the
    # configured AI provider. The authoritative IDs stay in the task record.
    snapshot_root = "private-project-snapshot"
    request = TaskPlanningRequest(
        workspace_root=snapshot_root,
        task_description=_task_description(task),
        planning_mode="bug_fix",
        additional_context=payload.additional_context,
        max_steps=payload.max_steps,
    )

    try:
        sources = await asyncio.to_thread(
            read_snapshot_sources,
            user.id,
            project_id,
            task.snapshot_id,
        )
        plan = await generate_snapshot_task_plan(request, sources)
        current = await asyncio.to_thread(
            read_snapshot_manifest,
            user.id,
            project_id,
        )

        if current is None or current.snapshot_id != task.snapshot_id:
            raise SnapshotValidationError("Project snapshot changed during planning")
    except (SnapshotConfigurationError, SnapshotValidationError):
        await _finish_planning_failure(
            db, task.id, user.id, planning_token,
            "The project snapshot changed or became unavailable.",
        )
        raise HTTPException(
            409,
            "Project snapshot changed during planning. Create a new review task.",
        ) from None
    except AIProviderError as exc:
        await _finish_planning_failure(
            db, task.id, user.id, planning_token, "AI provider failed during planning.",
        )
        raise HTTPException(502, "AI provider failed during planning") from exc
    except TaskPlanningError as exc:
        await _finish_planning_failure(
            db, task.id, user.id, planning_token,
            "The AI provider did not return a valid plan.",
        )
        raise HTTPException(502, "Could not generate a valid implementation plan") from exc
    except Exception as exc:
        await _finish_planning_failure(
            db, task.id, user.id, planning_token,
            "Task planning failed unexpectedly.",
        )
        raise HTTPException(500, "Task planning failed unexpectedly") from exc

    completed_task = (
        await db.execute(
            select(ProjectReviewTask)
            .where(
                ProjectReviewTask.id == task.id,
                ProjectReviewTask.user_id == user.id,
                ProjectReviewTask.planning_token == planning_token,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()

    if completed_task is None:
        raise HTTPException(409, "Planning result is no longer current")

    completed_task.plan = plan.model_dump(mode="json")
    completed_task.planning_status = "ready"
    completed_task.planning_token = None
    completed_task.planning_started_at = None
    completed_task.planning_error = None
    completed_task.planned_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(completed_task)
    return completed_task


@router.get(
    "/{project_id}/tasks",
    response_model=ProjectReviewTaskListResponse,
)
async def list_project_review_tasks(
    project_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ProjectReviewTaskListResponse:

    await _owned_project(db, user.id, project_id)

    result = await db.execute(
        select(ProjectReviewTask)
        .where(
            ProjectReviewTask.project_id == project_id,
            ProjectReviewTask.user_id == user.id,
        )
        .order_by(
            ProjectReviewTask.created_at.desc(),
            ProjectReviewTask.id.desc(),
        )
        .limit(limit)
        .offset(offset)
    )

    return ProjectReviewTaskListResponse(
        items=[
            ProjectReviewTaskResponse.model_validate(task)
            for task in result.scalars().all()
        ],
        limit=limit,
        offset=offset,
    )


@router.patch(
    "/{project_id}/tasks/{task_id}",
    response_model=ProjectReviewTaskResponse,
)
async def decide_project_review_task(
    project_id: UUID,
    task_id: UUID,
    payload: ProjectReviewTaskDecision,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectReviewTask:
    """Approve planning or dismiss a task.

    Approving a task never authorizes patch application.
    """

    await _owned_project(db, user.id, project_id)

    task = (
        await db.execute(
            select(ProjectReviewTask)
            .where(
                ProjectReviewTask.id == task_id,
                ProjectReviewTask.project_id == project_id,
                ProjectReviewTask.user_id == user.id,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )

    if task.status == payload.status:
        return task

    if task.status != "proposed":
        raise HTTPException(
            status_code=409,
            detail="Task decision cannot be changed",
        )

    task.status = payload.status

    await db.commit()
    await db.refresh(task)

    return task
