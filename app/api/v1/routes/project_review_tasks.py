"""Authenticated project-scoped review tasks.

These endpoints persist user decisions only.
They never call an AI model, run code, or apply patches.
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

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

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models.project import Project
from app.models.project_review import ProjectReview
from app.models.project_review_task import ProjectReviewTask
from app.models.user import User
from app.services.project_review_tasks import get_task_finding


router = APIRouter(
    prefix="/projects",
    tags=["Project Review Tasks"],
)


class ProjectReviewTaskCreate(BaseModel):
    review_id: UUID
    finding_index: int = Field(ge=0, le=100)


class ProjectReviewTaskDecision(BaseModel):
    status: Literal["approved", "dismissed"]


class ProjectReviewTaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    review_id: UUID
    finding_index: int
    title: str
    path: str
    severity: str
    evidence: str
    explanation: str
    status: Literal["proposed", "approved", "dismissed"]
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
        finding_index=payload.finding_index,
        status="proposed",
        **finding,
    )

    db.add(task)

    await db.commit()
    await db.refresh(task)

    return task


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