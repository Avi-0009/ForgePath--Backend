"""Owner-scoped uploaded-project workflow views and plan decisions.

This API is intentionally separate from filesystem-backed LangGraph workflows.
It never accepts workspace paths, applies patches, or executes commands.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.api.v1.routes.project_review_tasks import ProjectReviewTaskResponse
from app.db.session import get_db
from app.models.project import Project
from app.models.project_review_task import ProjectReviewTask
from app.models.user import User
from app.services.project_snapshots import SnapshotConfigurationError, read_snapshot_manifest


router = APIRouter(prefix="/project-workflows", tags=["Project Workflows"])

DecisionKind = Literal["finding", "plan", "patch"]
WorkflowStage = Literal[
    "dismissed", "finding_approval", "planning_not_started",
    "planning_planning", "planning_failed", "plan_approval",
    "plan_rejected", "patch_not_started", "patch_generating",
    "patch_ready", "patch_approved", "patch_rejected", "patch_failed",
    "patch_applied",
]


class ProjectPlanDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["approved", "rejected"]
    expected_snapshot_id: UUID
    expected_plan_id: UUID


class ProjectWorkflowItem(BaseModel):
    kind: Literal["project_task"] = "project_task"
    project_name: str
    stage: WorkflowStage
    pending_decision: DecisionKind | None
    task: ProjectReviewTaskResponse


class ProjectWorkflowList(BaseModel):
    items: list[ProjectWorkflowItem]
    total: int
    limit: int
    offset: int


def pending_for_task(task: ProjectReviewTask) -> DecisionKind | None:
    if task.status == "proposed":
        return "finding"
    if task.status != "approved":
        return None
    if (task.planning_status == "ready" and task.plan is not None and
            task.plan_decision in (None, "pending")):
        return "plan"
    if task.patch_status == "ready" and task.patch is not None:
        return "patch"
    return None


def stage_for_task(task: ProjectReviewTask) -> WorkflowStage:
    if task.status == "dismissed":
        return "dismissed"
    if task.status == "proposed":
        return "finding_approval"
    if task.planning_status in ("not_started", "planning", "failed"):
        return cast(WorkflowStage, f"planning_{task.planning_status}")
    if task.plan_decision in (None, "pending"):
        return "plan_approval"
    if task.plan_decision == "rejected":
        return "plan_rejected"
    return cast(WorkflowStage, f"patch_{task.patch_status}")


@router.get("", response_model=ProjectWorkflowList)
async def list_project_workflows(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=30, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    pending_only: bool = Query(default=False),
) -> ProjectWorkflowList:
    conditions = [
        ProjectReviewTask.user_id == user.id,
        Project.user_id == user.id,
    ]
    pending_condition = or_(
        ProjectReviewTask.status == "proposed",
        and_(
            ProjectReviewTask.status == "approved",
            ProjectReviewTask.planning_status == "ready",
            ProjectReviewTask.plan.is_not(None),
            or_(
                ProjectReviewTask.plan_decision.is_(None),
                ProjectReviewTask.plan_decision == "pending",
            ),
        ),
        and_(
            ProjectReviewTask.status == "approved",
            ProjectReviewTask.patch_status == "ready",
            ProjectReviewTask.patch.is_not(None),
        ),
    )
    current_snapshots: dict[UUID, UUID] = {}
    if pending_only:
        conditions.extend((Project.status == "ready", pending_condition))

        # Filter freshness before count/offset so stale rows cannot distort the
        # actionable queue or its pagination.
        candidate_result = await db.execute(
            select(Project.id)
            .join(ProjectReviewTask, ProjectReviewTask.project_id == Project.id)
            .where(*conditions)
            .distinct()
        )
        for (project_id,) in candidate_result.all():
            try:
                manifest = await asyncio.to_thread(
                    read_snapshot_manifest, user.id, project_id,
                )
            except (SnapshotConfigurationError, ValueError, OSError, KeyError):
                manifest = None
            if manifest is not None:
                current_snapshots[project_id] = manifest.snapshot_id

        if not current_snapshots:
            return ProjectWorkflowList(items=[], total=0, limit=limit, offset=offset)
        conditions.append(or_(*(
            and_(
                ProjectReviewTask.project_id == project_id,
                ProjectReviewTask.snapshot_id == snapshot_id,
            )
            for project_id, snapshot_id in current_snapshots.items()
        )))

    join = ProjectReviewTask.project_id == Project.id
    total = await db.scalar(
        select(func.count()).select_from(ProjectReviewTask).join(Project, join).where(*conditions)
    ) or 0
    result = await db.execute(
        select(ProjectReviewTask, Project.name)
        .join(Project, join)
        .where(*conditions)
        .order_by(ProjectReviewTask.created_at.desc(), ProjectReviewTask.id.desc())
        .offset(offset)
        .limit(limit)
    )

    snapshot_by_project: dict[UUID, UUID | None] = dict(current_snapshots)
    output: list[ProjectWorkflowItem] = []
    for task, project_name in result.all():
        if task.project_id not in snapshot_by_project:
            try:
                manifest = await asyncio.to_thread(
                    read_snapshot_manifest, user.id, task.project_id,
                )
                snapshot_by_project[task.project_id] = manifest.snapshot_id if manifest else None
            except (SnapshotConfigurationError, ValueError, OSError, KeyError):
                snapshot_by_project[task.project_id] = None
        current = task.snapshot_id == snapshot_by_project[task.project_id]
        output.append(ProjectWorkflowItem(
            project_name=project_name,
            stage=stage_for_task(task),
            pending_decision=pending_for_task(task) if current else None,
            task=ProjectReviewTaskResponse.model_validate(task).model_copy(
                update={"is_current_snapshot": current},
            ),
        ))

    return ProjectWorkflowList(items=output, total=total, limit=limit, offset=offset)


@router.patch(
    "/{project_id}/tasks/{task_id}/plan/decision",
    response_model=ProjectReviewTaskResponse,
)
async def decide_project_plan(
    project_id: UUID,
    task_id: UUID,
    payload: ProjectPlanDecision,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectReviewTask:
    # Project-before-task matches deletion and finding-decision lock order.
    project = (
        await db.execute(
            select(Project).where(
                Project.id == project_id,
                Project.user_id == user.id,
            ).with_for_update()
        )
    ).scalar_one_or_none()
    if project is None:
        raise HTTPException(404, "Project not found")
    if project.status != "ready":
        raise HTTPException(409, "Project is not ready")

    task = (
        await db.execute(
            select(ProjectReviewTask).where(
                ProjectReviewTask.id == task_id,
                ProjectReviewTask.project_id == project_id,
                ProjectReviewTask.user_id == user.id,
            ).with_for_update()
        )
    ).scalar_one_or_none()
    if task is None:
        raise HTTPException(404, "Task not found")
    if (task.status != "approved" or task.planning_status != "ready" or
            not isinstance(task.plan, dict) or not task.plan):
        raise HTTPException(409, "An approved task with a ready plan is required")
    if (task.snapshot_id != payload.expected_snapshot_id or
            str(task.plan.get("plan_id")) != str(payload.expected_plan_id)):
        raise HTTPException(409, "Plan or snapshot changed. Refresh before deciding")

    try:
        manifest = await asyncio.to_thread(read_snapshot_manifest, user.id, project_id)
    except (SnapshotConfigurationError, ValueError, OSError, KeyError):
        raise HTTPException(503, "Project snapshot is unavailable") from None
    if manifest is None or manifest.snapshot_id != task.snapshot_id:
        raise HTTPException(409, "This task belongs to an older project snapshot")

    existing = task.plan_decision or "pending"
    if existing == payload.status:
        return task
    if existing != "pending":
        raise HTTPException(409, "Plan approval cannot be changed")
    if task.patch is not None or task.patch_status not in ("not_started", "failed"):
        raise HTTPException(409, "A patch is already associated with this plan")

    task.plan_decision = payload.status
    task.plan_decided_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(task)
    return task
