"""Authenticated project-scoped review tasks.

Task decisions never start work automatically. The explicit planning endpoint
may call the configured AI provider, but it cannot run code or apply patches.
"""

import asyncio
import difflib
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
from app.schemas.code_patch import CodePatchRequest, CodePatchResponse
from app.services.code_patch import (
    CodePatchError,
    StaleSourceFileError,
    generate_snapshot_code_patch,
)
from app.services.task_planning.planner import (
    TaskPlanningError,
    generate_snapshot_task_plan,
)
from app.services.workspace import LANGUAGE_BY_EXTENSION


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


class ProjectReviewTaskPatchCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_snapshot_id: UUID
    expected_plan_id: UUID
    additional_context: str | None = Field(default=None, max_length=12_000)


class ProjectReviewTaskPatchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["approved", "rejected"]
    expected_snapshot_id: UUID
    expected_proposed_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


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
    patch_status: Literal[
        "not_started", "generating", "ready", "approved", "rejected", "failed"
    ]
    patch: CodePatchResponse | None
    patch_error: str | None
    patched_at: datetime | None
    is_current_snapshot: bool = True
    created_at: datetime
    updated_at: datetime


class ProjectReviewTaskListResponse(BaseModel):
    items: list[ProjectReviewTaskResponse]
    current_snapshot_id: UUID
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

    try:
        manifest = await asyncio.to_thread(
            read_snapshot_manifest, user.id, project_id,
        )
    except SnapshotConfigurationError:
        raise HTTPException(503, "Project snapshot storage is unavailable") from None

    if manifest is None or manifest.snapshot_id != review.snapshot_id:
        raise HTTPException(
            409,
            "This review belongs to an older project snapshot. Review the current snapshot.",
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

        if (
            current is None
            or current.snapshot_id != task.snapshot_id
            or current.sha256 != manifest.sha256
        ):
            raise SnapshotValidationError("Project snapshot changed during planning")
        await asyncio.to_thread(
            read_snapshot_sources,
            user.id,
            project_id,
            task.snapshot_id,
        )
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


def _patch_is_active(task: ProjectReviewTask) -> bool:
    if task.patch_status != "generating" or task.patch_started_at is None:
        return False

    started_at = task.patch_started_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)

    return datetime.now(timezone.utc) - started_at < timedelta(minutes=10)


def _suggested_fix(task: ProjectReviewTask) -> str:
    plan = task.plan if isinstance(task.plan, dict) else {}
    parts: list[str] = []
    objective = plan.get("objective")
    if isinstance(objective, str) and objective.strip():
        parts.append(objective.strip())

    steps = plan.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            candidates = step.get("candidate_files_to_inspect")
            description = step.get("description")
            if (
                isinstance(candidates, list)
                and task.path in candidates
                and isinstance(description, str)
                and description.strip()
            ):
                parts.append(description.strip())

    return "\n\n".join(parts)[:8_000] or task.explanation[:8_000]


async def _finish_patch_failure(
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
                ProjectReviewTask.patch_token == token,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()

    if task is not None:
        task.patch_status = "failed"
        task.patch_token = None
        task.patch_started_at = None
        task.patch_error = message[:300]
        await db.commit()


@router.post(
    "/{project_id}/tasks/{task_id}/patch",
    response_model=ProjectReviewTaskResponse,
)
async def create_project_review_task_patch(
    project_id: UUID,
    task_id: UUID,
    payload: ProjectReviewTaskPatchCreate,
    user: Annotated[User, Depends(get_current_user_with_usage_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectReviewTask:
    """Generate one reviewable proposal without applying it anywhere."""

    await _owned_project(db, user.id, project_id)

    try:
        manifest = await asyncio.to_thread(
            read_snapshot_manifest, user.id, project_id,
        )
    except SnapshotConfigurationError:
        raise HTTPException(503, "Project snapshot storage is unavailable") from None

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
    if task.status != "approved" or task.planning_status != "ready" or not task.plan:
        raise HTTPException(409, "Create and review the implementation plan first")
    if manifest is None or (
        payload.expected_snapshot_id != task.snapshot_id
        or manifest.snapshot_id != task.snapshot_id
    ):
        raise HTTPException(409, "This task belongs to an older project snapshot")
    if str(payload.expected_plan_id) != str(task.plan.get("plan_id")):
        raise HTTPException(409, "The implementation plan changed. Refresh this task")
    if task.patch is not None and task.patch_status in {"ready", "approved", "rejected"}:
        return task
    if _patch_is_active(task):
        raise HTTPException(409, "Patch generation is already in progress")

    patch_token = uuid4()
    task.patch_status = "generating"
    task.patch_token = patch_token
    task.patch_started_at = datetime.now(timezone.utc)
    task.patch_error = None
    await db.commit()

    try:
        sources = await asyncio.to_thread(
            read_snapshot_sources, user.id, project_id, task.snapshot_id,
        )
        source_content = dict(sources).get(task.path)
        if source_content is None:
            raise SnapshotValidationError("Finding file is absent from the snapshot")

        source_hash = hashlib.sha256(source_content.encode("utf-8")).hexdigest()
        request = CodePatchRequest(
            workspace_root="private-project-snapshot",
            relative_path=task.path,
            expected_sha256=source_hash,
            finding_title=task.title,
            finding_evidence=task.evidence[:4_000],
            finding_explanation=task.explanation[:8_000],
            suggested_fix=_suggested_fix(task),
            additional_context=payload.additional_context,
        )
        proposal = await generate_snapshot_code_patch(
            request,
            source_content,
            language=LANGUAGE_BY_EXTENSION.get(
                Path(task.path).suffix.casefold(), "unknown",
            ),
        )

        current = await asyncio.to_thread(
            read_snapshot_manifest, user.id, project_id,
        )
        if (
            current is None
            or current.snapshot_id != task.snapshot_id
            or current.sha256 != manifest.sha256
        ):
            raise SnapshotValidationError("Project snapshot changed during generation")
        verified_sources = await asyncio.to_thread(
            read_snapshot_sources, user.id, project_id, task.snapshot_id,
        )
        verified_by_path = dict(verified_sources)
        if task.path not in verified_by_path or hashlib.sha256(
            verified_by_path[task.path].encode("utf-8")
        ).hexdigest() != source_hash:
            raise SnapshotValidationError("Finding file changed during generation")
    except (SnapshotConfigurationError, SnapshotValidationError, StaleSourceFileError):
        await _finish_patch_failure(
            db, task.id, user.id, patch_token,
            "The project snapshot changed or the finding file is unavailable.",
        )
        raise HTTPException(409, "Project snapshot changed during patch generation") from None
    except AIProviderError as exc:
        await _finish_patch_failure(
            db, task.id, user.id, patch_token, "AI provider failed during patch generation.",
        )
        raise HTTPException(502, "AI provider failed during patch generation") from exc
    except CodePatchError as exc:
        await _finish_patch_failure(
            db, task.id, user.id, patch_token,
            "The AI provider did not return a valid patch proposal.",
        )
        raise HTTPException(502, "Could not generate a valid patch proposal") from exc
    except Exception as exc:
        await _finish_patch_failure(
            db, task.id, user.id, patch_token, "Patch generation failed unexpectedly.",
        )
        raise HTTPException(500, "Patch generation failed unexpectedly") from exc

    completed_task = (
        await db.execute(
            select(ProjectReviewTask)
            .where(
                ProjectReviewTask.id == task.id,
                ProjectReviewTask.user_id == user.id,
                ProjectReviewTask.patch_token == patch_token,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()
    if completed_task is None:
        raise HTTPException(409, "Patch proposal is no longer current")

    completed_task.patch = proposal.model_dump(mode="json")
    completed_task.patch_status = "ready"
    completed_task.patch_token = None
    completed_task.patch_started_at = None
    completed_task.patch_error = None
    completed_task.patched_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(completed_task)
    return completed_task


@router.patch(
    "/{project_id}/tasks/{task_id}/patch",
    response_model=ProjectReviewTaskResponse,
)
async def decide_project_review_task_patch(
    project_id: UUID,
    task_id: UUID,
    payload: ProjectReviewTaskPatchDecision,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectReviewTask:
    """Record a proposal decision; never apply snapshot files."""

    await _owned_project(db, user.id, project_id)

    try:
        manifest = await asyncio.to_thread(
            read_snapshot_manifest, user.id, project_id,
        )
    except SnapshotConfigurationError:
        raise HTTPException(503, "Project snapshot storage is unavailable") from None

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
    repeated_decision = task.patch_status == payload.status
    if task.patch is None or (
        not repeated_decision and task.patch_status != "ready"
    ):
        raise HTTPException(409, "A ready patch proposal is required")
    if (
        manifest is None
        or manifest.snapshot_id != task.snapshot_id
        or payload.expected_snapshot_id != task.snapshot_id
        or payload.expected_proposed_sha256 != task.patch.get("proposed_sha256")
    ):
        raise HTTPException(409, "The snapshot or patch proposal changed. Refresh this task")

    try:
        sources = await asyncio.to_thread(
            read_snapshot_sources, user.id, project_id, task.snapshot_id,
        )
    except (SnapshotConfigurationError, SnapshotValidationError):
        raise HTTPException(409, "Project snapshot is no longer valid") from None

    source_by_path = dict(sources)
    source_content = source_by_path.get(task.path)
    if source_content is None or hashlib.sha256(
        source_content.encode("utf-8")
    ).hexdigest() != task.patch.get("original_sha256"):
        raise HTTPException(409, "The proposal source is no longer current")

    if repeated_decision:
        return task

    task.patch_status = payload.status
    await db.commit()
    await db.refresh(task)
    return task


async def _build_verified_patch_export(
    user_id: UUID,
    project_id: UUID,
    task: ProjectReviewTask,
) -> str:
    if task.patch_status != "approved" or not isinstance(task.patch, dict):
        raise HTTPException(409, "Approve the patch proposal before exporting it")

    try:
        manifest = await asyncio.to_thread(
            read_snapshot_manifest, user_id, project_id,
        )
        sources = await asyncio.to_thread(
            read_snapshot_sources, user_id, project_id, task.snapshot_id,
        )
    except (SnapshotConfigurationError, SnapshotValidationError):
        raise HTTPException(409, "Project snapshot is no longer valid") from None

    if manifest is None or manifest.snapshot_id != task.snapshot_id:
        raise HTTPException(409, "The approved proposal belongs to an older snapshot")

    source_content = dict(sources).get(task.path)
    proposed_code = task.patch.get("proposed_code")
    proposed_sha256 = task.patch.get("proposed_sha256")

    if (
        source_content is None
        or not isinstance(proposed_code, str)
        or not isinstance(proposed_sha256, str)
        or task.patch.get("relative_path") != task.path
        or hashlib.sha256(source_content.encode("utf-8")).hexdigest()
        != task.patch.get("original_sha256")
        or hashlib.sha256(proposed_code.encode("utf-8")).hexdigest()
        != proposed_sha256
    ):
        raise HTTPException(409, "The approved proposal failed its integrity check")

    diff = "".join(
        difflib.unified_diff(
            source_content.splitlines(keepends=True),
            proposed_code.splitlines(keepends=True),
            fromfile=f"a/{task.path}",
            tofile=f"b/{task.path}",
        )
    )
    if not diff:
        raise HTTPException(409, "The approved proposal contains no changes")

    return diff


@router.get("/{project_id}/tasks/{task_id}/patch/download")
async def download_project_review_task_patch(
    project_id: UUID,
    task_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Export an approved, revalidated proposal without writing project files."""

    await _owned_project(db, user.id, project_id)
    task = (
        await db.execute(
            select(ProjectReviewTask).where(
                ProjectReviewTask.id == task_id,
                ProjectReviewTask.project_id == project_id,
                ProjectReviewTask.user_id == user.id,
            )
        )
    ).scalar_one_or_none()

    if task is None:
        raise HTTPException(404, "Task not found")

    diff = await _build_verified_patch_export(user.id, project_id, task)
    filename = f"forgepath-{task.id}.diff"

    return Response(
        content=diff,
        media_type="text/x-diff",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


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
    include_stale: bool = Query(default=False),
) -> ProjectReviewTaskListResponse:

    await _owned_project(db, user.id, project_id)

    try:
        manifest = await asyncio.to_thread(
            read_snapshot_manifest, user.id, project_id,
        )
    except SnapshotConfigurationError:
        raise HTTPException(503, "Project snapshot storage is unavailable") from None

    if manifest is None:
        raise HTTPException(409, "Project snapshot is unavailable")

    query = select(ProjectReviewTask).where(
        ProjectReviewTask.project_id == project_id,
        ProjectReviewTask.user_id == user.id,
    )

    if not include_stale:
        query = query.where(
            ProjectReviewTask.snapshot_id == manifest.snapshot_id,
        )

    result = await db.execute(
        query
        .order_by(
            ProjectReviewTask.created_at.desc(),
            ProjectReviewTask.id.desc(),
        )
        .limit(limit)
        .offset(offset)
    )

    return ProjectReviewTaskListResponse(
        items=[
            ProjectReviewTaskResponse.model_validate(task).model_copy(
                update={
                    "is_current_snapshot": task.snapshot_id == manifest.snapshot_id,
                }
            )
            for task in result.scalars().all()
        ],
        current_snapshot_id=manifest.snapshot_id,
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
