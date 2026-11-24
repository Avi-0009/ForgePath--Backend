"""User-scoped read-only initial review jobs for uploaded project snapshots."""
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models.project import Project
from app.models.project_review import ProjectReview
from app.models.user import User
from app.services.project_review_agent import run_initial_review, MAX_AI_CALLS, MAX_OUTPUT_TOKENS_PER_CALL
from app.services.project_snapshots import SnapshotConfigurationError, read_snapshot_manifest
from app.services.project_review_jobs import expire_reviews, LEASE_SECONDS


router = APIRouter(prefix="/projects", tags=["Project Reviews"])


def _result(review: ProjectReview) -> dict:
    return {
        "id": str(review.id),
        "project_id": str(review.project_id),
        "snapshot_id": str(review.snapshot_id),
        "status": review.status,
        "files_scanned": review.files_scanned,
        "files_reviewed": review.files_reviewed,
        "ai_calls": review.ai_calls,
        "max_ai_calls": MAX_AI_CALLS,
        "max_output_tokens_per_call": MAX_OUTPUT_TOKENS_PER_CALL,
        "result_text": review.result_text,
        "findings": review.findings or [],
        "error_message": review.error_message,
    }


@router.post("/{project_id}/reviews", status_code=202)
async def start_project_review(
    project_id: UUID,
    background_tasks: BackgroundTasks,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Queue one bounded review per user, reusing same-snapshot results."""
    # Serialize concurrent requests from the same user across projects.
    await db.execute(select(User).where(User.id == user.id).with_for_update())
    result = await db.execute(select(Project).where(Project.id == project_id, Project.user_id == user.id))
    project = result.scalar_one_or_none()
    if project is None or project.status != "ready":
        raise HTTPException(404, "Ready project not found")
    try:
        snapshot = await asyncio.to_thread(read_snapshot_manifest, user.id, project_id)
    except (SnapshotConfigurationError, OSError, ValueError, KeyError):
        raise HTTPException(503, "Project snapshot unavailable") from None
    if snapshot is None:
        raise HTTPException(409, "Upload a source snapshot before reviewing")

    await expire_reviews(db, user.id)

    previous = await db.execute(
        select(ProjectReview)
        .where(ProjectReview.project_id == project_id, ProjectReview.user_id == user.id,
               ProjectReview.snapshot_id == snapshot.snapshot_id)
        .order_by(ProjectReview.created_at.desc())
        .limit(1)
    )
    existing = previous.scalar_one_or_none()
    if existing and existing.status in {"queued", "running", "completed", "partial"}:
        await db.commit()
        return _result(existing)

    other = await db.execute(
        select(ProjectReview.id).where(ProjectReview.user_id == user.id,
            ProjectReview.status.in_(["queued", "running"])).limit(1)
    )
    if other.scalar_one_or_none() is not None:
        raise HTTPException(409, "Another review is already running for this account")

    review = ProjectReview(user_id=user.id, project_id=project_id, snapshot_id=snapshot.snapshot_id,
                           status="queued", lease_token=uuid4(),
                           lease_expires_at=func.now() + timedelta(seconds=LEASE_SECONDS))
    db.add(review)
    await db.commit()
    await db.refresh(review)
    background_tasks.add_task(run_initial_review, review.id, user.id, project_id, snapshot.snapshot_id, review.lease_token)
    return _result(review)


@router.get("/{project_id}/reviews/latest")
async def latest_project_review(
    project_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict | None :
    owner = await db.execute(select(Project.id).where(Project.id == project_id, Project.user_id == user.id))
    if owner.scalar_one_or_none() is None:
        raise HTTPException(404, "Project not found")
    await expire_reviews(db, user.id)
    await db.commit()
    result = await db.execute(
        select(ProjectReview).where(ProjectReview.project_id == project_id, ProjectReview.user_id == user.id)
        .order_by(ProjectReview.created_at.desc()).limit(1)
    )
    review = result.scalar_one_or_none()

    if review is None:
        return None

    return _result(review)
