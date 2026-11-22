"""Read-only, owner-scoped checks for local file references in saved snapshots."""

from __future__ import annotations

import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models.project import Project
from app.models.user import User
from app.services.project_reference_integrity import inspect_snapshot_references
from app.services.project_snapshots import (
    SnapshotConfigurationError,
    SnapshotValidationError,
    read_snapshot_manifest,
    read_snapshot_sources,
)


router = APIRouter(prefix="/projects", tags=["Project Code Intelligence"])


@router.get("/{project_id}/references")
async def check_project_references(
    project_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> dict:
    """Check static local imports without touching disk outside the snapshot."""
    owned = await db.execute(
        select(Project.id).where(
            Project.id == project_id,
            Project.user_id == user.id,
            Project.status == "ready",
        )
    )
    if owned.scalar_one_or_none() is None:
        raise HTTPException(404, "Ready project not found")

    try:
        manifest = await asyncio.to_thread(read_snapshot_manifest, user.id, project_id)
        if manifest is None:
            raise HTTPException(409, "Project has no source snapshot")
        sources = await asyncio.to_thread(
            read_snapshot_sources, user.id, project_id, manifest.snapshot_id,
        )
        report = await asyncio.to_thread(
            inspect_snapshot_references, sources, manifest.asset_paths, max_results=limit,
        )
    except HTTPException:
        raise
    except (SnapshotConfigurationError, SnapshotValidationError, OSError, ValueError, KeyError):
        raise HTTPException(503, "Project source snapshot is unavailable") from None

    return {
        "project_id": str(project_id),
        "snapshot_id": str(manifest.snapshot_id),
        **report,
    }