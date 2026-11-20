"""Authenticated search against immutable user-owned project snapshots."""

from __future__ import annotations

import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models.project import Project
from app.models.user import User
from app.services.project_code_search import search_snapshot_sources
from app.services.project_snapshots import (
    SnapshotConfigurationError,
    SnapshotValidationError,
    read_snapshot_manifest,
    read_snapshot_sources,
)


router = APIRouter(prefix="/projects", tags=["Project Code Intelligence"])


class ProjectCodeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=6, ge=1, le=20)


@router.post("/{project_id}/search")
async def search_project_code(
    project_id: UUID,
    payload: ProjectCodeSearchRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    owner = await db.execute(
        select(Project.id).where(
            Project.id == project_id,
            Project.user_id == user.id,
            Project.status == "ready",
        )
    )
    if owner.scalar_one_or_none() is None:
        raise HTTPException(404, "Ready project not found")

    try:
        manifest = await asyncio.to_thread(read_snapshot_manifest, user.id, project_id)
        if manifest is None:
            raise HTTPException(409, "Project has no source snapshot")
        sources = await asyncio.to_thread(
            read_snapshot_sources, user.id, project_id, manifest.snapshot_id
        )
        hits = await asyncio.to_thread(
            search_snapshot_sources, sources, payload.query, payload.top_k
        )
    except HTTPException:
        raise
    except (SnapshotConfigurationError, SnapshotValidationError, OSError, ValueError, KeyError):
        raise HTTPException(503, "Project source snapshot is unavailable") from None

    return {
        "project_id": str(project_id),
        "snapshot_id": str(manifest.snapshot_id),
        "workspace_root": f"project:{project_id}",
        "query": payload.query,
        "result_count": len(hits),
        "results": hits,
    }
