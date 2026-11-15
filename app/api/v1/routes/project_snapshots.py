"""Authenticated browser-to-backend source transfer for user-owned projects."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
from uuid import UUID

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models.project import Project
from app.models.user import User
from app.services.project_snapshots import (
    MAX_SNAPSHOT_BYTES,
    SnapshotConfigurationError,
    SnapshotValidationError,
    SourceSnapshot,
    read_snapshot_manifest,
    store_snapshot,
)


router = APIRouter(prefix="/projects", tags=["Project Sources"])
MAX_REQUEST_BYTES = MAX_SNAPSHOT_BYTES + 1_000_000


async def _owned_project(db: AsyncSession, user_id: UUID, project_id: UUID, *, lock: bool = False) -> Project:
    query = select(Project).where(Project.id == project_id, Project.user_id == user_id)
    if lock:
        query = query.with_for_update()
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    if project is None or project.status == "archived":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


@router.put("/{project_id}/snapshot", status_code=status.HTTP_201_CREATED)
async def upload_project_snapshot(
    project_id: UUID,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Accept only bounded text source files, never a filesystem path or archive."""
    project = await _owned_project(db, user.id, project_id, lock=True)
    declared_size = request.headers.get("content-length")
    if declared_size and (not declared_size.isdecimal() or int(declared_size) > MAX_REQUEST_BYTES):
        raise HTTPException(413, "Source upload exceeds the request limit")

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise HTTPException(413, "Source upload exceeds the request limit")
        body.extend(chunk)
    try:
        payload = SourceSnapshot.model_validate_json(body)
        manifest = await asyncio.to_thread(store_snapshot, user.id, project_id, payload)
    except (ValidationError, SnapshotValidationError, UnicodeError):
        raise HTTPException(422, "Source snapshot contains invalid or prohibited files") from None
    except SnapshotConfigurationError:
        raise HTTPException(503, "Private project snapshot storage is not configured") from None
    except OSError:
        raise HTTPException(503, "Could not safely store the source snapshot") from None

    project.status = "ready"
    await db.commit()
    return {
        "project_id": str(project_id),
        "snapshot_id": str(manifest.snapshot_id),
        "file_count": manifest.file_count,
        "total_bytes": manifest.total_bytes,
        "sha256": manifest.sha256,
        "status": "ready",
    }


@router.get("/{project_id}/snapshot")
async def get_project_snapshot(
    project_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Only disclose snapshot metadata after project ownership is verified."""
    await _owned_project(db, user.id, project_id)
    try:
        manifest = await asyncio.to_thread(read_snapshot_manifest, user.id, project_id)
    except (SnapshotConfigurationError, OSError, ValueError, KeyError):
        raise HTTPException(503, "Project snapshot is unavailable") from None
    if manifest is None:
        raise HTTPException(404, "Project snapshot not found")
    return {
        "project_id": str(project_id),
        "snapshot_id": str(manifest.snapshot_id),
        "file_count": manifest.file_count,
        "total_bytes": manifest.total_bytes,
        "sha256": manifest.sha256,
    }