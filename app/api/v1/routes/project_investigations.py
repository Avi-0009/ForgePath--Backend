"""Read-only, user-owned investigation of a file in an immutable snapshot."""

import asyncio
import hashlib
from pathlib import PurePosixPath
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_current_user_with_usage_context
from app.core.ai_provider import AIProviderError
from app.db.session import get_db
from app.models.project import Project
from app.models.user import User
from app.schemas.code_review import CodeReviewFileResponse, CodeReviewRequest, ReviewMode
from app.services.code_review import CodeReviewError, review_code
from app.services.project_code_search import search_snapshot_sources
from app.services.project_snapshots import (
    SnapshotConfigurationError, SnapshotValidationError,
    read_snapshot_manifest, read_snapshot_sources,
)

router = APIRouter(prefix="/projects", tags=["Project Investigation"])


class ProjectInvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1, max_length=1024)
    review_mode: ReviewMode = "general"
    task_description: str | None = Field(default=None, max_length=2000)


async def _owned_sources(project_id: UUID, user: User, db: AsyncSession):
    ownership = await db.execute(select(Project.id).where(
        Project.id == project_id,
        Project.user_id == user.id,
        Project.status == "ready",
    ))
    if ownership.scalar_one_or_none() is None:
        raise HTTPException(404, "Ready project not found")

    try:
        manifest = await asyncio.to_thread(read_snapshot_manifest, user.id, project_id)
        if manifest is None:
            raise HTTPException(409, "Project has no source snapshot")
        sources = await asyncio.to_thread(
            read_snapshot_sources, user.id, project_id, manifest.snapshot_id,
        )
        return manifest, sources
    except HTTPException:
        raise
    except (SnapshotConfigurationError, SnapshotValidationError, OSError, ValueError, KeyError):
        raise HTTPException(503, "Project source snapshot is unavailable") from None


@router.get("/{project_id}/investigation/files")
async def list_investigation_files(
    project_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    manifest, sources = await _owned_sources(project_id, user, db)
    return {
        "project_id": str(project_id),
        "snapshot_id": str(manifest.snapshot_id),
        "files": [path for path, _ in sources],
    }


@router.post("/{project_id}/investigation", response_model=CodeReviewFileResponse)
async def investigate_project_file(
    project_id: UUID,
    payload: ProjectInvestigationRequest,
    user: Annotated[User, Depends(get_current_user_with_usage_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CodeReviewFileResponse:
    _, sources = await _owned_sources(project_id, user, db)
    # Snapshot paths are POSIX relative paths. Reject traversal and ambiguous forms.
    path = payload.relative_path
    if (
        "\\" in path or ":" in path or path.startswith("/")
        or any(part in (".", "..") for part in path.split("/"))
        or PurePosixPath(path).as_posix() != path
    ):
        raise HTTPException(400, "Invalid source path")

    selected = next((content for source_path, content in sources if source_path == path), None)
    if selected is None:
        raise HTTPException(404, "File not present in project snapshot")
    if not selected.strip():
        raise HTTPException(400, "Selected source file is empty")
    if len(selected) > 100_000:
        raise HTTPException(413, "File exceeds the investigation limit")

    related = await asyncio.to_thread(
        search_snapshot_sources,
        [(source_path, content) for source_path, content in sources if source_path != path],
        payload.task_description or path.replace("/", " "),
        4,
    )
    context = "\n\n".join(
        f"Reference {hit['relative_path']} lines {hit['start_line']}-{hit['end_line']}:\n{hit['content']}"
        for hit in related
    )[:6000]
    try:
        reviewed = await review_code(
            CodeReviewRequest(
                code=selected,
                language=PurePosixPath(path).suffix.lstrip(".") or None,
                review_mode=payload.review_mode,
                task_description=payload.task_description,
            ),
            related_workspace_context=context or None,
        )
    except (AIProviderError, CodeReviewError) as exc:
        raise HTTPException(502, "AI provider could not complete the investigation") from exc

    return CodeReviewFileResponse(
        relative_path=path,
        language=PurePosixPath(path).suffix.lstrip(".") or None,
        sha256=hashlib.sha256(selected.encode("utf-8")).hexdigest(),
        size_bytes=len(selected.encode("utf-8")),
        line_count=len(selected.splitlines()),
        review=reviewed,
    )