"""Explicit semantic index lifecycle for current, user-owned project snapshots."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_current_user_with_usage_context
from app.core.ai_usage_context import ai_usage_context
from app.db.session import get_db
from app.models.project import Project
from app.models.user import User
from app.services.embedding_service import EmbeddingError
from app.services.project_semantic_index import (
    estimate_snapshot_chunks, index_project_snapshot, is_snapshot_index_ready,
    snapshot_vector_namespace, index_lifecycle_lock, remove_project_semantic_indexes,
)
from app.services.project_snapshots import (
    SnapshotConfigurationError, SnapshotValidationError,
    read_snapshot_manifest, read_snapshot_sources,
)
from app.services.vector_store import VectorStoreError

router = APIRouter(prefix="/projects", tags=["Project Semantic Index"])
logger = logging.getLogger(__name__)
MAX_INDEX_CHUNKS = 1200


class ConfirmIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_snapshot_id: UUID
    confirm_indexing: bool


class SemanticSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=8, ge=1, le=20)


async def _owned_snapshot(project_id: UUID, user: User, db: AsyncSession, *, read_sources=False):
    ownership = await db.execute(select(Project.id).where(
        Project.id == project_id, Project.user_id == user.id, Project.status == "ready",
    ))
    if ownership.scalar_one_or_none() is None:
        raise HTTPException(404, "Ready project not found")
    try:
        manifest = await asyncio.to_thread(read_snapshot_manifest, user.id, project_id)
        if manifest is None:
            raise HTTPException(409, "Project has no current snapshot")
        if not read_sources:
            return manifest, None
        sources = await asyncio.to_thread(
            read_snapshot_sources, user.id, project_id, manifest.snapshot_id,
        )
        return manifest, sources
    except HTTPException:
        raise
    except (SnapshotConfigurationError, SnapshotValidationError, OSError, ValueError, KeyError):
        raise HTTPException(503, "Project snapshot is unavailable") from None


def _service(request: Request):
    return getattr(request.app.state, "workspace_indexing_service", None)


def _jobs(request: Request):
    manager = getattr(request.app.state, "project_index_jobs", None)
    if manager is None:
        raise HTTPException(503, "Background indexing is not configured")
    return manager


@router.post("/{project_id}/semantic-index/jobs", status_code=202)
async def start_project_index_job(
    project_id: UUID, payload: ConfirmIndexRequest, request: Request,
    user: Annotated[User, Depends(get_current_user_with_usage_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    manifest, sources = await _owned_snapshot(project_id, user, db, read_sources=True)
    if not payload.confirm_indexing:
        raise HTTPException(400, "Explicit indexing confirmation is required")
    if payload.expected_snapshot_id != manifest.snapshot_id:
        raise HTTPException(409, "Snapshot changed. Preview indexing again")
    service = _service(request)
    if service is None:
        raise HTTPException(503, "Semantic embedding provider is not configured")
    estimate = await asyncio.to_thread(
        estimate_snapshot_chunks, service, user.id, project_id, manifest.snapshot_id, sources,
    )
    if estimate == 0:
        raise HTTPException(422, "No nonempty source chunks are available for indexing")
    if estimate > MAX_INDEX_CHUNKS:
        raise HTTPException(413, "Snapshot exceeds the 1,200-chunk embedding budget")
    try:
        return _jobs(request).start(
            user.id, project_id, manifest.snapshot_id, manifest.sha256, sources, estimate,
        )
    except ValueError:
        raise HTTPException(409, "An indexing job is already running for this project") from None
    except OSError:
        raise HTTPException(503, "Could not create indexing job") from None


@router.get("/{project_id}/semantic-index/jobs/latest")
async def get_latest_project_index_job(
    project_id: UUID, request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict | None:
    manifest, _ = await _owned_snapshot(project_id, user, db)
    try:
        return _jobs(request).latest(user.id, project_id, manifest.snapshot_id)
    except (OSError, ValueError):
        raise HTTPException(503, "Indexing job state is unavailable") from None


@router.get("/{project_id}/semantic-index/jobs/{job_id}")
async def get_project_index_job(
    project_id: UUID, job_id: UUID, request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    await _owned_snapshot(project_id, user, db)
    try:
        result = _jobs(request).get(user.id, project_id, job_id)
    except (OSError, ValueError, KeyError):
        raise HTTPException(503, "Indexing job state is unavailable") from None
    if result is None:
        raise HTTPException(404, "Indexing job not found")
    return result


@router.get("/{project_id}/semantic-index")
async def get_project_semantic_status(
    project_id: UUID, request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    manifest, _ = await _owned_snapshot(project_id, user, db)
    service = _service(request)
    if service is None:
        return {"status": "disabled", "snapshot_id": str(manifest.snapshot_id),
                "file_count": manifest.file_count, "estimated_chunks": None,
                "message": "Configure EMBEDDING_PROVIDER=openai and OPENAI_API_KEY on the backend."}
    ready = await asyncio.to_thread(
        is_snapshot_index_ready, service, user.id, project_id,
        manifest.snapshot_id, manifest.sha256, manifest.file_count,
    )
    return {"status": "ready" if ready else "not_indexed",
            "snapshot_id": str(manifest.snapshot_id), "file_count": manifest.file_count,
            "estimated_chunks": None,
            "embedding_model": service.vector_store.embedding_model}


@router.post("/{project_id}/semantic-index/preview")
async def preview_project_semantic_index(
    project_id: UUID, request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    manifest, sources = await _owned_snapshot(project_id, user, db, read_sources=True)
    service = _service(request)
    if service is None:
        raise HTTPException(503, "Semantic embedding provider is not configured")
    estimate = await asyncio.to_thread(
        estimate_snapshot_chunks, service, user.id, project_id, manifest.snapshot_id, sources,
    )
    return {"snapshot_id": str(manifest.snapshot_id), "file_count": len(sources),
            "estimated_chunks": estimate, "max_chunks": MAX_INDEX_CHUNKS,
            "embedding_model": service.vector_store.embedding_model,
            "requires_confirmation": True, "ai_calls": 0,
            "within_limit": 0 < estimate <= MAX_INDEX_CHUNKS}


@router.post("/{project_id}/semantic-index")
async def create_project_semantic_index(
    project_id: UUID, payload: ConfirmIndexRequest, request: Request,
    user: Annotated[User, Depends(get_current_user_with_usage_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    manifest, sources = await _owned_snapshot(project_id, user, db, read_sources=True)
    if not payload.confirm_indexing:
        raise HTTPException(400, "Explicit indexing confirmation is required")
    if payload.expected_snapshot_id != manifest.snapshot_id:
        raise HTTPException(409, "Snapshot changed. Preview indexing again.")
    service = _service(request)
    if service is None:
        raise HTTPException(503, "Semantic embedding provider is not configured")
    if index_lifecycle_lock.locked():
        raise HTTPException(409, "An indexing operation is already running")
    async with index_lifecycle_lock:
        estimate = await asyncio.to_thread(
            estimate_snapshot_chunks, service, user.id, project_id, manifest.snapshot_id, sources,
        )
        if estimate == 0:
            raise HTTPException(422, "No nonempty source chunks are available for indexing")
        if estimate > MAX_INDEX_CHUNKS:
            raise HTTPException(413, "Snapshot exceeds the 1,200-chunk embedding budget")
        try:
            with ai_usage_context(user_id=user.id, operation="project_embedding"):
                result = await index_project_snapshot(
                    service, user.id, project_id, manifest.snapshot_id, manifest.sha256, sources,
                )
        except (EmbeddingError, VectorStoreError, OSError, ValueError):
            logger.exception("Project semantic indexing failed")
            raise HTTPException(502, "Project semantic indexing failed; no ready index was recorded") from None
        current, _ = await _owned_snapshot(project_id, user, db)
        if current.snapshot_id != manifest.snapshot_id:
            raise HTTPException(409, "Snapshot changed while indexing; index is not active")
        return {"status": "ready", "snapshot_id": str(manifest.snapshot_id),
                "embedding_model": service.vector_store.embedding_model, **result}


@router.post("/{project_id}/semantic-search")
async def search_project_semantic_index(
    project_id: UUID, payload: SemanticSearchRequest, request: Request,
    user: Annotated[User, Depends(get_current_user_with_usage_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    manifest, sources = await _owned_snapshot(project_id, user, db, read_sources=True)
    service = _service(request)
    if service is None:
        raise HTTPException(503, "Semantic embedding provider is not configured")
    if not await asyncio.to_thread(
        is_snapshot_index_ready, service, user.id, project_id,
        manifest.snapshot_id, manifest.sha256, manifest.file_count,
    ):
        raise HTTPException(409, "Current project snapshot has not been indexed")
    namespace = snapshot_vector_namespace(service, user.id, project_id, manifest.snapshot_id)
    allowed = dict(sources)
    try:
        hits = await service.search(payload.query, top_k=payload.top_k, workspace_root=namespace)
    except (EmbeddingError, VectorStoreError, ValueError):
        raise HTTPException(502, "Semantic search failed") from None
    verified = []
    for hit in hits:
        meta = hit.metadata
        path = meta.get("relative_path")
        if meta.get("workspace_root") != namespace or path not in allowed:
            continue
        if not isinstance(hit.document, str) or hit.document not in allowed[path]:
            continue
        verified.append({"relative_path": path, "content": hit.document,
                         "start_line": meta.get("start_line"), "end_line": meta.get("end_line"),
                         "distance": hit.distance})
    return {"project_id": str(project_id), "snapshot_id": str(manifest.snapshot_id),
            "query": payload.query, "result_count": len(verified), "results": verified}


@router.delete("/{project_id}/semantic-index")
async def delete_project_semantic_index(
    project_id: UUID, request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Explicitly revoke every saved semantic index for an owned project."""
    await _owned_snapshot(project_id, user, db)
    service = _service(request)
    if service is None:
        raise HTTPException(503, "Semantic store must be configured to delete its vectors")
    async with index_lifecycle_lock:
        try:
            result = await asyncio.to_thread(
                remove_project_semantic_indexes, service, user.id, project_id,
            )
            jobs = getattr(request.app.state, "project_index_jobs", None)
            if jobs is not None:
                await asyncio.to_thread(jobs.prune, user.id, project_id)
        except (OSError, ValueError, SnapshotConfigurationError, VectorStoreError):
            logger.exception("Could not revoke project semantic index")
            raise HTTPException(503, "Could not safely remove project semantic index") from None
    return {"status": "not_indexed", **result}