import asyncio
import logging

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user, get_workspace_indexing_service
from app.models.user import User

from app.schemas.workspace import WorkspaceNode
from app.schemas.workspace_context import (
    WorkspaceContextRequest,
    WorkspaceContextResponse,
    WorkspaceDocumentSummary,
)
from app.schemas.workspace_indexing import (
    WorkspaceIndexRequest,
    WorkspaceIndexResponse,
    WorkspaceSearchRequest,
    WorkspaceSearchResponse,
    WorkspaceSearchHitResponse,
)

from app.services.embedding_service import EmbeddingError
from app.services.vector_store import VectorStoreError

from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceConfigurationError,
    WorkspaceReadError,
)

from app.services.workspace_tree import get_workspace_tree
from app.services.workspace_context import build_workspace_context

from app.services.workspace_ingestion import (
    ingest_workspace,
    resolve_allowed_workspace_root,
)

from app.services.workspace_indexing import (
    WorkspaceIndexingConfigurationError,
    WorkspaceIndexingError,
    WorkspaceIndexingService,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspace",
    tags=["Workspace"],
)


@router.get(
    "/tree",
    response_model=WorkspaceNode,
    status_code=status.HTTP_200_OK,
)
async def get_workspace_tree_endpoint(
    _current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> WorkspaceNode:
    """Return the file tree for configured, allowlisted workspaces."""

    try:
        return await asyncio.to_thread(get_workspace_tree)

    except WorkspaceConfigurationError as exc:
        logger.error("Workspace configuration error: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Workspace configuration is invalid or unavailable. "
                "Check WORKSPACE_ALLOWED_ROOTS."
            ),
        ) from exc

    except WorkspaceReadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception("Failed to build the workspace tree")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not load the workspace tree.",
        ) from None


@router.post(
    "/context/prepare",
    response_model=WorkspaceContextResponse,
    status_code=status.HTTP_200_OK,
)
async def prepare_workspace_context(
    payload: WorkspaceContextRequest,
    _current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> WorkspaceContextResponse:
    """
    Prepare in-memory workspace context and return safe metadata.

    Source-code contents and chunks are deliberately not included
    in the API response.
    """

    try:
        context = await asyncio.to_thread(
            build_workspace_context,
            payload.workspace_root,
        )

    except WorkspaceConfigurationError as exc:
        logger.error("Workspace configuration error: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workspace configuration is invalid or unavailable.",
        ) from exc

    except WorkspaceAccessDenied as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except WorkspaceReadError:
        logger.exception("Workspace context preparation failed")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The workspace context could not be prepared.",
        ) from None

    except Exception:
        logger.exception("Unexpected workspace context preparation failure")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workspace context preparation failed unexpectedly.",
        ) from None

    chunk_counts: dict[str, int] = {}

    for chunk in context.chunks:
        chunk_counts[chunk.relative_path] = (
            chunk_counts.get(chunk.relative_path, 0) + 1
        )

    documents = [
        WorkspaceDocumentSummary(
            relative_path=document.relative_path,
            language=document.language,
            sha256=document.sha256,
            size_bytes=document.size_bytes,
            line_count=document.line_count,
            chunk_count=chunk_counts.get(document.relative_path, 0),
        )
        for document in context.documents
    ]

    return WorkspaceContextResponse(
        document_count=context.document_count,
        chunk_count=context.chunk_count,
        files_scanned=context.files_scanned,
        files_skipped=context.files_skipped,
        total_bytes=context.total_bytes,
        truncated=context.truncated,
        documents=documents,
    )


@router.post(
    "/index",
    response_model=WorkspaceIndexResponse,
    status_code=status.HTTP_200_OK,
)
async def index_workspace(
    payload: WorkspaceIndexRequest,
    _current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
    indexing_service: Annotated[
        WorkspaceIndexingService,
        Depends(get_workspace_indexing_service),
    ],
) -> WorkspaceIndexResponse:
    """
    Ingest and semantically index an allowlisted workspace.

    Pipeline:
        Workspace ingestion
        -> Chunking
        -> Embedding generation
        -> ChromaDB persistence
    """

    try:
        # Safely read supported workspace documents.
        ingestion = await asyncio.to_thread(
            ingest_workspace,
            payload.workspace_root,
        )

        # Generate embeddings and persist the resulting chunks.
        result = await indexing_service.index_documents(
            ingestion.documents
        )

        return WorkspaceIndexResponse(
            files_processed=result.files_processed,
            chunks_processed=result.chunks_processed,
            vectors_upserted=result.vectors_upserted,
            stale_vectors_deleted=result.stale_vectors_deleted,
            empty_files=result.empty_files,
            files_scanned=ingestion.files_scanned,
            files_skipped=ingestion.files_skipped,
            total_bytes=ingestion.total_bytes,
            truncated=ingestion.truncated,
        )

    except WorkspaceConfigurationError as exc:
        logger.error("Workspace configuration error: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workspace configuration is invalid or unavailable.",
        ) from exc

    except WorkspaceAccessDenied as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except WorkspaceReadError:
        logger.exception("Workspace ingestion failed")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The workspace could not be safely ingested.",
        ) from None

    except WorkspaceIndexingConfigurationError as exc:
        logger.error("Workspace indexing configuration error: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workspace indexing is not correctly configured.",
        ) from exc

    except EmbeddingError:
        logger.exception("Embedding provider failed during indexing")

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The embedding provider could not process the workspace.",
        ) from None

    except (VectorStoreError, WorkspaceIndexingError):
        logger.exception("Workspace vector indexing failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not index the workspace.",
        ) from None

    except Exception:
        logger.exception("Unexpected workspace indexing failure")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workspace indexing failed unexpectedly.",
        ) from None


@router.post(
    "/search",
    response_model=WorkspaceSearchResponse,
    status_code=status.HTTP_200_OK,
)
async def search_workspace(
    payload: WorkspaceSearchRequest,
    _current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
    indexing_service: Annotated[
        WorkspaceIndexingService,
        Depends(get_workspace_indexing_service),
    ],
) -> WorkspaceSearchResponse:
    """
    Perform semantic search within one allowlisted workspace.

    Results include source paths, line ranges, and matching code chunks.
    """

    try:
        # Resolve the workspace against the configured allowlist.
        workspace_root = await asyncio.to_thread(
            resolve_allowed_workspace_root,
            payload.workspace_root,
        )

        hits = await indexing_service.search(
            payload.query,
            top_k=payload.top_k,
            workspace_root=str(workspace_root),
            relative_path=payload.relative_path,
            language=payload.language,
        )

        results = [
            WorkspaceSearchHitResponse(
                relative_path=hit.metadata["relative_path"],
                language=hit.metadata["language"],
                chunk_index=hit.metadata["chunk_index"],
                start_line=hit.metadata["start_line"],
                end_line=hit.metadata["end_line"],
                content=hit.document,
                distance=hit.distance,
            )
            for hit in hits
        ]

        return WorkspaceSearchResponse(
            workspace_root=str(workspace_root),
            query=payload.query,
            result_count=len(results),
            results=results,
        )

    except WorkspaceConfigurationError as exc:
        logger.error("Workspace configuration error: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workspace configuration is invalid or unavailable.",
        ) from exc

    except WorkspaceAccessDenied as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except EmbeddingError:
        logger.exception("Embedding provider failed during semantic search")

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The embedding provider could not process the query.",
        ) from None

    except (VectorStoreError, WorkspaceIndexingError):
        logger.exception("Workspace semantic search failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not search the workspace.",
        ) from None

    except Exception:
        logger.exception("Unexpected workspace search failure")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workspace semantic search failed unexpectedly.",
        ) from None