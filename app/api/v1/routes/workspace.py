import asyncio
import logging

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user
from app.models.user import User

from app.schemas.workspace import WorkspaceNode
from app.schemas.workspace_context import (
    WorkspaceContextRequest,
    WorkspaceContextResponse,
    WorkspaceDocumentSummary,
)

from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceConfigurationError,
    WorkspaceReadError,
)

from app.services.workspace_tree import get_workspace_tree
from app.services.workspace_context import build_workspace_context


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