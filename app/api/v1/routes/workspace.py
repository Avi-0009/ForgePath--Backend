import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user
from app.models.user import User
from app.schemas.workspace import WorkspaceNode
from app.services.workspace import WorkspaceConfigurationError, WorkspaceReadError
from app.services.workspace_tree import get_workspace_tree


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspace",
    tags=["Workspace"]
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

    except Exception as exc:
        logger.exception("Failed to build the workspace tree")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not load the workspace tree.",
        ) from exc