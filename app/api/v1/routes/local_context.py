from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_current_user
from app.models.user import User
from app.schemas.local_context import (
    ProjectInstructionsRequest,
    ProjectInstructionsResponse,
)
from app.services.local_context import (
    LocalContextError,
    get_project_instructions,
    save_project_instructions,
)


router = APIRouter(
    prefix="/local-context",
    tags=["Local Context"],
)


@router.put(
    "/instructions",
    response_model=ProjectInstructionsResponse,
    status_code=status.HTTP_200_OK,
)
async def save_instructions(
    payload: ProjectInstructionsRequest,
    _current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> ProjectInstructionsResponse:
    """Save or update instructions for an allowed local project."""

    try:
        context = await save_project_instructions(
            project_root=payload.workspace_root,
            instructions=payload.instructions,
        )
    except LocalContextError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return ProjectInstructionsResponse(
        project_name=context.project_name,
        instructions=context.instructions,
        created_at=context.created_at,
        updated_at=context.updated_at,
    )


@router.get(
    "/instructions",
    response_model=ProjectInstructionsResponse,
    status_code=status.HTTP_200_OK,
)
async def retrieve_instructions(
    workspace_root: Annotated[
        str,
        Query(min_length=1, max_length=1024),
    ],
    _current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> ProjectInstructionsResponse:
    """Retrieve saved instructions for an allowed local project."""

    try:
        context = await get_project_instructions(
            project_root=workspace_root,
        )
    except LocalContextError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if context is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No instructions have been saved for this project yet.",
        )

    return ProjectInstructionsResponse(
        project_name=context.project_name,
        instructions=context.instructions,
        created_at=context.created_at,
        updated_at=context.updated_at,
    )
