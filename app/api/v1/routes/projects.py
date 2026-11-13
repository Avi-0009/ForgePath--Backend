
"""Authenticated, user-owned project management APIs.

Repository ingestion, source snapshots, and indexing are intentionally
outside the scope of this module.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models.project import Project
from app.models.user import User


router = APIRouter(
    prefix="/projects",
    tags=["Projects"],
)


# Request and response schemas

class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Project name cannot be empty.")

        return value


class ProjectUpdateRequest(BaseModel):
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
    )

    archive: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError("Project name cannot be empty.")

        return value

    @model_validator(mode="after")
    def validate_update(self):
        if self.name is None and not self.archive:
            raise ValueError(
                "Provide a project name or request archiving."
            )

        return self


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    status: str
    created_at: datetime
    updated_at: datetime


class ProjectListResponse(BaseModel):
    items: list[ProjectResponse]
    limit: int
    offset: int


# Ownership helper

async def _get_owned_project(
    db: AsyncSession,
    *,
    project_id: UUID,
    user_id: UUID,
) -> Project:
    """Fetch a project only when owned by the authenticated user."""

    result = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.user_id == user_id,
        )
    )

    project = result.scalar_one_or_none()

    if project is None:
        # Identical response for nonexistent and foreign projects.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found.",
        )

    return project


# Create project

@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_project(
    payload: ProjectCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Project:
    """Create a persistent project owned by the current user."""

    project = Project(
        user_id=current_user.id,
        name=payload.name,
        status="created",
    )

    db.add(project)

    await db.commit()
    await db.refresh(project)

    return project


# List projects

@router.get(
    "",
    response_model=ProjectListResponse,
)
async def list_projects(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ProjectListResponse:
    """List only projects belonging to the current user."""

    result = await db.execute(
        select(Project)
        .where(Project.user_id == current_user.id)
        .order_by(
            Project.created_at.desc(),
            Project.id.desc(),
        )
        .limit(limit)
        .offset(offset)
    )

    projects = result.scalars().all()

    return ProjectListResponse(
        items=[
            ProjectResponse.model_validate(project)
            for project in projects
        ],
        limit=limit,
        offset=offset,
    )


# Get project

@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
)
async def get_project(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Project:
    """Retrieve an existing project owned by the current user."""

    return await _get_owned_project(
        db,
        project_id=project_id,
        user_id=current_user.id,
    )


# Update / archive project

@router.patch(
    "/{project_id}",
    response_model=ProjectResponse,
)
async def update_project(
    project_id: UUID,
    payload: ProjectUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Project:
    """Rename or archive an owned project."""

    project = await _get_owned_project(
        db,
        project_id=project_id,
        user_id=current_user.id,
    )

    if payload.name is not None:
        project.name = payload.name

    if payload.archive:
        project.status = "archived"

    await db.commit()
    await db.refresh(project)

    return project


# Delete project

@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_project(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete an owned project.

    Currently deletes only the project database record.
    Source snapshots and external resources are not yet attached.
    """

    project = await _get_owned_project(
        db,
        project_id=project_id,
        user_id=current_user.id,
    )

    await db.delete(project)
    await db.commit()