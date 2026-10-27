from pydantic import BaseModel, ConfigDict, Field


class ProjectInstructionsRequest(BaseModel):
    """Request to save or update instructions for a local project."""

    model_config = ConfigDict(extra="forbid")

    workspace_root: str = Field(
        min_length=1,
        max_length=1024,
        description="A configured, allowlisted workspace directory.",
    )

    instructions: str = Field(
        max_length=50_000,
        description="Project-specific instructions and coding conventions.",
    )


class ProjectInstructionsResponse(BaseModel):
    """Saved project instructions, without exposing the absolute path."""

    project_name: str
    instructions: str
    created_at: str
    updated_at: str
