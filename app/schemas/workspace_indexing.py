
from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkspaceIndexRequest(BaseModel):
    """Request to index an allowlisted workspace."""

    model_config = ConfigDict(extra="forbid")

    workspace_root: str = Field(
        min_length=1,
        max_length=1024,
        description="An explicitly configured workspace root.",
    )

    @field_validator("workspace_root")
    @classmethod
    def validate_workspace_root(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Workspace root cannot be empty.")

        return value


class WorkspaceIndexResponse(BaseModel):
    """Summary of a workspace indexing operation."""

    files_processed: int = Field(ge=0)
    chunks_processed: int = Field(ge=0)
    vectors_upserted: int = Field(ge=0)
    stale_vectors_deleted: int = Field(ge=0)
    empty_files: int = Field(ge=0)

    files_scanned: int = Field(ge=0)
    files_skipped: int = Field(ge=0)
    total_bytes: int = Field(ge=0)

    truncated: bool


class WorkspaceSearchRequest(BaseModel):
    """Request to search indexed workspace code semantically."""

    model_config = ConfigDict(extra="forbid")

    workspace_root: str = Field(
        min_length=1,
        max_length=1024,
    )

    query: str = Field(
        min_length=1,
        max_length=4000,
        description="Natural-language or code search query.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of matching chunks.",
    )

    relative_path: str | None = Field(
        default=None,
        min_length=1,
        max_length=2048,
    )

    language: str | None = Field(
        default=None,
        min_length=1,
        max_length=32,
    )

    @field_validator("workspace_root", "query")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("This field cannot be empty.")

        return value

    @field_validator("relative_path", "language")
    @classmethod
    def validate_optional_text(
        cls,
        value: str | None,
    ) -> str | None:

        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError("This field cannot be empty.")

        return value


class WorkspaceSearchHitResponse(BaseModel):
    """One matching source-code chunk."""

    relative_path: str
    language: str

    chunk_index: int = Field(ge=0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    content: str

    distance: float | None = None


class WorkspaceSearchResponse(BaseModel):
    """Semantic search results for one workspace."""

    workspace_root: str
    query: str

    result_count: int = Field(ge=0)

    results: list[WorkspaceSearchHitResponse] = Field(
        default_factory=list,
    )
