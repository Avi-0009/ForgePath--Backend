from pydantic import BaseModel, ConfigDict, Field


class WorkspaceContextRequest(BaseModel):
    """Request to prepare context for one configured workspace."""

    model_config = ConfigDict(extra="forbid")

    workspace_root: str = Field(
        min_length=1,
        max_length=1024,
        description="An explicitly configured workspace root.",
    )


class WorkspaceDocumentSummary(BaseModel):
    """Metadata for one ingested workspace document."""

    relative_path: str
    language: str
    sha256: str
    size_bytes: int = Field(ge=0)
    line_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)


class WorkspaceContextResponse(BaseModel):
    """Safe summary of the prepared workspace context."""

    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)

    files_scanned: int = Field(ge=0)
    files_skipped: int = Field(ge=0)

    total_bytes: int = Field(ge=0)
    truncated: bool

    documents: list[WorkspaceDocumentSummary] = Field(
        default_factory=list
    )