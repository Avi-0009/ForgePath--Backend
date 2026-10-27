from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class CodePatchRequest(BaseModel):
    """Request to generate a proposed fix for a reviewed local file."""

    model_config = ConfigDict(extra="forbid")

    workspace_root: str = Field(
        min_length=1,
        max_length=1024,
        description="A configured, allowlisted project directory.",
    )

    relative_path: str = Field(
        min_length=1,
        max_length=1024,
        description="File path relative to the project root.",
    )

    # Hash returned by the earlier file-review request.
    expected_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 of the exact file version that was reviewed.",
    )

    finding_title: str = Field(
        min_length=1,
        max_length=200,
    )

    finding_evidence: str = Field(
        min_length=1,
        max_length=4000,
    )

    finding_explanation: str = Field(
        min_length=1,
        max_length=8000,
    )

    suggested_fix: str = Field(
        min_length=1,
        max_length=8000,
    )

    additional_context: str | None = Field(
        default=None,
        max_length=12000,
    )


class CodePatchResponse(BaseModel):
    """A proposed patch for human review; never applied automatically."""

    relative_path: str
    original_sha256: str
    proposed_sha256: str
    original_line_count: int = Field(ge=0)
    proposed_line_count: int = Field(ge=0)
    summary: str
    diff: str
    proposed_code: str
    patch_applied: bool = False


class CodePatchApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_root: str = Field(min_length=1, max_length=1024)
    relative_path: str = Field(min_length=1, max_length=1024)

    expected_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$"
    )
    proposed_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$"
    )

    proposed_code: str = Field(
        min_length=1,
        max_length=1_000_000,
    )

    # Applying a patch requires an explicit approval value.
    approval: Literal["apply"]


class CodePatchApplyResponse(BaseModel):
    relative_path: str
    original_sha256: str
    applied_sha256: str
    backup_id: str
    patch_applied: bool