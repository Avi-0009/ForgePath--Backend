from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TestExecutionRequest(BaseModel):
    """Request to execute a predefined test runner locally."""

    model_config = ConfigDict(extra="forbid")

    workspace_root: str = Field(
        min_length=1,
        max_length=1024,
        description="An allowlisted local project directory.",
    )

    runner: Literal[
        "pytest",
        "npm",
        "go",
    ]

    timeout_seconds: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Maximum allowed execution time.",
    )

    approval: Literal["run"]


class TestExecutionResponse(BaseModel):
    """Result of a local test execution."""

    execution_id: UUID

    workspace_root: str

    runner: Literal[
        "pytest",
        "npm",
        "go",
    ]

    status: Literal[
        "passed",
        "failed",
        "timed_out",
        "error",
    ]

    command: list[str]

    exit_code: int | None

    duration_ms: int = Field(ge=0)

    stdout: str = ""
    stderr: str = ""

    stdout_truncated: bool = False
    stderr_truncated: bool = False

    started_at: datetime
    finished_at: datetime