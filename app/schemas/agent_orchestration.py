from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.code_patch import CodePatchResponse
from app.schemas.execution import TestExecutionResponse
from app.schemas.task_planning import (
    TaskPlanningRequest,
    TaskPlanningResponse,
)
from app.schemas.error_analysis import TestFailureAnalysis


AgentWorkflowStatus = Literal[
    "running",
    "awaiting_plan_approval",
    "awaiting_file_selection",
    "awaiting_patch_approval",
    "awaiting_test_approval",
    "completed",
    "completed_without_tests",
    "tests_failed",
    "cancelled",
    "failed",
    "awaiting_repair_file_selection",
]


class AgentWorkflowStartRequest(TaskPlanningRequest):
    """Input for starting an orchestrated coding workflow."""

    test_runner: Literal[
        "pytest",
        "npm",
        "go",
    ] = "pytest"

    test_timeout_seconds: int = Field(
        default=120,
        ge=1,
        le=600,
    )


class AgentWorkflowResumeRequest(BaseModel):
    """Decision supplied when resuming a paused workflow."""

    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "approve",
        "reject",
        "cancel",
        "select_file",
    ]

    selected_file: str | None = Field(
        default=None,
        min_length=1,
        max_length=1024,
    )

    @model_validator(mode="after")
    def validate_action_payload(self):
        if self.action == "select_file":
            if not self.selected_file:
                raise ValueError(
                    "selected_file is required when action is select_file."
                )

        elif self.selected_file is not None:
            raise ValueError(
                "selected_file is only allowed with select_file."
            )

        return self


class AgentPendingAction(BaseModel):
    """Information about the decision currently required from the user."""

    kind: Literal[
        "plan_approval",
        "file_selection",
        "repair_file_selection",
        "patch_approval",
        "test_approval",
    ]

    message: str

    step_number: int | None = Field(
        default=None,
        ge=1,
    )

    available_files: list[str] = Field(
        default_factory=list,
        max_length=20,
    )


class AgentAppliedPatch(BaseModel):
    """Metadata for a patch applied during an approved workflow."""

    relative_path: str

    backup_id: str = Field(
        pattern=r"^[a-f0-9]{32}$",
    )

    original_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
    )

    applied_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
    )


class AgentWorkflowResponse(BaseModel):
    """Current state and latest result of an agent workflow."""

    thread_id: str

    status: AgentWorkflowStatus

    workspace_root: str

    message: str

    plan: TaskPlanningResponse | None = None

    current_step_number: int | None = Field(
        default=None,
        ge=1,
    )

    pending_action: AgentPendingAction | None = None

    patch_proposal: CodePatchResponse | None = None

    applied_patches: list[AgentAppliedPatch] = Field(
        default_factory=list,
    )

    manual_steps: list[int] = Field(
        default_factory=list,
    )

    skipped_steps: list[int] = Field(
        default_factory=list,
    )

    test_result: TestExecutionResponse | None = None

    error_message: str | None = None

    failure_analysis: TestFailureAnalysis | None = None

    repair_attempts: int = Field(
        default=0,
        ge=0,
    )

    max_repair_attempts: int = Field(
        default=2,
        ge=0,
        le=5,
    )