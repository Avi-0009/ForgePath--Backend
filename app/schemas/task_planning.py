from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


PlanningMode = Literal[
    "auto",
    "feature",
    "bug_fix",
    "refactor",
    "testing",
]

PlanComplexity = Literal[
    "low",
    "medium",
    "high",
]

RiskSeverity = Literal[
    "low",
    "medium",
    "high",
]


class TaskPlanningRequest(BaseModel):
    """Input contract for generating a local project implementation plan."""

    model_config = ConfigDict(extra="forbid")

    workspace_root: str = Field(
        min_length=1,
        max_length=1024,
        description="An allowlisted local project directory.",
    )

    task_description: str = Field(
        min_length=10,
        max_length=10_000,
        description="The task that ForgePath should break into steps.",
    )

    planning_mode: PlanningMode = "auto"

    additional_context: str | None = Field(
        default=None,
        max_length=12_000,
        description="Optional requirements, constraints, or preferences.",
    )

    max_steps: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of implementation steps.",
    )

    @field_validator("workspace_root", "task_description")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("This field cannot be empty or whitespace only.")

        return value.strip()


class TaskPlanStep(BaseModel):
    """One ordered and actionable step in the generated plan."""

    model_config = ConfigDict(extra="forbid")

    step_number: int = Field(
        ge=1,
        le=20,
        strict=True,
    )

    title: str = Field(
        min_length=1,
        max_length=160,
    )

    description: str = Field(
        min_length=1,
        max_length=3000,
    )

    dependencies: list[
        Annotated[int, Field(strict=True, ge=1, le=20)]
    ] = Field(
        default_factory=list,
        max_length=20,
        description="Earlier step numbers that must finish first.",
    )

    candidate_files_to_inspect: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Suggested relative project paths to inspect. "
            "These are candidates, not verified files."
        ),
    )

    acceptance_criteria: list[str] = Field(
        min_length=1,
        max_length=8,
    )

    verification_actions: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Suggested checks; they are not executed by the planner.",
    )

    @field_validator(
        "title",
        "description",
        mode="before",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if isinstance(value, str) and not value.strip():
            raise ValueError("This field cannot be empty or whitespace only.")

        return value.strip() if isinstance(value, str) else value


class TaskPlanRisk(BaseModel):
    """One identified implementation risk and its mitigation."""

    model_config = ConfigDict(extra="forbid")

    severity: RiskSeverity

    title: str = Field(
        min_length=1,
        max_length=160,
    )

    description: str = Field(
        min_length=1,
        max_length=2000,
    )

    mitigation: str = Field(
        min_length=1,
        max_length=2000,
    )


class TaskPlanningResponse(BaseModel):
    """Structured implementation plan returned by ForgePath."""

    model_config = ConfigDict(extra="forbid")

    plan_id: UUID = Field(default_factory=uuid4)

    workspace_root: str

    task_summary: str = Field(
        min_length=1,
        max_length=1000,
    )

    objective: str = Field(
        min_length=1,
        max_length=3000,
    )

    complexity: PlanComplexity

    assumptions: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    steps: list[TaskPlanStep] = Field(
        min_length=1,
        max_length=20,
    )

    risks: list[TaskPlanRisk] = Field(
        default_factory=list,
        max_length=15,
    )

    open_questions: list[str] = Field(
        default_factory=list,
        max_length=15,
    )

    overall_verification_strategy: list[str] = Field(
        default_factory=list,
        max_length=15,
    )

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @model_validator(mode="after")
    def validate_step_dependencies(self):
        """Require ordered steps and valid backward-only dependencies."""

        step_numbers = [
            step.step_number
            for step in self.steps
        ]

        expected_numbers = list(
            range(1, len(self.steps) + 1)
        )

        if step_numbers != expected_numbers:
            raise ValueError(
                "Steps must be ordered consecutively, starting at 1."
            )

        completed_steps: set[int] = set()

        for step in self.steps:
            if len(step.dependencies) != len(set(step.dependencies)):
                raise ValueError(
                    f"Step {step.step_number} has duplicate dependencies."
                )

            for dependency in step.dependencies:
                if dependency not in completed_steps:
                    raise ValueError(
                        f"Step {step.step_number} depends on invalid or "
                        "non-preceding step "
                        f"{dependency}."
                    )

            completed_steps.add(step.step_number)

        return self