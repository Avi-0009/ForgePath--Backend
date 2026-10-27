
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ReviewMode = Literal[
    "general",
    "dsa",
    "bug_hunt",
    "security",
    "quality",
]

FindingSeverity = Literal[
    "critical",
    "high",
    "medium",
    "low",
    "info",
]

FindingCategory = Literal[
    "correctness",
    "algorithm",
    "complexity",
    "security",
    "performance",
    "readability",
    "maintainability",
    "testing",
    "other",
]


class CodeReviewRequest(BaseModel):
    """Input contract for reviewing code from any source."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        min_length=1,
        max_length=100_000,
        description="Source code to review.",
    )

    language: str | None = Field(
        default=None,
        max_length=50,
        description="Programming language; infer it when omitted.",
    )

    review_mode: ReviewMode = "general"

    task_description: str | None = Field(
        default=None,
        max_length=10_000,
        description="Problem statement, requirements, or intended behavior.",
    )

    additional_context: str | None = Field(
        default=None,
        max_length=12_000,
        description="Relevant constraints, conventions, or context.",
    )

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Code cannot be empty or whitespace only.")
        return value


class ReviewFinding(BaseModel):
    """One evidence-backed issue or improvement."""

    severity: FindingSeverity
    category: FindingCategory
    title: str = Field(min_length=1, max_length=200)

    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)

    evidence: str = Field(
        min_length=1,
        description="The code or behavior supporting this finding.",
    )

    explanation: str = Field(min_length=1)
    suggested_fix: str | None = None


class CodeReviewResponse(BaseModel):
    """Structured results from the code review engine."""

    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    suggested_tests: list[str] = Field(default_factory=list)