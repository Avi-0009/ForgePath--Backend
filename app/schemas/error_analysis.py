from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TestFailureAnalysis(BaseModel):
    """Structured diagnosis of a failed local test execution."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1000)

    root_cause: str = Field(min_length=1, max_length=3000)

    explanation: str = Field(min_length=1, max_length=4000)

    candidate_files: list[str] = Field(
        default_factory=list,
        max_length=10,
    )

    suggested_fix: str = Field(min_length=1, max_length=4000)

    confidence: Literal["low", "medium", "high"]

    verification_notes: list[str] = Field(
        default_factory=list,
        max_length=8,
    )

    inventory_truncated: bool = False

    known_file_count: int = Field(default=0, ge=0)

    @field_validator(
        "summary",
        "root_cause",
        "explanation",
        "suggested_fix",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("This field cannot be empty.")

        return value

    @field_validator("candidate_files")
    @classmethod
    def validate_candidate_files(cls, value: list[str]) -> list[str]:
        cleaned = []
        seen = set()

        for path in value:
            path = path.strip()

            if not path:
                continue

            normalized = path.replace("\\", "/").casefold()

            if normalized not in seen:
                seen.add(normalized)
                cleaned.append(path)

        return cleaned