from datetime import datetime
from uuid import UUID
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AIUsageSummaryResponse(BaseModel):
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cached_input_tokens: int = 0
    total_reasoning_tokens: int = 0

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    total_ai_duration_ms: int = 0

    total_test_runs: int = 0
    passed_test_runs: int = 0
    failed_test_runs: int = 0
    timed_out_test_runs: int = 0
    errored_test_runs: int = 0

    total_test_duration_ms: int = 0


class AIUsageEventItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    provider: str
    model_id: str
    api_model: str
    operation: str
    request_type: str
    status: Literal["succeeded", "failed"]

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    cached_input_tokens: int | None
    reasoning_tokens: int | None

    duration_ms: int | None
    created_at: datetime


class AITestRunItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    execution_id: UUID
    runner: Literal["pytest", "npm", "go"]

    status: Literal[
        "passed",
        "failed",
        "timed_out",
        "error",
    ]

    exit_code: int | None
    duration_ms: int
    started_at: datetime
    finished_at: datetime
    created_at: datetime


class PaginatedAIUsageEventsResponse(BaseModel):
    items: list[AIUsageEventItem]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class PaginatedAITestRunsResponse(BaseModel):
    items: list[AITestRunItem]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)