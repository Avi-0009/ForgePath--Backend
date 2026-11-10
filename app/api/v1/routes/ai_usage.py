import logging
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_with_usage_context
from app.db.session import get_db
from app.models.user import User
from app.schemas.ai_usage import (
    AIUsageSummaryResponse,
    PaginatedAIUsageEventsResponse,
    PaginatedAITestRunsResponse,
)
from app.services.ai_usage_metrics import (
    get_ai_usage_summary,
    list_ai_usage_events,
    list_agent_test_runs,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai/usage",
    tags=["AI Usage"],
)


def validate_date_range(
    start_at: datetime | None,
    end_at: datetime | None,
) -> None:
    """Validate timezone-aware date filters."""

    for name, value in (
        ("start_at", start_at),
        ("end_at", end_at),
    ):
        if value is not None and value.utcoffset() is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{name} must include a timezone offset.",
            )

    if start_at and end_at and end_at <= start_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="end_at must be later than start_at.",
        )


@router.get(
    "/summary",
    response_model=AIUsageSummaryResponse,
)
async def get_usage_summary(
    current_user: Annotated[
        User,
        Depends(get_current_user_with_usage_context),
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> AIUsageSummaryResponse:

    validate_date_range(start_at, end_at)

    result = await get_ai_usage_summary(
        db,
        user_id=current_user.id,
        start_at=start_at,
        end_at=end_at,
    )

    return AIUsageSummaryResponse(**result)


@router.get(
    "/events",
    response_model=PaginatedAIUsageEventsResponse,
)
async def get_usage_events(
    current_user: Annotated[
        User,
        Depends(get_current_user_with_usage_context),
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    provider: Literal[
        "nvidia",
        "openai",
        "huggingface",
    ] | None = None,
    model_id: str | None = Query(default=None, max_length=128),
    operation: str | None = Query(default=None, max_length=64),
    event_status: Literal[
        "succeeded",
        "failed",
    ] | None = Query(default=None, alias="status"),
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> PaginatedAIUsageEventsResponse:

    validate_date_range(start_at, end_at)

    records, total = await list_ai_usage_events(
        db,
        user_id=current_user.id,
        limit=limit,
        offset=offset,
        provider=provider,
        model_id=model_id,
        operation=operation,
        event_status=event_status,
        start_at=start_at,
        end_at=end_at,
    )

    return PaginatedAIUsageEventsResponse(
        items=records,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/test-runs",
    response_model=PaginatedAITestRunsResponse,
)
async def get_usage_test_runs(
    current_user: Annotated[
        User,
        Depends(get_current_user_with_usage_context),
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    runner: Literal[
        "pytest",
        "npm",
        "go",
    ] | None = None,
    run_status: Literal[
        "passed",
        "failed",
        "timed_out",
        "error",
    ] | None = Query(default=None, alias="status"),
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> PaginatedAITestRunsResponse:

    validate_date_range(start_at, end_at)

    records, total = await list_agent_test_runs(
        db,
        user_id=current_user.id,
        limit=limit,
        offset=offset,
        runner=runner,
        run_status=run_status,
        start_at=start_at,
        end_at=end_at,
    )

    return PaginatedAITestRunsResponse(
        items=records,
        total=total,
        limit=limit,
        offset=offset,
    )