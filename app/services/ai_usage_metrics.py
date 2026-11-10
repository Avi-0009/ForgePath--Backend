from datetime import datetime
from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ai_usage_event import AIUsageEvent
from app.models.agent_test_run import AgentTestRun


def _validate_date_range(
    start_at: datetime | None,
    end_at: datetime | None,
) -> None:
    if start_at and end_at and end_at <= start_at:
        raise ValueError("end_at must be later than start_at.")


def _apply_date_filters(
    statement,
    model,
    start_at: datetime | None,
    end_at: datetime | None,
):
    if start_at is not None:
        statement = statement.where(model.created_at >= start_at)

    if end_at is not None:
        statement = statement.where(model.created_at < end_at)

    return statement


async def get_ai_usage_summary(
    db: AsyncSession,
    *,
    user_id: UUID,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> dict:
    """Return aggregated AI usage and test execution metrics for one user."""

    _validate_date_range(start_at, end_at)

    ai_statement = select(
        func.coalesce(
            func.sum(AIUsageEvent.input_tokens), 0
        ).label("total_input_tokens"),
        func.coalesce(
            func.sum(AIUsageEvent.output_tokens), 0
        ).label("total_output_tokens"),
        func.coalesce(
            func.sum(AIUsageEvent.total_tokens), 0
        ).label("total_tokens"),
        func.coalesce(
            func.sum(AIUsageEvent.cached_input_tokens), 0
        ).label("total_cached_input_tokens"),
        func.coalesce(
            func.sum(AIUsageEvent.reasoning_tokens), 0
        ).label("total_reasoning_tokens"),
        func.count(AIUsageEvent.id).label("total_requests"),
        func.coalesce(
            func.sum(
                case(
                    (AIUsageEvent.status == "succeeded", 1),
                    else_=0,
                )
            ),
            0,
        ).label("successful_requests"),
        func.coalesce(
            func.sum(
                case(
                    (AIUsageEvent.status == "failed", 1),
                    else_=0,
                )
            ),
            0,
        ).label("failed_requests"),
        func.coalesce(
            func.sum(AIUsageEvent.duration_ms), 0
        ).label("total_ai_duration_ms"),
    ).where(
        AIUsageEvent.user_id == user_id
    )

    ai_statement = _apply_date_filters(
        ai_statement,
        AIUsageEvent,
        start_at,
        end_at,
    )

    ai_result = (
        await db.execute(ai_statement)
    ).mappings().one()

    test_statement = select(
        func.count(AgentTestRun.id).label("total_test_runs"),
        func.coalesce(
            func.sum(
                case(
                    (AgentTestRun.status == "passed", 1),
                    else_=0,
                )
            ),
            0,
        ).label("passed_test_runs"),
        func.coalesce(
            func.sum(
                case(
                    (AgentTestRun.status == "failed", 1),
                    else_=0,
                )
            ),
            0,
        ).label("failed_test_runs"),
        func.coalesce(
            func.sum(
                case(
                    (AgentTestRun.status == "timed_out", 1),
                    else_=0,
                )
            ),
            0,
        ).label("timed_out_test_runs"),
        func.coalesce(
            func.sum(
                case(
                    (AgentTestRun.status == "error", 1),
                    else_=0,
                )
            ),
            0,
        ).label("errored_test_runs"),
        func.coalesce(
            func.sum(AgentTestRun.duration_ms), 0
        ).label("total_test_duration_ms"),
    ).where(
        AgentTestRun.user_id == user_id
    )

    test_statement = _apply_date_filters(
        test_statement,
        AgentTestRun,
        start_at,
        end_at,
    )

    test_result = (
        await db.execute(test_statement)
    ).mappings().one()

    return {
        **dict(ai_result),
        **dict(test_result),
    }


async def list_ai_usage_events(
    db: AsyncSession,
    *,
    user_id: UUID,
    limit: int = 20,
    offset: int = 0,
    provider: str | None = None,
    model_id: str | None = None,
    operation: str | None = None,
    event_status: str | None = None,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> tuple[list[AIUsageEvent], int]:
    """Return paginated AI usage events belonging to one user."""

    _validate_date_range(start_at, end_at)

    filters = [
        AIUsageEvent.user_id == user_id,
    ]

    if provider:
        filters.append(AIUsageEvent.provider == provider)

    if model_id:
        filters.append(AIUsageEvent.model_id == model_id)

    if operation:
        filters.append(AIUsageEvent.operation == operation)

    if event_status:
        filters.append(AIUsageEvent.status == event_status)

    statement = (
        select(AIUsageEvent)
        .where(*filters)
    )

    statement = _apply_date_filters(
        statement,
        AIUsageEvent,
        start_at,
        end_at,
    )

    statement = statement.order_by(
        AIUsageEvent.created_at.desc(),
        AIUsageEvent.id.desc(),
    ).offset(offset).limit(limit)

    count_statement = (
        select(func.count())
        .select_from(AIUsageEvent)
        .where(*filters)
    )

    count_statement = _apply_date_filters(
        count_statement,
        AIUsageEvent,
        start_at,
        end_at,
    )

    result = await db.scalars(statement)
    total = await db.scalar(count_statement)

    return list(result.all()), total or 0


async def list_agent_test_runs(
    db: AsyncSession,
    *,
    user_id: UUID,
    limit: int = 20,
    offset: int = 0,
    runner: str | None = None,
    run_status: str | None = None,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> tuple[list[AgentTestRun], int]:
    """Return paginated test executions belonging to one user."""

    _validate_date_range(start_at, end_at)

    filters = [
        AgentTestRun.user_id == user_id,
    ]

    if runner:
        filters.append(AgentTestRun.runner == runner)

    if run_status:
        filters.append(AgentTestRun.status == run_status)

    statement = (
        select(AgentTestRun)
        .where(*filters)
    )

    statement = _apply_date_filters(
        statement,
        AgentTestRun,
        start_at,
        end_at,
    )

    statement = statement.order_by(
        AgentTestRun.created_at.desc(),
        AgentTestRun.id.desc(),
    ).offset(offset).limit(limit)

    count_statement = (
        select(func.count())
        .select_from(AgentTestRun)
        .where(*filters)
    )

    count_statement = _apply_date_filters(
        count_statement,
        AgentTestRun,
        start_at,
        end_at,
    )

    result = await db.scalars(statement)
    total = await db.scalar(count_statement)

    return list(result.all()), total or 0