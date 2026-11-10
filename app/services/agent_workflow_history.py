from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_workflow_history import AgentWorkflowHistory


TERMINAL_STATUSES = {
    "completed",
    "completed_without_tests",
    "tests_failed",
    "cancelled",
    "failed",
}


async def create_agent_workflow_history(
    db: AsyncSession,
    *,
    thread_id: UUID,
    user_id: UUID,
    workspace_root: str,
    task_description: str,
    planning_mode: str,
    test_runner: str,
) -> AgentWorkflowHistory:
    """Create a persistent record when an agent workflow starts."""

    record = AgentWorkflowHistory(
        thread_id=thread_id,
        user_id=user_id,
        workspace_root=workspace_root,
        task_description=task_description,
        planning_mode=planning_mode,
        test_runner=test_runner,
    )

    try:
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return record

    except Exception:
        await db.rollback()
        raise


async def sync_agent_workflow_history(
    db: AsyncSession,
    record: AgentWorkflowHistory,
    state: dict,
) -> AgentWorkflowHistory:
    """Synchronize searchable history metadata from LangGraph state."""

    status = state.get("status", "running")
    plan = state.get("plan")

    record.status = status
    record.message = (state.get("message") or "")[:2000] or None

    if isinstance(plan, dict):
        record.total_steps = len(plan.get("steps", []))

    completed_steps = state.get("completed_steps", [])
    record.completed_steps = len(set(completed_steps))

    record.repair_attempts = state.get("repair_attempts", 0)

    if status in TERMINAL_STATUSES:
        if record.finished_at is None:
            record.finished_at = datetime.now(timezone.utc)

    else:
        record.finished_at = None

    try:
        await db.commit()
        await db.refresh(record)
        return record

    except Exception:
        await db.rollback()
        raise


async def list_agent_workflow_history(
    db: AsyncSession,
    *,
    user_id: UUID,
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
) -> tuple[list[AgentWorkflowHistory], int]:
    """Return paginated workflow history belonging to one user."""

    filters = [
        AgentWorkflowHistory.user_id == user_id,
    ]

    if status:
        filters.append(
            AgentWorkflowHistory.status == status
        )

    statement = (
        select(AgentWorkflowHistory)
        .where(*filters)
        .order_by(
            AgentWorkflowHistory.created_at.desc(),
            AgentWorkflowHistory.id.desc(),
        )
        .offset(offset)
        .limit(limit)
    )

    count_statement = (
        select(func.count())
        .select_from(AgentWorkflowHistory)
        .where(*filters)
    )

    try:
        result = await db.scalars(statement)
        total = await db.scalar(count_statement)

        return list(result.all()), total or 0

    except Exception:
        await db.rollback()
        raise


async def get_agent_workflow_history(
    db: AsyncSession,
    *,
    user_id: UUID,
    thread_id: UUID,
) -> AgentWorkflowHistory | None:
    """Retrieve one workflow history record while enforcing ownership."""

    statement = select(AgentWorkflowHistory).where(
        AgentWorkflowHistory.user_id == user_id,
        AgentWorkflowHistory.thread_id == thread_id,
    )

    return await db.scalar(statement)