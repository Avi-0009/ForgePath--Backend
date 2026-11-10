"""Best-effort persistence for AI usage and local test-run metadata.

This module deliberately stores metadata only. It never accepts or persists
prompts, model responses, source contents, commands, workspace paths, or test
stdout/stderr.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import update

from app.core.ai_usage_context import get_ai_usage_context
from app.db.session import SessionLocal
from app.models.agent_test_run import AgentTestRun
from app.models.agent_workflow_history import AgentWorkflowHistory
from app.models.ai_usage_event import AIUsageEvent


logger = logging.getLogger(__name__)


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


async def record_ai_usage_event(
    *,
    provider: str,
    model_id: str,
    api_model: str,
    status: str,
    request_type: str = "chat_completion",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    total_tokens_calculated: bool = False,
    duration_ms: int | None = None,
    provider_request_id: str | None = None,
    finish_reason: str | None = None,
    error_type: str | None = None,
    operation: str | None = None,
) -> bool:
    """Persist one metadata-only AI request and update workflow aggregates.

    Persistence is deliberately best-effort: telemetry failures are logged but
    never replace the user's successful or failed AI operation.
    """

    context = get_ai_usage_context()
    event_operation = operation or context.operation or "unspecified"

    if status not in {"succeeded", "failed"}:
        raise ValueError("AI usage status must be 'succeeded' or 'failed'.")
    if len(event_operation) > 64:
        raise ValueError("AI usage operation cannot exceed 64 characters.")

    safe_input = _optional_nonnegative_int(input_tokens)
    safe_output = _optional_nonnegative_int(output_tokens)
    safe_total = _optional_nonnegative_int(total_tokens)
    safe_cached = _optional_nonnegative_int(cached_input_tokens)
    safe_reasoning = _optional_nonnegative_int(reasoning_tokens)
    safe_duration = _optional_nonnegative_int(duration_ms)

    event = AIUsageEvent(
        user_id=context.user_id,
        thread_id=context.thread_id,
        provider=str(provider)[:32],
        model_id=str(model_id)[:128],
        api_model=str(api_model)[:255],
        operation=event_operation,
        request_type=str(request_type)[:32],
        status=status,
        input_tokens=safe_input,
        output_tokens=safe_output,
        total_tokens=safe_total,
        cached_input_tokens=safe_cached,
        reasoning_tokens=safe_reasoning,
        total_tokens_calculated=bool(total_tokens_calculated),
        duration_ms=safe_duration,
        provider_request_id=(str(provider_request_id)[:255] if provider_request_id else None),
        finish_reason=(str(finish_reason)[:64] if finish_reason else None),
        # Only the exception class is accepted. No exception text is persisted.
        error_type=(str(error_type)[:128] if error_type else None),
    )

    try:
        async with SessionLocal() as db:
            db.add(event)

            if context.thread_id is not None:
                await db.execute(
                    update(AgentWorkflowHistory)
                    .where(AgentWorkflowHistory.thread_id == context.thread_id)
                    .values(
                        total_input_tokens=(
                            AgentWorkflowHistory.total_input_tokens + (safe_input or 0)
                        ),
                        total_output_tokens=(
                            AgentWorkflowHistory.total_output_tokens + (safe_output or 0)
                        ),
                        total_tokens=(
                            AgentWorkflowHistory.total_tokens + (safe_total or 0)
                        ),
                        total_cached_input_tokens=(
                            AgentWorkflowHistory.total_cached_input_tokens + (safe_cached or 0)
                        ),
                        total_reasoning_tokens=(
                            AgentWorkflowHistory.total_reasoning_tokens + (safe_reasoning or 0)
                        ),
                        ai_request_count=AgentWorkflowHistory.ai_request_count + 1,
                        failed_ai_request_count=(
                            AgentWorkflowHistory.failed_ai_request_count
                            + (1 if status == "failed" else 0)
                        ),
                        total_ai_duration_ms=(
                            AgentWorkflowHistory.total_ai_duration_ms + (safe_duration or 0)
                        ),
                    )
                )

            await db.commit()
        return True

    except Exception:
        logger.warning(
            "Could not persist AI usage telemetry; continuing without telemetry.",
            exc_info=True,
        )
        return False


async def record_agent_test_run(
    result: Any,
    *,
    user_id: UUID | None = None,
    thread_id: UUID | None = None,
) -> bool:
    """Persist one test execution's safe metadata and update workflow totals.

    ``result`` may be the runner dataclass or a dictionary with matching keys.
    Output text, command arguments, and workspace paths are never read.
    """

    context = get_ai_usage_context()

    def read(name: str, default: Any = None) -> Any:
        if isinstance(result, dict):
            return result.get(name, default)
        return getattr(result, name, default)

    execution_id = read("execution_id")
    runner = read("runner")
    run_status = read("status")
    duration_ms = read("duration_ms")
    started_at = read("started_at")
    finished_at = read("finished_at")

    if not isinstance(execution_id, UUID):
        logger.warning("Skipped test-run telemetry because execution_id was invalid.")
        return False
    if runner not in {"pytest", "npm", "go"}:
        logger.warning("Skipped test-run telemetry because runner was unsupported.")
        return False
    if run_status not in {"passed", "failed", "timed_out", "error"}:
        logger.warning("Skipped test-run telemetry because status was unsupported.")
        return False
    if not isinstance(started_at, datetime):
        logger.warning("Skipped test-run telemetry because started_at was invalid.")
        return False
    if not isinstance(finished_at, datetime):
        logger.warning("Skipped test-run telemetry because finished_at was invalid.")
        return False

    safe_duration = _nonnegative_int(duration_ms)
    resolved_user_id = user_id or context.user_id
    resolved_thread_id = thread_id or context.thread_id

    event = AgentTestRun(
        execution_id=execution_id,
        user_id=resolved_user_id,
        thread_id=resolved_thread_id,
        runner=runner,
        status=run_status,
        exit_code=read("exit_code"),
        duration_ms=safe_duration,
        stdout_truncated=bool(read("stdout_truncated", False)),
        stderr_truncated=bool(read("stderr_truncated", False)),
        started_at=started_at,
        finished_at=finished_at,
    )

    try:
        async with SessionLocal() as db:
            db.add(event)

            if resolved_thread_id is not None:
                await db.execute(
                    update(AgentWorkflowHistory)
                    .where(AgentWorkflowHistory.thread_id == resolved_thread_id)
                    .values(
                        test_run_count=AgentWorkflowHistory.test_run_count + 1,
                        test_pass_count=(
                            AgentWorkflowHistory.test_pass_count
                            + (1 if run_status == "passed" else 0)
                        ),
                        test_fail_count=(
                            AgentWorkflowHistory.test_fail_count
                            + (1 if run_status == "failed" else 0)
                        ),
                        test_timeout_count=(
                            AgentWorkflowHistory.test_timeout_count
                            + (1 if run_status == "timed_out" else 0)
                        ),
                        test_error_count=(
                            AgentWorkflowHistory.test_error_count
                            + (1 if run_status == "error" else 0)
                        ),
                        total_test_duration_ms=(
                            AgentWorkflowHistory.total_test_duration_ms + safe_duration
                        ),
                    )
                )

            await db.commit()
        return True

    except Exception:
        logger.warning(
            "Could not persist test-run telemetry; continuing without telemetry.",
            exc_info=True,
        )
        return False