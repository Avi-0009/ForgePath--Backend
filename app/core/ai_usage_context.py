"""Request-scoped identity and operation context for usage telemetry.

Context is propagated through async tasks and asyncio.to_thread(). It contains
only IDs and a short operation label; never place prompts or source content here.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator
from uuid import UUID


@dataclass(frozen=True, slots=True)
class AIUsageContext:
    user_id: UUID | None = None
    thread_id: UUID | None = None
    operation: str = "unspecified"


_CURRENT_AI_USAGE_CONTEXT: ContextVar[AIUsageContext] = ContextVar(
    "forgepath_ai_usage_context",
    default=AIUsageContext(),
)


def get_ai_usage_context() -> AIUsageContext:
    """Return the current request's usage attribution metadata."""

    return _CURRENT_AI_USAGE_CONTEXT.get()


def _normalize_optional_uuid(value: UUID | str | None, field_name: str) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        raise ValueError(f"{field_name} must be a valid UUID or None.") from None


@contextmanager
def ai_usage_context(
    *,
    user_id: UUID | str | None = None,
    thread_id: UUID | str | None = None,
    operation: str | None = None,
) -> Iterator[AIUsageContext]:
    """Temporarily set or override usage attribution while preserving parents.

    Passing ``None`` keeps an existing ID, which makes nested scopes suitable
    for operation labels inside authenticated requests and agent workflows.
    """

    current = get_ai_usage_context()
    resolved_operation = current.operation if operation is None else operation.strip()

    if not resolved_operation or len(resolved_operation) > 64:
        raise ValueError("operation must contain between 1 and 64 characters.")

    updated = AIUsageContext(
        user_id=(
            current.user_id
            if user_id is None
            else _normalize_optional_uuid(user_id, "user_id")
        ),
        thread_id=(
            current.thread_id
            if thread_id is None
            else _normalize_optional_uuid(thread_id, "thread_id")
        ),
        operation=resolved_operation,
    )

    token: Token[AIUsageContext] = _CURRENT_AI_USAGE_CONTEXT.set(updated)
    try:
        yield updated
    finally:
        _CURRENT_AI_USAGE_CONTEXT.reset(token)