from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AIUsageEvent(Base):
    """One metadata-only record for a text-generation or embedding request."""

    __tablename__ = "ai_usage_events"

    __table_args__ = (
        CheckConstraint(
            "status IN ('succeeded', 'failed')",
            name="ck_ai_usage_events_status",
        ),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_ai_usage_events_input_tokens",
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_ai_usage_events_output_tokens",
        ),
        CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0",
            name="ck_ai_usage_events_total_tokens",
        ),
        CheckConstraint(
            "cached_input_tokens IS NULL OR cached_input_tokens >= 0",
            name="ck_ai_usage_events_cached_input_tokens",
        ),
        CheckConstraint(
            "reasoning_tokens IS NULL OR reasoning_tokens >= 0",
            name="ck_ai_usage_events_reasoning_tokens",
        ),
        CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name="ck_ai_usage_events_duration_ms",
        ),
        Index(
            "ix_ai_usage_events_user_created_at",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_ai_usage_events_thread_created_at",
            "thread_id",
            "created_at",
        ),
        Index(
            "ix_ai_usage_events_provider_model_created_at",
            "provider",
            "api_model",
            "created_at",
        ),
        Index(
            "ix_ai_usage_events_operation_created_at",
            "operation",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    # Nullable so system/background usage can be retained without a user owner.
    # SET NULL preserves aggregate records if an account is deleted.
    user_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # References the unique LangGraph thread ID, not its internal checkpoint ID.
    thread_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("agent_workflows.thread_id", ondelete="SET NULL"),
        nullable=True,
    )

    provider: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    # Public allowlisted ForgePath model ID, e.g. nvidia.nemotron-3-ultra.
    model_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    # Actual provider model name returned by the adapter when available.
    api_model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # Logical operation, e.g. code_review, code_patch, agent_planning.
    operation: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    request_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="chat_completion",
        server_default=text("'chat_completion'"),
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    # Provider-reported values remain NULL when the provider omits them.
    input_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    output_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    total_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    cached_input_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    reasoning_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # True means the adapter derived total_tokens from provider-reported
    # input_tokens + output_tokens; it does not mean tokenization was estimated.
    total_tokens_calculated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    provider_request_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    finish_reason: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    # Store only the exception class, never raw exception text or request data.
    error_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )