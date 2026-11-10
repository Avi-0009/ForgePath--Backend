from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AgentWorkflowHistory(Base):
    __tablename__ = "agent_workflows"

    __table_args__ = (
        UniqueConstraint(
            "thread_id",
            name="uq_agent_workflows_thread_id",
        ),
        CheckConstraint(
            "completed_steps >= 0",
            name="ck_agent_workflows_completed_steps",
        ),
        CheckConstraint(
            "total_steps >= 0",
            name="ck_agent_workflows_total_steps",
        ),
        CheckConstraint(
            "repair_attempts >= 0",
            name="ck_agent_workflows_repair_attempts",
        ),
        CheckConstraint(
            "total_input_tokens >= 0",
            name="ck_agent_workflows_total_input_tokens",
        ),
        CheckConstraint(
            "total_output_tokens >= 0",
            name="ck_agent_workflows_total_output_tokens",
        ),
        CheckConstraint(
            "total_tokens >= 0",
            name="ck_agent_workflows_total_tokens",
        ),
        CheckConstraint(
            "total_cached_input_tokens >= 0",
            name="ck_agent_workflows_total_cached_input_tokens",
        ),
        CheckConstraint(
            "total_reasoning_tokens >= 0",
            name="ck_agent_workflows_total_reasoning_tokens",
        ),
        CheckConstraint(
            "ai_request_count >= 0 AND failed_ai_request_count >= 0",
            name="ck_agent_workflows_ai_request_counts",
        ),
        CheckConstraint(
            "total_ai_duration_ms >= 0",
            name="ck_agent_workflows_total_ai_duration_ms",
        ),
        CheckConstraint(
            "test_run_count >= 0 AND test_pass_count >= 0 "
            "AND test_fail_count >= 0 AND test_timeout_count >= 0 "
            "AND test_error_count >= 0",
            name="ck_agent_workflows_test_run_counts",
        ),
        CheckConstraint(
            "total_test_duration_ms >= 0",
            name="ck_agent_workflows_total_test_duration_ms",
        ),
        Index(
            "ix_agent_workflows_user_created_at",
            "user_id",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    thread_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
    )

    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    workspace_root: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )

    task_description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    planning_mode: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    test_runner: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="running",
        server_default=text("'running'"),
    )

    message: Mapped[str | None] = mapped_column(
        String(2000),
        nullable=True,
    )

    total_steps: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    completed_steps: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    repair_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    # Aggregate AI-usage metrics. Individual events remain the source of truth.
    total_input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    total_output_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    total_cached_input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    total_reasoning_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    ai_request_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    failed_ai_request_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    total_ai_duration_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    # Aggregate local test execution metrics.
    test_run_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    test_pass_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    test_fail_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    test_timeout_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    test_error_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    total_test_duration_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )