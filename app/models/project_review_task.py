"""User-owned, manually selected tasks from review findings."""

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
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ProjectReviewTask(Base):
    __tablename__ = "project_review_tasks"

    __table_args__ = (
        UniqueConstraint(
            "review_id",
            "finding_index",
            name="uq_project_review_tasks_finding",
        ),
        CheckConstraint(
            "finding_index >= 0",
            name="ck_project_review_tasks_finding_index",
        ),
        CheckConstraint(
            "status IN ('proposed', 'approved', 'dismissed')",
            name="ck_project_review_tasks_status",
        ),
        CheckConstraint(
            "planning_status IN ('not_started', 'planning', 'ready', 'failed')",
            name="ck_project_review_tasks_planning_status",
        ),
        CheckConstraint(
            "patch_status IN ('not_started', 'generating', 'ready', 'approved', 'rejected', 'failed')",
            name="ck_project_review_tasks_patch_status",
        ),
        CheckConstraint(
            "severity IN ('critical', 'high', 'medium', 'low', 'info')",
            name="ck_project_review_tasks_severity",
        ),
        Index(
            "ix_project_review_tasks_owner_project_created",
            "user_id",
            "project_id",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    project_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )

    review_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("project_reviews.id", ondelete="CASCADE"),
        nullable=False,
    )

    snapshot_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
    )

    finding_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
    )

    path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(12),
        nullable=False,
    )

    evidence: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    explanation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="proposed",
        server_default=text("'proposed'"),
    )

    planning_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="not_started",
        server_default=text("'not_started'"),
    )

    planning_token: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    planning_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    plan: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    planning_error: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
    )

    planned_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    patch_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="not_started",
        server_default=text("'not_started'"),
    )

    patch_token: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    patch_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    patch: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    patch_error: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
    )

    patched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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
