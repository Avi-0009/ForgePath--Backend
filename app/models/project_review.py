"""Durable metadata and results for user-owned snapshot reviews."""
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, Index, func, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from sqlalchemy.dialects.postgresql import JSONB


class ProjectReview(Base):
    __tablename__ = "project_reviews"
    __table_args__ = (Index("uq_project_reviews_active_user", "user_id", unique=True,
                           postgresql_where=text("status IN ('queued','running')")),)

    lease_token: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4, 
        server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("users.id", 
        ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    project_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("projects.id", 
        ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )
    snapshot_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(20), 
        nullable=False, 
        server_default=text("'queued'")
    )
    files_scanned: Mapped[int] = mapped_column(
        Integer, 
        nullable=False, 
        server_default=text("0")
    )
    files_reviewed: Mapped[int] = mapped_column(
        Integer, 
        nullable=False, 
        server_default=text("0")
    )
    ai_calls: Mapped[int] = mapped_column(
        Integer, 
        nullable=False, 
        server_default=text("0")
    )
    result_text: Mapped[str | None] = mapped_column(
        Text, 
        nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(
        String(300), 
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now(), 
        onupdate=func.now()
    )
    findings: Mapped[list[dict]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
