from app.db.base import Base
from sqlalchemy import (
    UniqueConstraint,
    CheckConstraint,
    ForeignKey,
    String,
    DateTime,
    Integer,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column
from uuid import UUID, uuid4
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from datetime import datetime



class PasswordResetOTP(Base):
    __tablename__ = "password_reset_otps"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            name="uq_password_reset_otps_user_id",
        ),
        CheckConstraint(
            "attempts >= 0",
            name="ck_password_reset_otps_attempts_nonnegative",
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

    otp_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    attempts: Mapped[int] = mapped_column(
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

    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )