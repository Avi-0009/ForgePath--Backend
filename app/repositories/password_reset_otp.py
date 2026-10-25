
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.password_reset_otp import PasswordResetOTP


async def get_by_user_id(
    db: AsyncSession,
    user_id: UUID,
    *,
    for_update: bool = False,
) -> PasswordResetOTP | None:
    """Find a user's password-reset OTP record."""

    statement = select(PasswordResetOTP).where(
        PasswordResetOTP.user_id == user_id
    )

    if for_update:
        statement = statement.with_for_update()

    result = await db.execute(statement)
    return result.scalar_one_or_none()


async def create_otp_record(
    db: AsyncSession,
    user_id: UUID,
    otp_hash: str,
    expires_at: datetime,
) -> PasswordResetOTP:
    """Create a password-reset OTP record."""

    record = PasswordResetOTP(
        user_id=user_id,
        otp_hash=otp_hash,
        expires_at=expires_at,
        attempts=0,
    )

    db.add(record)
    await db.flush()

    return record