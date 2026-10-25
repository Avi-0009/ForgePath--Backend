
import hashlib
import hmac
import logging
import re
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import hash_password
from app.models.auth_session import AuthSession
from app.models.password_reset_otp import PasswordResetOTP
from app.repositories import user as user_repository
from app.repositories import password_reset_otp as otp_repository
from app.services.brevo_email import (
    BrevoEmailError,
    send_password_reset_otp,
)


logger = logging.getLogger(__name__)

OTP_LENGTH = 6
OTP_RESEND_COOLDOWN_SECONDS = 60


def _generate_otp() -> str:
    """Generate a cryptographically secure six-digit OTP."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _hash_otp(otp: str) -> str:
    """Create an HMAC-SHA256 hash of the OTP."""
    secret = settings.password_reset_secret_key.get_secret_value()

    return hmac.new(
        secret.encode("utf-8"),
        otp.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


async def request_password_reset_otp(
    db: AsyncSession,
    email: str,
) -> bool:
    """
    Generate and send a password-reset OTP.

    Returns True when an email was successfully requested from Brevo.
    Returns False when the account is unavailable or the request is
    rate-limited. The API must return the same public response either way.
    """
    normalized_email = email.strip().lower()

    user = await user_repository.get_by_email(db, normalized_email)

    if user is None or not user.is_active:
        await db.rollback()
        return False

    now = datetime.now(timezone.utc)

    try:
        record = await otp_repository.get_by_user_id(
            db,
            user.id,
            for_update=True,
        )

        # Only allow one new OTP request per user every 60 seconds.
        if record is not None:
            last_request = record.created_at

            if (
                last_request is not None
                and now - last_request
                < timedelta(seconds=OTP_RESEND_COOLDOWN_SECONDS)
            ):
                await db.rollback()
                return False

        otp = _generate_otp()
        otp_hash = _hash_otp(otp)
        expires_at = now + timedelta(
            minutes=settings.password_reset_otp_expire_minutes
        )

        if record is None:
            record = await otp_repository.create_otp_record(
                db=db,
                user_id=user.id,
                otp_hash=otp_hash,
                expires_at=expires_at,
            )
        else:
            record.otp_hash = otp_hash
            record.expires_at = expires_at
            record.attempts = 0
            record.used_at = None

        # Track the most recent issuance for resend throttling.
        record.created_at = now

        await db.commit()

    except IntegrityError:
        await db.rollback()
        # Another concurrent request may have created the row first.
        return False

    except Exception:
        await db.rollback()
        raise

    # Send only after the OTP record has been committed.
    try:
        await send_password_reset_otp(
            recipient_email=normalized_email,
            otp=otp,
        )

    except BrevoEmailError:
        # Invalidate this OTP if the email could not be sent.
        try:
            current_record = await otp_repository.get_by_user_id(
                db,
                user.id,
                for_update=True,
            )

            if (
                current_record is not None
                and hmac.compare_digest(
                    current_record.otp_hash,
                    otp_hash,
                )
                and current_record.used_at is None
            ):
                current_record.used_at = datetime.now(timezone.utc)
                await db.commit()
            else:
                await db.rollback()

        except Exception:
            await db.rollback()
            logger.exception(
                "Could not invalidate an OTP after an email delivery failure"
            )

        raise

    return True


async def reset_password_with_otp(
    db: AsyncSession,
    email: str,
    otp: str,
    new_password: str,
) -> bool:
    """
    Validate an OTP and reset the password in one transaction.

    On success, consumes the OTP and revokes all existing login sessions.
    """
    normalized_email = email.strip().lower()

    if not re.fullmatch(r"[0-9]{6}", otp):
        return False

    user = await user_repository.get_by_email(db, normalized_email)

    if user is None or not user.is_active:
        await db.rollback()
        return False

    try:
        record = await otp_repository.get_by_user_id(
            db,
            user.id,
            for_update=True,
        )

        now = datetime.now(timezone.utc)

        if (
            record is None
            or record.used_at is not None
            or record.expires_at <= now
            or record.attempts >= settings.password_reset_otp_max_attempts
        ):
            await db.rollback()
            return False

        expected_hash = _hash_otp(otp)

        if not hmac.compare_digest(record.otp_hash, expected_hash):
            record.attempts += 1

            if (
                record.attempts
                >= settings.password_reset_otp_max_attempts
            ):
                record.used_at = now

            await db.commit()
            return False

        # Update the password and consume the OTP atomically.
        user.password_hash = hash_password(new_password)
        record.used_at = now

        # Invalidate all existing login sessions for this account.
        await db.execute(
            update(AuthSession)
            .where(
                AuthSession.user_id == user.id,
                AuthSession.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )

        await db.commit()
        return True

    except Exception:
        await db.rollback()
        raise