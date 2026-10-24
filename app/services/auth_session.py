from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import (
    create_refresh_token,
    hash_refresh_token,
)
from app.models.auth_session import AuthSession
from app.repositories.auth_session import (
    create_session,
    get_session_by_refresh_hash,
)


@dataclass(frozen=True, slots=True)
class CreatedAuthSession:
    """A persisted login session and its one-time raw refresh token."""

    session: AuthSession
    refresh_token: str


async def create_auth_session(
    db: AsyncSession,
    user_id: UUID,
) -> CreatedAuthSession:
    """Create a session and persist only the refresh-token hash."""

    refresh_token = create_refresh_token()
    refresh_token_hash = hash_refresh_token(refresh_token)

    expires_at = datetime.now(timezone.utc) + timedelta(
        days=settings.refresh_token_expire_days
    )

    try:
        auth_session = await create_session(
            db=db,
            user_id=user_id,
            refresh_token_hash=refresh_token_hash,
            expires_at=expires_at,
        )

        await db.commit()
        await db.refresh(auth_session)

    except Exception:
        await db.rollback()
        raise

    return CreatedAuthSession(
        session=auth_session,
        refresh_token=refresh_token,
    )


@dataclass(frozen=True, slots=True)
class RotatedAuthSession:
    session: AuthSession
    refresh_token: str


async def rotate_auth_session(
    db: AsyncSession,
    refresh_token: str,
) -> RotatedAuthSession | None:
    """Validate and rotate a refresh token exactly once."""

    token_hash = hash_refresh_token(refresh_token)
    now = datetime.now(timezone.utc)

    try:
        # Lock the matching session row to prevent concurrent reuse.
        auth_session = await get_session_by_refresh_hash(
            db,
            token_hash,
            for_update=True,
        )

        if (
            auth_session is None
            or auth_session.revoked_at is not None
            or auth_session.expires_at <= now
        ):
            await db.rollback()
            return None

        new_refresh_token = create_refresh_token()
        auth_session.refresh_token_hash = hash_refresh_token(
            new_refresh_token
        )
        auth_session.expires_at = now + timedelta(
            days=settings.refresh_token_expire_days
        )
        auth_session.last_used_at = now

        await db.commit()
        await db.refresh(auth_session)

        return RotatedAuthSession(
            session=auth_session,
            refresh_token=new_refresh_token,
        )

    except Exception:
        await db.rollback()
        raise



async def revoke_auth_session(
    db: AsyncSession,
    refresh_token: str | None,
) -> bool:
    """Revoke the session associated with a refresh token."""

    if not refresh_token:
        return False

    token_hash = hash_refresh_token(refresh_token)

    try:
        # Lock the session row while revoking it.
        auth_session = await get_session_by_refresh_hash(
            db,
            token_hash,
            for_update=True,
        )

        if (
            auth_session is None
            or auth_session.revoked_at is not None
        ):
            await db.rollback()
            return False

        auth_session.revoked_at = datetime.now(timezone.utc)

        await db.commit()
        return True

    except Exception:
        await db.rollback()
        raise