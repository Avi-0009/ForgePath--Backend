from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from datetime import datetime
from app.models.auth_session import AuthSession
from sqlalchemy import select

async def create_session(
    db: AsyncSession,
    user_id: UUID,
    refresh_token_hash: str,
    expires_at: datetime,
) -> AuthSession:
    """Add a new authentication session to the database."""
    auth_session = AuthSession(
        user_id=user_id,
        refresh_token_hash=refresh_token_hash,
        expires_at=expires_at,
    )

    db.add(auth_session)
    await db.flush()

    return auth_session


async def get_session_by_id(
    db: AsyncSession,
    session_id: UUID,
) -> AuthSession | None:
    """Retrieve an authentication session by its ID."""
    result = await db.execute(
        select(AuthSession).where(AuthSession.id == session_id)
    )

    return result.scalar_one_or_none()


async def get_session_by_refresh_hash(
    db: AsyncSession,
    refresh_token_hash: str,
    *,
    for_update: bool = False,
) -> AuthSession | None:
    statement = select(AuthSession).where(
        AuthSession.refresh_token_hash == refresh_token_hash
    )

    if for_update:
        statement = statement.with_for_update()

    result = await db.execute(statement)
    return result.scalar_one_or_none()