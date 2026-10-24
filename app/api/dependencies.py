
from typing import Annotated
from uuid import UUID
from datetime import datetime, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.user import User
from app.repositories import user as user_repository
from app.repositories.auth_session import get_session_by_id


bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired access token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise unauthorized

    try:
        payload = decode_access_token(credentials.credentials)

        if payload.get("type") != "access":
            raise ValueError("Unexpected token type")

        subject = payload.get("sub")
        session_subject = payload.get("sid")

        if not isinstance(subject, str):
            raise ValueError("Invalid token subject")

        if not isinstance(session_subject, str):
            raise ValueError("Missing authentication session")

        user_id = UUID(subject)
        session_id = UUID(session_subject)

    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise unauthorized from None

    user = await user_repository.get_by_id(db, user_id)

    if user is None or not user.is_active:
        raise unauthorized

    auth_session = await get_session_by_id(db, session_id)

    if (
        auth_session is None
        or auth_session.user_id != user.id
        or auth_session.revoked_at is not None
        or auth_session.expires_at <= datetime.now(timezone.utc)
    ):
        raise unauthorized

    return user