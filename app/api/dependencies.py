from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.user import User
from app.repositories import user as user_repository


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

        if not isinstance(subject, str):
            raise ValueError("Invalid token subject")

        user_id = UUID(subject)

    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise unauthorized from None

    user = await user_repository.get_by_id(db, user_id)

    if user is None or not user.is_active:
        raise unauthorized

    return user