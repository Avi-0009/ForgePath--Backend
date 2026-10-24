from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import jwt
import hashlib
import secrets
from pwdlib import PasswordHash

from app.core.config import settings


password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Hash a plaintext password using the recommended password hasher."""
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Check a supplied password against its stored hash."""
    return password_hasher.verify(password, password_hash)


def create_access_token(
    user_id: UUID,
    session_id: UUID,
) -> str:
    """Create a short-lived access token tied to a login session."""
    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(user_id),
        "sid": str(session_id),
        "iat": now,
        "exp": now + timedelta(
            minutes=settings.access_token_expire_minutes
        ),
        "type": "access",
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """Validate a token's signature, algorithm, and expiration."""
    return jwt.decode(
        token,
        settings.jwt_secret_key.get_secret_value(),
        algorithms=[settings.jwt_algorithm],
    )

def create_refresh_token() -> str:
    """Generate a cryptographically secure refresh token."""
    return secrets.token_urlsafe(64)

def hash_refresh_token(refresh_token: str) -> str :
    """Hash a refresh token before storing it in the database."""
    return hashlib.sha256(
        refresh_token.encode("utf-8")
    ).hexdigest()