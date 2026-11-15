"""Encrypted per-user credentials. Plaintext is never stored or serialized."""
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.user_ai_context import UserAIChoice
from app.models.user_ai_configuration import UserAIConfiguration


def credential_cipher() -> Fernet:
    secret = settings.ai_credentials_encryption_key
    if secret is None:
        raise RuntimeError("AI_CREDENTIALS_ENCRYPTION_KEY is not configured")
    try:
        return Fernet(secret.get_secret_value().encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("AI_CREDENTIALS_ENCRYPTION_KEY must be a Fernet key") from exc


async def load_user_ai_choice(db: AsyncSession, user_id) -> UserAIChoice | None:
    record = await db.get(UserAIConfiguration, user_id)
    if record is None:
        return None
    try:
        api_key = credential_cipher().decrypt(record.encrypted_api_key.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError) as exc:
        raise RuntimeError("Stored AI credentials could not be decrypted") from exc
    return UserAIChoice(record.provider, record.model_identifier, api_key)