"""Authenticated personal AI configuration: never disclose or log credentials."""
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.core.ai_provider import AIProviderError, generate_completion
from app.core.user_ai_context import user_ai_choice
from app.db.session import get_db
from app.models.user import User
from app.models.user_ai_configuration import UserAIConfiguration
from app.services.user_ai_configuration import credential_cipher, load_user_ai_choice

router = APIRouter(prefix="/ai/configuration", tags=["AI Configuration"])
Provider = Literal["nvidia", "openai", "huggingface"]


class ConfigurationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model_identifier: str = Field(min_length=1, max_length=200)
    api_key: SecretStr | None = None

    @field_validator("model_identifier")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ch.isspace() or ord(ch) < 33 for ch in value):
            raise ValueError("Model identifier must be a nonempty identifier without whitespace")
        return value


class ConfigurationResponse(BaseModel):
    provider: Provider | None
    model_identifier: str | None
    has_api_key: bool


@router.get("", response_model=ConfigurationResponse)
async def read_configuration(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ConfigurationResponse:
    record = await db.get(UserAIConfiguration, user.id)
    if record is None:
        return ConfigurationResponse(provider=None, model_identifier=None, has_api_key=False)
    return ConfigurationResponse(provider=record.provider, model_identifier=record.model_identifier, has_api_key=True)


@router.put("", response_model=ConfigurationResponse)
async def save_configuration(
    payload: ConfigurationUpdate,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ConfigurationResponse:
    record = await db.get(UserAIConfiguration, user.id)
    key = payload.api_key.get_secret_value() if payload.api_key is not None else None
    if key is not None:
        key = key.strip()
        if (
            not key
            or len(key) > 4096
            or any(
                ord(character) < 33 or ord(character) > 126
                for character in key
            )
        ):
            raise HTTPException(
                422,
                "API key must contain 1–4096 printable ASCII characters",
            )
    if record is None and key is None:
        raise HTTPException(422, "API key is required for a new configuration")
    if record is not None and record.provider != payload.provider and key is None:
        raise HTTPException(422, "A new API key is required when changing providers")
    if key is not None:
        try:
            ciphertext = credential_cipher().encrypt(key.encode("utf-8")).decode("ascii")
        except RuntimeError:
            raise HTTPException(503, "Credential encryption is not configured") from None
    else:
        ciphertext = record.encrypted_api_key
    if record is None:
        record = UserAIConfiguration(user_id=user.id, provider=payload.provider,
                                     model_identifier=payload.model_identifier, encrypted_api_key=ciphertext)
        db.add(record)
    else:
        record.provider = payload.provider
        record.model_identifier = payload.model_identifier
        record.encrypted_api_key = ciphertext
    await db.commit()
    return ConfigurationResponse(provider=record.provider, model_identifier=record.model_identifier, has_api_key=True)


@router.delete("", status_code=204)
async def remove_configuration(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    record = await db.get(UserAIConfiguration, user.id)
    if record is not None:
        await db.delete(record)
        await db.commit()


@router.post("/test")
async def test_configuration(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, bool]:
    try:
        choice = await load_user_ai_choice(db, user.id)
    except RuntimeError:
        raise HTTPException(503, "Personal AI credentials are unavailable") from None
    if choice is None:
        raise HTTPException(404, "Save a personal AI configuration first")
    try:
        with user_ai_choice(choice):
            await generate_completion(instructions="Reply briefly.", prompt="Reply with OK.", max_output_tokens=128)
    except (AIProviderError, ValueError):
        raise HTTPException(422, "Provider connection failed. Check your key, model ID and account access") from None
    return {"connected": True}
