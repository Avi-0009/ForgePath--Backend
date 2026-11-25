"""Personal AI settings stay user-scoped and never disclose credentials."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.v1.routes import ai_configuration as routes
from app.core.ai_provider import AIProviderError
from app.core.user_ai_context import UserAIChoice


class FakeDatabase:
    def __init__(self, record=None):
        self.record = record
        self.added = []
        self.commits = 0
        self.deleted = []

    async def get(self, _model, _user_id):
        return self.record

    def add(self, record):
        self.record = record
        self.added.append(record)

    async def commit(self):
        self.commits += 1

    async def delete(self, record):
        self.deleted.append(record)


def test_read_configuration_returns_metadata_without_ciphertext():
    user = SimpleNamespace(id=uuid4())
    db = FakeDatabase(SimpleNamespace(
        provider="openai",
        model_identifier="gpt-6-luna",
        encrypted_api_key="private-ciphertext",
    ))

    response = asyncio.run(routes.read_configuration(user, db))
    data = response.model_dump()

    assert data == {
        "provider": "openai",
        "model_identifier": "gpt-6-luna",
        "has_api_key": True,
    }
    assert "private-ciphertext" not in repr(response)


def test_configuration_update_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        routes.ConfigurationUpdate.model_validate({
            "provider": "openai",
            "model_identifier": "gpt-6-luna",
            "api_key": "secret-value",
            "return_api_key": True,
        })


@pytest.mark.parametrize(
    "api_key",
    ["line\nbreak", "tab\tkey", "space key", "non-ascii-kéy"],
)
def test_save_rejects_non_header_safe_api_keys(api_key):
    user = SimpleNamespace(id=uuid4())
    db = FakeDatabase()

    with pytest.raises(HTTPException) as exception:
        asyncio.run(routes.save_configuration(
            routes.ConfigurationUpdate(
                provider="openai",
                model_identifier="gpt-6-luna",
                api_key=api_key,
            ),
            user,
            db,
        ))

    assert exception.value.status_code == 422
    assert db.commits == 0


def test_changing_provider_requires_a_new_key():
    user = SimpleNamespace(id=uuid4())
    db = FakeDatabase(SimpleNamespace(
        provider="nvidia",
        model_identifier="nvidia/model",
        encrypted_api_key="encrypted",
    ))

    with pytest.raises(HTTPException) as exception:
        asyncio.run(routes.save_configuration(
            routes.ConfigurationUpdate(
                provider="openai",
                model_identifier="gpt-6-luna",
            ),
            user,
            db,
        ))

    assert exception.value.status_code == 422
    assert db.commits == 0


def test_connection_failure_does_not_disclose_provider_error(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    db = FakeDatabase()

    async def load_choice(_db, _user_id):
        return UserAIChoice("openai", "gpt-6-luna", "private-api-key")

    async def fail_completion(**_kwargs):
        raise AIProviderError("private upstream detail")

    monkeypatch.setattr(routes, "load_user_ai_choice", load_choice)
    monkeypatch.setattr(routes, "generate_completion", fail_completion)

    with pytest.raises(HTTPException) as exception:
        asyncio.run(routes.test_configuration(user, db))

    assert exception.value.status_code == 422
    assert "private upstream detail" not in exception.value.detail
