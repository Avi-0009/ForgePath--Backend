"""Authenticated profile updates preserve validation and uniqueness guarantees."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from app.api.v1.routes import auth as auth_routes
from app.schemas.auth import ProfileUpdateRequest
from app.services import auth as auth_service


class Database:
    def __init__(self, commit_error=None):
        self.commit_error = commit_error
        self.commits = 0
        self.refreshes = 0
        self.rollbacks = 0

    async def commit(self):
        self.commits += 1
        if self.commit_error is not None:
            raise self.commit_error

    async def refresh(self, _value):
        self.refreshes += 1

    async def rollback(self):
        self.rollbacks += 1


def user():
    return SimpleNamespace(
        id=uuid4(),
        full_name="Existing Name",
        username="existing_user",
    )


def test_profile_payload_normalizes_name_and_username():
    payload = ProfileUpdateRequest(
        full_name="  Updated Name  ",
        username="  UPDATED_USER  ",
    )

    assert payload.full_name == "Updated Name"
    assert payload.username == "updated_user"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"full_name": "   "},
        {"username": "invalid-name"},
        {"username": "ab"},
    ],
)
def test_profile_payload_rejects_empty_or_invalid_changes(payload):
    with pytest.raises(ValidationError):
        ProfileUpdateRequest(**payload)


def test_profile_update_persists_normalized_values(monkeypatch):
    current_user = user()
    db = Database()

    async def no_existing_user(_db, username):
        assert username == "updated_user"
        return None

    monkeypatch.setattr(
        auth_service.user_repository,
        "get_by_username",
        no_existing_user,
    )

    result = asyncio.run(auth_service.update_user_profile(
        db,
        current_user,
        ProfileUpdateRequest(
            full_name="Updated Name",
            username="updated_user",
        ),
    ))

    assert result is current_user
    assert result.full_name == "Updated Name"
    assert result.username == "updated_user"
    assert db.commits == 1
    assert db.refreshes == 1
    assert db.rollbacks == 0


def test_profile_update_rejects_existing_username_before_write(monkeypatch):
    current_user = user()
    db = Database()

    async def existing_user(_db, _username):
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(
        auth_service.user_repository,
        "get_by_username",
        existing_user,
    )

    with pytest.raises(auth_service.AccountAlreadyExistsError) as exception:
        asyncio.run(auth_service.update_user_profile(
            db,
            current_user,
            ProfileUpdateRequest(username="taken_user"),
        ))

    assert exception.value.field == "username"
    assert current_user.username == "existing_user"
    assert db.commits == 0


def test_profile_update_maps_concurrent_unique_conflict(monkeypatch):
    current_user = user()
    original_error = SimpleNamespace(
        diag=SimpleNamespace(constraint_name="uq_users_username")
    )
    db = Database(IntegrityError("UPDATE users", {}, original_error))

    async def no_existing_user(_db, _username):
        return None

    monkeypatch.setattr(
        auth_service.user_repository,
        "get_by_username",
        no_existing_user,
    )

    with pytest.raises(auth_service.AccountAlreadyExistsError) as exception:
        asyncio.run(auth_service.update_user_profile(
            db,
            current_user,
            ProfileUpdateRequest(username="racing_user"),
        ))

    assert exception.value.field == "username"
    assert db.rollbacks == 1


def test_profile_route_returns_readable_username_conflict(monkeypatch):
    async def conflict(*_args):
        raise auth_service.AccountAlreadyExistsError("username")

    monkeypatch.setattr(auth_routes, "update_user_profile", conflict)

    with pytest.raises(HTTPException) as exception:
        asyncio.run(auth_routes.update_me(
            ProfileUpdateRequest(username="taken_user"),
            user(),
            Database(),
        ))

    assert exception.value.status_code == 409
    assert exception.value.detail == "This username is already taken."


def test_profile_route_updates_only_authenticated_user(monkeypatch):
    current_user = user()

    async def update(db, passed_user, payload):
        assert passed_user is current_user
        assert payload.full_name == "Updated Name"
        passed_user.full_name = payload.full_name
        return passed_user

    monkeypatch.setattr(auth_routes, "update_user_profile", update)

    result = asyncio.run(auth_routes.update_me(
        ProfileUpdateRequest(full_name="Updated Name"),
        current_user,
        Database(),
    ))

    assert result is current_user
    assert result.full_name == "Updated Name"
