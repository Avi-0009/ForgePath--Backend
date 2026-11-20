import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.v1.routes.auth import signup
from app.schemas.auth import SignupRequest
from app.services import auth as auth_service


def make_payload():
    return SignupRequest(
        email="example@email.com",
        username="example_user",
        password="CorrectHorseBatteryStaple",
        full_name="Example User",
    )


def test_duplicate_email_is_checked_first(monkeypatch):
    calls = []

    async def fake_email_lookup(db, email):
        calls.append("email")
        return SimpleNamespace(id="existing")

    async def fake_username_lookup(db, username):
        calls.append("username")
        return SimpleNamespace(id="existing")

    monkeypatch.setattr(
        auth_service.user_repository,
        "get_by_email",
        fake_email_lookup,
    )

    monkeypatch.setattr(
        auth_service.user_repository,
        "get_by_username",
        fake_username_lookup,
    )

    with pytest.raises(
        auth_service.AccountAlreadyExistsError
    ) as exc:
        asyncio.run(
            auth_service.register_user(
                SimpleNamespace(),
                make_payload(),
            )
        )

    assert exc.value.field == "email"
    assert calls == ["email"]


def test_duplicate_username_is_detected(monkeypatch):
    async def fake_email_lookup(db, email):
        return None

    async def fake_username_lookup(db, username):
        return SimpleNamespace(id="existing")

    monkeypatch.setattr(
        auth_service.user_repository,
        "get_by_email",
        fake_email_lookup,
    )

    monkeypatch.setattr(
        auth_service.user_repository,
        "get_by_username",
        fake_username_lookup,
    )

    with pytest.raises(
        auth_service.AccountAlreadyExistsError
    ) as exc:
        asyncio.run(
            auth_service.register_user(
                SimpleNamespace(),
                make_payload(),
            )
        )

    assert exc.value.field == "username"


@pytest.mark.parametrize(
    "field, expected",
    [
        ("email", "This email address is already in use."),
        ("username", "This username is already taken."),
    ],
)
def test_signup_returns_readable_conflict(
    monkeypatch,
    field,
    expected,
):
    from app.api.v1.routes import auth as auth_routes

    async def fake_register(db, payload):
        raise auth_service.AccountAlreadyExistsError(field)

    monkeypatch.setattr(
        auth_routes,
        "register_user",
        fake_register,
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            signup(
                make_payload(),
                SimpleNamespace(),
            )
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == expected