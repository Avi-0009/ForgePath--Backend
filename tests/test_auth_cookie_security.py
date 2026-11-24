"""Verify refresh cookie transport flags in login/refresh/logout without paid calls."""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import Response

from app.api.v1.routes import auth
from app.schemas.auth import LoginRequest


def _user():
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=uuid4(), full_name="Test User", username="test_user",
        email="test@example.com", is_active=True, is_email_verified=True,
        created_at=now, updated_at=now,
    )


def _assert_cookie(response: Response, secure: bool):
    cookie = response.headers["set-cookie"].lower()
    assert "forgepath_refresh=" in cookie
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "path=/api/v1/auth" in cookie
    assert ("; secure" in cookie) is secure


@pytest.mark.parametrize("secure", [True, False])
def test_login_cookie_follows_deployment_policy(monkeypatch, secure):
    user = _user()
    monkeypatch.setattr(auth.settings, "refresh_cookie_secure", secure)
    monkeypatch.setattr(auth, "authenticate_user", AsyncMock(return_value=user))
    monkeypatch.setattr(auth, "create_auth_session", AsyncMock(return_value=SimpleNamespace(
        session=SimpleNamespace(id=uuid4()), refresh_token="refresh-token",
    )))
    monkeypatch.setattr(auth, "create_access_token", lambda **_: "access-token")
    response = Response()
    result = asyncio.run(auth.login(LoginRequest(identifier="test_user", password="Password123"), response, SimpleNamespace()))
    assert result.access_token == "access-token"
    _assert_cookie(response, secure)


@pytest.mark.parametrize("secure", [True, False])
def test_refresh_cookie_follows_deployment_policy(monkeypatch, secure):
    user = _user()
    monkeypatch.setattr(auth.settings, "refresh_cookie_secure", secure)
    monkeypatch.setattr(auth, "rotate_auth_session", AsyncMock(return_value=SimpleNamespace(
        session=SimpleNamespace(id=uuid4(), user_id=user.id), refresh_token="next-token",
    )))
    monkeypatch.setattr(auth.user_repository, "get_by_id", AsyncMock(return_value=user))
    monkeypatch.setattr(auth, "create_access_token", lambda **_: "access-token")
    response = Response()
    result = asyncio.run(auth.refresh_access_token(response, SimpleNamespace(), "old-token"))
    assert result.access_token == "access-token"
    _assert_cookie(response, secure)


@pytest.mark.parametrize("secure", [True, False])
def test_logout_deletes_cookie_using_same_policy(monkeypatch, secure):
    monkeypatch.setattr(auth.settings, "refresh_cookie_secure", secure)
    monkeypatch.setattr(auth, "revoke_auth_session", AsyncMock(return_value=None))
    response = Response()
    asyncio.run(auth.logout(response, SimpleNamespace(), "old-token"))
    cookie = response.headers["set-cookie"].lower()
    assert "max-age=0" in cookie
    _assert_cookie(response, secure)
