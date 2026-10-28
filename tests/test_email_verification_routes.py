
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.main import app
from app.services import email_verification as verification_service
from app.services import brevo_email


REQUEST_URL = "/api/v1/auth/email-verification/request"
VERIFY_URL = "/api/v1/auth/email-verification/verify"


@pytest.fixture
def route_client(monkeypatch):
    user = SimpleNamespace(
        id=uuid4(),
        full_name="Test User",
        username="test_user",
        email="test@example.com",
        is_active=True,
        is_email_verified=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    db = SimpleNamespace(refresh=AsyncMock())

    async def override_db():
        yield db

    monkeypatch.setitem(
        app.dependency_overrides,
        get_db,
        override_db,
    )

    with TestClient(app) as client:
        yield client, db, user


def authenticate_user(monkeypatch, user):
    async def override_current_user():
        return user

    monkeypatch.setitem(
        app.dependency_overrides,
        get_current_user,
        override_current_user,
    )


def test_request_requires_authentication(route_client):
    client, _, _ = route_client

    response = client.post(REQUEST_URL)

    assert response.status_code == 401


def test_request_sends_otp_for_authenticated_user(
    route_client,
    monkeypatch,
):
    client, db, user = route_client
    authenticate_user(monkeypatch, user)

    send_otp = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "app.api.v1.routes.auth.request_email_verification_otp",
        send_otp,
    )

    response = client.post(
        REQUEST_URL,
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "Verification code sent to your email address."
    }

    send_otp.assert_awaited_once_with(
        db=db,
        user_id=user.id,
    )


def test_request_maps_resend_cooldown_to_429(
    route_client,
    monkeypatch,
):
    client, _, user = route_client
    authenticate_user(monkeypatch, user)

    send_otp = AsyncMock(
        side_effect=verification_service.EmailVerificationRateLimitedError(
            "Please wait before requesting another verification code."
        )
    )
    monkeypatch.setattr(
        "app.api.v1.routes.auth.request_email_verification_otp",
        send_otp,
    )

    response = client.post(
        REQUEST_URL,
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"


def test_verify_requires_authentication(route_client):
    client, _, _ = route_client

    response = client.post(
        VERIFY_URL,
        json={"otp": "123456"},
    )

    assert response.status_code == 401


def test_verify_returns_updated_user(
    route_client,
    monkeypatch,
):
    client, db, user = route_client
    authenticate_user(monkeypatch, user)

    async def fake_verify(*, db, user_id, otp):
        assert user_id == user.id
        assert otp == "123456"
        user.is_email_verified = True
        return True

    verify_otp = AsyncMock(side_effect=fake_verify)
    monkeypatch.setattr(
        "app.api.v1.routes.auth.verify_email_with_otp",
        verify_otp,
    )

    response = client.post(
        VERIFY_URL,
        headers={"Authorization": "Bearer test-token"},
        json={"otp": "123456"},
    )

    assert response.status_code == 200
    assert response.json()["is_email_verified"] is True
    assert response.json()["email"] == user.email

    verify_otp.assert_awaited_once_with(
        db=db,
        user_id=user.id,
        otp="123456",
    )
    db.refresh.assert_awaited_once_with(user)


def test_verify_returns_400_for_invalid_otp(
    route_client,
    monkeypatch,
):
    client, _, user = route_client
    authenticate_user(monkeypatch, user)

    verify_otp = AsyncMock(return_value=False)
    monkeypatch.setattr(
        "app.api.v1.routes.auth.verify_email_with_otp",
        verify_otp,
    )

    response = client.post(
        VERIFY_URL,
        headers={"Authorization": "Bearer test-token"},
        json={"otp": "000000"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid or expired verification code."
    )


def test_verify_rejects_malformed_otp(
    route_client,
    monkeypatch,
):
    client, _, user = route_client
    authenticate_user(monkeypatch, user)

    verify_otp = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "app.api.v1.routes.auth.verify_email_with_otp",
        verify_otp,
    )

    response = client.post(
        VERIFY_URL,
        headers={"Authorization": "Bearer test-token"},
        json={"otp": "12ab"},
    )

    assert response.status_code == 422
    verify_otp.assert_not_awaited()