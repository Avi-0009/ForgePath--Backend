
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import SecretStr

from app.core.config import settings
from app.services import email_verification as verification
from app.services.brevo_email import BrevoEmailError


class FakeSession:
    def __init__(self):
        self.added = []
        self.commits = 0
        self.rollbacks = 0

    def add(self, record):
        self.added.append(record)

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def verification_setup(monkeypatch):
    db = FakeSession()

    user = SimpleNamespace(
        id=uuid4(),
        email="test@example.com",
        is_active=True,
        is_email_verified=False,
    )

    async def get_user(_db, user_id):
        return user if user_id == user.id else None

    monkeypatch.setattr(
        verification.user_repository,
        "get_by_id",
        get_user,
    )
    monkeypatch.setattr(
        settings,
        "password_reset_secret_key",
        SecretStr("test-only-secret-for-email-verification"),
    )

    return db, user


def make_record(
    otp="123456",
    *,
    expires_at=None,
    attempts=0,
    used_at=None,
    created_at=None,
):
    now = datetime.now(timezone.utc)

    return SimpleNamespace(
        user_id=uuid4(),
        otp_hash=verification._hash_otp(otp),
        expires_at=expires_at or now + timedelta(minutes=5),
        attempts=attempts,
        used_at=used_at,
        created_at=created_at or now - timedelta(minutes=2),
    )


def test_generates_and_sends_six_digit_otp(
    verification_setup,
    monkeypatch,
):
    db, user = verification_setup
    sent = {}

    async def no_existing_record(*args, **kwargs):
        return None

    async def fake_send_email(recipient_email, otp):
        sent["email"] = recipient_email
        sent["otp"] = otp

    monkeypatch.setattr(
        verification,
        "_get_otp_record",
        no_existing_record,
    )
    monkeypatch.setattr(
        verification,
        "send_email_verification_otp",
        fake_send_email,
    )

    asyncio.run(
        verification.request_email_verification_otp(
            db=db,
            user_id=user.id,
        )
    )

    assert sent["email"] == user.email
    assert len(sent["otp"]) == 6
    assert sent["otp"].isdigit()

    record = db.added[0]
    assert record.otp_hash == verification._hash_otp(sent["otp"])
    assert record.otp_hash != sent["otp"]
    assert db.commits == 1


def test_rejects_request_during_resend_cooldown(
    verification_setup,
    monkeypatch,
):
    db, user = verification_setup
    record = make_record(
        created_at=datetime.now(timezone.utc)
        - timedelta(seconds=10)
    )
    sent = []

    async def get_record(*args, **kwargs):
        return record

    async def fake_send_email(*args, **kwargs):
        sent.append(True)

    monkeypatch.setattr(
        verification,
        "_get_otp_record",
        get_record,
    )
    monkeypatch.setattr(
        verification,
        "send_email_verification_otp",
        fake_send_email,
    )

    with pytest.raises(
        verification.EmailVerificationRateLimitedError
    ):
        asyncio.run(
            verification.request_email_verification_otp(
                db=db,
                user_id=user.id,
            )
        )

    assert db.rollbacks == 1
    assert sent == []


def test_email_delivery_failure_invalidates_new_otp(
    verification_setup,
    monkeypatch,
):
    db, user = verification_setup
    record = make_record(
        created_at=datetime.now(timezone.utc)
        - timedelta(minutes=2)
    )

    async def get_record(*args, **kwargs):
        return record

    async def fail_sending_email(*args, **kwargs):
        raise BrevoEmailError("Simulated email failure")

    monkeypatch.setattr(
        verification,
        "_get_otp_record",
        get_record,
    )
    monkeypatch.setattr(
        verification,
        "send_email_verification_otp",
        fail_sending_email,
    )

    with pytest.raises(BrevoEmailError):
        asyncio.run(
            verification.request_email_verification_otp(
                db=db,
                user_id=user.id,
            )
        )

    assert record.used_at is not None
    assert db.commits == 2


def test_successful_verification_updates_user_and_consumes_otp(
    verification_setup,
    monkeypatch,
):
    db, user = verification_setup
    record = make_record("123456")

    async def get_record(*args, **kwargs):
        return record

    monkeypatch.setattr(
        verification,
        "_get_otp_record",
        get_record,
    )

    result = asyncio.run(
        verification.verify_email_with_otp(
            db=db,
            user_id=user.id,
            otp="123456",
        )
    )

    assert result is True
    assert user.is_email_verified is True
    assert record.used_at is not None
    assert db.commits == 1


def test_wrong_otp_increments_attempt_count(
    verification_setup,
    monkeypatch,
):
    db, user = verification_setup
    record = make_record("123456")

    async def get_record(*args, **kwargs):
        return record

    monkeypatch.setattr(
        verification,
        "_get_otp_record",
        get_record,
    )

    result = asyncio.run(
        verification.verify_email_with_otp(
            db=db,
            user_id=user.id,
            otp="654321",
        )
    )

    assert result is False
    assert record.attempts == 1
    assert user.is_email_verified is False
    assert db.commits == 1


def test_expired_otp_is_rejected_and_consumed(
    verification_setup,
    monkeypatch,
):
    db, user = verification_setup
    record = make_record(
        expires_at=datetime.now(timezone.utc)
        - timedelta(seconds=1)
    )

    async def get_record(*args, **kwargs):
        return record

    monkeypatch.setattr(
        verification,
        "_get_otp_record",
        get_record,
    )

    result = asyncio.run(
        verification.verify_email_with_otp(
            db=db,
            user_id=user.id,
            otp="123456",
        )
    )

    assert result is False
    assert record.used_at is not None
    assert user.is_email_verified is False
    assert db.commits == 1


def test_malformed_otp_is_rejected_without_database_write(
    verification_setup,
):
    db, user = verification_setup

    result = asyncio.run(
        verification.verify_email_with_otp(
            db=db,
            user_id=user.id,
            otp="12ab",
        )
    )

    assert result is False
    assert user.is_email_verified is False
    assert db.commits == 0