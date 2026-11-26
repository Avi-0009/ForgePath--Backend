"""Refresh-session revocation is durable and idempotent."""

import asyncio
from types import SimpleNamespace

from app.services import auth_session


class Database:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


def test_revoke_refresh_session_locks_and_commits(monkeypatch):
    db = Database()
    session = SimpleNamespace(revoked_at=None)

    async def get_session(passed_db, token_hash, *, for_update):
        assert passed_db is db
        assert token_hash == "hashed-token"
        assert for_update is True
        return session

    monkeypatch.setattr(auth_session, "hash_refresh_token", lambda _: "hashed-token")
    monkeypatch.setattr(auth_session, "get_session_by_refresh_hash", get_session)

    revoked = asyncio.run(auth_session.revoke_auth_session(db, "refresh-token"))

    assert revoked is True
    assert session.revoked_at is not None
    assert db.commits == 1
    assert db.rollbacks == 0


def test_revoke_refresh_session_is_safe_without_cookie():
    db = Database()

    revoked = asyncio.run(auth_session.revoke_auth_session(db, None))

    assert revoked is False
    assert db.commits == 0
    assert db.rollbacks == 0


def test_revoke_refresh_session_is_idempotent(monkeypatch):
    db = Database()
    session = SimpleNamespace(revoked_at=object())

    async def get_session(*_args, **_kwargs):
        return session

    monkeypatch.setattr(auth_session, "get_session_by_refresh_hash", get_session)

    revoked = asyncio.run(auth_session.revoke_auth_session(db, "refresh-token"))

    assert revoked is False
    assert db.commits == 0
    assert db.rollbacks == 1
