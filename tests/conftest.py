"""Shared test fixtures for ForgePath."""

from types import SimpleNamespace

import pytest

from app.api import dependencies


@pytest.fixture(autouse=True)
def support_legacy_database_mocks(monkeypatch):
    """
    Preserve existing isolated route tests.

    Older route tests use SimpleNamespace as a database
    placeholder. Those tests do not exercise AI credentials.

    Real database sessions continue through the actual
    credential-loading implementation.
    """

    original_loader = dependencies.load_user_ai_choice

    async def load_test_ai_choice(db, user_id):
        if isinstance(db, SimpleNamespace):
            return None

        return await original_loader(db, user_id)

    monkeypatch.setattr(
        dependencies,
        "load_user_ai_choice",
        load_test_ai_choice,
    )