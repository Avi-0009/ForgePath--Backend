from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.base import Base
import app.models


TEST_DATABASE_URL = os.getenv("FORGEPATH_M10_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="Dedicated M10 PostgreSQL database not configured",
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def migrated_database():
    """Run Alembic migrations against the dedicated M10 database."""

    if not TEST_DATABASE_URL:
        pytest.skip("Test database not configured")

    url = make_url(TEST_DATABASE_URL)

    assert (
        url.drivername == "postgresql+asyncpg"
        and url.host in {"127.0.0.1", "localhost"}
        and url.port == 5544
        and url.database == "forgepath_m10"
        and url.username == "forgepath_test"
    ), "Refusing to migrate a non-M10 database"

    root = Path(__file__).resolve().parents[1]

    environment = os.environ.copy()
    environment["DATABASE_URL"] = TEST_DATABASE_URL

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        "Alembic migration failed:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    return TEST_DATABASE_URL


def _user_values(*, prefix: str = "test") -> dict:
    """Generate valid user data respecting the 20-character username limit."""

    suffix = uuid4().hex[:10]
    username = f"{prefix}_{suffix}"

    assert len(username) <= 20

    return {
        "id": uuid4(),
        "name": "Integration Test",
        "username": username,
        "email": f"{username}@example.com",
        "password": "test-hash",
    }


async def _insert_user(connection, values: dict) -> None:
    """Insert a temporary integration-test user."""

    await connection.execute(
        text("""
            INSERT INTO users
                (id, full_name, username, email, password_hash)
            VALUES
                (:id, :name, :username, :email, :password)
        """),
        values,
    )


async def _insert_session(
    connection,
    *,
    user_id,
    token_hash: str,
    session_id=None,
) -> None:
    """Insert a temporary authentication session."""

    await connection.execute(
        text("""
            INSERT INTO auth_sessions
                (id, user_id, refresh_token_hash, expires_at)
            VALUES
                (:id, :user_id, :token_hash, NOW() + INTERVAL '1 day')
        """),
        {
            "id": session_id or uuid4(),
            "user_id": user_id,
            "token_hash": token_hash,
        },
    )


# PostgreSQL connectivity

@pytest.mark.anyio
async def test_postgres_connection(migrated_database):
    """Verify PostgreSQL accepts real queries."""

    engine = create_async_engine(migrated_database)

    try:
        async with engine.connect() as connection:
            result = await connection.execute(text("SELECT 1"))

            assert result.scalar_one() == 1

    finally:
        await engine.dispose()


# Application tables

@pytest.mark.anyio
async def test_all_application_tables_exist(migrated_database):
    """Every registered SQLAlchemy table must exist after migration."""

    engine = create_async_engine(migrated_database)

    try:
        async with engine.connect() as connection:

            def inspect_tables(sync_connection):
                inspector = inspect(sync_connection)
                return set(inspector.get_table_names())

            existing_tables = await connection.run_sync(inspect_tables)

        expected_tables = set(Base.metadata.tables.keys())

        assert expected_tables, "No SQLAlchemy models registered"

        missing = expected_tables - existing_tables

        assert not missing, (
            f"Missing PostgreSQL tables: {sorted(missing)}"
        )

    finally:
        await engine.dispose()


# Alembic revision

@pytest.mark.anyio
async def test_alembic_version_exists(migrated_database):
    """Verify Alembic recorded the applied migration."""

    engine = create_async_engine(migrated_database)

    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text("SELECT version_num FROM alembic_version")
            )

            revisions = result.scalars().all()

            assert revisions, "Alembic revision was not recorded"

    finally:
        await engine.dispose()


# Foreign-key relationships

@pytest.mark.anyio
async def test_postgres_foreign_keys(migrated_database):
    """Verify PostgreSQL foreign keys match registered SQLAlchemy models."""

    engine = create_async_engine(migrated_database)

    try:
        async with engine.connect() as connection:

            def check_foreign_keys(sync_connection):
                inspector = inspect(sync_connection)

                for table in Base.metadata.sorted_tables:
                    expected = {
                        (
                            tuple(fk.column_keys),
                            fk.referred_table.name,
                        )
                        for fk in table.foreign_key_constraints
                    }

                    actual = {
                        (
                            tuple(fk["constrained_columns"]),
                            fk["referred_table"],
                        )
                        for fk in inspector.get_foreign_keys(table.name)
                    }

                    assert expected == actual, (
                        f"Foreign-key mismatch: {table.name}\n"
                        f"Expected: {expected}\n"
                        f"Actual: {actual}"
                    )

            await connection.run_sync(check_foreign_keys)

    finally:
        await engine.dispose()


# Username/email uniqueness

@pytest.mark.anyio
async def test_postgres_user_unique_constraints(migrated_database):
    """Username and email must have database-level uniqueness."""

    engine = create_async_engine(migrated_database)

    try:
        async with engine.connect() as connection:

            def check_uniqueness(sync_connection):
                inspector = inspect(sync_connection)

                constraints = inspector.get_unique_constraints("users")
                indexes = inspector.get_indexes("users")

                unique_columns = {
                    column
                    for constraint in constraints
                    for column in constraint["column_names"]
                    if column is not None
                }

                unique_columns.update(
                    column
                    for index in indexes
                    if index["unique"]
                    for column in index["column_names"]
                    if column is not None
                )

                assert "username" in unique_columns
                assert "email" in unique_columns

            await connection.run_sync(check_uniqueness)

    finally:
        await engine.dispose()


# Duplicate username rejection

@pytest.mark.anyio
async def test_postgres_rejects_duplicate_username(migrated_database):
    """PostgreSQL must reject duplicate usernames."""

    engine = create_async_engine(migrated_database)

    first = _user_values(prefix="dup")
    second = _user_values(prefix="dup")

    second["username"] = first["username"]

    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()

            try:
                await _insert_user(connection, first)

                with pytest.raises(IntegrityError):
                    async with connection.begin_nested():
                        await _insert_user(connection, second)

            finally:
                await transaction.rollback()

    finally:
        await engine.dispose()


# Invalid foreign-key rejection

@pytest.mark.anyio
async def test_postgres_rejects_invalid_session_user(migrated_database):
    """An auth session must reference an existing user."""

    engine = create_async_engine(migrated_database)

    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()

            try:
                with pytest.raises(IntegrityError):
                    async with connection.begin_nested():
                        await _insert_session(
                            connection,
                            user_id=uuid4(),
                            token_hash=uuid4().hex,
                        )

            finally:
                await transaction.rollback()

    finally:
        await engine.dispose()


# User deletion cascades to sessions

@pytest.mark.anyio
async def test_postgres_user_session_cascade(migrated_database):
    """Deleting a user must delete their refresh sessions."""

    engine = create_async_engine(migrated_database)

    user = _user_values(prefix="cascade")
    session_id = uuid4()

    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()

            try:
                await _insert_user(connection, user)

                await _insert_session(
                    connection,
                    user_id=user["id"],
                    token_hash=uuid4().hex,
                    session_id=session_id,
                )

                await connection.execute(
                    text("DELETE FROM users WHERE id = :id"),
                    {"id": user["id"]},
                )

                result = await connection.execute(
                    text("""
                        SELECT COUNT(*)
                        FROM auth_sessions
                        WHERE id = :id
                    """),
                    {"id": session_id},
                )

                assert result.scalar_one() == 0

            finally:
                await transaction.rollback()

    finally:
        await engine.dispose()


# Duplicate email rejection

@pytest.mark.anyio
async def test_postgres_rejects_duplicate_email(migrated_database):
    """PostgreSQL must reject duplicate email addresses."""

    engine = create_async_engine(migrated_database)

    first = _user_values(prefix="email")
    second = _user_values(prefix="email")

    second["email"] = first["email"]

    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()

            try:
                await _insert_user(connection, first)

                with pytest.raises(IntegrityError):
                    async with connection.begin_nested():
                        await _insert_user(connection, second)

            finally:
                await transaction.rollback()

    finally:
        await engine.dispose()


# Duplicate refresh-token rejection

@pytest.mark.anyio
async def test_postgres_rejects_duplicate_refresh_token(migrated_database):
    """Two auth sessions must not share a refresh-token hash."""

    engine = create_async_engine(migrated_database)

    user = _user_values(prefix="session")
    token_hash = uuid4().hex

    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()

            try:
                await _insert_user(connection, user)

                await _insert_session(
                    connection,
                    user_id=user["id"],
                    token_hash=token_hash,
                )

                with pytest.raises(IntegrityError):
                    async with connection.begin_nested():
                        await _insert_session(
                            connection,
                            user_id=user["id"],
                            token_hash=token_hash,
                        )

            finally:
                await transaction.rollback()

    finally:
        await engine.dispose()


# ON DELETE SET NULL

@pytest.mark.anyio
async def test_postgres_ai_usage_user_set_null(migrated_database):
    """Deleting a user must preserve AI usage with user_id set to NULL."""

    engine = create_async_engine(migrated_database)

    user = _user_values(prefix="usage")
    usage_id = uuid4()

    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()

            try:
                await _insert_user(connection, user)

                await connection.execute(
                    text("""
                        INSERT INTO ai_usage_events
                            (
                                id, user_id, provider, model_id,
                                api_model, operation, status
                            )
                        VALUES
                            (
                                :id, :user_id, :provider, :model_id,
                                :api_model, :operation, :status
                            )
                    """),
                    {
                        "id": usage_id,
                        "user_id": user["id"],
                        "provider": "openai",
                        "model_id": "integration-test",
                        "api_model": "integration-test",
                        "operation": "code_review",
                        "status": "succeeded",
                    },
                )

                await connection.execute(
                    text("DELETE FROM users WHERE id = :id"),
                    {"id": user["id"]},
                )

                result = await connection.execute(
                    text("""
                        SELECT user_id
                        FROM ai_usage_events
                        WHERE id = :id
                    """),
                    {"id": usage_id},
                )

                row = result.one_or_none()

                assert row is not None, (
                    "AI usage event was incorrectly deleted"
                )

                assert row.user_id is None, (
                    "User reference was not set to NULL"
                )

            finally:
                await transaction.rollback()

    finally:
        await engine.dispose()


# Transaction rollback integrity

@pytest.mark.anyio
async def test_postgres_transaction_rollback(migrated_database):
    """Rolled-back database writes must not persist."""

    engine = create_async_engine(migrated_database)

    user = _user_values(prefix="rb")

    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()

            try:
                await _insert_user(connection, user)

                result = await connection.execute(
                    text("""
                        SELECT COUNT(*)
                        FROM users
                        WHERE id = :id
                    """),
                    {"id": user["id"]},
                )

                assert result.scalar_one() == 1

            finally:
                await transaction.rollback()

        async with engine.connect() as connection:
            result = await connection.execute(
                text("""
                    SELECT COUNT(*)
                    FROM users
                    WHERE id = :id
                """),
                {"id": user["id"]},
            )

            assert result.scalar_one() == 0, (
                "Rolled-back user record still exists"
            )

    finally:
        await engine.dispose()