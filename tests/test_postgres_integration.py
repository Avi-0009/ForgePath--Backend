
"""M10: Real PostgreSQL and Alembic integration tests.

Requires a dedicated disposable PostgreSQL database.
Never run against production or development application data.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
from uuid import uuid4
from sqlalchemy.exc import IntegrityError

from app.db.base import Base
import app.models  # noqa: F401 - registers SQLAlchemy models


TEST_DATABASE_URL = os.getenv(
    "FORGEPATH_M10_TEST_DATABASE_URL"
)

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="Dedicated M10 PostgreSQL database not configured",
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def migrated_database():
    """Apply Alembic migrations to the dedicated test database."""

    if not TEST_DATABASE_URL:
        pytest.skip("Test database not configured")

    # Prevent accidental execution against other databases.
    assert (
        "forgepath_m10" in TEST_DATABASE_URL
        and "5544" in TEST_DATABASE_URL
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
        f"Alembic migration failed:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    return TEST_DATABASE_URL


@pytest.mark.anyio
async def test_postgres_connection(migrated_database):
    """Verify the database accepts actual SQL queries."""

    engine = create_async_engine(migrated_database)

    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text("SELECT 1")
            )
            assert result.scalar_one() == 1
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_all_application_tables_exist(migrated_database):
    """Every registered SQLAlchemy table must exist after migration."""

    engine = create_async_engine(migrated_database)

    try:
        async with engine.connect() as connection:
            existing_tables = await connection.run_sync(
                lambda sync_connection: set(
                    inspect(sync_connection).get_table_names()
                )
            )

        expected_tables = set(Base.metadata.tables.keys())

        assert expected_tables, "No application models registered"

        missing = expected_tables - existing_tables

        assert not missing, (
            f"Missing PostgreSQL tables: {sorted(missing)}"
        )

    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_alembic_version_exists(migrated_database):
    """Verify Alembic recorded the applied migration revision."""

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



@pytest.mark.anyio
async def test_postgres_foreign_keys(migrated_database):
    """Verify actual PostgreSQL foreign keys match SQLAlchemy models."""

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
                        f"Foreign key mismatch: {table.name}\n"
                        f"Expected: {expected}\n"
                        f"Actual: {actual}"
                    )

            await connection.run_sync(check_foreign_keys)

    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_postgres_user_unique_constraints(migrated_database):
    """Usernames and emails must be unique in PostgreSQL."""

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


@pytest.mark.anyio
async def test_postgres_rejects_duplicate_username(migrated_database):
    """PostgreSQL must enforce username uniqueness."""

    engine = create_async_engine(migrated_database)
    unique = uuid4().hex[:12]

    try:
        async with engine.connect() as connection:
            async with connection.begin():
                await connection.execute(
                    text("""
                        INSERT INTO users
                        (full_name, username, email, password_hash)
                        VALUES
                        (:name, :username, :email, :password)
                    """),
                    {
                        "name": "Integration Test",
                        "username": f"test_{unique}",
                        "email": f"first_{unique}@example.com",
                        "password": "test-hash",
                    },
                )

                # Attempt to insert a duplicate username.
                # PostgreSQL should reject it.
                with pytest.raises(IntegrityError):
                    async with connection.begin_nested():
                        await connection.execute(
                            text("""
                                INSERT INTO users
                                (full_name, username, email, password_hash)
                                VALUES
                                (:name, :username, :email, :password)
                            """),
                            {
                                "name": "Duplicate Test",
                                "username": f"test_{unique}",
                                "email": f"second_{unique}@example.com",
                                "password": "test-hash",
                            },
                        )

    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_postgres_rejects_invalid_session_user(migrated_database):
    """Auth sessions must reference an existing user."""

    engine = create_async_engine(migrated_database)

    try:
        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text("""
                        INSERT INTO auth_sessions
                        (user_id, refresh_token_hash, expires_at)
                        VALUES
                        (:user_id, :token_hash, NOW() + INTERVAL '1 day')
                    """),
                    {
                        "user_id": uuid4(),
                        "token_hash": uuid4().hex,
                    },
                )
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_postgres_user_session_cascade(migrated_database):
    """Deleting a user must delete their refresh sessions."""

    engine = create_async_engine(migrated_database)
    user_id = uuid4()
    session_id = uuid4()
    suffix = uuid4().hex[:12]

    try:
        async with engine.begin() as connection:
            await connection.execute(
                text("""
                    INSERT INTO users
                    (id, full_name, username, email, password_hash)
                    VALUES (:id, :name, :username, :email, :password)
                """),
                {
                    "id": user_id,
                    "name": "Cascade Test",
                    "username": f"test_{suffix}",
                    "email": f"cascade_{suffix}@example.com",
                    "password": "test-hash",
                },
            )

            await connection.execute(
                text("""
                    INSERT INTO auth_sessions
                    (id, user_id, refresh_token_hash, expires_at)
                    VALUES
                    (:id, :user_id, :token_hash, NOW() + INTERVAL '1 day')
                """),
                {
                    "id": session_id,
                    "user_id": user_id,
                    "token_hash": uuid4().hex,
                },
            )

            await connection.execute(
                text("DELETE FROM users WHERE id = :id"),
                {"id": user_id},
            )

            result = await connection.execute(
                text("SELECT COUNT(*) FROM auth_sessions WHERE id = :id"),
                {"id": session_id},
            )

            assert result.scalar_one() == 0

    finally:
        await engine.dispose()