import asyncio

import pytest

from app.core.config import settings
from app.services.local_context import (
    LocalContextError,
    get_project_instructions,
    save_project_instructions,
)


@pytest.fixture
def allowed_project(tmp_path, monkeypatch):
    project = tmp_path / "my-project"
    project.mkdir()

    # Use a temporary location so tests never touch real local memory.
    monkeypatch.setenv(
        "LOCALAPPDATA",
        str(tmp_path / "local-data"),
    )
    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(project),
    )

    return project


def test_save_and_retrieve_project_instructions(allowed_project):
    async def scenario():
        saved = await save_project_instructions(
            project_root=allowed_project,
            instructions="Use Python 3.13 and type hints.",
        )

        retrieved = await get_project_instructions(
            project_root=allowed_project,
        )

        assert saved.instructions == (
            "Use Python 3.13 and type hints."
        )
        assert retrieved is not None
        assert retrieved.instructions == saved.instructions
        assert retrieved.project_name == "my-project"

    asyncio.run(scenario())


def test_updating_instructions_does_not_create_duplicates(
    allowed_project,
):
    async def scenario():
        await save_project_instructions(
            project_root=allowed_project,
            instructions="Use Python.",
        )

        updated = await save_project_instructions(
            project_root=allowed_project,
            instructions="Use Python 3.13 and pytest.",
        )

        retrieved = await get_project_instructions(
            project_root=allowed_project,
        )

        assert retrieved is not None
        assert retrieved.instructions == (
            "Use Python 3.13 and pytest."
        )
        assert retrieved.created_at == updated.created_at
        assert retrieved.updated_at == updated.updated_at

    asyncio.run(scenario())


def test_rejects_project_outside_allowlist(
    allowed_project,
    tmp_path,
):
    other_project = tmp_path / "other-project"
    other_project.mkdir()

    async def scenario():
        with pytest.raises(LocalContextError):
            await save_project_instructions(
                project_root=other_project,
                instructions="These instructions must not be saved.",
            )

    asyncio.run(scenario())


def test_rejects_excessively_long_instructions(allowed_project):
    async def scenario():
        with pytest.raises(ValueError):
            await save_project_instructions(
                project_root=allowed_project,
                instructions="x" * 50_001,
            )

    asyncio.run(scenario())
