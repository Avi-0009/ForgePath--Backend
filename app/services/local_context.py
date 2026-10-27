import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from app.core.config import settings


MAX_INSTRUCTIONS_LENGTH = 50_000


class LocalContextError(Exception):
    """Raised when local project context cannot be accessed or saved."""


@dataclass(frozen=True, slots=True)
class ProjectContext:
    """Persisted instructions and metadata for one project."""

    project_root: str
    project_name: str
    instructions: str
    created_at: str
    updated_at: str


def _database_path() -> Path:
    """Return a database location outside the source repository."""

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")

        if not base:
            base = str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_DATA_HOME")

        if not base:
            base = str(Path.home() / ".local" / "share")

    return Path(base) / "ForgePath" / "context.sqlite3"


def _resolve_project_root(project_root: str | Path) -> Path:
    """Require the project root to match a configured workspace."""

    if not settings.workspace_root_list:
        raise LocalContextError(
            "No workspace roots are configured. "
            "Set WORKSPACE_ALLOWED_ROOTS in the backend environment."
        )

    try:
        requested_root = Path(project_root).expanduser().resolve(
            strict=True
        )
    except (OSError, RuntimeError, ValueError):
        raise LocalContextError(
            "The requested project root is unavailable or invalid."
        ) from None

    if not requested_root.is_dir():
        raise LocalContextError(
            "The requested project root is not a directory."
        )

    allowed_roots: set[Path] = set()

    for configured_root in settings.workspace_root_list:
        try:
            resolved_root = Path(configured_root).expanduser().resolve(
                strict=True
            )
        except (OSError, RuntimeError, ValueError):
            raise LocalContextError(
                "A configured workspace root is unavailable. "
                "Check WORKSPACE_ALLOWED_ROOTS."
            ) from None

        if not resolved_root.is_dir():
            raise LocalContextError(
                "A configured workspace root is not a directory."
            )

        allowed_roots.add(resolved_root)

    if requested_root not in allowed_roots:
        raise LocalContextError(
            "Project context is only available for configured "
            "workspace roots."
        )

    return requested_root


async def initialize_local_context() -> Path:
    """Create the local database and its tables if necessary."""

    db_path = _database_path()

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(
            db_path,
            timeout=5.0,
        ) as db:
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute("PRAGMA journal_mode = WAL")

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS project_context (
                    project_root TEXT PRIMARY KEY NOT NULL,
                    project_name TEXT NOT NULL,
                    instructions TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            await db.commit()

        # Best-effort owner-only permissions on systems that support them.
        if os.name != "nt":
            db_path.chmod(0o600)

    except (OSError, aiosqlite.Error) as exc:
        raise LocalContextError(
            "Could not initialize ForgePath's local context database."
        ) from exc

    return db_path


async def save_project_instructions(
    *,
    project_root: str | Path,
    instructions: str,
) -> ProjectContext:
    """Create or update instructions for an allowed project."""

    if not isinstance(instructions, str):
        raise ValueError("Instructions must be text.")

    if len(instructions) > MAX_INSTRUCTIONS_LENGTH:
        raise ValueError(
            f"Instructions cannot exceed "
            f"{MAX_INSTRUCTIONS_LENGTH} characters."
        )

    root = _resolve_project_root(project_root)
    db_path = await initialize_local_context()

    now = datetime.now(timezone.utc).isoformat()
    root_text = str(root)

    try:
        async with aiosqlite.connect(
            db_path,
            timeout=5.0,
        ) as db:
            await db.execute("PRAGMA busy_timeout = 5000")
            db.row_factory = aiosqlite.Row

            await db.execute(
                """
                INSERT INTO project_context (
                    project_root,
                    project_name,
                    instructions,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_root) DO UPDATE SET
                    project_name = excluded.project_name,
                    instructions = excluded.instructions,
                    updated_at = excluded.updated_at
                """,
                (
                    root_text,
                    root.name or str(root),
                    instructions,
                    now,
                    now,
                ),
            )

            await db.commit()

            cursor = await db.execute(
                """
                SELECT project_root, project_name, instructions, created_at, updated_at
                FROM project_context
                WHERE project_root = ?
                """,
                (root_text,),
            )

            row = await cursor.fetchone()

            if row is None:
                raise LocalContextError(
                    "The saved project context could not be retrieved."
                )

            return ProjectContext(
                project_root=row["project_root"],
                project_name=row["project_name"],
                instructions=row["instructions"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    except aiosqlite.Error as exc:
        raise LocalContextError(
            "Could not save project instructions."
        ) from exc


async def get_project_instructions(
    *,
    project_root: str | Path,
) -> ProjectContext | None:
    """Retrieve saved instructions for an allowed project."""

    root = _resolve_project_root(project_root)
    db_path = await initialize_local_context()

    try:
        async with aiosqlite.connect(
            db_path,
            timeout=5.0,
        ) as db:
            await db.execute("PRAGMA busy_timeout = 5000")
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                """
                SELECT project_root, project_name, instructions, created_at, updated_at
                FROM project_context
                WHERE project_root = ?
                """,
                (str(root),),
            )

            row = await cursor.fetchone()

            if row is None:
                return None

            return ProjectContext(
                project_root=row["project_root"],
                project_name=row["project_name"],
                instructions=row["instructions"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    except aiosqlite.Error as exc:
        raise LocalContextError(
            "Could not retrieve project instructions."
        ) from exc