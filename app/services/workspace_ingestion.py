import os
import stat

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.services.workspace import (
    DEFAULT_MAX_FILE_BYTES,
    EXCLUDED_DIRECTORIES,
    LANGUAGE_BY_EXTENSION,
    WorkspaceAccessDenied,
    WorkspaceConfigurationError,
    WorkspaceReadError,
    _validate_relative_path,
    read_configured_workspace_file,
)


MAX_INGESTION_DEPTH = 12
MAX_INGESTION_FILES = 1_000
MAX_INGESTION_TOTAL_BYTES = 20_000_000

IGNORED_FILENAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "cargo.lock",
    "composer.lock",
    "bun.lock",
    "bun.lockb",
}


@dataclass(frozen=True, slots=True)
class IngestedWorkspaceDocument:
    """A safely read document prepared for later AI processing."""

    workspace_root: str
    relative_path: str
    language: str
    content: str
    sha256: str
    size_bytes: int
    line_count: int


@dataclass(frozen=True, slots=True)
class WorkspaceIngestionResult:
    """A bounded snapshot of ingestible workspace documents."""

    workspace_root: str
    documents: tuple[IngestedWorkspaceDocument, ...]
    files_scanned: int
    files_skipped: int
    total_bytes: int
    truncated: bool


def _is_link_or_junction(path: Path) -> bool:
    """Avoid traversing symbolic links and Windows junctions."""

    try:
        if path.is_symlink():
            return True

        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction and is_junction())

    except OSError:
        return True


def _resolve_allowed_workspace_root(
    workspace_root: str | Path,
) -> Path:
    """Resolve a workspace and require an exact configured-root match."""

    configured_roots = settings.workspace_root_list

    if not configured_roots:
        raise WorkspaceConfigurationError(
            "No workspace roots are configured."
        )

    try:
        root = Path(workspace_root).expanduser().resolve(strict=True)

        if not root.is_dir():
            raise ValueError("Workspace root is not a directory.")

        allowed_roots: set[Path] = set()

        for configured_root in configured_roots:
            resolved = (
                Path(configured_root)
                .expanduser()
                .resolve(strict=True)
            )

            if not resolved.is_dir():
                raise ValueError(
                    "A configured workspace root is not a directory."
                )

            allowed_roots.add(resolved)

    except (OSError, RuntimeError, ValueError):
        raise WorkspaceConfigurationError(
            "A workspace root is invalid or unavailable."
        ) from None

    if root not in allowed_roots:
        raise WorkspaceAccessDenied(
            "The selected workspace is not allowlisted."
        )

    return root


def resolve_allowed_workspace_root(
    workspace_root: str | Path,
) -> Path:
    """
    Public interface for validating and resolving an allowed workspace.

    Requires an exact match against configured workspace roots.
    Returns the canonical resolved workspace path.
    """

    return _resolve_allowed_workspace_root(workspace_root)


def _is_ingestable_file(path: Path) -> bool:
    """Accept supported source, configuration, and documentation files."""

    filename = path.name.casefold()

    if filename in IGNORED_FILENAMES:
        return False

    if filename.endswith((".min.js", ".min.css", ".map")):
        return False

    return path.suffix.lower() in LANGUAGE_BY_EXTENSION


def ingest_workspace(
    workspace_root: str | Path,
    *,
    max_files: int = MAX_INGESTION_FILES,
    max_total_bytes: int = MAX_INGESTION_TOTAL_BYTES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> WorkspaceIngestionResult:
    """
    Safely collect supported text documents from one configured workspace.

    The service:
    - Requires an exact allowlisted workspace root.
    - Never follows symbolic links or directory junctions.
    - Skips excluded directories and sensitive paths.
    - Reads only supported file extensions.
    - Enforces per-file, total-size, file-count, and depth limits.
    - Reuses the workspace reader for file validation and hashing.

    This service does not persist files, call an AI provider, generate
    embeddings, or modify the original workspace.
    """

    if max_files < 1:
        raise ValueError("max_files must be greater than zero.")

    if max_total_bytes < 1:
        raise ValueError("max_total_bytes must be greater than zero.")

    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be greater than zero.")

    root = _resolve_allowed_workspace_root(workspace_root)

    pending_directories = deque([(root, 0)])

    documents: list[IngestedWorkspaceDocument] = []

    files_scanned = 0
    files_skipped = 0
    total_bytes = 0
    truncated = False

    while pending_directories:
        directory, depth = pending_directories.popleft()

        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda item: item.name.casefold(),
            )
        except OSError:
            truncated = True
            continue

        for entry in entries:
            if _is_link_or_junction(entry):
                continue

            try:
                relative_path = entry.relative_to(root).as_posix()

                _validate_relative_path(relative_path)

                entry_stat = entry.stat(follow_symlinks=False)

            except WorkspaceReadError:
                files_skipped += 1
                continue

            except (OSError, RuntimeError, ValueError):
                files_skipped += 1
                continue

            if stat.S_ISDIR(entry_stat.st_mode):
                if entry.name.casefold() in EXCLUDED_DIRECTORIES:
                    continue

                if depth >= MAX_INGESTION_DEPTH:
                    truncated = True
                    continue

                pending_directories.append((entry, depth + 1))
                continue

            if not stat.S_ISREG(entry_stat.st_mode):
                continue

            if not _is_ingestable_file(entry):
                files_skipped += 1
                continue

            if files_scanned >= max_files:
                truncated = True
                break

            if entry_stat.st_size > max_file_bytes:
                files_scanned += 1
                files_skipped += 1
                continue

            remaining_bytes = max_total_bytes - total_bytes

            if entry_stat.st_size > remaining_bytes:
                truncated = True
                break

            files_scanned += 1

            try:
                source = read_configured_workspace_file(
                    workspace_root=str(root),
                    relative_path=relative_path,
                    max_bytes=max_file_bytes,
                )

            except WorkspaceConfigurationError:
                raise

            except WorkspaceReadError:
                files_skipped += 1
                continue

            if not source.content.strip():
                files_skipped += 1
                continue

            if source.size_bytes > remaining_bytes:
                truncated = True
                break

            document = IngestedWorkspaceDocument(
                workspace_root=str(root),
                relative_path=source.relative_path,
                language=source.language or "text",
                content=source.content,
                sha256=source.sha256,
                size_bytes=source.size_bytes,
                line_count=source.line_count,
            )

            documents.append(document)
            total_bytes += source.size_bytes

        if truncated:
            break

    return WorkspaceIngestionResult(
        workspace_root=str(root),
        documents=tuple(documents),
        files_scanned=files_scanned,
        files_skipped=files_skipped,
        total_bytes=total_bytes,
        truncated=truncated,
    )