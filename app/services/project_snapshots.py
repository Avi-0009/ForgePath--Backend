"""Bounded, user-owned, read-only source snapshots from browser folder picks.

No shell commands, AI calls, network fetching or user-selected server paths occur
here. A snapshot is immutable once written; the active pointer is atomic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.services.workspace import (
    EXCLUDED_DIRECTORIES,
    LANGUAGE_BY_EXTENSION,
    SENSITIVE_FILENAMES,
    SENSITIVE_SUFFIXES,
)


MAX_SNAPSHOT_FILES = 1_000
MAX_SNAPSHOT_BYTES = 20 * 1024 * 1024
MAX_FILE_BYTES = 1_000_000
MAX_DEPTH = 12

_SKIP_DIRECTORIES = EXCLUDED_DIRECTORIES | {
    ".svn", ".hg", ".next", ".nuxt", "coverage", "vendor", "target",
    ".config", ".idea", ".vscode", ".terraform", ".local", "tmp",
}
_SKIP_FILENAMES = SENSITIVE_FILENAMES | {
    ".env", ".npmrc", ".pypirc", ".netrc", "secrets.json",
    "credentials.json", "service-account.json", "docker-compose.override.yml",
}
_SKIP_SUFFIXES = SENSITIVE_SUFFIXES | {
    ".p12", ".pfx", ".jks", ".keystore", ".sqlite", ".db",
}
_SUSPICIOUS = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?i:aws_secret_access_key\s*[:=]|authorization\s*[:=]\s*bearer\s+\S+|"
    r"(?:api[_-]?key|password|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_+\-/]{18,})"
)


class SnapshotValidationError(ValueError):
    """The candidate snapshot is not safe to store."""


class SnapshotConfigurationError(RuntimeError):
    """The dedicated server snapshot directory is missing or unsafe."""


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    relative_path: str = Field(min_length=1, max_length=1024)
    content: str


class SourceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    files: list[SourceFile] = Field(min_length=1, max_length=MAX_SNAPSHOT_FILES)


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: UUID
    file_count: int
    total_bytes: int
    sha256: str


def _safe_path(relative_path: str) -> PurePosixPath:
    if "\\" in relative_path or ":" in relative_path or "\x00" in relative_path:
        raise SnapshotValidationError("Source path contains an unsafe character")
    path = PurePosixPath(relative_path)
    if path.is_absolute() or len(path.parts) > MAX_DEPTH or not path.parts:
        raise SnapshotValidationError("Invalid source path")
    if any(
        part in {"", ".", ".."} or part.startswith(".") or
        part.casefold() in _SKIP_DIRECTORIES
        for part in path.parts[:-1]
    ):
        raise SnapshotValidationError("Unsafe source directory")
    name = path.name.casefold()
    if (
        name.startswith(".") or name in _SKIP_FILENAMES or
        "secret" in name or "credential" in name or "private_key" in name or
        any(name.endswith(suffix) for suffix in _SKIP_SUFFIXES) or
        path.suffix.casefold() not in LANGUAGE_BY_EXTENSION or
        name.endswith((".min.js", ".min.css"))
    ):
        raise SnapshotValidationError("Sensitive or unsupported source filename")
    return path


def _prepare(snapshot: SourceSnapshot) -> tuple[list[tuple[PurePosixPath, bytes]], SnapshotManifest]:
    seen: set[str] = set()
    files: list[tuple[PurePosixPath, bytes]] = []
    total = 0
    digest = hashlib.sha256()
    for item in snapshot.files:
        path = _safe_path(item.relative_path)
        canonical = path.as_posix().casefold()
        if canonical in seen:
            raise SnapshotValidationError("Duplicate source filename")
        seen.add(canonical)
        if "\x00" in item.content or _SUSPICIOUS.search(item.content):
            raise SnapshotValidationError("A file contains binary data or suspected credentials")
        data = item.content.encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise SnapshotValidationError("A source file exceeds the per-file size limit")
        total += len(data)
        if total > MAX_SNAPSHOT_BYTES:
            raise SnapshotValidationError("Codebase exceeds the snapshot size limit")
        files.append((path, data))
    files.sort(key=lambda entry: entry[0].as_posix())
    for path, data in files:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(data).digest())
    return files, SnapshotManifest(uuid4(), len(files), total, digest.hexdigest())


def _storage_root() -> Path:
    configured = settings.project_snapshot_directory
    if not configured or not Path(configured).is_absolute():
        raise SnapshotConfigurationError("PROJECT_SNAPSHOT_DIRECTORY must be an absolute private path")
    path = Path(configured)
    try:
        if path.is_symlink() or (getattr(path, "is_junction", None) and path.is_junction()):
            raise SnapshotConfigurationError("Snapshot storage cannot be a symlink or junction")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved = path.resolve(strict=True)
        if resolved != path.absolute() or not resolved.is_dir():
            raise SnapshotConfigurationError("Unsafe snapshot storage root")
        if os.name != "nt" and (resolved.stat().st_mode & 0o077):
            raise SnapshotConfigurationError("Snapshot storage must not be accessible by group or others")
        return resolved
    except OSError as exc:
        raise SnapshotConfigurationError("Snapshot storage is unavailable") from exc


def _project_directory(user_id: UUID, project_id: UUID) -> Path:
    return _storage_root() / str(user_id) / str(project_id)


def store_snapshot(user_id: UUID, project_id: UUID, snapshot: SourceSnapshot) -> SnapshotManifest:
    """Persist one immutable version and atomically select it for the project."""
    files, manifest = _prepare(snapshot)
    previous = read_snapshot_manifest(user_id, project_id)
    if (
        previous is not None and previous.sha256 == manifest.sha256 and
        previous.file_count == manifest.file_count and
        previous.total_bytes == manifest.total_bytes
    ):
        return previous
    project_directory = _project_directory(user_id, project_id)
    project_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    version_directory = project_directory / str(manifest.snapshot_id)
    version_directory.mkdir(mode=0o700)
    try:
        for relative, data in files:
            destination = version_directory.joinpath(*relative.parts)
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with destination.open("xb") as destination_file:
                os.chmod(destination, 0o600)
                destination_file.write(data)
        manifest_path = version_directory / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as metadata_file:
            json.dump({
                "snapshot_id": str(manifest.snapshot_id),
                "file_count": manifest.file_count,
                "total_bytes": manifest.total_bytes,
                "sha256": manifest.sha256,
            }, metadata_file)
        os.chmod(manifest_path, 0o600)
        temporary_pointer = project_directory / f".current-{uuid4().hex}"
        temporary_pointer.write_text(str(manifest.snapshot_id), encoding="ascii")
        os.replace(temporary_pointer, project_directory / "CURRENT")
    except BaseException:
        shutil.rmtree(version_directory)
        raise
    return manifest


def read_snapshot_manifest(user_id: UUID, project_id: UUID) -> SnapshotManifest | None:
    project_directory = _project_directory(user_id, project_id)
    pointer = project_directory / "CURRENT"
    if not pointer.is_file():
        return None
    snapshot_id = UUID(pointer.read_text(encoding="ascii").strip())
    manifest_path = project_directory / str(snapshot_id) / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return SnapshotManifest(
        UUID(data["snapshot_id"]), int(data["file_count"]),
        int(data["total_bytes"]), str(data["sha256"]),
    )


def read_snapshot_sources(
    user_id: UUID, project_id: UUID, snapshot_id: UUID,
) -> list[tuple[str, str]]:
    """Read a pinned immutable snapshot; refuse links and changed contents."""
    root = _project_directory(user_id, project_id) / str(snapshot_id)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise SnapshotValidationError("Snapshot is unavailable")
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    results: list[tuple[str, str]] = []
    digest = hashlib.sha256()
    total = 0
    for path in sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        if path == manifest_path:
            continue
        if path.is_symlink():
            raise SnapshotValidationError("Snapshot contains an invalid entry")
        if path.is_dir():
            continue
        if not path.is_file():
            raise SnapshotValidationError("Snapshot contains an invalid entry")
        relative = path.relative_to(root)
        if any(part.is_symlink() for part in (root, *path.parents)):
            raise SnapshotValidationError("Snapshot contains a symbolic link")
        safe = _safe_path(relative.as_posix())
        if path.stat().st_size > MAX_FILE_BYTES:
            raise SnapshotValidationError("Snapshot file limit exceeded")
        data = path.read_bytes()
        total += len(data)
        if total > MAX_SNAPSHOT_BYTES or len(results) >= MAX_SNAPSHOT_FILES:
            raise SnapshotValidationError("Snapshot limit exceeded")
        digest.update(safe.as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(data).digest())
        results.append((safe.as_posix(), data.decode("utf-8")))
    if (
        len(results) != expected["file_count"] or
        total != expected["total_bytes"] or
        digest.hexdigest() != expected["sha256"]
    ):
        raise SnapshotValidationError("Snapshot checksum mismatch")
    return results