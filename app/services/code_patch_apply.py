import asyncio
import hashlib
import json
import os
import stat
import tempfile
import uuid

from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.schemas.code_patch import (
    CodePatchApplyRequest,
    CodePatchApplyResponse,
)
from app.services.workspace import (
    DEFAULT_MAX_FILE_BYTES,
    WorkspaceAccessDenied,
    WorkspaceConfigurationError,
    WorkspaceFileNotFound,
    WorkspaceInvalidPath,
    _validate_relative_path,
    read_configured_workspace_file,
)


class PatchApplyError(Exception):
    """Base exception for patch-application failures."""


class StalePatchSourceError(PatchApplyError):
    """The source file changed after the patch was generated."""


class ProposedPatchHashMismatch(PatchApplyError):
    """The proposed content does not match its supplied SHA-256."""


def _is_junction(path: Path) -> bool:
    """Detect Windows directory junctions where supported."""
    check = getattr(path, "is_junction", None)
    return bool(check and check())


def _reject_links(root: Path, parts: tuple[str, ...]) -> None:
    """Reject symlinks and junctions in the requested file path."""
    current = root

    for part in parts:
        current = current / part

        if current.is_symlink() or _is_junction(current):
            raise WorkspaceAccessDenied(
                "Patch application does not follow symbolic links or junctions."
            )


def _resolve_allowlisted_root(workspace_root: str) -> Path:
    """Resolve the requested root and require an exact allowlist match."""
    if not settings.workspace_root_list:
        raise WorkspaceConfigurationError(
            "No workspace roots are configured."
        )

    try:
        root = Path(workspace_root).expanduser().resolve(strict=True)

        if not root.is_dir():
            raise ValueError("Workspace root is not a directory.")

        allowed_roots: set[Path] = set()

        for configured_root in settings.workspace_root_list:
            allowed = Path(configured_root).expanduser().resolve(strict=True)

            if not allowed.is_dir():
                raise ValueError("Configured root is not a directory.")

            allowed_roots.add(allowed)

    except WorkspaceConfigurationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise WorkspaceConfigurationError(
            "A workspace root is invalid or unavailable."
        ) from exc

    if root not in allowed_roots:
        raise WorkspaceAccessDenied(
            "The selected workspace is not allowlisted."
        )

    return root


def _backup_directory() -> Path:
    """Return ForgePath's local backup directory."""
    local_app_data = os.environ.get("LOCALAPPDATA")

    if local_app_data:
        base = Path(local_app_data)
    elif os.environ.get("XDG_DATA_HOME"):
        base = Path(os.environ["XDG_DATA_HOME"])
    else:
        base = Path.home() / ".local" / "share"

    directory = base / "ForgePath" / "backups"
    directory.mkdir(parents=True, exist_ok=True)

    if not directory.is_dir():
        raise PatchApplyError("The backup directory is unavailable.")

    if os.name != "nt":
        directory.chmod(0o700)

    return directory


def _write_backup(
    *,
    original_bytes: bytes,
    backup_id: str,
    workspace_root: Path,
    relative_path: str,
    original_sha256: str,
    proposed_sha256: str,
) -> None:
    """Save the original bytes and a local manifest before modifying the file."""
    directory = _backup_directory()
    backup_path = directory / f"{backup_id}.bak"
    manifest_path = directory / f"{backup_id}.json"

    manifest = {
        "backup_id": backup_id,
        "workspace_root": str(workspace_root),
        "relative_path": relative_path,
        "original_sha256": original_sha256,
        "proposed_sha256": proposed_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with backup_path.open("xb") as backup_file:
            backup_file.write(original_bytes)
            backup_file.flush()
            os.fsync(backup_file.fileno())

        if os.name != "nt":
            backup_path.chmod(0o600)

        with manifest_path.open("x", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
            manifest_file.flush()
            os.fsync(manifest_file.fileno())

        if os.name != "nt":
            manifest_path.chmod(0o600)

    except OSError as exc:
        # Do not modify the source when a complete backup cannot be made.
        for path in (backup_path, manifest_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        raise PatchApplyError(
            "The original file could not be backed up. No patch was applied."
        ) from exc


def _read_current_bytes(target: Path, max_bytes: int) -> tuple[bytes, os.stat_result]:
    """Read a regular file with a size limit and inspect the opened file."""
    try:
        with target.open("rb") as source_file:
            opened_stat = os.fstat(source_file.fileno())

            if not stat.S_ISREG(opened_stat.st_mode):
                raise WorkspaceInvalidPath(
                    "The selected path is not a regular file."
                )

            if opened_stat.st_size > max_bytes:
                raise WorkspaceInvalidPath(
                    "The source file exceeds the supported size."
                )

            raw = source_file.read(max_bytes + 1)

            if len(raw) > max_bytes:
                raise WorkspaceInvalidPath(
                    "The source file exceeds the supported size."
                )

    except (WorkspaceInvalidPath, WorkspaceAccessDenied):
        raise
    except OSError as exc:
        raise PatchApplyError(
            "The source file could not be read."
        ) from exc

    return raw, opened_stat


def _apply_code_patch_sync(
    request: CodePatchApplyRequest,
) -> CodePatchApplyResponse:
    proposed_bytes = request.proposed_code.encode("utf-8")

    if len(proposed_bytes) > DEFAULT_MAX_FILE_BYTES:
        raise PatchApplyError(
            "The proposed file exceeds the supported size."
        )

    actual_proposed_sha256 = hashlib.sha256(proposed_bytes).hexdigest()

    if actual_proposed_sha256 != request.proposed_sha256:
        raise ProposedPatchHashMismatch(
            "The proposed code does not match proposed_sha256."
        )

    root = _resolve_allowlisted_root(request.workspace_root)
    parts = _validate_relative_path(request.relative_path)

    _reject_links(root, parts)

    # Reuse the workspace service's file, exclusion, and size validations.
    source = read_configured_workspace_file(
        workspace_root=str(root),
        relative_path=request.relative_path,
    )

    # Use the canonical relative path returned by the workspace service.
    canonical_parts = tuple(source.relative_path.split("/"))
    _reject_links(root, canonical_parts)
    target = root.joinpath(*canonical_parts)

    if not target.is_file():
        raise WorkspaceFileNotFound(
            "The source file is no longer available."
        )

    original_bytes, original_stat = _read_current_bytes(
        target,
        DEFAULT_MAX_FILE_BYTES,
    )
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()

    if (
        original_sha256 != request.expected_sha256
        or original_sha256 != source.sha256
    ):
        raise StalePatchSourceError(
            "The source file changed after the patch was generated. "
            "Generate a new patch before applying."
        )

    # Preserve the original file's permissions on POSIX systems.
    original_mode = stat.S_IMODE(original_stat.st_mode)
    backup_id = uuid.uuid4().hex

    # The backup must be complete before a source file can be changed.
    _write_backup(
        original_bytes=original_bytes,
        backup_id=backup_id,
        workspace_root=root,
        relative_path=source.relative_path,
        original_sha256=original_sha256,
        proposed_sha256=actual_proposed_sha256,
    )

    temporary_path: Path | None = None

    try:
        # Create the temporary file in the target directory so replacement
        # stays on the same filesystem.
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".forgepath-",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary_path = Path(temporary_name)

        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(proposed_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        if os.name != "nt":
            os.chmod(temporary_path, original_mode)

        # Recheck immediately before replacement to reduce the chance of
        # overwriting changes made by another process.
        _reject_links(root, canonical_parts)

        current_bytes, current_stat = _read_current_bytes(
            target,
            DEFAULT_MAX_FILE_BYTES,
        )
        current_sha256 = hashlib.sha256(current_bytes).hexdigest()

        if (
            current_sha256 != request.expected_sha256
            or current_stat.st_dev != original_stat.st_dev
            or current_stat.st_ino != original_stat.st_ino
        ):
            raise StalePatchSourceError(
                "The source file changed before the patch could be applied. "
                "The original backup has been retained."
            )

        os.replace(temporary_path, target)
        temporary_path = None

        # Verify the final on-disk file.
        applied_bytes, _ = _read_current_bytes(
            target,
            DEFAULT_MAX_FILE_BYTES,
        )
        applied_sha256 = hashlib.sha256(applied_bytes).hexdigest()

        if applied_sha256 != actual_proposed_sha256:
            raise PatchApplyError(
                "Post-write verification failed. The original backup "
                f"is available as {backup_id}; inspect the file before "
                "continuing."
            )

        return CodePatchApplyResponse(
            relative_path=source.relative_path,
            original_sha256=original_sha256,
            applied_sha256=applied_sha256,
            backup_id=backup_id,
            patch_applied=True,
        )

    except (PatchApplyError, WorkspaceAccessDenied, WorkspaceInvalidPath):
        raise
    except OSError as exc:
        raise PatchApplyError(
            "The patch could not be written. The original backup has "
            f"been retained as {backup_id}."
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


async def apply_code_patch(
    request: CodePatchApplyRequest,
) -> CodePatchApplyResponse:
    """Apply an explicitly approved patch without blocking the event loop."""
    return await asyncio.to_thread(_apply_code_patch_sync, request)