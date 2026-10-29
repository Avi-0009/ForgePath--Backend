import os
import stat
import hashlib

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Sequence
from app.core.config import settings


DEFAULT_MAX_FILE_BYTES = 1_000_000  # 1 MB

EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
}

SENSITIVE_FILENAMES = {
    "credentials.json",
    "secrets.json",
    "service-account.json",
    "id_rsa",
    "id_ed25519",
    "pgadmin-servers.json",
}

SENSITIVE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".pyi": "python",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".h": "c",
    ".c": "c",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sh": "shell",
    ".ps1": "powershell",
}


class WorkspaceReadError(Exception):
    """Base exception for safe workspace-reading failures."""


class WorkspaceAccessDenied(WorkspaceReadError):
    """The requested workspace or path is not permitted."""


class WorkspaceInvalidPath(WorkspaceReadError):
    """The supplied relative path is invalid."""


class WorkspaceFileNotFound(WorkspaceReadError):
    """The requested file does not exist."""


class WorkspaceFileTooLarge(WorkspaceReadError):
    """The requested file exceeds the permitted size."""


class WorkspaceBinaryFile(WorkspaceReadError):
    """The selected file is not valid UTF-8 text."""


class WorkspaceConfigurationError(WorkspaceReadError):
    """The configured workspace allowlist is invalid."""


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    """Read-only result containing the selected file's contents."""

    relative_path: str
    language: str | None
    content: str
    sha256: str
    size_bytes: int
    line_count: int


def _resolve_directory(path: str | Path) -> Path | None:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None

    return resolved if resolved.is_dir() else None


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Compatibility helper for path-containment checks."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_relative_path(relative_path: str) -> tuple[str, ...]:
    if not relative_path or not relative_path.strip():
        raise WorkspaceInvalidPath("A relative file path is required.")

    if "\x00" in relative_path:
        raise WorkspaceInvalidPath("The file path is invalid.")

    # Normalize Windows separators so paths are handled consistently.
    normalized = relative_path.strip().replace("\\", "/")
    windows_path = PureWindowsPath(relative_path)

    if (
        normalized.startswith("/")
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    ):
        raise WorkspaceInvalidPath(
            "Provide a relative path inside the selected workspace."
        )

    parts = tuple(normalized.split("/"))

    if any(part in {"", ".", ".."} for part in parts):
        raise WorkspaceInvalidPath(
            "The file path contains an invalid component."
        )

    if any(part.lower() in EXCLUDED_DIRECTORIES for part in parts):
        raise WorkspaceAccessDenied(
            "Files in excluded directories cannot be reviewed."
        )

    filename = parts[-1].lower()

    # Block environment files, including environment-specific variants.
    if filename == ".env" or filename.startswith(".env."):
        raise WorkspaceAccessDenied(
            "Environment files cannot be reviewed."
        )

    file_path = Path(filename)

    if (
        filename in SENSITIVE_FILENAMES
        or file_path.stem.lower() in SENSITIVE_FILENAMES
        or file_path.suffix.lower() in SENSITIVE_SUFFIXES
    ):
        raise WorkspaceAccessDenied(
            "This potentially sensitive file cannot be reviewed."
        )

    return parts


def read_workspace_file(
    *,
    workspace_root: str | Path,
    relative_path: str,
    allowed_workspace_roots: Sequence[str | Path],
    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> WorkspaceFile:
    """
    Read a UTF-8 text file from an explicitly allowlisted workspace.

    This function never writes to the source file. The workspace root
    must match one of the configured allowlisted roots exactly.
    """

    if max_bytes < 1:
        raise ValueError("max_bytes must be greater than zero.")

    root = _resolve_directory(workspace_root)

    if root is None:
        raise WorkspaceAccessDenied(
            "The selected workspace is unavailable or invalid."
        )

    # Require explicit approval for each workspace root.
    if not allowed_workspace_roots:
        raise WorkspaceConfigurationError(
            "No allowed workspace roots are configured."
        )

    allowed_roots: set[Path] = set()

    for candidate in allowed_workspace_roots:
        resolved = _resolve_directory(candidate)

        if resolved is None:
            raise WorkspaceConfigurationError(
                "A configured workspace root is invalid or unavailable. "
                "Check WORKSPACE_ALLOWED_ROOTS in your backend configuration."
            )

        allowed_roots.add(resolved)

    if root not in allowed_roots:
        raise WorkspaceAccessDenied(
            "The selected workspace is not allowlisted."
        )

    parts = _validate_relative_path(relative_path)
    target = root.joinpath(*parts)

    try:
        # Resolve symlinks before checking containment.
        resolved_target = target.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise WorkspaceFileNotFound(
            "The requested file does not exist."
        ) from None

    # Reject symlinks that point outside the permitted workspace.
    if not _is_relative_to(resolved_target, root):
        raise WorkspaceAccessDenied(
            "The requested file is outside the selected workspace."
        )

    if not resolved_target.is_file():
        raise WorkspaceInvalidPath(
            "The selected path is not a regular file."
        )

    # Re-check exclusions against the resolved relative path.
    resolved_parts = resolved_target.relative_to(root).parts
    _validate_relative_path("/".join(resolved_parts))

    
    try:
        with resolved_target.open("rb") as file:
            # Inspect the file descriptor actually opened.
            opened_stat = os.fstat(file.fileno())

            if not stat.S_ISREG(opened_stat.st_mode):
                raise WorkspaceInvalidPath(
                    "The selected path is not a regular file."
                )

            # Re-resolve and validate the path after opening.
            try:
                opened_path = resolved_target.resolve(strict=True)

                if not _is_relative_to(opened_path, root):
                    raise WorkspaceAccessDenied(
                        "The opened file is outside the selected workspace."
                    )

                opened_parts = opened_path.relative_to(root).parts
                _validate_relative_path("/".join(opened_parts))

                path_stat = opened_path.stat()

            except WorkspaceReadError:
                raise
            except (OSError, RuntimeError, ValueError):
                raise WorkspaceAccessDenied(
                    "The file path changed during validation. Retry the request."
                ) from None

            # Ensure the pathname still identifies the opened file.
            if (
                opened_stat.st_dev != path_stat.st_dev
                or opened_stat.st_ino != path_stat.st_ino
            ):
                raise WorkspaceAccessDenied(
                    "The file changed during validation. Retry the request."
                )

            if opened_stat.st_size > max_bytes:
                raise WorkspaceFileTooLarge(
                    f"Files larger than {max_bytes} bytes are not supported."
                )

            # Read only after the post-open checks have passed.
            raw_content = file.read(max_bytes + 1)

    except WorkspaceReadError:
        raise
    except OSError:
        raise WorkspaceReadError(
            "The requested file could not be read."
        ) from None


    if len(raw_content) > max_bytes:
        raise WorkspaceFileTooLarge(
            f"Files larger than {max_bytes} bytes are not supported."
        )

    if b"\x00" in raw_content:
        raise WorkspaceBinaryFile(
            "Binary files cannot be reviewed as source code."
        )

    try:
        content = raw_content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise WorkspaceBinaryFile(
            "The selected file is not valid UTF-8 text."
        ) from None

    suffix = resolved_target.suffix.lower()

    return WorkspaceFile(
        relative_path=resolved_target.relative_to(root).as_posix(),
        language=LANGUAGE_BY_EXTENSION.get(suffix),
        content=content,
        sha256=hashlib.sha256(raw_content).hexdigest(),
        size_bytes=len(raw_content),
        line_count=len(content.splitlines()),
    )

def read_configured_workspace_file(
    *,
    workspace_root: str | Path,
    relative_path: str,
    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> WorkspaceFile:
    """Read a file using the roots configured in backend settings."""

    allowed_roots = settings.workspace_root_list

    if not allowed_roots:
        raise WorkspaceAccessDenied(
            "No workspace roots are configured."
        )

    return read_workspace_file(
        workspace_root=workspace_root,
        relative_path=relative_path,
        allowed_workspace_roots=allowed_roots,
        max_bytes=max_bytes,
    )