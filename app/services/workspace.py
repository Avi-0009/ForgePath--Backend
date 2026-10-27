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

    if (
        filename in SENSITIVE_FILENAMES
        or Path(filename).suffix.lower() in SENSITIVE_SUFFIXES
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
    allowed_roots = {
        resolved
        for candidate in allowed_workspace_roots
        if (resolved := _resolve_directory(candidate)) is not None
    }

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
        if resolved_target.stat().st_size > max_bytes:
            raise WorkspaceFileTooLarge(
                f"Files larger than {max_bytes} bytes are not supported."
            )

        # Read no more than max_bytes + 1, even if the file grows.
        with resolved_target.open("rb") as file:
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