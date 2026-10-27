
import hashlib

import pytest

from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceBinaryFile,
    WorkspaceFileTooLarge,
    WorkspaceInvalidPath,
    read_workspace_file,
)


def test_reads_file_and_returns_metadata(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    source = root / "solution.py"
    content = "def add(a, b):\n    return a + b\n"
    source.write_bytes(content.encode("utf-8"))

    result = read_workspace_file(
        workspace_root=root,
        relative_path="solution.py",
        allowed_workspace_roots=[root],
    )

    assert result.content == content
    assert result.relative_path == "solution.py"
    assert result.language == "python"
    assert result.line_count == 2
    assert result.size_bytes == len(content.encode("utf-8"))
    assert result.sha256 == hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    # Reading must not modify the original file.
    assert source.read_text(encoding="utf-8") == content


def test_rejects_workspace_not_in_allowlist(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    other = tmp_path / "other"
    other.mkdir()

    (project / "main.py").write_text(
        "print('hello')",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceAccessDenied):
        read_workspace_file(
            workspace_root=project,
            relative_path="main.py",
            allowed_workspace_roots=[other],
        )


def test_rejects_parent_directory_traversal(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    with pytest.raises(WorkspaceInvalidPath):
        read_workspace_file(
            workspace_root=root,
            relative_path="../secret.txt",
            allowed_workspace_roots=[root],
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        ".env",
        ".env.production",
        ".git/config",
        ".venv/pyvenv.cfg",
        "node_modules/package.json",
        "credentials.json",
        "private.key",
    ],
)
def test_rejects_sensitive_or_excluded_paths(
    tmp_path,
    relative_path,
):
    root = tmp_path / "project"
    root.mkdir()

    with pytest.raises(WorkspaceAccessDenied):
        read_workspace_file(
            workspace_root=root,
            relative_path=relative_path,
            allowed_workspace_roots=[root],
        )


def test_rejects_oversized_file(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    (root / "large.py").write_text(
        "x" * 100,
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceFileTooLarge):
        read_workspace_file(
            workspace_root=root,
            relative_path="large.py",
            allowed_workspace_roots=[root],
            max_bytes=20,
        )


def test_rejects_binary_file(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    (root / "binary.dat").write_bytes(b"\x00\x01\x02")

    with pytest.raises(WorkspaceBinaryFile):
        read_workspace_file(
            workspace_root=root,
            relative_path="binary.dat",
            allowed_workspace_roots=[root],
        )


def test_rejects_file_outside_workspace_via_symlink(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    outside = tmp_path / "secret.txt"
    outside.write_text("private", encoding="utf-8")

    link = root / "linked.txt"

    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable on this system.")

    with pytest.raises(WorkspaceAccessDenied):
        read_workspace_file(
            workspace_root=root,
            relative_path="linked.txt",
            allowed_workspace_roots=[root],
        )
