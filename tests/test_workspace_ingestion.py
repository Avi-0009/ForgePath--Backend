import pytest

from app.core.config import settings
from app.services.workspace import WorkspaceAccessDenied
from app.services.workspace_ingestion import (
    ingest_workspace,
)


@pytest.fixture
def allowed_workspace(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(root),
    )

    return root


def test_ingests_supported_workspace_files(allowed_workspace):
    root = allowed_workspace

    (root / "main.py").write_bytes(
        b"print('hello')\n"
    )
    (root / "README.md").write_text(
        "# Project\n",
        encoding="utf-8",
    )
    (root / "settings.toml").write_text(
        'name = "ForgePath"\n',
        encoding="utf-8",
    )

    result = ingest_workspace(root)

    assert result.workspace_root == str(root)
    assert result.files_scanned == 3
    assert result.files_skipped == 0
    assert result.total_bytes > 0
    assert result.truncated is False

    documents = {
        document.relative_path: document
        for document in result.documents
    }

    assert set(documents) == {
        "main.py",
        "README.md",
        "settings.toml",
    }

    python_document = documents["main.py"]

    assert python_document.language == "python"
    assert python_document.content == "print('hello')\n"
    assert python_document.sha256
    assert python_document.line_count == 1
    assert python_document.workspace_root == str(root)


def test_excludes_sensitive_generated_and_unsupported_files(
    allowed_workspace,
):
    root = allowed_workspace

    (root / "main.py").write_text(
        "print('safe')\n",
        encoding="utf-8",
    )
    (root / "notes.txt").write_text(
        "Not a supported source format",
        encoding="utf-8",
    )
    (root / "package-lock.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (root / ".env").write_text(
        "SECRET=value",
        encoding="utf-8",
    )
    (root / ".env.local").write_text(
        "SECRET=value",
        encoding="utf-8",
    )
    (root / "credentials.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (root / "private.pem").write_text(
        "private key",
        encoding="utf-8",
    )
    (root / "pgadmin-servers.json").write_text(
        '{"Servers": {}}',
        encoding="utf-8",
    )

    for directory_name in (
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        "dist",
    ):
        directory = root / directory_name
        directory.mkdir()

        (directory / "generated.py").write_text(
            "print('must not be ingested')",
            encoding="utf-8",
        )

    result = ingest_workspace(root)

    visible_paths = {
        document.relative_path
        for document in result.documents
    }

    assert visible_paths == {"main.py"}
    assert result.files_skipped >= 7

    assert all(
        "node_modules" not in path
        and ".git" not in path
        and ".venv" not in path
        and "__pycache__" not in path
        and "dist" not in path
        for path in visible_paths
    )


def test_rejects_workspace_outside_allowlist(
    allowed_workspace,
    tmp_path,
):
    outside_workspace = tmp_path / "outside"
    outside_workspace.mkdir()

    (outside_workspace / "main.py").write_text(
        "print('private')",
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceAccessDenied):
        ingest_workspace(outside_workspace)


def test_respects_maximum_file_count(
    allowed_workspace,
):
    root = allowed_workspace

    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    (root / "b.py").write_text("b = 2\n", encoding="utf-8")

    result = ingest_workspace(
        root,
        max_files=1,
    )

    assert len(result.documents) == 1
    assert result.documents[0].relative_path == "a.py"
    assert result.files_scanned == 1
    assert result.truncated is True


def test_respects_total_ingestion_size(
    allowed_workspace,
):
    root = allowed_workspace

    first_content = b"a = 1\n"
    second_content = b"b = 222\n"

    (root / "a.py").write_bytes(first_content)
    (root / "b.py").write_bytes(second_content)

    result = ingest_workspace(
        root,
        max_total_bytes=len(first_content),
    )

    assert len(result.documents) == 1
    assert result.documents[0].relative_path == "a.py"
    assert result.total_bytes == len(first_content)
    assert result.truncated is True


def test_skips_files_exceeding_individual_size_limit(
    allowed_workspace,
):
    root = allowed_workspace

    (root / "large.py").write_text(
        "x = " + ("1" * 100),
        encoding="utf-8",
    )
    (root / "small.py").write_text(
        "x = 1\n",
        encoding="utf-8",
    )

    result = ingest_workspace(
        root,
        max_file_bytes=20,
    )

    visible_paths = {
        document.relative_path
        for document in result.documents
    }

    assert visible_paths == {"small.py"}
    assert result.files_skipped == 1
    assert result.truncated is False


def test_respects_maximum_directory_depth(
    allowed_workspace,
    monkeypatch,
):
    root = allowed_workspace

    source_dir = root / "src"
    source_dir.mkdir()

    (source_dir / "main.py").write_text(
        "print('hello')",
        encoding="utf-8",
    )

    import app.services.workspace_ingestion as ingestion

    monkeypatch.setattr(
        ingestion,
        "MAX_INGESTION_DEPTH",
        0,
    )

    result = ingestion.ingest_workspace(root)

    assert result.documents == ()
    assert result.truncated is True


def test_skips_binary_source_files(
    allowed_workspace,
):
    root = allowed_workspace

    (root / "binary.py").write_bytes(
        b"\x00\x01\x02\x03",
    )
    (root / "valid.py").write_text(
        "print('valid')\n",
        encoding="utf-8",
    )

    result = ingest_workspace(root)

    visible_paths = {
        document.relative_path
        for document in result.documents
    }

    assert visible_paths == {"valid.py"}
    assert result.files_skipped == 1


def test_does_not_ingest_symbolic_links(
    allowed_workspace,
    tmp_path,
):
    root = allowed_workspace

    outside_file = tmp_path / "outside.py"
    outside_file.write_text(
        "print('outside workspace')",
        encoding="utf-8",
    )

    link = root / "linked.py"

    try:
        link.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic link creation is unavailable.")

    result = ingest_workspace(root)

    assert result.documents == ()
    assert result.files_scanned == 0


def test_skips_empty_and_whitespace_only_files(allowed_workspace):
    root = allowed_workspace

    (root / "empty.py").write_bytes(b"")
    (root / "whitespace.py").write_bytes(b" \t\r\n")

    result = ingest_workspace(root)

    assert result.documents == ()
    assert result.files_scanned == 2
    assert result.files_skipped == 2
    assert result.total_bytes == 0
    assert result.truncated is False
