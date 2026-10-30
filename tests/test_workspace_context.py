import hashlib

import pytest

from app.core.config import settings
from app.services.workspace_context import build_workspace_context


@pytest.fixture
def allowed_workspace(tmp_path, monkeypatch):
    root = tmp_path / "forgepath"
    root.mkdir()

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(root),
    )

    return root


def test_builds_context_from_multiple_documents(allowed_workspace):
    root = allowed_workspace

    python_content = b"def add(a, b):\n    return a + b\n"
    markdown_content = b"# ForgePath\n\nLocal AI coding assistant.\n"

    (root / "main.py").write_bytes(python_content)

    (root / "README.md").write_bytes(markdown_content)

    context = build_workspace_context(str(root))

    assert context.workspace_root == str(root)
    assert context.document_count == 2
    assert context.chunk_count == 2

    assert context.files_scanned == 2
    assert context.files_skipped == 0

    assert context.total_bytes == (
        len(python_content) + len(markdown_content)
    )

    assert context.truncated is False

    documents = {
        document.relative_path: document
        for document in context.documents
    }

    assert set(documents) == {"main.py", "README.md"}

    assert documents["main.py"].language == "python"
    assert documents["README.md"].language == "markdown"

    assert documents["main.py"].sha256 == hashlib.sha256(
        python_content
    ).hexdigest()


def test_preserves_chunk_provenance_and_line_metadata(
    allowed_workspace,
):
    root = allowed_workspace

    content = (
        b"def first():\n"
        b"    return 1\n"
        b"\n"
        b"def second():\n"
        b"    return 2\n"
    )

    (root / "example.py").write_bytes(content)

    context = build_workspace_context(
        str(root),
        max_chunk_chars=25,
        overlap_chars=0,
    )

    assert context.document_count == 1
    assert context.chunk_count >= 2

    document = context.documents[0]

    for index, chunk in enumerate(context.chunks):
        assert chunk.workspace_root == str(root)
        assert chunk.relative_path == "example.py"
        assert chunk.language == "python"

        assert chunk.chunk_index == index
        assert chunk.document_sha256 == document.sha256
        assert chunk.chunk_sha256 == hashlib.sha256(
            chunk.content.encode("utf-8")
        ).hexdigest()

        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_propagates_ingestion_resource_limits(
    allowed_workspace,
):
    root = allowed_workspace

    (root / "a.py").write_bytes(b"a = 1\n")
    (root / "b.py").write_bytes(b"b = 2\n")

    context = build_workspace_context(
        str(root),
        max_files=1,
    )

    assert context.document_count == 1
    assert context.files_scanned == 1
    assert context.truncated is True


def test_empty_workspace_returns_empty_context(
    allowed_workspace,
):
    context = build_workspace_context(str(allowed_workspace))

    assert context.document_count == 0
    assert context.chunk_count == 0

    assert context.documents == ()
    assert context.chunks == ()

    assert context.files_scanned == 0
    assert context.files_skipped == 0
    assert context.total_bytes == 0

    assert context.truncated is False


@pytest.mark.parametrize(
    ("max_chunk_chars", "overlap_chars"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_rejects_invalid_chunk_configuration(
    allowed_workspace,
    max_chunk_chars,
    overlap_chars,
):
    with pytest.raises(ValueError):
        build_workspace_context(
            str(allowed_workspace),
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )


def test_preserves_skipped_file_metadata(
    allowed_workspace,
):
    root = allowed_workspace

    (root / "main.py").write_bytes(b"print('hello')\n")
    (root / "notes.txt").write_bytes(b"Unsupported format")

    context = build_workspace_context(str(root))

    assert context.document_count == 1
    assert context.files_scanned == 1
    assert context.files_skipped == 1

    assert context.documents[0].relative_path == "main.py"