import hashlib

import pytest

from app.services.workspace_ingestion import (
    IngestedWorkspaceDocument,
)
from app.services.workspace_chunking import chunk_document


def make_document(content: str) -> IngestedWorkspaceDocument:
    raw = content.encode("utf-8")

    return IngestedWorkspaceDocument(
        workspace_root="C:/projects/forgepath",
        relative_path="src/example.py",
        language="python",
        content=content,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        line_count=len(content.splitlines()),
    )


def test_empty_document_returns_no_chunks():
    document = make_document("")

    chunks = chunk_document(document)

    assert chunks == ()


def test_small_document_returns_one_chunk():
    content = "def add(a, b):\n    return a + b\n"
    document = make_document(content)

    chunks = chunk_document(document)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.content == content
    assert chunk.chunk_index == 0
    assert chunk.start_line == 1
    assert chunk.end_line == 2
    assert chunk.relative_path == "src/example.py"
    assert chunk.language == "python"
    assert chunk.workspace_root == document.workspace_root
    assert chunk.document_sha256 == document.sha256

    assert chunk.chunk_sha256 == hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def test_splits_document_with_line_overlap():
    content = "alpha\nbravo\ncharlie\ndelta\n"
    document = make_document(content)

    chunks = chunk_document(
        document,
        max_chars=14,
        overlap_chars=6,
    )

    assert len(chunks) == 3

    assert chunks[0].content == "alpha\nbravo\n"
    assert chunks[1].content == "bravo\ncharlie\n"
    assert chunks[2].content == "delta\n"

    assert chunks[0].end_line == 2
    assert chunks[1].start_line == 2
    assert chunks[1].end_line == 3
    assert chunks[2].start_line == 4

    assert all(len(chunk.content) <= 14 for chunk in chunks)


def test_prefers_blank_line_before_next_declaration():
    content = (
        "def alpha():\n"
        "    return 1\n"
        "\n"
        "class Beta:\n"
        "    pass\n"
    )

    document = make_document(content)

    chunks = chunk_document(
        document,
        max_chars=35,
        overlap_chars=0,
    )

    assert len(chunks) == 2

    assert chunks[0].content == (
        "def alpha():\n"
        "    return 1\n"
    )

    assert chunks[0].end_line == 2
    assert chunks[1].start_line == 3


def test_splits_very_long_single_line():
    content = "abcdefghijklmno\n"
    document = make_document(content)

    chunks = chunk_document(
        document,
        max_chars=8,
        overlap_chars=2,
    )

    assert len(chunks) == 3

    assert all(len(chunk.content) <= 8 for chunk in chunks)

    assert chunks[0].content == "abcdefgh"
    assert chunks[1].content == "ghijklmn"
    assert chunks[2].content == "mno\n"

    assert all(
        chunk.start_line == 1 and chunk.end_line == 1
        for chunk in chunks
    )


def test_chunk_hashes_are_deterministic():
    document = make_document(
        "def add(a, b):\n    return a + b\n"
    )

    first = chunk_document(document)
    second = chunk_document(document)

    assert [chunk.chunk_sha256 for chunk in first] == [
        chunk.chunk_sha256 for chunk in second
    ]

    assert [chunk.content for chunk in first] == [
        chunk.content for chunk in second
    ]


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_rejects_invalid_chunk_configuration(
    max_chars,
    overlap_chars,
):
    document = make_document("print('hello')\n")

    with pytest.raises(ValueError):
        chunk_document(
            document,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )


def test_empty_chunks_are_not_created_for_trailing_content():
    document = make_document("first\nsecond\n")

    chunks = chunk_document(
        document,
        max_chars=100,
        overlap_chars=10,
    )

    assert len(chunks) == 1
    assert all(chunk.content for chunk in chunks)