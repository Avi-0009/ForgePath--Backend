import asyncio
from types import SimpleNamespace

import pytest

from app.services.workspace import WorkspaceFileNotFound
from app.services.workspace_review_context import (
    WorkspaceReviewContextBuilder,
)


WORKSPACE_ROOT = "C:/projects/forgepath"
SELECTED_FILE = "app/main.py"
VALID_HASH = "a" * 64


class FakeIndexingService:
    """Mock semantic indexing service."""

    def __init__(self, hits=None):
        self.hits = hits or []
        self.calls = []

    async def search(self, query, **kwargs):
        self.calls.append({
            "query": query,
            **kwargs,
        })

        return self.hits


def make_hit(
    *,
    path="app/auth.py",
    content="def authenticate():\n    return True\n",
    workspace_root=WORKSPACE_ROOT,
    document_sha256=VALID_HASH,
    chunk_index=0,
    start_line=1,
    end_line=2,
    language="python",
    distance=0.1,
):
    return SimpleNamespace(
        document=content,
        distance=distance,
        metadata={
            "workspace_root": workspace_root,
            "relative_path": path,
            "document_sha256": document_sha256,
            "chunk_index": chunk_index,
            "start_line": start_line,
            "end_line": end_line,
            "language": language,
        },
    )


def make_builder(
    hits=None,
    *,
    top_k=5,
    max_context_chars=12_000,
):
    indexing_service = FakeIndexingService(hits)

    builder = WorkspaceReviewContextBuilder(
        indexing_service,
        top_k=top_k,
        max_context_chars=max_context_chars,
    )

    return builder, indexing_service


def build_context(builder, **overrides):
    arguments = {
        "workspace_root": WORKSPACE_ROOT,
        "relative_path": SELECTED_FILE,
        "source_content": "def main():\n    return authenticate()\n",
        "language": "python",
    }

    arguments.update(overrides)

    return asyncio.run(builder.build(**arguments))


# ─────────────────────────────────────────────
# SUCCESSFUL RETRIEVAL
# ─────────────────────────────────────────────

def test_build_returns_verified_workspace_context(
    monkeypatch,
):
    hit = make_hit()

    builder, indexing_service = make_builder([hit])

    read_calls = []

    def fake_read(**kwargs):
        read_calls.append(kwargs)

        return SimpleNamespace(
            sha256=VALID_HASH,
        )

    monkeypatch.setattr(
        "app.services.workspace_review_context.read_configured_workspace_file",
        fake_read,
    )

    result = build_context(builder)

    assert result.chunk_count == 1
    assert result.is_empty is False

    assert result.chunks[0].relative_path == "app/auth.py"
    assert result.chunks[0].language == "python"
    assert result.chunks[0].start_line == 1
    assert result.chunks[0].end_line == 2

    assert result.chunks[0].content == hit.document
    assert result.stale_chunks_skipped == 0
    assert result.unavailable_chunks_skipped == 0

    assert result.total_chars == len(hit.document)

    # Verify that retrieval was restricted to this workspace.
    assert indexing_service.calls[0]["workspace_root"] == WORKSPACE_ROOT

    # Verify that the selected source file was excluded.
    assert (
        indexing_service.calls[0]["exclude_relative_path"]
        == SELECTED_FILE
    )

    assert len(read_calls) == 1


# ─────────────────────────────────────────────
# STALE INDEXED CONTENT
# ─────────────────────────────────────────────

def test_build_skips_stale_indexed_chunks(
    monkeypatch,
):
    hit = make_hit(
        document_sha256="b" * 64,
    )

    builder, _ = make_builder([hit])

    def fake_read(**kwargs):
        return SimpleNamespace(
            sha256=VALID_HASH,
        )

    monkeypatch.setattr(
        "app.services.workspace_review_context.read_configured_workspace_file",
        fake_read,
    )

    result = build_context(builder)

    assert result.chunk_count == 0
    assert result.is_empty is True

    assert result.stale_chunks_skipped == 1
    assert result.unavailable_chunks_skipped == 0


# ─────────────────────────────────────────────
# MISSING SOURCE FILE
# ─────────────────────────────────────────────

def test_build_skips_unavailable_source_file(
    monkeypatch,
):
    hit = make_hit()

    builder, _ = make_builder([hit])

    def fake_read(**kwargs):
        raise WorkspaceFileNotFound(
            "The requested file does not exist."
        )

    monkeypatch.setattr(
        "app.services.workspace_review_context.read_configured_workspace_file",
        fake_read,
    )

    result = build_context(builder)

    assert result.chunk_count == 0
    assert result.unavailable_chunks_skipped == 1
    assert result.stale_chunks_skipped == 0


# ─────────────────────────────────────────────
# WORKSPACE ISOLATION
# ─────────────────────────────────────────────

def test_build_rejects_chunks_from_another_workspace(
    monkeypatch,
):
    hit = make_hit(
        workspace_root="D:/another-project",
    )

    builder, _ = make_builder([hit])

    def fake_read(**kwargs):
        pytest.fail(
            "The source file must not be read for an untrusted workspace result."
        )

    monkeypatch.setattr(
        "app.services.workspace_review_context.read_configured_workspace_file",
        fake_read,
    )

    result = build_context(builder)

    assert result.chunk_count == 0
    assert result.unavailable_chunks_skipped == 1


# ─────────────────────────────────────────────
# SELECTED FILE EXCLUSION
# ─────────────────────────────────────────────

def test_build_rejects_selected_file_even_if_search_returns_it(
    monkeypatch,
):
    hit = make_hit(
        path=SELECTED_FILE,
    )

    builder, _ = make_builder([hit])

    def fake_read(**kwargs):
        pytest.fail(
            "The selected source file must not be read as reference context."
        )

    monkeypatch.setattr(
        "app.services.workspace_review_context.read_configured_workspace_file",
        fake_read,
    )

    result = build_context(builder)

    assert result.chunk_count == 0
    assert result.unavailable_chunks_skipped == 1


# ─────────────────────────────────────────────
# CONTEXT SIZE LIMIT
# ─────────────────────────────────────────────

def test_build_respects_max_context_characters(
    monkeypatch,
):
    first_hit = make_hit(
        path="app/auth.py",
        content="A" * 7,
        chunk_index=0,
        start_line=1,
        end_line=1,
    )

    second_hit = make_hit(
        path="app/database.py",
        content="B" * 7,
        chunk_index=0,
        start_line=1,
        end_line=1,
    )

    builder, _ = make_builder(
        [first_hit, second_hit],
        max_context_chars=10,
    )

    def fake_read(**kwargs):
        return SimpleNamespace(
            sha256=VALID_HASH,
        )

    monkeypatch.setattr(
        "app.services.workspace_review_context.read_configured_workspace_file",
        fake_read,
    )

    result = build_context(builder)

    assert result.chunk_count == 1
    assert result.total_chars == 7

    assert result.chunks[0].relative_path == "app/auth.py"


# ─────────────────────────────────────────────
# DUPLICATE FILE VALIDATION
# ─────────────────────────────────────────────

def test_build_validates_same_file_only_once(
    monkeypatch,
):
    first_hit = make_hit(
        path="app/auth.py",
        chunk_index=0,
        content="def authenticate():\n    pass\n",
    )

    second_hit = make_hit(
        path="app/auth.py",
        chunk_index=1,
        content="def validate_token():\n    pass\n",
        start_line=10,
        end_line=11,
    )

    builder, _ = make_builder([
        first_hit,
        second_hit,
    ])

    read_calls = []

    def fake_read(**kwargs):
        read_calls.append(kwargs)

        return SimpleNamespace(
            sha256=VALID_HASH,
        )

    monkeypatch.setattr(
        "app.services.workspace_review_context.read_configured_workspace_file",
        fake_read,
    )

    result = build_context(builder)

    assert result.chunk_count == 2

    # Same file and indexed document hash: one disk read.
    assert len(read_calls) == 1


# ─────────────────────────────────────────────
# INDEXING DISABLED
# ─────────────────────────────────────────────

def test_build_returns_empty_context_when_indexing_is_disabled():
    builder = WorkspaceReviewContextBuilder(
        indexing_service=None,
    )

    result = build_context(builder)

    assert result.is_empty is True
    assert result.chunk_count == 0
    assert result.total_chars == 0


# ─────────────────────────────────────────────
# EMPTY SOURCE
# ─────────────────────────────────────────────

def test_build_returns_empty_context_for_empty_source():
    builder, indexing_service = make_builder([])

    result = build_context(
        builder,
        source_content="   ",
    )

    assert result.is_empty is True
    assert result.chunk_count == 0
    assert result.total_chars == 0

    # No need to perform semantic retrieval.
    assert indexing_service.calls == []


# ─────────────────────────────────────────────
# CONTEXT SERIALIZATION
# ─────────────────────────────────────────────

def test_context_serializes_verified_chunks_as_json(
    monkeypatch,
):
    hit = make_hit()

    builder, _ = make_builder([hit])

    monkeypatch.setattr(
        "app.services.workspace_review_context.read_configured_workspace_file",
        lambda **kwargs: SimpleNamespace(
            sha256=VALID_HASH,
        ),
    )

    result = build_context(builder)

    serialized = result.to_prompt_text()

    assert '"relative_path": "app/auth.py"' in serialized
    assert '"language": "python"' in serialized
    assert '"content":' in serialized