import hashlib
import pytest

from app.services.workspace_chunking import WorkspaceChunk
from app.services.vector_store import (
    ChromaVectorStore,
    VectorStoreConfigurationError,
    create_chunk_id,
)


@pytest.fixture
def vector_store(tmp_path):
    store = ChromaVectorStore(
        dimensions=3,
        embedding_model="fake-embedding-model",
        persist_directory=tmp_path / "chroma",
    )

    yield store

    store.close()


def make_chunk(
    *,
    workspace_root="C:/projects/forgepath",
    relative_path="app/main.py",
    language="python",
    chunk_index=0,
    content="def main():\n    return 42\n",
    start_line=1,
    end_line=2,
):
    content_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    return WorkspaceChunk(
        workspace_root=workspace_root,
        relative_path=relative_path,
        language=language,
        chunk_index=chunk_index,
        start_line=start_line,
        end_line=end_line,
        content=content,
        document_sha256="a" * 64,
        chunk_sha256=content_hash,
    )


def test_initializes_empty_persistent_collection(vector_store):
    assert vector_store.count() == 0
    assert vector_store.dimensions == 3
    assert vector_store.embedding_model == "fake-embedding-model"


def test_creates_deterministic_chunk_ids():
    chunk = make_chunk()

    first_id = create_chunk_id(chunk)
    second_id = create_chunk_id(chunk)

    assert first_id == second_id
    assert len(first_id) == 64


def test_upserts_workspace_chunks(vector_store):
    chunks = [
        make_chunk(
            relative_path="app/main.py",
            chunk_index=0,
        ),
        make_chunk(
            relative_path="app/auth.py",
            chunk_index=0,
            content="def authenticate():\n    pass\n",
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    inserted = vector_store.upsert_chunks(
        chunks,
        embeddings,
    )

    assert inserted == 2
    assert vector_store.count() == 2


def test_search_returns_document_and_metadata(vector_store):
    chunk = make_chunk()

    vector_store.upsert_chunks(
        [chunk],
        [[1.0, 0.0, 0.0]],
    )

    results = vector_store.search(
        [1.0, 0.0, 0.0],
        top_k=5,
    )

    assert len(results) == 1

    result = results[0]

    assert result.document == chunk.content
    assert result.metadata["relative_path"] == "app/main.py"
    assert result.metadata["workspace_root"] == (
        "C:/projects/forgepath"
    )
    assert result.metadata["language"] == "python"
    assert result.metadata["start_line"] == 1
    assert result.metadata["end_line"] == 2
    assert result.metadata["embedding_model"] == (
        "fake-embedding-model"
    )

    assert result.distance is not None
    assert result.distance == pytest.approx(0.0)


def test_filters_search_by_workspace_and_language(vector_store):
    chunks = [
        make_chunk(
            workspace_root="C:/projects/forgepath",
            relative_path="app/main.py",
            language="python",
            content="def main(): pass",
        ),
        make_chunk(
            workspace_root="C:/projects/other",
            relative_path="app/main.py",
            language="python",
            content="def other(): pass",
        ),
        make_chunk(
            workspace_root="C:/projects/forgepath",
            relative_path="src/main.ts",
            language="typescript",
            content="function main() {}",
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
    ]

    vector_store.upsert_chunks(chunks, embeddings)

    results = vector_store.search(
        [1.0, 0.0, 0.0],
        workspace_root="C:/projects/forgepath",
        language="python",
        top_k=10,
    )

    assert len(results) == 1
    assert results[0].metadata["relative_path"] == "app/main.py"
    assert results[0].metadata["language"] == "python"


def test_upsert_updates_existing_chunk_instead_of_duplicating(
    vector_store,
):
    original = make_chunk(
        content="def calculate(): return 1",
    )

    updated = make_chunk(
        content="def calculate(): return 2",
    )

    first_id = create_chunk_id(original)
    second_id = create_chunk_id(updated)

    assert first_id == second_id

    vector_store.upsert_chunks(
        [original],
        [[1.0, 0.0, 0.0]],
    )

    vector_store.upsert_chunks(
        [updated],
        [[0.0, 1.0, 0.0]],
    )

    assert vector_store.count() == 1

    results = vector_store.search(
        [0.0, 1.0, 0.0],
    )

    assert results[0].document == updated.content
    assert results[0].metadata["chunk_sha256"] == (
        updated.chunk_sha256
    )


def test_rejects_mismatched_chunk_and_embedding_counts(
    vector_store,
):
    with pytest.raises(ValueError):
        vector_store.upsert_chunks(
            [make_chunk()],
            [],
        )

    assert vector_store.count() == 0


def test_rejects_invalid_embedding_dimensions(vector_store):
    with pytest.raises(ValueError):
        vector_store.upsert_chunks(
            [make_chunk()],
            [[1.0, 0.0]],
        )

    assert vector_store.count() == 0


def test_rejects_empty_chunks(vector_store):
    empty_chunk = make_chunk(content="   ")

    with pytest.raises(ValueError):
        vector_store.upsert_chunks(
            [empty_chunk],
            [[1.0, 0.0, 0.0]],
        )

    assert vector_store.count() == 0


def test_deletes_chunks_for_one_file(vector_store):
    chunks = [
        make_chunk(
            relative_path="app/main.py",
            chunk_index=0,
        ),
        make_chunk(
            relative_path="app/main.py",
            chunk_index=1,
            content="def second(): pass",
        ),
        make_chunk(
            relative_path="app/auth.py",
            content="def authenticate(): pass",
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    vector_store.upsert_chunks(chunks, embeddings)

    deleted = vector_store.delete_file_chunks(
        workspace_root="C:/projects/forgepath",
        relative_path="app/main.py",
    )

    assert deleted == 2
    assert vector_store.count() == 1

    remaining = vector_store.search(
        [0.0, 0.0, 1.0],
    )

    assert remaining[0].metadata["relative_path"] == "app/auth.py"


def test_deletes_only_selected_workspace(vector_store):
    chunks = [
        make_chunk(
            workspace_root="C:/projects/forgepath",
            relative_path="main.py",
        ),
        make_chunk(
            workspace_root="C:/projects/other",
            relative_path="main.py",
        ),
    ]

    vector_store.upsert_chunks(
        chunks,
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    deleted = vector_store.delete_workspace(
        "C:/projects/forgepath",
    )

    assert deleted == 1
    assert vector_store.count() == 1


def test_persists_vectors_after_reopening_client(tmp_path):
    database_path = tmp_path / "persistent-chroma"

    first_store = ChromaVectorStore(
        dimensions=3,
        embedding_model="fake-embedding-model",
        persist_directory=database_path,
    )

    chunk = make_chunk()

    first_store.upsert_chunks(
        [chunk],
        [[1.0, 0.0, 0.0]],
    )

    first_store.close()

    second_store = ChromaVectorStore(
        dimensions=3,
        embedding_model="fake-embedding-model",
        persist_directory=database_path,
    )

    try:
        assert second_store.count() == 1

        results = second_store.search(
            [1.0, 0.0, 0.0],
        )

        assert len(results) == 1
        assert results[0].document == chunk.content

    finally:
        second_store.close()


def test_rejects_different_model_for_existing_collection(
    tmp_path,
):
    database_path = tmp_path / "model-check"

    first_store = ChromaVectorStore(
        dimensions=3,
        embedding_model="model-a",
        persist_directory=database_path,
    )

    first_store.close()

    with pytest.raises(VectorStoreConfigurationError):
        ChromaVectorStore(
            dimensions=3,
            embedding_model="model-b",
            persist_directory=database_path,
        )


def test_close_is_safe_to_call_multiple_times(vector_store):
    vector_store.close()
    vector_store.close()

    assert vector_store._closed is True


def test_search_rejects_relative_path_without_workspace(
    vector_store,
):
    with pytest.raises(ValueError):
        vector_store.search(
            [1.0, 0.0, 0.0],
            relative_path="app/main.py",
        )


def test_empty_collection_search_returns_no_results(vector_store):
    results = vector_store.search(
        [1.0, 0.0, 0.0],
    )

    assert results == []


def test_rejects_invalid_vector_values(vector_store):
    with pytest.raises(ValueError):
        vector_store.upsert_chunks(
            [make_chunk()],
            [[1.0, float("nan"), 0.0]],
        )

    assert vector_store.count() == 0