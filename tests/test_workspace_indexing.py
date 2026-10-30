import asyncio
import hashlib
from typing import Sequence

import pytest

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import ChromaVectorStore
from app.services.workspace_indexing import (
    WorkspaceIndexingConfigurationError,
    WorkspaceIndexingService,
    get_embedding_model_identifier,
)
from app.services.workspace_ingestion import (
    IngestedWorkspaceDocument,
)


class FakeEmbeddingProvider:
    """Deterministic embedding provider for integration tests."""

    def __init__(self) -> None:
        self.fail_documents = False
        self.fail_queries = False
        self.document_batches: list[list[str]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

    @property
    def dimensions(self) -> int:
        return 3

    @staticmethod
    def _embed(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()

        return [
            int.from_bytes(digest[i:i + 4], "big") / 2**32 + 0.001
            for i in (0, 4, 8)
        ]

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:

        if self.fail_documents:
            raise RuntimeError("Fake embedding provider failure.")

        self.document_batches.append(list(texts))

        return [
            self._embed(text)
            for text in texts
        ]

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:

        if self.fail_queries:
            raise RuntimeError("Fake query embedding failure.")

        return self._embed(text)


def make_document(
    content: str,
    *,
    workspace_root: str,
    relative_path: str = "src/example.py",
    language: str = "python",
) -> IngestedWorkspaceDocument:

    raw = content.encode("utf-8")

    return IngestedWorkspaceDocument(
        workspace_root=workspace_root,
        relative_path=relative_path,
        language=language,
        content=content,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        line_count=len(content.splitlines()),
    )


@pytest.fixture
def indexing_environment(tmp_path):
    provider = FakeEmbeddingProvider()

    embedding_service = EmbeddingService(
        provider=provider,
    )

    vector_store = ChromaVectorStore(
        dimensions=embedding_service.dimensions,
        embedding_model=get_embedding_model_identifier(
            embedding_service
        ),
        persist_directory=tmp_path / "chroma",
    )

    indexing_service = WorkspaceIndexingService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    yield indexing_service, vector_store, provider

    vector_store.close()


def run(coroutine):
    """Run an async service method in a synchronous pytest test."""

    return asyncio.run(coroutine)


def test_index_single_document(
    indexing_environment,
    tmp_path,
):
    indexing_service, vector_store, _ = indexing_environment

    root = str(tmp_path / "workspace")

    document = make_document(
        "def add(a, b):\n    return a + b\n",
        workspace_root=root,
    )

    result = run(
        indexing_service.index_document(document)
    )

    assert result.files_processed == 1
    assert result.chunks_processed == 1
    assert result.vectors_upserted == 1
    assert result.stale_vectors_deleted == 0
    assert result.empty_files == 0

    assert vector_store.count() == 1


def test_index_multiple_documents(
    indexing_environment,
    tmp_path,
):
    indexing_service, vector_store, _ = indexing_environment

    root = str(tmp_path / "workspace")

    documents = [
        make_document(
            "def add(a, b):\n    return a + b\n",
            workspace_root=root,
            relative_path="src/math.py",
        ),
        make_document(
            "class User:\n    pass\n",
            workspace_root=root,
            relative_path="src/user.py",
        ),
    ]

    result = run(
        indexing_service.index_documents(documents)
    )

    assert result.files_processed == 2
    assert result.chunks_processed == 2
    assert result.vectors_upserted == 2

    assert vector_store.count() == 2


def test_semantic_search_returns_indexed_document(
    indexing_environment,
    tmp_path,
):
    indexing_service, _, _ = indexing_environment

    root = str(tmp_path / "workspace")

    content = "def calculate_total(price, tax):\n    return price + tax\n"

    document = make_document(
        content,
        workspace_root=root,
        relative_path="src/billing.py",
    )

    run(
        indexing_service.index_document(document)
    )

    hits = run(
        indexing_service.search(
            "calculate total price and tax",
            workspace_root=root,
            relative_path="src/billing.py",
            top_k=5,
        )
    )

    assert len(hits) == 1

    assert hits[0].document == content
    assert hits[0].metadata["relative_path"] == "src/billing.py"
    assert hits[0].metadata["language"] == "python"
    assert hits[0].distance is not None


def test_search_respects_workspace_filter(
    indexing_environment,
    tmp_path,
):
    indexing_service, _, _ = indexing_environment

    root_a = str(tmp_path / "workspace_a")
    root_b = str(tmp_path / "workspace_b")

    documents = [
        make_document(
            "def workspace_a_function():\n    return 1\n",
            workspace_root=root_a,
        ),
        make_document(
            "def workspace_b_function():\n    return 2\n",
            workspace_root=root_b,
        ),
    ]

    run(
        indexing_service.index_documents(documents)
    )

    hits = run(
        indexing_service.search(
            "find a function",
            workspace_root=root_a,
            top_k=10,
        )
    )

    assert len(hits) == 1
    assert hits[0].metadata["workspace_root"] == root_a


def test_reindexing_deletes_obsolete_chunks(
    indexing_environment,
    tmp_path,
):
    indexing_service, vector_store, _ = indexing_environment

    root = str(tmp_path / "workspace")

    old_content = "".join(
        f"value_{i} = {i}\n"
        for i in range(1200)
    )

    old_document = make_document(
        old_content,
        workspace_root=root,
    )

    first_result = run(
        indexing_service.index_document(old_document)
    )

    old_count = vector_store.count()

    assert old_count > 1
    assert first_result.chunks_processed == old_count

    updated_document = make_document(
        "answer = 42\n",
        workspace_root=root,
    )

    second_result = run(
        indexing_service.index_document(updated_document)
    )

    assert second_result.files_processed == 1
    assert second_result.chunks_processed == 1
    assert second_result.vectors_upserted == 1

    assert second_result.stale_vectors_deleted == old_count - 1
    assert vector_store.count() == 1


def test_empty_document_removes_previously_indexed_chunks(
    indexing_environment,
    tmp_path,
):
    indexing_service, vector_store, _ = indexing_environment

    root = str(tmp_path / "workspace")

    original_document = make_document(
        "def hello():\n    return 'hello'\n",
        workspace_root=root,
    )

    run(
        indexing_service.index_document(original_document)
    )

    assert vector_store.count() == 1

    empty_document = make_document(
        "",
        workspace_root=root,
    )

    result = run(
        indexing_service.index_document(empty_document)
    )

    assert result.files_processed == 1
    assert result.chunks_processed == 0
    assert result.vectors_upserted == 0
    assert result.stale_vectors_deleted == 1
    assert result.empty_files == 1

    assert vector_store.count() == 0


def test_embedding_failure_preserves_existing_vectors(
    indexing_environment,
    tmp_path,
):
    indexing_service, vector_store, provider = indexing_environment

    root = str(tmp_path / "workspace")

    original_document = make_document(
        "def original():\n    return 1\n",
        workspace_root=root,
    )

    run(
        indexing_service.index_document(original_document)
    )

    original_ids = vector_store.get_file_chunk_ids(
        workspace_root=root,
        relative_path="src/example.py",
    )

    assert len(original_ids) == 1

    provider.fail_documents = True

    updated_document = make_document(
        "def updated():\n    return 999\n",
        workspace_root=root,
    )

    with pytest.raises(RuntimeError):
        run(
            indexing_service.index_document(updated_document)
        )

    current_ids = vector_store.get_file_chunk_ids(
        workspace_root=root,
        relative_path="src/example.py",
    )

    assert current_ids == original_ids
    assert vector_store.count() == 1


def test_duplicate_documents_are_rejected(
    indexing_environment,
    tmp_path,
):
    indexing_service, vector_store, _ = indexing_environment

    root = str(tmp_path / "workspace")

    first = make_document(
        "value = 1\n",
        workspace_root=root,
    )

    second = make_document(
        "value = 2\n",
        workspace_root=root,
    )

    with pytest.raises(ValueError, match="Duplicate workspace document"):
        run(
            indexing_service.index_documents([first, second])
        )

    assert vector_store.count() == 0


def test_empty_document_list_returns_empty_result(
    indexing_environment,
):
    indexing_service, vector_store, _ = indexing_environment

    result = run(
        indexing_service.index_documents([])
    )

    assert result.files_processed == 0
    assert result.chunks_processed == 0
    assert result.vectors_upserted == 0
    assert result.stale_vectors_deleted == 0
    assert result.empty_files == 0

    assert vector_store.count() == 0


def test_embedding_dimension_mismatch_is_rejected(
    tmp_path,
):
    provider = FakeEmbeddingProvider()

    embedding_service = EmbeddingService(
        provider=provider,
    )

    vector_store = ChromaVectorStore(
        dimensions=4,
        embedding_model=get_embedding_model_identifier(
            embedding_service
        ),
        persist_directory=tmp_path / "mismatched-chroma",
    )

    try:
        with pytest.raises(
            WorkspaceIndexingConfigurationError,
            match="dimensions",
        ):
            WorkspaceIndexingService(
                embedding_service=embedding_service,
                vector_store=vector_store,
            )
    finally:
        vector_store.close()


def test_embedding_model_mismatch_is_rejected(
    tmp_path,
):
    provider = FakeEmbeddingProvider()

    embedding_service = EmbeddingService(
        provider=provider,
    )

    vector_store = ChromaVectorStore(
        dimensions=3,
        embedding_model="another:model",
        persist_directory=tmp_path / "mismatched-model-chroma",
    )

    try:
        with pytest.raises(
            WorkspaceIndexingConfigurationError,
            match="model",
        ):
            WorkspaceIndexingService(
                embedding_service=embedding_service,
                vector_store=vector_store,
            )
    finally:
        vector_store.close()