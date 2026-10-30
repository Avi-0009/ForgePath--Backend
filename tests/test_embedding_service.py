import asyncio
import pytest

from app.services.embedding_service import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    EmbeddingService,
)


class FakeEmbeddingProvider:
    """Deterministic provider for testing service behavior."""

    provider_name = "fake"
    model_name = "fake-embedding-model"
    dimensions = 3

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    async def embed_documents(self, texts):
        self.document_calls.append(list(texts))

        return [
            [float(index + 1), 0.5, 0.25]
            for index, _ in enumerate(texts)
        ]

    async def embed_query(self, text):
        self.query_calls.append(text)

        return [1.0, 0.5, 0.25]


def test_initializes_embedding_service():
    provider = FakeEmbeddingProvider()

    service = EmbeddingService(provider)

    assert service.provider_name == "fake"
    assert service.model_name == "fake-embedding-model"
    assert service.dimensions == 3
    assert service.batch_size == 32


def test_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        EmbeddingService(
            FakeEmbeddingProvider(),
            batch_size=0,
        )


def test_rejects_invalid_dimensions():
    provider = FakeEmbeddingProvider()
    provider.dimensions = 0

    with pytest.raises(EmbeddingConfigurationError):
        EmbeddingService(provider)


def test_generates_document_embeddings():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    vectors = asyncio.run(
        service.embed_documents(
            [
                "def add(a, b): return a + b",
                "class User: pass",
            ]
        )
    )

    assert len(vectors) == 2
    assert vectors[0] == [1.0, 0.5, 0.25]
    assert vectors[1] == [2.0, 0.5, 0.25]

    assert len(provider.document_calls) == 1


def test_batches_document_embeddings():
    provider = FakeEmbeddingProvider()

    service = EmbeddingService(
        provider,
        batch_size=2,
    )

    vectors = asyncio.run(
        service.embed_documents(
            ["one", "two", "three", "four", "five"]
        )
    )

    assert len(vectors) == 5
    assert len(provider.document_calls) == 3

    assert provider.document_calls == [
        ["one", "two"],
        ["three", "four"],
        ["five"],
    ]


def test_empty_document_collection_returns_empty_list():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    vectors = asyncio.run(
        service.embed_documents([])
    )

    assert vectors == []
    assert provider.document_calls == []


def test_generates_query_embedding():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    vector = asyncio.run(
        service.embed_query(
            "Where is JWT authentication validated?"
        )
    )

    assert vector == [1.0, 0.5, 0.25]

    assert provider.query_calls == [
        "Where is JWT authentication validated?"
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n\t",
    ],
)
def test_rejects_empty_embedding_input(text):
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    with pytest.raises(ValueError):
        asyncio.run(service.embed_query(text))


def test_rejects_wrong_vector_dimensions():
    class InvalidProvider(FakeEmbeddingProvider):
        async def embed_query(self, text):
            return [1.0, 2.0]

    service = EmbeddingService(InvalidProvider())

    with pytest.raises(EmbeddingProviderError):
        asyncio.run(service.embed_query("test query"))


def test_rejects_non_finite_vector_values():
    class InvalidProvider(FakeEmbeddingProvider):
        async def embed_query(self, text):
            return [1.0, float("nan"), 0.5]

    service = EmbeddingService(InvalidProvider())

    with pytest.raises(EmbeddingProviderError):
        asyncio.run(service.embed_query("test query"))


def test_rejects_wrong_number_of_document_embeddings():
    class InvalidProvider(FakeEmbeddingProvider):
        async def embed_documents(self, texts):
            return [[1.0, 0.5, 0.25]]

    service = EmbeddingService(
        InvalidProvider(),
        batch_size=2,
    )

    with pytest.raises(EmbeddingProviderError):
        asyncio.run(
            service.embed_documents(["one", "two"])
        )