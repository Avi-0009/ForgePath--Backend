import asyncio

from types import SimpleNamespace

import pytest

from app.services.embedding_service import EmbeddingProviderError
from app.services.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)


class FakeEmbeddingsAPI:
    """Mock OpenAI embeddings endpoint."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        if self.error:
            raise self.error

        return self.response


class FakeOpenAIClient:
    """Mock asynchronous OpenAI client."""

    def __init__(self, response=None, error=None):
        self.embeddings = FakeEmbeddingsAPI(
            response=response,
            error=error,
        )


def make_embedding_response(*embeddings):
    """Build a mock OpenAI embedding response."""

    return SimpleNamespace(
        data=[
            SimpleNamespace(
                index=index,
                embedding=embedding,
            )
            for index, embedding in embeddings
        ]
    )


def make_provider(
    client=None,
    dimensions=3,
    model_name="text-embedding-3-small",
):
    client = client or FakeOpenAIClient(
        response=make_embedding_response(
            (0, [0.1, 0.2, 0.3]),
        )
    )

    provider = OpenAIEmbeddingProvider(
        api_key=None,
        model_name=model_name,
        dimensions=dimensions,
        client=client,
    )

    return provider, client


def test_provider_configuration():
    provider, _ = make_provider()

    assert provider.provider_name == "openai"
    assert provider.model_name == "text-embedding-3-small"
    assert provider.dimensions == 3


def test_embed_documents_preserves_input_order():
    response = make_embedding_response(
        (1, [0.4, 0.5, 0.6]),
        (0, [0.1, 0.2, 0.3]),
    )

    client = FakeOpenAIClient(response=response)

    provider, _ = make_provider(client=client)

    result = asyncio.run(
        provider.embed_documents(
            ["first document", "second document"]
        )
    )

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    call = client.embeddings.calls[0]

    assert call["model"] == "text-embedding-3-small"
    assert call["input"] == [
        "first document",
        "second document",
    ]
    assert call["dimensions"] == 3
    assert call["encoding_format"] == "float"


def test_embed_query_returns_one_embedding():
    provider, client = make_provider()

    result = asyncio.run(
        provider.embed_query("find authentication logic")
    )

    assert result == [0.1, 0.2, 0.3]

    assert client.embeddings.calls[0]["input"] == [
        "find authentication logic"
    ]


def test_embed_documents_returns_empty_list_for_empty_input():
    provider, client = make_provider()

    result = asyncio.run(
        provider.embed_documents([])
    )

    assert result == []
    assert client.embeddings.calls == []


def test_rejects_empty_document():
    provider, client = make_provider()

    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(
            provider.embed_documents(["valid text", "   "])
        )

    assert client.embeddings.calls == []


def test_rejects_empty_query():
    provider, client = make_provider()

    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(
            provider.embed_query("   ")
        )

    assert client.embeddings.calls == []


def test_rejects_more_than_2048_inputs():
    provider, client = make_provider()

    texts = ["document"] * 2049

    with pytest.raises(ValueError, match="2048"):
        asyncio.run(
            provider.embed_documents(texts)
        )

    assert client.embeddings.calls == []


def test_wraps_openai_api_errors():
    client = FakeOpenAIClient(
        error=RuntimeError("Simulated API failure")
    )

    provider, _ = make_provider(client=client)

    with pytest.raises(
        EmbeddingProviderError,
        match="request failed",
    ):
        asyncio.run(
            provider.embed_documents(["some source code"])
        )


def test_rejects_invalid_embedding_dimensions():
    response = make_embedding_response(
        (0, [0.1, 0.2]),
    )

    provider, _ = make_provider(
        client=FakeOpenAIClient(response=response),
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="unexpected embedding dimension",
    ):
        asyncio.run(
            provider.embed_documents(["some code"])
        )


def test_rejects_invalid_response_indices():
    response = make_embedding_response(
        (0, [0.1, 0.2, 0.3]),
        (0, [0.4, 0.5, 0.6]),
    )

    provider, _ = make_provider(
        client=FakeOpenAIClient(response=response),
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="invalid embedding indices",
    ):
        asyncio.run(
            provider.embed_documents(["first", "second"])
        )


def test_rejects_dimension_mismatch_for_legacy_model():
    with pytest.raises(
        Exception,
        match="does not support custom dimensions",
    ):
        OpenAIEmbeddingProvider(
            api_key=None,
            model_name="text-embedding-ada-002",
            dimensions=3,
            client=FakeOpenAIClient(),
        )


def test_requires_api_key_when_creating_own_client():
    with pytest.raises(
        Exception,
        match="OPENAI_API_KEY",
    ):
        OpenAIEmbeddingProvider(
            api_key=None,
            model_name="text-embedding-3-small",
            dimensions=1536,
        )


def test_close_releases_owned_client():
    provider = OpenAIEmbeddingProvider(
        api_key="test-api-key",
    )

    asyncio.run(provider.close())

    assert provider._closed is True


def test_close_is_safe_to_call_multiple_times():
    provider = OpenAIEmbeddingProvider(
        api_key="test-api-key",
    )

    asyncio.run(provider.close())
    asyncio.run(provider.close())

    assert provider._closed is True


def test_closed_provider_rejects_embedding_requests():
    provider = OpenAIEmbeddingProvider(
        api_key="test-api-key",
    )

    async def run():
        await provider.close()

        with pytest.raises(
            EmbeddingProviderError,
            match="has been closed",
        ):
            await provider.embed_documents(["some code"])

    asyncio.run(run())


def test_external_client_is_not_closed_by_provider():
    provider, client = make_provider()

    asyncio.run(provider.close())

    # The provider does not own an injected client.
    assert provider._closed is True
    assert not hasattr(client, "close")