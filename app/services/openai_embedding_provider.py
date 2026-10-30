from __future__ import annotations

from typing import Sequence
from openai import AsyncOpenAI

from app.services.embedding_service import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
)


class OpenAIEmbeddingProvider:
    """
    OpenAI implementation of ForgePath's embedding provider interface.

    Uses the asynchronous OpenAI API.
    Does not download or execute models locally.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str = "text-embedding-3-small",
        dimensions: int = 1536,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: AsyncOpenAI | None = None,
    ) -> None:

        if not model_name.strip():
            raise EmbeddingConfigurationError(
                "An OpenAI embedding model name is required."
            )

        if dimensions < 1:
            raise EmbeddingConfigurationError(
                "Embedding dimensions must be greater than zero."
            )

        if timeout_seconds <= 0:
            raise EmbeddingConfigurationError(
                "OpenAI timeout must be greater than zero."
            )

        if max_retries < 0:
            raise EmbeddingConfigurationError(
                "OpenAI max retries cannot be negative."
            )

        self._model_name = model_name.strip()
        self._dimensions = dimensions
        self._closed = False

        # text-embedding-3 models support configurable dimensions.
        self._supports_custom_dimensions = (
            self._model_name.startswith("text-embedding-3-")
        )

        if (
            not self._supports_custom_dimensions
            and dimensions != 1536
        ):
            raise EmbeddingConfigurationError(
                "This OpenAI embedding model does not support "
                "custom dimensions. Use 1536 dimensions or select "
                "a text-embedding-3 model."
            )

        self._owns_client = client is None

        if client is not None:
            self._client = client

        else:
            if not api_key or not api_key.strip():
                raise EmbeddingConfigurationError(
                    "OPENAI_API_KEY is required to initialize "
                    "the OpenAI embedding provider."
                )

            self._client = AsyncOpenAI(
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple documents."""

        if self._closed:
            raise EmbeddingProviderError(
                "The OpenAI embedding provider has been closed."
            )

        if not texts:
            return []

        if len(texts) > 2048:
            raise ValueError(
                "An OpenAI embedding request cannot contain "
                "more than 2048 inputs."
            )

        for text in texts:
            if not isinstance(text, str):
                raise ValueError(
                    "Embedding input must be text."
                )

            if not text.strip():
                raise ValueError(
                    "Embedding input cannot be empty."
                )

        request_params = {
            "model": self._model_name,
            "input": list(texts),
            "encoding_format": "float",
        }

        if self._supports_custom_dimensions:
            request_params["dimensions"] = self._dimensions

        try:
            response = await self._client.embeddings.create(
                **request_params,
            )

        except Exception as exc:
            raise EmbeddingProviderError(
                "The OpenAI embedding request failed."
            ) from exc

        try:
            if len(response.data) != len(texts):
                raise ValueError(
                    "The OpenAI API returned an unexpected "
                    "number of embeddings."
                )

            # Preserve input ordering using the response indices.
            ordered_data = sorted(
                response.data,
                key=lambda item: item.index,
            )

            expected_indices = list(range(len(texts)))

            actual_indices = [
                item.index
                for item in ordered_data
            ]

            if actual_indices != expected_indices:
                raise ValueError(
                    "The OpenAI API returned invalid embedding indices."
                )

            embeddings = [
                list(item.embedding)
                for item in ordered_data
            ]

            for embedding in embeddings:
                if len(embedding) != self._dimensions:
                    raise ValueError(
                        "The OpenAI API returned an unexpected "
                        "embedding dimension."
                    )

            return embeddings

        except (AttributeError, TypeError, ValueError) as exc:
            raise EmbeddingProviderError(
                f"The OpenAI API returned an invalid embedding response: {exc}"
            ) from exc

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate one embedding for a search query."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Embedding query cannot be empty."
            )

        embeddings = await self.embed_documents([text])

        return embeddings[0]

    async def close(self) -> None:
        """Release the asynchronous OpenAI client."""

        if self._closed:
            return

        if self._owns_client:
            await self._client.close()

        self._closed = True