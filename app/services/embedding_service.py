import math
from typing import Protocol, Sequence


class EmbeddingError(RuntimeError):
    """Base error for embedding operations."""

class EmbeddingConfigurationError(EmbeddingError):
    """The embedding provider configuration is invalid."""

class EmbeddingProviderError(EmbeddingError):
    """The embedding provider returned an invalid result."""

class EmbeddingProvider(Protocol):
    """
    Standard interface implemented by every embedding provider.

    A provider may use OpenAI, Voyage, or another compatible API.
    """

    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimensions(self) -> int:
        ...

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        ...


class EmbeddingService:
    """
    Provider-independent embedding service.

    Responsibilities:
    - Validate incoming text.
    - Process documents in bounded batches.
    - Validate vector dimensions and numeric values.
    - Preserve the order of input documents.

    Actual model inference is delegated to the configured provider.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        batch_size: int = 32,
    ) -> None:

        if batch_size < 1:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if provider.dimensions < 1:
            raise EmbeddingConfigurationError(
                "Embedding dimensions must be greater than zero."
            )

        if not provider.provider_name.strip():
            raise EmbeddingConfigurationError(
                "Embedding provider name cannot be empty."
            )

        if not provider.model_name.strip():
            raise EmbeddingConfigurationError(
                "Embedding model name cannot be empty."
            )

        self.provider = provider
        self.batch_size = batch_size
        self.dimensions = provider.dimensions

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    def _validate_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise ValueError("Embedding input must be text.")

        if not text.strip():
            raise ValueError(
                "Embedding input cannot be empty or whitespace only."
            )

    def _validate_vector(
        self,
        vector: Sequence[float],
    ) -> list[float]:

        if len(vector) != self.dimensions:
            raise EmbeddingProviderError(
                f"Expected {self.dimensions} embedding dimensions, "
                f"received {len(vector)}."
            )

        validated: list[float] = []

        for value in vector:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise EmbeddingProviderError(
                    "Embedding vectors must contain numeric values."
                )

            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise EmbeddingProviderError(
                    "Embedding vectors cannot contain NaN or infinity."
                )

            validated.append(numeric_value)

        return validated

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.

        Input order is preserved across all batches.
        """

        if not texts:
            return []

        for text in texts:
            self._validate_text(text)

        results: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]

            vectors = await self.provider.embed_documents(batch)

            if len(vectors) != len(batch):
                raise EmbeddingProviderError(
                    "The provider returned an unexpected number "
                    "of document embeddings."
                )

            for vector in vectors:
                results.append(
                    self._validate_vector(vector)
                )

        return results

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate one query embedding for semantic search."""

        self._validate_text(text)

        vector = await self.provider.embed_query(text)

        return self._validate_vector(vector)