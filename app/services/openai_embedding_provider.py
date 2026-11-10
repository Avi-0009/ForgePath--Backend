from __future__ import annotations

import logging
import time
from typing import Sequence

from openai import AsyncOpenAI

from app.core.ai_usage_context import ai_usage_context
from app.services.embedding_service import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
)


logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider with metadata-only usage tracking."""

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
        self._supports_custom_dimensions = self._model_name.startswith(
            "text-embedding-3-"
        )

        if not self._supports_custom_dimensions and dimensions != 1536:
            raise EmbeddingConfigurationError(
                "This OpenAI embedding model does not support custom dimensions. "
                "Use 1536 dimensions or select a text-embedding-3 model."
            )

        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            if not api_key or not api_key.strip():
                raise EmbeddingConfigurationError(
                    "OPENAI_API_KEY is required to initialize the OpenAI embedding provider."
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

    async def _record_embedding_usage(
        self,
        *,
        status: str,
        duration_ms: int,
        input_tokens: int | None = None,
        total_tokens: int | None = None,
        request_id: str | None = None,
        error_type: str | None = None,
    ) -> None:
        try:
            from app.services.ai_usage_tracking import record_ai_usage_event

            await record_ai_usage_event(
                provider="openai",
                model_id=f"embedding.openai.{self._model_name}",
                api_model=self._model_name,
                operation="workspace_embedding",
                request_type="embedding",
                status=status,
                input_tokens=input_tokens,
                # Embedding endpoints do not generate output tokens.
                output_tokens=0 if input_tokens is not None else None,
                total_tokens=total_tokens,
                duration_ms=duration_ms,
                provider_request_id=request_id,
                error_type=error_type,
            )
        except Exception:
            logger.warning(
                "Embedding usage tracking hook failed; continuing with embeddings.",
                exc_info=True,
            )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents and record aggregate usage."""

        if self._closed:
            raise EmbeddingProviderError(
                "The OpenAI embedding provider has been closed."
            )
        if not texts:
            return []
        if len(texts) > 2048:
            raise ValueError(
                "An OpenAI embedding request cannot contain more than 2048 inputs."
            )
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("Embedding input must be text.")
            if not text.strip():
                raise ValueError("Embedding input cannot be empty.")

        request_params = {
            "model": self._model_name,
            "input": list(texts),
            "encoding_format": "float",
        }
        if self._supports_custom_dimensions:
            request_params["dimensions"] = self._dimensions

        started_at = time.perf_counter()
        response = None

        with ai_usage_context(operation="workspace_embedding"):
            try:
                response = await self._client.embeddings.create(**request_params)
            except Exception as exc:
                duration_ms = max(0, round((time.perf_counter() - started_at) * 1000))
                await self._record_embedding_usage(
                    status="failed",
                    duration_ms=duration_ms,
                    error_type=type(exc).__name__,
                )
                raise EmbeddingProviderError(
                    "The OpenAI embedding request failed."
                ) from exc

            try:
                if len(response.data) != len(texts):
                    raise ValueError(
                        "The OpenAI API returned an unexpected number of embeddings."
                    )

                ordered_data = sorted(response.data, key=lambda item: item.index)
                actual_indices = [item.index for item in ordered_data]
                if actual_indices != list(range(len(texts))):
                    raise ValueError(
                        "The OpenAI API returned invalid embedding indices."
                    )

                embeddings = [list(item.embedding) for item in ordered_data]
                for embedding in embeddings:
                    if len(embedding) != self._dimensions:
                        raise ValueError(
                            "The OpenAI API returned an unexpected embedding dimension."
                        )

                duration_ms = max(0, round((time.perf_counter() - started_at) * 1000))
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                request_id = getattr(response, "_request_id", None)
                if not isinstance(request_id, str):
                    request_id = None

                await self._record_embedding_usage(
                    status="succeeded",
                    duration_ms=duration_ms,
                    input_tokens=input_tokens,
                    total_tokens=total_tokens,
                    request_id=request_id,
                )
                return embeddings

            except (AttributeError, TypeError, ValueError) as exc:
                duration_ms = max(0, round((time.perf_counter() - started_at) * 1000))
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                await self._record_embedding_usage(
                    status="failed",
                    duration_ms=duration_ms,
                    input_tokens=input_tokens,
                    total_tokens=total_tokens,
                    error_type=type(exc).__name__,
                )
                raise EmbeddingProviderError(
                    f"The OpenAI API returned an invalid embedding response: {exc}"
                ) from exc

    async def embed_query(self, text: str) -> list[float]:
        """Generate one embedding for a search query."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding query cannot be empty.")
        embeddings = await self.embed_documents([text])
        return embeddings[0]

    async def close(self) -> None:
        """Release the asynchronous OpenAI client."""

        if self._closed:
            return
        if self._owns_client:
            await self._client.close()
        self._closed = True