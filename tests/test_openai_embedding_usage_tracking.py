import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.ai_usage_context import ai_usage_context
from app.services import ai_usage_tracking
from app.services.embedding_service import EmbeddingProviderError
from app.services.openai_embedding_provider import OpenAIEmbeddingProvider


class FakeEmbeddingsAPI:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.request_params = None

    async def create(self, **kwargs):
        self.request_params = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.embeddings = FakeEmbeddingsAPI(response=response, error=error)



def test_embedding_success_records_provider_usage(monkeypatch):
    recorded = []

    async def fake_record(**kwargs):
        recorded.append(kwargs)
        return True

    monkeypatch.setattr(ai_usage_tracking, "record_ai_usage_event", fake_record)

    response = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
            SimpleNamespace(index=1, embedding=[0.3, 0.4]),
        ],
        usage=SimpleNamespace(prompt_tokens=17, total_tokens=17),
        _request_id="req_embed_123",
    )
    client = FakeClient(response=response)
    provider = OpenAIEmbeddingProvider(
        api_key=None,
        model_name="text-embedding-3-small",
        dimensions=2,
        client=client,
    )
    user_id = uuid4()

    async def run():
        with ai_usage_context(user_id=user_id, operation="workspace_index"):
            return await provider.embed_documents(["first", "second"])

    embeddings = asyncio.run(run())

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert len(recorded) == 1
    assert recorded[0]["provider"] == "openai"
    assert recorded[0]["request_type"] == "embedding"
    assert recorded[0]["operation"] == "workspace_embedding"
    assert recorded[0]["status"] == "succeeded"
    assert recorded[0]["input_tokens"] == 17
    assert recorded[0]["output_tokens"] == 0
    assert recorded[0]["total_tokens"] == 17
    assert recorded[0]["provider_request_id"] == "req_embed_123"


def test_embedding_failure_records_failed_request(monkeypatch):
    recorded = []

    async def fake_record(**kwargs):
        recorded.append(kwargs)
        return True

    monkeypatch.setattr(ai_usage_tracking, "record_ai_usage_event", fake_record)
    client = FakeClient(error=RuntimeError("private provider details"))
    provider = OpenAIEmbeddingProvider(
        api_key=None,
        model_name="text-embedding-3-small",
        dimensions=2,
        client=client,
    )

    with pytest.raises(EmbeddingProviderError):
        asyncio.run(provider.embed_documents(["document"]))

    assert len(recorded) == 1
    assert recorded[0]["status"] == "failed"
    assert recorded[0]["error_type"] == "RuntimeError"
    assert "private provider details" not in str(recorded[0])