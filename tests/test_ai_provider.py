import asyncio

import httpx
import pytest

from app.core import ai_provider
from app.core.ai_provider import AIProviderError


# MOCK HTTP CLIENT

class FakeResponse:
    def __init__(
        self,
        payload=None,
        status_code=200,
        json_error=False,
    ):
        self.payload = payload
        self.json_error = json_error

        self.request = httpx.Request(
            "POST",
            "https://test-provider.example/v1",
        )

        self.response = httpx.Response(
            status_code,
            request=self.request,
        )

    def raise_for_status(self):
        self.response.raise_for_status()

    def json(self):
        if self.json_error:
            raise ValueError("Invalid JSON")

        return self.payload


class FakeAsyncClient:
    def __init__(
        self,
        response=None,
        error=None,
        **kwargs,
    ):
        self.response = response
        self.error = error
        self.timeout = kwargs.get("timeout")
        self.request_data = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, headers, json):
        self.request_data = {
            "url": url,
            "headers": headers,
            "json": json,
        }

        if self.error:
            raise self.error

        return self.response


def install_fake_client(
    monkeypatch,
    response=None,
    error=None,
):
    clients = []

    def fake_factory(**kwargs):
        client = FakeAsyncClient(
            response=response,
            error=error,
            **kwargs,
        )

        clients.append(client)
        return client

    monkeypatch.setattr(
        ai_provider.httpx,
        "AsyncClient",
        fake_factory,
    )

    return clients


# API KEY VALIDATION

def test_get_api_key_returns_key(monkeypatch):
    monkeypatch.setenv(
        "NVIDIA_API_KEY",
        "test-api-key",
    )

    assert (
        ai_provider._get_api_key("NVIDIA_API_KEY")
        == "test-api-key"
    )


def test_get_api_key_rejects_missing_key(monkeypatch):
    monkeypatch.delenv(
        "NVIDIA_API_KEY",
        raising=False,
    )

    with pytest.raises(
        AIProviderError,
        match="Missing NVIDIA_API_KEY",
    ):
        ai_provider._get_api_key("NVIDIA_API_KEY")


# NVIDIA RESPONSE EXTRACTION

def test_extract_nvidia_text_from_string():
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"summary": "Review complete"}',
                }
            }
        ]
    }

    result = ai_provider._extract_nvidia_text(response)

    assert result == '{"summary": "Review complete"}'


def test_extract_nvidia_text_from_content_list():
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Part one ",
                        },
                        {
                            "type": "text",
                            "text": "part two",
                        },
                    ]
                }
            }
        ]
    }

    result = ai_provider._extract_nvidia_text(response)

    assert result == "Part one part two"


def test_extract_nvidia_text_rejects_empty_response():
    response = {
        "choices": [
            {
                "message": {
                    "content": "",
                }
            }
        ]
    }

    with pytest.raises(
        AIProviderError,
        match="no text output",
    ):
        ai_provider._extract_nvidia_text(response)


def test_extract_nvidia_text_rejects_missing_choices():
    with pytest.raises(
        AIProviderError,
        match="no completion choices",
    ):
        ai_provider._extract_nvidia_text({})


# OPENAI RESPONSE EXTRACTION

def test_extract_openai_text_from_output_text():
    response = {
        "output_text": "OpenAI generated response"
    }

    result = ai_provider._extract_openai_text(response)

    assert result == "OpenAI generated response"


def test_extract_openai_text_from_output_items():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Generated ",
                    },
                    {
                        "type": "output_text",
                        "text": "response",
                    },
                ],
            }
        ]
    }

    result = ai_provider._extract_openai_text(response)

    assert result == "Generated response"


def test_extract_openai_text_rejects_empty_response():
    with pytest.raises(
        AIProviderError,
        match="no text output",
    ):
        ai_provider._extract_openai_text({})


# HTTP REQUEST HANDLING

def test_post_json_returns_valid_response(monkeypatch):
    expected = {"status": "ok"}

    response = FakeResponse(payload=expected)

    clients = install_fake_client(
        monkeypatch,
        response=response,
    )

    result = asyncio.run(
        ai_provider._post_json(
            url="https://test-provider.example/v1",
            api_key="test-key",
            payload={"input": "hello"},
            provider_name="NVIDIA",
        )
    )

    assert result == expected
    assert len(clients) == 1

    assert clients[0].timeout == (
        ai_provider.REQUEST_TIMEOUT_SECONDS
    )

    assert clients[0].request_data["headers"]["Authorization"] == (
        "Bearer test-key"
    )

    assert clients[0].request_data["json"] == {
        "input": "hello"
    }


def test_post_json_rejects_non_dictionary_response(monkeypatch):
    response = FakeResponse(payload=["unexpected"])

    install_fake_client(
        monkeypatch,
        response=response,
    )

    with pytest.raises(
        AIProviderError,
        match="unexpected response",
    ):
        asyncio.run(
            ai_provider._post_json(
                url="https://test-provider.example/v1",
                api_key="test-key",
                payload={},
                provider_name="NVIDIA",
            )
        )


def test_post_json_rejects_invalid_json(monkeypatch):
    response = FakeResponse(json_error=True)

    install_fake_client(
        monkeypatch,
        response=response,
    )

    with pytest.raises(
        AIProviderError,
        match="invalid JSON",
    ):
        asyncio.run(
            ai_provider._post_json(
                url="https://test-provider.example/v1",
                api_key="test-key",
                payload={},
                provider_name="NVIDIA",
            )
        )


# HTTP STATUS ERRORS

@pytest.mark.parametrize(
    ("status_code", "expected_detail"),
    [
        (401, "API key and model access"),
        (403, "API key and model access"),
        (429, "Rate limit or provider quota"),
        (500, "provider configuration"),
    ],
)
def test_post_json_maps_http_errors(
    monkeypatch,
    status_code,
    expected_detail,
):
    response = FakeResponse(status_code=status_code)

    install_fake_client(
        monkeypatch,
        response=response,
    )

    with pytest.raises(AIProviderError) as exc_info:
        asyncio.run(
            ai_provider._post_json(
                url="https://test-provider.example/v1",
                api_key="test-key",
                payload={},
                provider_name="NVIDIA",
            )
        )

    assert f"HTTP {status_code}" in str(exc_info.value)
    assert expected_detail in str(exc_info.value)


def test_post_json_maps_network_timeout(monkeypatch):
    request = httpx.Request(
        "POST",
        "https://test-provider.example/v1",
    )

    error = httpx.ReadTimeout(
        "Request timed out",
        request=request,
    )

    install_fake_client(
        monkeypatch,
        error=error,
    )

    with pytest.raises(
        AIProviderError,
        match="ReadTimeout",
    ):
        asyncio.run(
            ai_provider._post_json(
                url="https://test-provider.example/v1",
                api_key="test-key",
                payload={},
                provider_name="NVIDIA",
            )
        )


# NVIDIA GENERATION CONFIGURATION

def test_generate_text_uses_nvidia_non_streaming(
    monkeypatch,
):
    captured = {}

    async def fake_post_json(**kwargs):
        captured.update(kwargs)

        return {
            "choices": [
                {
                    "message": {
                        "content": "Successful AI response",
                    }
                }
            ]
        }

    monkeypatch.setattr(
        ai_provider,
        "AI_PROVIDER",
        "nvidia",
    )

    monkeypatch.setattr(
        ai_provider,
        "_get_api_key",
        lambda _: "test-nvidia-key",
    )

    monkeypatch.setattr(
        ai_provider,
        "_post_json",
        fake_post_json,
    )

    result = asyncio.run(
        ai_provider.generate_text(
            instructions="Review code carefully.",
            prompt="Review this function.",
        )
    )

    assert result == "Successful AI response"

    payload = captured["payload"]

    assert payload["model"] == ai_provider.NVIDIA_MODEL
    assert payload["stream"] is False
    assert payload["max_tokens"] == ai_provider.NVIDIA_MAX_TOKENS

    assert (
        payload["chat_template_kwargs"]["enable_thinking"]
        == ai_provider.NVIDIA_ENABLE_THINKING
    )

    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"

    assert captured["api_key"] == "test-nvidia-key"
    assert captured["provider_name"] == "NVIDIA"


def test_generate_text_rejects_empty_instructions():
    with pytest.raises(
        ValueError,
        match="Instructions cannot be empty",
    ):
        asyncio.run(
            ai_provider.generate_text(
                instructions=" ",
                prompt="Review this code.",
            )
        )


def test_generate_text_rejects_empty_prompt():
    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty",
    ):
        asyncio.run(
            ai_provider.generate_text(
                instructions="Review carefully.",
                prompt=" ",
            )
        )