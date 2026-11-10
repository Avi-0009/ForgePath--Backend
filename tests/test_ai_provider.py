import asyncio

import httpx
import pytest

from app.core import ai_provider
from app.core.ai_provider import (
    AICompletionResult,
    AIProviderError,
    AIUsage,
)


class FakeResponse:
    def __init__(self, payload=None, status_code=200, json_error=False):
        self.payload = payload
        self.json_error = json_error
        self.request = httpx.Request(
            "POST", "https://test-provider.example/v1"
        )
        self.response = httpx.Response(status_code, request=self.request)

    def raise_for_status(self):
        self.response.raise_for_status()

    def json(self):
        if self.json_error:
            raise ValueError("invalid json")
        return self.payload


class FakeAsyncClient:
    def __init__(self, response=None, error=None, **kwargs):
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


def install_fake_client(monkeypatch, response=None, error=None):
    clients = []

    def fake_factory(**kwargs):
        client = FakeAsyncClient(
            response=response,
            error=error,
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(ai_provider.httpx, "AsyncClient", fake_factory)
    return clients


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Registry and configuration
# ---------------------------------------------------------------------------


def test_model_registry_contains_three_supported_providers():
    profiles = ai_provider.list_available_models()

    assert {profile.provider for profile in profiles} == {
        "nvidia",
        "openai",
        "huggingface",
    }
    assert ai_provider.DEFAULT_MODEL_ID in ai_provider.MODEL_PROFILES


def test_get_model_profile_rejects_unregistered_model():
    with pytest.raises(AIProviderError, match="Unsupported model_id"):
        ai_provider.get_model_profile("provider.arbitrary-model")


def test_output_token_limit_rejects_values_above_model_limit():
    profile = ai_provider.get_model_profile("nvidia.nemotron-3-ultra")

    with pytest.raises(ValueError, match="cannot exceed"):
        ai_provider._resolve_max_output_tokens(profile, 32_769)


def test_output_token_limit_rejects_non_positive_values():
    profile = ai_provider.get_model_profile("nvidia.nemotron-3-ultra")

    with pytest.raises(ValueError, match="positive integer"):
        ai_provider._resolve_max_output_tokens(profile, 0)


def test_reasoning_effort_is_validated_per_model():
    profile = ai_provider.get_model_profile("huggingface.qwen3-coder-next")

    assert ai_provider._resolve_reasoning_effort(profile, None) is None

    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        ai_provider._resolve_reasoning_effort(profile, "high")


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_get_api_key_uses_first_available_alias(monkeypatch):
    monkeypatch.delenv("MISSING_PRIMARY", raising=False)
    monkeypatch.setenv("SECONDARY_TOKEN", "  test-secret  ")

    assert ai_provider._get_api_key(
        "MISSING_PRIMARY", "SECONDARY_TOKEN"
    ) == "test-secret"


def test_get_api_key_rejects_missing_all_aliases(monkeypatch):
    monkeypatch.delenv("MISSING_PRIMARY", raising=False)
    monkeypatch.delenv("MISSING_SECONDARY", raising=False)

    with pytest.raises(AIProviderError, match="Missing MISSING_PRIMARY or MISSING_SECONDARY"):
        ai_provider._get_api_key("MISSING_PRIMARY", "MISSING_SECONDARY")


# ---------------------------------------------------------------------------
# Response parsing and usage normalization
# ---------------------------------------------------------------------------


def test_extract_chat_completion_text_from_string():
    response = {
        "choices": [{"message": {"content": '{"summary": "Review complete"}'}}]
    }

    assert ai_provider._extract_chat_completion_text(response, "NVIDIA") == (
        '{"summary": "Review complete"}'
    )


def test_extract_chat_completion_text_from_content_list():
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "Part one "},
                        {"type": "output_text", "text": "part two"},
                    ]
                }
            }
        ]
    }

    assert ai_provider._extract_chat_completion_text(
        response, "Hugging Face"
    ) == "Part one part two"


def test_extract_chat_completion_does_not_return_reasoning_content():
    response = {
        "choices": [
            {
                "message": {
                    "content": "Final answer",
                    "reasoning_content": "Internal reasoning must not be exposed",
                }
            }
        ]
    }

    assert ai_provider._extract_chat_completion_text(response, "NVIDIA") == (
        "Final answer"
    )


def test_extract_chat_completion_rejects_empty_content():
    response = {"choices": [{"message": {"content": ""}}]}

    with pytest.raises(AIProviderError, match="no text output"):
        ai_provider._extract_chat_completion_text(response, "NVIDIA")


def test_extract_chat_completion_rejects_missing_choices():
    with pytest.raises(AIProviderError, match="no completion choices"):
        ai_provider._extract_chat_completion_text({}, "NVIDIA")


def test_extract_openai_text_from_output_text():
    assert ai_provider._extract_openai_text(
        {"output_text": "  Generated answer  "}
    ) == "Generated answer"


def test_extract_openai_text_from_output_items():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Generated "},
                    {"type": "output_text", "text": "answer"},
                ],
            }
        ]
    }

    assert ai_provider._extract_openai_text(response) == "Generated answer"


def test_extract_openai_text_rejects_empty_response():
    with pytest.raises(AIProviderError, match="no text output"):
        ai_provider._extract_openai_text({})


def test_normalize_usage_from_chat_completion_payload():
    usage = ai_provider._normalize_usage(
        {
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 45,
                "total_tokens": 165,
                "prompt_tokens_details": {"cached_tokens": 20},
                "completion_tokens_details": {"reasoning_tokens": 12},
            }
        }
    )

    assert usage == AIUsage(
        input_tokens=120,
        output_tokens=45,
        total_tokens=165,
        cached_input_tokens=20,
        reasoning_tokens=12,
        total_tokens_calculated=False,
    )


def test_normalize_usage_calculates_total_only_from_reported_counts():
    usage = ai_provider._normalize_usage(
        {"usage": {"input_tokens": 7, "output_tokens": 3}}
    )

    assert usage.input_tokens == 7
    assert usage.output_tokens == 3
    assert usage.total_tokens == 10
    assert usage.total_tokens_calculated is True


def test_normalize_usage_does_not_estimate_missing_counts():
    usage = ai_provider._normalize_usage({"usage": {"input_tokens": 8}})

    assert usage.input_tokens == 8
    assert usage.output_tokens is None
    assert usage.total_tokens is None


def test_normalize_usage_returns_empty_when_provider_omits_usage():
    assert ai_provider._normalize_usage({}) == AIUsage()


# ---------------------------------------------------------------------------
# HTTP transport and error mapping
# ---------------------------------------------------------------------------


def test_post_json_returns_response_and_uses_explicit_timeout(monkeypatch):
    expected = {"status": "ok"}
    clients = install_fake_client(
        monkeypatch, response=FakeResponse(payload=expected)
    )

    result = run(
        ai_provider._post_json(
            url="https://test-provider.example/v1",
            api_key="test-key",
            payload={"input": "hello"},
            provider_name="NVIDIA",
        )
    )

    assert result == expected
    assert len(clients) == 1
    assert isinstance(clients[0].timeout, httpx.Timeout)
    assert clients[0].timeout.read == ai_provider.REQUEST_TIMEOUT_SECONDS
    assert clients[0].timeout.connect == ai_provider.CONNECT_TIMEOUT_SECONDS
    assert clients[0].request_data["headers"]["Authorization"] == "Bearer test-key"
    assert clients[0].request_data["json"] == {"input": "hello"}


def test_post_json_rejects_non_dictionary_response(monkeypatch):
    install_fake_client(
        monkeypatch, response=FakeResponse(payload=["unexpected"])
    )

    with pytest.raises(AIProviderError, match="unexpected response format"):
        run(
            ai_provider._post_json(
                url="https://test-provider.example/v1",
                api_key="test-key",
                payload={},
                provider_name="NVIDIA",
            )
        )


def test_post_json_rejects_invalid_json(monkeypatch):
    install_fake_client(
        monkeypatch, response=FakeResponse(json_error=True)
    )

    with pytest.raises(AIProviderError, match="invalid JSON"):
        run(
            ai_provider._post_json(
                url="https://test-provider.example/v1",
                api_key="test-key",
                payload={},
                provider_name="NVIDIA",
            )
        )


@pytest.mark.parametrize(
    ("status_code", "expected_detail"),
    [
        (401, "access to this model"),
        (403, "access to this model"),
        (429, "rate limit or account quota"),
        (400, "model ID and provider request configuration"),
        (500, "server-side error"),
    ],
)
def test_post_json_maps_http_errors(
    monkeypatch, status_code, expected_detail
):
    install_fake_client(
        monkeypatch, response=FakeResponse(status_code=status_code)
    )

    with pytest.raises(AIProviderError) as exc_info:
        run(
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
    request = httpx.Request("POST", "https://test-provider.example/v1")
    error = httpx.ReadTimeout("Request timed out", request=request)
    install_fake_client(monkeypatch, error=error)

    with pytest.raises(AIProviderError, match="timed out after 240 seconds"):
        run(
            ai_provider._post_json(
                url="https://test-provider.example/v1",
                api_key="test-key",
                payload={},
                provider_name="NVIDIA",
            )
        )


def test_post_json_maps_non_timeout_network_error(monkeypatch):
    request = httpx.Request("POST", "https://test-provider.example/v1")
    error = httpx.ConnectError("Connection refused", request=request)
    install_fake_client(monkeypatch, error=error)

    with pytest.raises(AIProviderError, match="ConnectError"):
        run(
            ai_provider._post_json(
                url="https://test-provider.example/v1",
                api_key="test-key",
                payload={},
                provider_name="NVIDIA",
            )
        )


# ---------------------------------------------------------------------------
# Provider adapters and public interface
# ---------------------------------------------------------------------------


def test_generate_completion_uses_nvidia_adapter(monkeypatch):
    captured = {}

    async def fake_post_json(**kwargs):
        captured.update(kwargs)
        return {
            "id": "req-nvidia-1",
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
            "choices": [
                {
                    "message": {"content": "NVIDIA response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    requested_keys = []
    monkeypatch.setattr(ai_provider, "_post_json", fake_post_json)
    monkeypatch.setattr(
        ai_provider,
        "_get_api_key",
        lambda *names: requested_keys.append(names) or "test-nvidia-key",
    )

    result = run(
        ai_provider.generate_completion(
            instructions="Follow the system instructions.",
            prompt="Review this code.",
            model_id="nvidia.nemotron-3-ultra",
        )
    )

    assert isinstance(result, AICompletionResult)
    assert result.text == "NVIDIA response"
    assert result.provider == "nvidia"
    assert result.model_id == "nvidia.nemotron-3-ultra"
    assert result.request_id == "req-nvidia-1"
    assert result.finish_reason == "stop"
    assert result.usage.total_tokens == 15
    assert requested_keys == [("NVIDIA_API_KEY",)]
    assert captured["url"] == ai_provider.NVIDIA_CHAT_URL
    assert captured["payload"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert captured["payload"]["max_tokens"] == 16_384
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["reasoning_effort"] == "medium"
    assert captured["payload"]["messages"][0]["role"] == "system"


def test_generate_completion_uses_openai_adapter(monkeypatch):
    captured = {}

    async def fake_post_json(**kwargs):
        captured.update(kwargs)
        return {
            "id": "req-openai-1",
            "output_text": "OpenAI response",
            "usage": {
                "input_tokens": 30,
                "output_tokens": 10,
                "total_tokens": 40,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens_details": {"reasoning_tokens": 6},
            },
        }

    monkeypatch.setattr(ai_provider, "_post_json", fake_post_json)
    monkeypatch.setattr(ai_provider, "_get_api_key", lambda *names: "test-openai-key")

    result = run(
        ai_provider.generate_completion(
            instructions="Be precise.",
            prompt="Explain this code.",
            model_id="openai.gpt-6-luna",
            max_output_tokens=20_000,
            reasoning_effort="high",
        )
    )

    assert result.text == "OpenAI response"
    assert result.provider == "openai"
    assert result.usage.cached_input_tokens == 4
    assert result.usage.reasoning_tokens == 6
    assert captured["url"] == ai_provider.OPENAI_RESPONSES_URL
    assert captured["payload"]["model"] == "gpt-6-luna"
    assert captured["payload"]["max_output_tokens"] == 20_000
    assert captured["payload"]["reasoning"] == {"effort": "high"}


def test_generate_completion_uses_huggingface_and_hg_token(monkeypatch):
    captured = {}
    requested_keys = []

    async def fake_post_json(**kwargs):
        captured.update(kwargs)
        return {
            "id": "req-hf-1",
            "model": "Qwen/Qwen3-Coder-Next",
            "choices": [
                {
                    "message": {"content": "Hugging Face response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 8,
                "total_tokens": 33,
            },
        }

    monkeypatch.setattr(ai_provider, "_post_json", fake_post_json)
    monkeypatch.setattr(
        ai_provider,
        "_get_api_key",
        lambda *names: requested_keys.append(names) or "test-hf-token",
    )

    result = run(
        ai_provider.generate_completion(
            instructions="Be a coding assistant.",
            prompt="Inspect this project.",
            model_id="huggingface.qwen3-coder-next",
        )
    )

    assert result.text == "Hugging Face response"
    assert result.provider == "huggingface"
    assert result.usage.total_tokens == 33
    assert requested_keys == [
        ("HG_TOKEN", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
    ]
    assert captured["url"] == ai_provider.HUGGINGFACE_CHAT_URL
    assert captured["payload"]["model"] == "Qwen/Qwen3-Coder-Next"
    assert captured["payload"]["stream"] is False


def test_generate_completion_rejects_empty_instructions():
    with pytest.raises(ValueError, match="Instructions cannot be empty"):
        run(
            ai_provider.generate_completion(
                instructions=" ",
                prompt="Valid prompt",
            )
        )


def test_generate_completion_rejects_empty_prompt():
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        run(
            ai_provider.generate_completion(
                instructions="Valid instructions",
                prompt=" ",
            )
        )


def test_generate_text_remains_backward_compatible(monkeypatch):
    async def fake_generate_completion(**kwargs):
        assert kwargs["instructions"] == "System instructions"
        assert kwargs["prompt"] == "User prompt"
        return AICompletionResult(
            text="Only the generated text",
            provider="nvidia",
            model_id="nvidia.nemotron-3-ultra",
            api_model="nvidia/nemotron-3-ultra-550b-a55b",
            usage=AIUsage(input_tokens=1, output_tokens=2, total_tokens=3),
            duration_ms=10,
        )

    monkeypatch.setattr(
        ai_provider, "generate_completion", fake_generate_completion
    )

    result = run(
        ai_provider.generate_text(
            instructions="System instructions",
            prompt="User prompt",
        )
    )

    assert result == "Only the generated text"