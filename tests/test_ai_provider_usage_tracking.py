import asyncio

from app.core import ai_provider
from app.core.ai_provider import AIProviderError
from app.services import ai_usage_tracking


def test_successful_completion_records_normalized_usage(monkeypatch):
    recorded = []

    async def fake_record(**kwargs):
        recorded.append(kwargs)
        return True

    async def fake_nvidia_request(**kwargs):
        return {
            "id": "req_ai_001",
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
            "choices": [
                {
                    "message": {"content": "Generated result"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }

    monkeypatch.setattr(ai_usage_tracking, "record_ai_usage_event", fake_record)
    monkeypatch.setattr(ai_provider, "_request_nvidia", fake_nvidia_request)

    result = asyncio.run(
        ai_provider.generate_completion(
            instructions="Do the work.",
            prompt="Sensitive source content must not be recorded.",
        )
    )

    assert result.text == "Generated result"
    assert len(recorded) == 1
    event = recorded[0]
    assert event["provider"] == "nvidia"
    assert event["model_id"] == "nvidia.nemotron-3-ultra"
    assert event["status"] == "succeeded"
    assert event["input_tokens"] == 11
    assert event["output_tokens"] == 7
    assert event["total_tokens"] == 18
    assert event["total_tokens_calculated"] is True
    assert event["cached_input_tokens"] == 3
    assert event["reasoning_tokens"] == 2
    assert event["provider_request_id"] == "req_ai_001"
    assert event["finish_reason"] == "stop"
    assert "prompt" not in event
    assert "Sensitive source content" not in str(event)


def test_failed_completion_records_exception_class_only(monkeypatch):
    recorded = []

    async def fake_record(**kwargs):
        recorded.append(kwargs)
        return True

    async def fake_nvidia_request(**kwargs):
        raise AIProviderError("Do not persist this error detail or credentials")

    monkeypatch.setattr(ai_usage_tracking, "record_ai_usage_event", fake_record)
    monkeypatch.setattr(ai_provider, "_request_nvidia", fake_nvidia_request)

    try:
        asyncio.run(
            ai_provider.generate_completion(
                instructions="Do the work.",
                prompt="Private input",
            )
        )
        raise AssertionError("AIProviderError was expected")
    except AIProviderError:
        pass

    assert len(recorded) == 1
    event = recorded[0]
    assert event["status"] == "failed"
    assert event["error_type"] == "AIProviderError"
    assert "Do not persist" not in str(event)
    assert "Private input" not in str(event)