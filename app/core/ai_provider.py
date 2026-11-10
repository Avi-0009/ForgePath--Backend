"""Central AI model registry and provider adapters for ForgePath.

Existing services can continue using ``generate_text(instructions=..., prompt=...)``.
New code that needs token and execution metadata should use ``generate_completion``.

No provider fallback is performed automatically. A failed request stays a failed
request, avoiding unexpected provider changes or charges.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Literal, TypeAlias

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider/model registry
# ---------------------------------------------------------------------------

AIProviderName: TypeAlias = Literal["nvidia", "openai", "huggingface"]
ReasoningEffort: TypeAlias = Literal["none", "low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True, slots=True)
class AIModelProfile:
    """Public, allowlisted model information. Never put provider API keys here."""

    model_id: str
    provider: AIProviderName
    api_model: str
    display_name: str
    description: str
    context_window_tokens: int
    max_output_tokens: int | None
    default_max_output_tokens: int
    reasoning_efforts: tuple[str, ...]
    default_reasoning_effort: str | None


# Keep this setting manual until the per-user model preference API is implemented.
# Change only this value to switch the default provider for existing callers.
AI_PROVIDER: AIProviderName = "nvidia"

MODEL_PROFILES: dict[str, AIModelProfile] = {
    "nvidia.nemotron-3-ultra": AIModelProfile(
        model_id="nvidia.nemotron-3-ultra",
        provider="nvidia",
        api_model="nvidia/nemotron-3-ultra-550b-a55b",
        display_name="NVIDIA Nemotron 3 Ultra",
        description="Long-context reasoning and agentic coding workflows.",
        context_window_tokens=1_000_000,
        max_output_tokens=32_768,
        default_max_output_tokens=16_384,
        reasoning_efforts=("none", "medium", "high"),
        default_reasoning_effort="medium",
    ),
    "openai.gpt-6-luna": AIModelProfile(
        model_id="openai.gpt-6-luna",
        provider="openai",
        api_model="gpt-6-luna",
        display_name="OpenAI GPT-6 Luna",
        description="General-purpose model for complex reasoning and coding tasks.",
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        default_max_output_tokens=32_768,
        reasoning_efforts=("none", "low", "medium", "high", "xhigh", "max"),
        default_reasoning_effort="medium",
    ),
    "huggingface.qwen3-coder-next": AIModelProfile(
        model_id="huggingface.qwen3-coder-next",
        provider="huggingface",
        api_model="Qwen/Qwen3-Coder-Next",
        display_name="Qwen3 Coder Next",
        description="Agentic coding model served through Hugging Face Inference Providers.",
        context_window_tokens=262_144,
        # The hosted provider determines the actual maximum output length.
        max_output_tokens=None,
        default_max_output_tokens=16_384,
        reasoning_efforts=(),
        default_reasoning_effort=None,
    ),
}

_DEFAULT_MODEL_BY_PROVIDER: dict[AIProviderName, str] = {
    "nvidia": "nvidia.nemotron-3-ultra",
    "openai": "openai.gpt-6-luna",
    "huggingface": "huggingface.qwen3-coder-next",
}

if AI_PROVIDER not in _DEFAULT_MODEL_BY_PROVIDER:
    raise RuntimeError(f"Unsupported configured AI provider: {AI_PROVIDER}")

DEFAULT_MODEL_ID = _DEFAULT_MODEL_BY_PROVIDER[AI_PROVIDER]


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT_SECONDS = 240.0
CONNECT_TIMEOUT_SECONDS = 20.0

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HUGGINGFACE_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


# ---------------------------------------------------------------------------
# Result types and errors
# ---------------------------------------------------------------------------


class AIProviderError(RuntimeError):
    """Raised when an AI provider request fails or returns an invalid response."""


@dataclass(frozen=True, slots=True)
class AIUsage:
    """Normalized provider-reported token usage.

    Individual fields remain ``None`` when a provider does not report them.
    ``total_tokens_calculated`` is true only when total was derived by adding
    provider-reported input and output counts; no token estimation is performed.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens_calculated: bool = False


@dataclass(frozen=True, slots=True)
class AICompletionResult:
    """Text plus safe metadata from one successful model request."""

    text: str
    provider: AIProviderName
    model_id: str
    api_model: str
    usage: AIUsage
    duration_ms: int
    request_id: str | None = None
    finish_reason: str | None = None


def list_available_models() -> tuple[AIModelProfile, ...]:
    """Return the allowlisted models in stable registry order."""

    return tuple(MODEL_PROFILES.values())


def get_model_profile(model_id: str) -> AIModelProfile:
    """Resolve a public model ID without accepting arbitrary model names."""

    try:
        return MODEL_PROFILES[model_id]
    except KeyError:
        supported = ", ".join(MODEL_PROFILES)
        raise AIProviderError(
            f"Unsupported model_id '{model_id}'. Supported models: {supported}."
        ) from None


def _resolve_profile(model_id: str | None) -> AIModelProfile:
    selected_model_id = model_id or DEFAULT_MODEL_ID
    return get_model_profile(selected_model_id)


def _resolve_max_output_tokens(
    profile: AIModelProfile,
    requested_max_output_tokens: int | None,
) -> int:
    value = (
        profile.default_max_output_tokens
        if requested_max_output_tokens is None
        else requested_max_output_tokens
    )

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_output_tokens must be a positive integer.")

    if profile.max_output_tokens is not None and value > profile.max_output_tokens:
        raise ValueError(
            f"max_output_tokens for {profile.display_name} cannot exceed "
            f"{profile.max_output_tokens:,}."
        )

    return value


def _resolve_reasoning_effort(
    profile: AIModelProfile,
    requested_reasoning_effort: str | None,
) -> str | None:
    effort = (
        profile.default_reasoning_effort
        if requested_reasoning_effort is None
        else requested_reasoning_effort
    )

    if effort is None:
        if requested_reasoning_effort is not None:
            raise ValueError(
                f"{profile.display_name} does not expose a configurable reasoning effort."
            )
        return None

    if effort not in profile.reasoning_efforts:
        supported = ", ".join(profile.reasoning_efforts) or "none"
        raise ValueError(
            f"Unsupported reasoning effort '{effort}' for {profile.display_name}. "
            f"Supported values: {supported}."
        )

    return effort


# ---------------------------------------------------------------------------
# Credentials and HTTP requests
# ---------------------------------------------------------------------------


def _get_api_key(*env_vars: str) -> str:
    for env_var in env_vars:
        api_key = os.getenv(env_var, "").strip()
        if api_key:
            return api_key

    expected = " or ".join(env_vars)
    raise AIProviderError(
        f"Missing {expected}. Add the provider token to your backend .env file."
    )


async def _post_json(
    *,
    url: str,
    api_key: str,
    payload: dict,
    provider_name: str,
) -> dict:
    timeout = httpx.Timeout(
        REQUEST_TIMEOUT_SECONDS,
        connect=CONNECT_TIMEOUT_SECONDS,
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            if not isinstance(result, dict):
                raise AIProviderError(
                    f"{provider_name} returned an unexpected response format."
                )

            return result

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code

        if status_code in (401, 403):
            detail = "Check the API token and whether it has access to this model."
        elif status_code == 429:
            detail = "The provider rate limit or account quota has been reached."
        elif status_code in (400, 404, 422):
            detail = "Check the model ID and provider request configuration."
        elif status_code >= 500:
            detail = "The provider reported a server-side error."
        else:
            detail = "Check the provider configuration and request."

        raise AIProviderError(
            f"{provider_name} request failed (HTTP {status_code}). {detail}"
        ) from None

    except httpx.TimeoutException:
        raise AIProviderError(
            f"{provider_name} request timed out after "
            f"{REQUEST_TIMEOUT_SECONDS:g} seconds."
        ) from None

    except httpx.RequestError as exc:
        raise AIProviderError(
            f"Could not connect to {provider_name} ({type(exc).__name__})."
        ) from None

    except (ValueError, TypeError):
        raise AIProviderError(f"{provider_name} returned invalid JSON.") from None


# ---------------------------------------------------------------------------
# Provider response parsing
# ---------------------------------------------------------------------------


def _extract_openai_text(response: dict) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    text_parts: list[str] = []
    output_items = response.get("output", [])

    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue

            content_items = item.get("content", [])
            if not isinstance(content_items, list):
                continue

            for content in content_items:
                if (
                    isinstance(content, dict)
                    and content.get("type") == "output_text"
                    and isinstance(content.get("text"), str)
                ):
                    text_parts.append(content["text"])

    text = "".join(text_parts).strip()
    if not text:
        raise AIProviderError("OpenAI returned no text output.")

    return text


def _extract_chat_completion_text(response: dict, provider_name: str) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AIProviderError(f"{provider_name} returned no completion choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise AIProviderError(f"{provider_name} returned an invalid completion choice.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise AIProviderError(f"{provider_name} returned an invalid message.")

    # Deliberately read only message.content. Internal reasoning fields, when
    # present in a provider response, must never be surfaced as the answer.
    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") in ("text", "output_text")
            and isinstance(item.get("text"), str)
        ]
        combined = "".join(text_parts).strip()
        if combined:
            return combined

    raise AIProviderError(f"{provider_name} returned no text output.")


def _read_nonnegative_int(source: object, *keys: str) -> int | None:
    if not isinstance(source, dict):
        return None

    for key in keys:
        value = source.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value

    return None


def _normalize_usage(response: dict) -> AIUsage:
    """Normalize Responses API and OpenAI-compatible chat usage payloads."""

    raw_usage = response.get("usage")
    if not isinstance(raw_usage, dict):
        return AIUsage()

    input_tokens = _read_nonnegative_int(
        raw_usage,
        "input_tokens",
        "prompt_tokens",
    )
    output_tokens = _read_nonnegative_int(
        raw_usage,
        "output_tokens",
        "completion_tokens",
    )
    total_tokens = _read_nonnegative_int(raw_usage, "total_tokens")

    input_details = raw_usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = raw_usage.get("prompt_tokens_details")

    output_details = raw_usage.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = raw_usage.get("completion_tokens_details")

    cached_input_tokens = _read_nonnegative_int(
        input_details,
        "cached_tokens",
        "cache_read_input_tokens",
    )
    reasoning_tokens = _read_nonnegative_int(
        output_details,
        "reasoning_tokens",
    )

    total_tokens_calculated = False
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
        total_tokens_calculated = True

    return AIUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens_calculated=total_tokens_calculated,
    )


def _extract_finish_reason(response: dict, provider: AIProviderName) -> str | None:
    if provider == "openai":
        # The Responses API does not use the Chat Completions finish_reason field.
        return None

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    value = first_choice.get("finish_reason")
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------


async def _request_openai(
    *,
    profile: AIModelProfile,
    instructions: str,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> dict:
    payload: dict = {
        "model": profile.api_model,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }

    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}

    return await _post_json(
        url=OPENAI_RESPONSES_URL,
        api_key=_get_api_key("OPENAI_API_KEY"),
        payload=payload,
        provider_name="OpenAI",
    )


async def _request_nvidia(
    *,
    profile: AIModelProfile,
    instructions: str,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> dict:
    payload: dict = {
        "model": profile.api_model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_output_tokens,
        "stream": False,
        "chat_template_kwargs": {
            # Useful for coding-agent calls where a non-empty assistant body is
            # required. Do not expose reasoning_content as the model's answer.
            "force_nonempty_content": True,
        },
    }

    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    return await _post_json(
        url=NVIDIA_CHAT_URL,
        api_key=_get_api_key("NVIDIA_API_KEY"),
        payload=payload,
        provider_name="NVIDIA",
    )


async def _request_huggingface(
    *,
    profile: AIModelProfile,
    instructions: str,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> dict:
    # Qwen3-Coder-Next is configured as a non-thinking model in this registry.
    if reasoning_effort is not None:
        raise ValueError(
            f"{profile.display_name} does not support a reasoning_effort setting."
        )

    return await _post_json(
        url=HUGGINGFACE_CHAT_URL,
        api_key=_get_api_key("HG_TOKEN", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
        payload={
            "model": profile.api_model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_output_tokens,
            "stream": False,
        },
        provider_name="Hugging Face",
    )


# ---------------------------------------------------------------------------
# Public generation interface
# ---------------------------------------------------------------------------


async def _record_completion_usage_best_effort(
    *,
    profile: AIModelProfile,
    status: str,
    usage: AIUsage,
    duration_ms: int,
    request_id: str | None = None,
    finish_reason: str | None = None,
    error_type: str | None = None,
) -> None:
    """Persist provider metadata without allowing telemetry to break AI work."""

    try:
        # Import lazily to keep the core provider independent of DB startup.
        from app.services.ai_usage_tracking import record_ai_usage_event

        await record_ai_usage_event(
            provider=profile.provider,
            model_id=profile.model_id,
            api_model=profile.api_model,
            request_type="chat_completion",
            status=status,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            total_tokens_calculated=usage.total_tokens_calculated,
            duration_ms=duration_ms,
            provider_request_id=request_id,
            finish_reason=finish_reason,
            error_type=error_type,
        )
    except Exception:
        logger.warning(
            "AI usage tracking hook failed; continuing with the AI result.",
            exc_info=True,
        )


async def generate_completion(
    *,
    instructions: str,
    prompt: str,
    model_id: str | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> AICompletionResult:
    """Generate text, record safe telemetry, and return normalized metadata.

    ``model_id`` must be one of the allowlisted IDs in ``MODEL_PROFILES``.
    Existing callers do not need to pass it; the configured default is used.
    The database event contains token counts and request metadata only.
    """

    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("Instructions cannot be empty.")

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    profile = _resolve_profile(model_id)
    output_limit = _resolve_max_output_tokens(profile, max_output_tokens)
    effort = _resolve_reasoning_effort(profile, reasoning_effort)

    started_at = time.perf_counter()
    response: dict | None = None

    try:
        if profile.provider == "openai":
            response = await _request_openai(
                profile=profile,
                instructions=instructions,
                prompt=prompt,
                max_output_tokens=output_limit,
                reasoning_effort=effort,
            )
            generated_text = _extract_openai_text(response)

        elif profile.provider == "nvidia":
            response = await _request_nvidia(
                profile=profile,
                instructions=instructions,
                prompt=prompt,
                max_output_tokens=output_limit,
                reasoning_effort=effort,
            )
            generated_text = _extract_chat_completion_text(response, "NVIDIA")

        elif profile.provider == "huggingface":
            response = await _request_huggingface(
                profile=profile,
                instructions=instructions,
                prompt=prompt,
                max_output_tokens=output_limit,
                reasoning_effort=effort,
            )
            generated_text = _extract_chat_completion_text(response, "Hugging Face")

        else:  # Defensive guard if the registry is extended incorrectly.
            raise AIProviderError(f"Unsupported AI provider: {profile.provider}")

    except Exception as exc:
        duration_ms = max(0, round((time.perf_counter() - started_at) * 1000))
        failed_usage = _normalize_usage(response) if isinstance(response, dict) else AIUsage()
        request_id = response.get("id") if isinstance(response, dict) else None
        if not isinstance(request_id, str):
            request_id = None

        await _record_completion_usage_best_effort(
            profile=profile,
            status="failed",
            usage=failed_usage,
            duration_ms=duration_ms,
            request_id=request_id,
            finish_reason=(
                _extract_finish_reason(response, profile.provider)
                if isinstance(response, dict)
                else None
            ),
            error_type=type(exc).__name__,
        )
        raise

    duration_ms = max(0, round((time.perf_counter() - started_at) * 1000))
    response_model = response.get("model")
    actual_api_model = (
        response_model.strip()
        if isinstance(response_model, str) and response_model.strip()
        else profile.api_model
    )
    request_id = response.get("id")
    if not isinstance(request_id, str):
        request_id = None

    usage = _normalize_usage(response)
    finish_reason = _extract_finish_reason(response, profile.provider)

    await _record_completion_usage_best_effort(
        profile=profile,
        status="succeeded",
        usage=usage,
        duration_ms=duration_ms,
        request_id=request_id,
        finish_reason=finish_reason,
    )

    return AICompletionResult(
        text=generated_text,
        provider=profile.provider,
        model_id=profile.model_id,
        api_model=actual_api_model,
        usage=usage,
        duration_ms=duration_ms,
        request_id=request_id,
        finish_reason=finish_reason,
    )


async def generate_text(
    *,
    instructions: str,
    prompt: str,
    model_id: str | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Backward-compatible text-only interface used by existing ForgePath services.

    Use ``generate_completion`` when the caller needs token usage, model identity,
    request ID, or duration metadata.
    """

    result = await generate_completion(
        instructions=instructions,
        prompt=prompt,
        model_id=model_id,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    return result.text