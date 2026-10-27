import os
from typing import Literal, cast

import httpx
from dotenv import load_dotenv

AI_PROVIDER: Literal["openai", "nvidia"] = "nvidia"

OPENAI_MODEL = "gpt-6-luna"
OPENAI_REASONING_EFFORT = "low"

NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_ENABLE_THINKING = True
NVIDIA_MAX_TOKENS = 16384

REQUEST_TIMEOUT_SECONDS = 240

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
# HUGGINGFACE_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

load_dotenv()

class AIProviderError(RuntimeError): 
    """Raised when an AI provider request fails."""

def _get_api_key(env_var: str) -> str:
    api_key = os.getenv(env_var, "").strip()

    if not api_key:
        raise AIProviderError(
            f"Missing {env_var}. "
            "Add it to your backend .env file."
        )

    return api_key

async def _post_json(
    url: str,
    api_key: str,
    payload: dict,
    provider_name: str,
) -> dict :
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
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
                raise AIProviderError(f"{provider_name} returned an unexpected response.")

            return result

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code

        if status_code in (401, 403):
            detail = "Check the API key and model access."
        elif status_code == 429:
            detail = "Rate limit or provider quota reached."
        else:
            detail = "Check the provider configuration and request."

        raise AIProviderError(
            f"{provider_name} request failed "
            f"(HTTP {status_code}). {detail}"
        ) from None

    except httpx.RequestError as exc:
        raise AIProviderError(
            f"Could not connect to {provider_name} "
            f"({type(exc).__name__})."
        ) from None

    except ValueError:
        raise AIProviderError(f"{provider_name} returned invalid JSON.") from None


def _extract_openai_text(response: dict) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"].strip()

    text_parts: list[str] = []

    for item in response.get("output", []):
        if item.get("type") != "message":
            continue

        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text_parts.append(content.get("text", ""))

    text = "".join(text_parts).strip()

    if not text:
        raise AIProviderError(
            "OpenAI returned no text output."
        )

    return text


def _extract_nvidia_text(response: dict) -> str:
    """Extract generated text from an NVIDIA chat-completions response."""

    choices = response.get("choices")

    if not isinstance(choices, list) or not choices:
        raise AIProviderError(
            "NVIDIA returned no completion choices."
        )

    message = choices[0].get("message")

    if not isinstance(message, dict):
        raise AIProviderError(
            "NVIDIA returned an invalid message."
        )

    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ]

        result = "".join(text_parts).strip()

        if result:
            return result

    raise AIProviderError(
        "NVIDIA returned no text output."
    )


async def generate_text(
    *,
    instructions: str,
    prompt: str,
) -> str:
    """
    Generate text using the configured AI provider.

    Other ForgePath services should call this function rather
    than creating provider clients themselves.
    """

    if not instructions.strip():
        raise ValueError("Instructions cannot be empty.")

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    if AI_PROVIDER == "openai":
        api_key = _get_api_key("OPENAI_API_KEY")

        response = await _post_json(
            url=OPENAI_RESPONSES_URL,
            api_key=api_key,
            provider_name="OpenAI",
            payload={
                "model": OPENAI_MODEL,
                "instructions": instructions,
                "input": prompt,
                "reasoning": {
                    "effort": OPENAI_REASONING_EFFORT,
                },
            },
        )

        return _extract_openai_text(response)

    if AI_PROVIDER == "nvidia":
        api_key = _get_api_key("NVIDIA_API_KEY")

        response = await _post_json(
            url=NVIDIA_CHAT_URL,
            api_key=api_key,
            provider_name="NVIDIA",
            payload={
                "model": NVIDIA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": instructions,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": 1,
                "top_p": 0.95,
                "max_tokens": NVIDIA_MAX_TOKENS,
                "chat_template_kwargs": {
                    "enable_thinking": NVIDIA_ENABLE_THINKING,
                },
            },
        )

        return _extract_nvidia_text(response)
    
    raise AIProviderError(
        f"Unsupported AI provider: {cast(str, AI_PROVIDER)}"
    )