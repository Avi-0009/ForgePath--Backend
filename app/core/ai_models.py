from dataclasses import dataclass
from typing import Literal


AIProviderName = Literal[
    "nvidia",
    "openai",
    "huggingface",
]


class UnknownAIModelError(ValueError):
    """Raised when a requested model is not registered."""


@dataclass(frozen=True, slots=True)
class AIModelProfile:
    """Metadata describing one supported AI model."""

    model_id: str
    provider: AIProviderName
    api_model: str
    display_name: str
    description: str
    context_window_tokens: int | None = None
    capabilities: frozenset[str] = frozenset()


DEFAULT_MODEL_ID = "nvidia.nemotron-3-ultra"


MODEL_REGISTRY: dict[str, AIModelProfile] = {
    "nvidia.nemotron-3-ultra": AIModelProfile(
        model_id="nvidia.nemotron-3-ultra",
        provider="nvidia",
        api_model="nvidia/nemotron-3-ultra-550b-a55b",
        display_name="Nemotron 3 Ultra",
        description="NVIDIA general-purpose reasoning and coding model.",
        capabilities=frozenset({
            "code_review",
            "code_generation",
            "task_planning",
            "failure_analysis",
            "structured_output",
        }),
    ),

    "openai.gpt-6-luna": AIModelProfile(
        model_id="openai.gpt-6-luna",
        provider="openai",
        api_model="gpt-6-luna",
        display_name="GPT-6 Luna",
        description="OpenAI model for reasoning and software development.",
        capabilities=frozenset({
            "code_review",
            "code_generation",
            "task_planning",
            "failure_analysis",
            "structured_output",
        }),
    ),

    "huggingface.qwen3-coder-480b": AIModelProfile(
        model_id="huggingface.qwen3-coder-480b",
        provider="huggingface",
        api_model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        display_name="Qwen3 Coder 480B",
        description=(
            "Coding-focused model for repository understanding, "
            "code generation and agentic coding workflows."
        ),
        context_window_tokens=262_144,
        capabilities=frozenset({
            "code_review",
            "code_generation",
            "task_planning",
            "failure_analysis",
            "agentic_coding",
            "structured_output",
        }),
    ),
}


def get_model_profile(model_id: str) -> AIModelProfile:
    """Retrieve a registered model by its public identifier."""

    profile = MODEL_REGISTRY.get(model_id)

    if profile is None:
        raise UnknownAIModelError(
            f"Unsupported AI model: {model_id}"
        )

    return profile


def list_model_profiles(
    provider: AIProviderName | None = None,
) -> list[AIModelProfile]:
    """List registered models, optionally filtered by provider."""

    profiles = list(MODEL_REGISTRY.values())

    if provider is not None:
        profiles = [
            profile
            for profile in profiles
            if profile.provider == provider
        ]

    return profiles