"""Request-scoped AI configuration for ForgePath."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class UserAIChoice:
    provider: str
    model_identifier: str
    api_key: str


_CURRENT_USER_AI_CHOICE: ContextVar[
    UserAIChoice | None
] = ContextVar(
    "forgepath_user_ai_choice",
    default=None,
)


def get_user_ai_choice() -> UserAIChoice | None:
    """Return the AI configuration for the current request."""

    return _CURRENT_USER_AI_CHOICE.get()


@contextmanager
def user_ai_choice(
    choice: UserAIChoice | None,
) -> Iterator[None]:
    """
    Temporarily activate a user's AI configuration.

    The previous configuration is restored automatically.
    """

    token = _CURRENT_USER_AI_CHOICE.set(choice)

    try:
        yield

    finally:
        _CURRENT_USER_AI_CHOICE.reset(token)