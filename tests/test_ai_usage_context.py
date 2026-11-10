import asyncio
from uuid import uuid4

from app.core.ai_usage_context import ai_usage_context, get_ai_usage_context


def test_usage_context_defaults_and_resets():
    before = get_ai_usage_context()
    user_id = uuid4()

    with ai_usage_context(user_id=user_id, operation="code_review") as current:
        assert current.user_id == user_id
        assert current.thread_id is None
        assert current.operation == "code_review"

    assert get_ai_usage_context() == before


def test_nested_context_preserves_identity_and_restores_operation():
    user_id = uuid4()
    thread_id = uuid4()

    with ai_usage_context(user_id=user_id, thread_id=thread_id, operation="agent_workflow"):
        with ai_usage_context(operation="agent_planning") as nested:
            assert nested.user_id == user_id
            assert nested.thread_id == thread_id
            assert nested.operation == "agent_planning"

        restored = get_ai_usage_context()
        assert restored.operation == "agent_workflow"
        assert restored.user_id == user_id
        assert restored.thread_id == thread_id


def test_context_propagates_to_asyncio_to_thread():
    user_id = uuid4()

    async def run():
        with ai_usage_context(user_id=user_id, operation="workspace_embedding"):
            return await asyncio.to_thread(
                lambda: get_ai_usage_context().user_id
            )

    assert asyncio.run(run()) == user_id


def test_invalid_context_values_are_rejected():
    try:
        with ai_usage_context(user_id="not-a-uuid"):
            raise AssertionError("invalid UUID should fail before entering scope")
    except ValueError as exc:
        assert "user_id" in str(exc)

    try:
        with ai_usage_context(operation="x" * 65):
            raise AssertionError("long operation should fail before entering scope")
    except ValueError as exc:
        assert "operation" in str(exc)