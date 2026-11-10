import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.ai_usage_context import ai_usage_context
from app.services import ai_usage_tracking


class FakeSession:
    def __init__(self):
        self.added = []
        self.statements = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, item):
        self.added.append(item)

    async def execute(self, statement):
        self.statements.append(statement)

    async def commit(self):
        self.commits += 1


def install_session(monkeypatch):
    session = FakeSession()

    @asynccontextmanager
    async def factory():
        yield session

    monkeypatch.setattr(ai_usage_tracking, "SessionLocal", factory)
    return session


def test_record_ai_usage_event_stores_metadata_and_updates_workflow(monkeypatch):
    session = install_session(monkeypatch)
    user_id = uuid4()
    thread_id = uuid4()

    async def run():
        with ai_usage_context(
            user_id=user_id,
            thread_id=thread_id,
            operation="agent_planning",
        ):
            return await ai_usage_tracking.record_ai_usage_event(
                provider="nvidia",
                model_id="nvidia.nemotron-3-ultra",
                api_model="nvidia/nemotron-3-ultra-550b-a55b",
                status="succeeded",
                input_tokens=120,
                output_tokens=80,
                total_tokens=200,
                duration_ms=1350,
            )

    recorded = asyncio.run(run())
    assert recorded is True
    assert len(session.added) == 1
    assert session.commits == 1
    event = session.added[0]
    assert event.user_id == user_id
    assert event.thread_id == thread_id
    assert event.operation == "agent_planning"
    assert event.input_tokens == 120
    assert event.output_tokens == 80
    assert event.total_tokens == 200
    assert len(session.statements) == 1


def test_record_ai_usage_event_rejects_invalid_status():
    async def run():
        await ai_usage_tracking.record_ai_usage_event(
            provider="nvidia",
            model_id="nvidia.test",
            api_model="test",
            status="unknown",
        )

    with pytest.raises(ValueError, match="status"):
        asyncio.run(run())


def test_record_test_run_does_not_copy_command_or_output(monkeypatch):
    session = install_session(monkeypatch)
    user_id = uuid4()
    thread_id = uuid4()
    execution_id = uuid4()
    now = datetime.now(timezone.utc)

    result = SimpleNamespace(
        execution_id=execution_id,
        runner="pytest",
        status="failed",
        exit_code=1,
        duration_ms=450,
        started_at=now,
        finished_at=now,
        stdout="SENSITIVE TEST OUTPUT",
        stderr="ANOTHER SECRET",
        command=["pytest", "-q"],
        stdout_truncated=True,
        stderr_truncated=False,
        workspace_root="C:/private/project",
    )

    async def run():
        with ai_usage_context(user_id=user_id, thread_id=thread_id):
            return await ai_usage_tracking.record_agent_test_run(result)

    recorded = asyncio.run(run())
    assert recorded is True
    assert len(session.added) == 1
    assert len(session.statements) == 1
    event = session.added[0]
    assert event.execution_id == execution_id
    assert event.user_id == user_id
    assert event.thread_id == thread_id
    assert event.status == "failed"
    assert event.duration_ms == 450
    assert event.stdout_truncated is True
    assert not hasattr(event, "stdout")
    assert not hasattr(event, "stderr")
    assert not hasattr(event, "command")
    assert not hasattr(event, "workspace_root")