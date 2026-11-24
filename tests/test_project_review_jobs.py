"""Offline lease SQL and worker behavior checks; no model or database connections."""
import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from sqlalchemy.dialects import postgresql
from app.services import project_review_agent as worker
from app.services.project_review_jobs import expire_reviews, write_owned_review


def test_claim_and_progress_are_fenced_by_token_status_and_database_time():
    async def check():
        db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(rowcount=1)), commit=AsyncMock())
        token = uuid4()
        assert await write_owned_review(db, uuid4(), token, claim=True, status="running")
        statement = db.execute.call_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        assert "lease_token =" in sql and "lease_expires_at > now()" in sql
        assert "queued" in compiled.params.values() and token in compiled.params.values()
        db.execute.return_value.rowcount = 0
        assert not await write_owned_review(db, uuid4(), token, status="completed")
        assert db.execute.call_args.args[0].compile().params["lease_expires_at"] is None
    asyncio.run(check())


def test_recovery_is_owner_scoped_and_does_not_requeue_work():
    async def check():
        db = SimpleNamespace(execute=AsyncMock())
        owner = uuid4()
        await expire_reviews(db, owner)
        compiled = db.execute.call_args.args[0].compile(dialect=postgresql.dialect())
        assert owner in compiled.params.values()
        assert "interrupted" in compiled.params.values()
        assert "lease_expires_at <= now()" in str(compiled)
        assert "lease_expires_at IS NULL" in str(compiled)
    asyncio.run(check())


def setup_worker(monkeypatch, writes, answer=None):
    class Session:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            pass
    monkeypatch.setattr(worker, "SessionLocal", Session)
    monkeypatch.setattr(worker, "write_owned_review", writes)
    monkeypatch.setattr(worker, "load_user_ai_choice", AsyncMock(return_value=None))
    monkeypatch.setattr(worker, "read_snapshot_sources", lambda *_: [("src/a.py", "value = 1")])
    completion = AsyncMock(return_value=answer or SimpleNamespace(text='{"summary":"OK","findings":[]}', finish_reason="stop"))
    monkeypatch.setattr(worker, "generate_completion", completion)
    return completion


def run_worker():
    asyncio.run(worker.run_initial_review(*(uuid4() for _ in range(5))))


def test_duplicate_or_expired_worker_never_reads_sources_or_calls_ai(monkeypatch):
    completion = setup_worker(monkeypatch, AsyncMock(return_value=False))
    def forbidden(*_):
        raise AssertionError("Unclaimed worker must not read private snapshots")
    monkeypatch.setattr(worker, "read_snapshot_sources", forbidden)
    run_worker()
    completion.assert_not_awaited()


def test_lost_lease_stops_before_paid_call(monkeypatch):
    completion = setup_worker(monkeypatch, AsyncMock(side_effect=[True, True, False]))
    run_worker()
    completion.assert_not_awaited()


def test_late_result_cannot_complete_interrupted_attempt(monkeypatch):
    writes = AsyncMock(side_effect=[True, True, True, False])
    completion = setup_worker(monkeypatch, writes)
    run_worker()
    completion.assert_awaited_once()
    assert all(call.kwargs.get("status") != "completed" for call in writes.await_args_list)


def test_invalid_ai_output_does_not_count_as_reviewed(monkeypatch):
    writes = AsyncMock(return_value=True)
    setup_worker(monkeypatch, writes, SimpleNamespace(text="invalid", finish_reason="stop"))
    run_worker()
    final = writes.await_args_list[-1].kwargs
    assert final["status"] == "partial"
    assert final["files_reviewed"] == 0


def test_timeout_fails_attempt_without_retry(monkeypatch):
    writes = AsyncMock(return_value=True)
    completion = setup_worker(monkeypatch, writes)
    completion.side_effect = TimeoutError()
    run_worker()
    completion.assert_awaited_once()
    assert writes.await_args_list[-1].kwargs["status"] == "failed"


def test_graceful_cancellation_marks_interrupted_and_does_not_retry(monkeypatch):
    writes = AsyncMock(return_value=True)
    completion = setup_worker(monkeypatch, writes)
    completion.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        run_worker()
    completion.assert_awaited_once()
    assert writes.await_args_list[-1].kwargs["status"] == "interrupted"


@pytest.mark.parametrize("previous_status,queued", [("interrupted", True), ("running", False), ("completed", False)])
def test_explicit_start_recovers_before_reuse_and_queues_only_retry(monkeypatch, previous_status, queued):
    from fastapi import BackgroundTasks
    from app.api.v1.routes import project_reviews as routes

    owner, project, snapshot, previous_id = (uuid4() for _ in range(4))
    previous = SimpleNamespace(id=previous_id, project_id=project, snapshot_id=snapshot,
                               status=previous_status, files_scanned=1, files_reviewed=0,
                               ai_calls=0, result_text=None, findings=[], error_message=None)
    events = []
    replies = iter([None, SimpleNamespace(status="ready"), previous, None])
    class DB:
        async def execute(self, statement):
            events.append(str(statement))
            value = next(replies)
            return SimpleNamespace(scalar_one_or_none=lambda: value)
        def add(self, review):
            review.id = uuid4()
            review.files_scanned = review.files_reviewed = review.ai_calls = 0
            self.added = review
        async def commit(self):
            pass
        async def refresh(self, _):
            pass
    async def expire(_db, user_id):
        assert user_id == owner
        events.append("recover-expired")
    monkeypatch.setattr(routes, "expire_reviews", expire)
    monkeypatch.setattr(routes, "read_snapshot_manifest", lambda *_: SimpleNamespace(snapshot_id=snapshot))
    tasks = BackgroundTasks()
    result = asyncio.run(routes.start_project_review(project, tasks, SimpleNamespace(id=owner), DB()))
    assert "FOR UPDATE" in events[0]
    assert events[2] == "recover-expired"
    assert len(tasks.tasks) == int(queued)
    if queued:
        assert result["id"] != str(previous_id)
        assert result["status"] == "queued"
        assert tasks.tasks[0].args[-1] is not None
    else:
        assert result["id"] == str(previous_id)
