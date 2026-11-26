"""Project workflow contracts without network, providers, or a live database."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.routes import project_workflows as workflow


def make_task(*, status="approved", decision="pending", planning="ready",
              patch_status="not_started"):
    return SimpleNamespace(
        id=uuid4(), project_id=uuid4(), snapshot_id=uuid4(),
        status=status, planning_status=planning, plan_decision=decision,
        plan={"plan_id": str(uuid4())}, plan_decided_at=None,
        patch_status=patch_status, patch=None,
    )


class FakeDatabase:
    def __init__(self, project, task):
        self.execute = AsyncMock(side_effect=[
            SimpleNamespace(scalar_one_or_none=lambda: project),
            SimpleNamespace(scalar_one_or_none=lambda: task),
        ])
        self.commit = AsyncMock()
        self.refresh = AsyncMock()


class FakeListDatabase:
    def __init__(self, project_ids):
        self.execute = AsyncMock(side_effect=[
            SimpleNamespace(all=lambda: [(project_id,) for project_id in project_ids]),
            SimpleNamespace(all=lambda: []),
        ])
        self.scalar = AsyncMock(return_value=0)


async def decide(monkeypatch, task, db, *, requested="approved",
                 snapshot=None, plan=None):
    monkeypatch.setattr(
        workflow, "read_snapshot_manifest",
        lambda *_: SimpleNamespace(snapshot_id=task.snapshot_id),
    )
    return await workflow.decide_project_plan(
        task.project_id,
        task.id,
        workflow.ProjectPlanDecision(
            status=requested,
            expected_snapshot_id=snapshot or task.snapshot_id,
            expected_plan_id=plan or task.plan["plan_id"],
        ),
        SimpleNamespace(id=uuid4()),
        db,
    )


def test_stage_machine_is_distinct_from_legacy_langgraph():
    task = make_task(status="proposed")
    assert workflow.pending_for_task(task) == "finding"
    task.status = "approved"
    assert workflow.pending_for_task(task) == "plan"
    assert workflow.stage_for_task(task) == "plan_approval"
    task.plan_decision = "rejected"
    assert workflow.pending_for_task(task) is None
    assert workflow.stage_for_task(task) == "plan_rejected"
    task.plan_decision = "approved"
    task.patch_status = "ready"
    task.patch = {"diff": "safe preview"}
    assert workflow.pending_for_task(task) == "patch"
    assert workflow.stage_for_task(task) == "patch_ready"


def test_plan_approval_persists_and_duplicate_is_idempotent(monkeypatch):
    task = make_task()
    first = FakeDatabase(SimpleNamespace(status="ready"), task)
    result = asyncio.run(decide(monkeypatch, task, first))
    assert result.plan_decision == "approved"
    assert result.plan_decided_at is not None
    first.commit.assert_awaited_once()
    project_query = str(first.execute.await_args_list[0].args[0])
    task_query = str(first.execute.await_args_list[1].args[0])
    assert "projects.user_id" in project_query
    assert "project_review_tasks.user_id" in task_query
    assert "FOR UPDATE" in project_query
    assert "FOR UPDATE" in task_query

    duplicate = FakeDatabase(SimpleNamespace(status="ready"), task)
    asyncio.run(decide(monkeypatch, task, duplicate))
    duplicate.commit.assert_not_awaited()


def test_resolved_plan_decision_cannot_be_reversed(monkeypatch):
    task = make_task(decision="approved")
    db = FakeDatabase(SimpleNamespace(status="ready"), task)
    with pytest.raises(HTTPException) as error:
        asyncio.run(decide(monkeypatch, task, db, requested="rejected"))
    assert error.value.status_code == 409
    db.commit.assert_not_awaited()


def test_foreign_project_is_hidden(monkeypatch):
    task = make_task()
    db = FakeDatabase(None, task)
    with pytest.raises(HTTPException) as error:
        asyncio.run(decide(monkeypatch, task, db))
    assert error.value.status_code == 404
    assert db.execute.await_count == 1


def test_payload_and_active_snapshot_are_revalidated(monkeypatch):
    task = make_task()
    wrong_payload = FakeDatabase(SimpleNamespace(status="ready"), task)
    with pytest.raises(HTTPException) as error:
        asyncio.run(decide(monkeypatch, task, wrong_payload, snapshot=uuid4()))
    assert error.value.status_code == 409

    stale = FakeDatabase(SimpleNamespace(status="ready"), task)
    monkeypatch.setattr(
        workflow, "read_snapshot_manifest",
        lambda *_: SimpleNamespace(snapshot_id=uuid4()),
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(workflow.decide_project_plan(
            task.project_id, task.id,
            workflow.ProjectPlanDecision(
                status="approved",
                expected_snapshot_id=task.snapshot_id,
                expected_plan_id=task.plan["plan_id"],
            ),
            SimpleNamespace(id=uuid4()), stale,
        ))
    assert error.value.status_code == 409
    stale.commit.assert_not_awaited()


def test_plan_decision_cannot_change_after_patch_exists(monkeypatch):
    task = make_task()
    task.patch_status = "ready"
    task.patch = {"diff": "proposal"}
    db = FakeDatabase(SimpleNamespace(status="ready"), task)
    with pytest.raises(HTTPException) as error:
        asyncio.run(decide(monkeypatch, task, db))
    assert error.value.status_code == 409


def test_pending_queue_filters_stale_projects_before_count(monkeypatch):
    db = FakeListDatabase([uuid4()])
    monkeypatch.setattr(workflow, "read_snapshot_manifest", lambda *_: None)
    result = asyncio.run(workflow.list_project_workflows(
        SimpleNamespace(id=uuid4()), db, limit=30, offset=0, pending_only=True,
    ))
    assert result.items == []
    assert result.total == 0
    db.scalar.assert_not_awaited()
    assert db.execute.await_count == 1
    candidate_query = str(db.execute.await_args_list[0].args[0])
    assert "project_review_tasks.user_id" in candidate_query
    assert "projects.user_id" in candidate_query


def test_pending_queue_counts_after_snapshot_filter(monkeypatch):
    db = FakeListDatabase([uuid4()])
    monkeypatch.setattr(
        workflow, "read_snapshot_manifest",
        lambda *_: SimpleNamespace(snapshot_id=uuid4()),
    )
    result = asyncio.run(workflow.list_project_workflows(
        SimpleNamespace(id=uuid4()), db, limit=30, offset=0, pending_only=True,
    ))
    assert result.total == 0
    db.scalar.assert_awaited_once()
    assert db.execute.await_count == 2


class ConcurrentDatabase:
    def __init__(self, project, task, lock):
        self.project = project
        self.task = task
        self.lock = lock
        self.calls = 0
        self.commits = 0

    async def execute(self, _statement):
        if self.calls == 0:
            await self.lock.acquire()
            value = self.project
        else:
            value = self.task
        self.calls += 1
        return SimpleNamespace(scalar_one_or_none=lambda: value)

    async def commit(self):
        self.commits += 1
        self.lock.release()

    async def refresh(self, _task):
        return None


def test_concurrent_duplicate_decisions_persist_one_transition(monkeypatch):
    task = make_task()
    project = SimpleNamespace(status="ready")

    async def run():
        lock = asyncio.Lock()
        first = ConcurrentDatabase(project, task, lock)
        second = ConcurrentDatabase(project, task, lock)
        monkeypatch.setattr(
            workflow, "read_snapshot_manifest",
            lambda *_: SimpleNamespace(snapshot_id=task.snapshot_id),
        )
        payload = workflow.ProjectPlanDecision(
            status="approved",
            expected_snapshot_id=task.snapshot_id,
            expected_plan_id=task.plan["plan_id"],
        )
        results = await asyncio.gather(
            workflow.decide_project_plan(
                task.project_id, task.id, payload, SimpleNamespace(id=uuid4()), first,
            ),
            workflow.decide_project_plan(
                task.project_id, task.id, payload, SimpleNamespace(id=uuid4()), second,
            ),
        )
        return first, second, results

    first, second, results = asyncio.run(run())
    assert [result.plan_decision for result in results] == ["approved", "approved"]
    assert first.commits + second.commits == 1


def test_migration_preserves_patched_plans_and_chains_from_current_head():
    migration = Path(__file__).resolve().parents[1] / "alembic" / "versions" / \
        "c98bf6d4021a_project_plan_decisions.py"
    source = migration.read_text(encoding="utf-8")
    assert 'down_revision = "e91b7a4c2d60"' in source
    assert "jsonb_typeof(patch) = 'object'" in source
    assert "SET plan_decision = 'approved'" in source
    assert "SET plan_decision = 'pending'" in source
