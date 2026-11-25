"""Task APIs require owned projects/reviews and never start agent work."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Response

from app.api.v1.routes.project_review_tasks import (
    ProjectReviewTaskCreate,
    ProjectReviewTaskDecision,
    ProjectReviewTaskPlanCreate,
    create_project_review_task,
    create_project_review_task_plan,
    decide_project_review_task,
)


FINDING = {
    "path": "src/main.ts",
    "title": "Review this call",
    "severity": "low",
    "evidence": "request()",
    "explanation": "Consider a timeout.",
    "verification_status": "unverified",
}


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeDatabase:
    def __init__(self, *results):
        self.results = iter(results)
        self.queries = []
        self.added = []
        self.commits = 0

    async def execute(self, query):
        self.queries.append(str(query))
        return FakeResult(next(self.results))

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commits += 1

    async def refresh(self, value):
        pass


def test_create_task_checks_owner_and_uses_saved_finding_without_ai():
    user = SimpleNamespace(id=uuid4())

    project_id = uuid4()
    review_id = uuid4()

    db = FakeDatabase(
        SimpleNamespace(
            id=project_id,
            status="ready",
        ),
        SimpleNamespace(
            id=review_id,
            snapshot_id=uuid4(),
            status="completed",
            findings=[FINDING],
        ),
        None,
    )

    result = asyncio.run(
        create_project_review_task(
            project_id,
            ProjectReviewTaskCreate(
                review_id=review_id,
                finding_index=0,
            ),
            Response(),
            user,
            db,
        )
    )

    assert result.user_id == user.id
    assert result.project_id == project_id
    assert result.review_id == review_id
    assert result.title == FINDING["title"]
    assert result.status == "proposed"
    assert db.commits == 1

    assert "projects.user_id" in db.queries[0]
    assert "project_reviews.user_id" in db.queries[1]
    assert "project_review_tasks.user_id" in db.queries[2]


def test_creating_task_from_foreign_project_is_hidden():
    user = SimpleNamespace(id=uuid4())
    db = FakeDatabase(None)

    with pytest.raises(HTTPException) as exception:
        asyncio.run(
            create_project_review_task(
                uuid4(),
                ProjectReviewTaskCreate(
                    review_id=uuid4(),
                    finding_index=0,
                ),
                Response(),
                user,
                db,
            )
        )

    assert exception.value.status_code == 404
    assert db.added == []


def test_creating_task_from_foreign_review_is_hidden():
    user = SimpleNamespace(id=uuid4())

    db = FakeDatabase(
        SimpleNamespace(status="ready"),
        None,
    )

    with pytest.raises(HTTPException) as exception:
        asyncio.run(
            create_project_review_task(
                uuid4(),
                ProjectReviewTaskCreate(
                    review_id=uuid4(),
                    finding_index=0,
                ),
                Response(),
                user,
                db,
            )
        )

    assert exception.value.status_code == 404
    assert db.added == []


def test_duplicate_creation_returns_existing_task():
    review_id = uuid4()

    existing = SimpleNamespace(
        status="approved",
    )

    db = FakeDatabase(
        SimpleNamespace(status="ready"),
        SimpleNamespace(
            id=review_id,
            snapshot_id=uuid4(),
            status="completed",
            findings=[FINDING],
        ),
        existing,
    )

    response = Response()

    result = asyncio.run(
        create_project_review_task(
            uuid4(),
            ProjectReviewTaskCreate(
                review_id=review_id,
                finding_index=0,
            ),
            response,
            SimpleNamespace(id=uuid4()),
            db,
        )
    )

    assert result is existing
    assert response.status_code == 200
    assert db.added == []


def test_task_approval_is_explicit_and_keeps_existing_task_data():
    task = SimpleNamespace(
        status="proposed",
        title="Review this call",
    )

    db = FakeDatabase(
        SimpleNamespace(status="ready"),
        task,
    )

    result = asyncio.run(
        decide_project_review_task(
            uuid4(),
            uuid4(),
            ProjectReviewTaskDecision(
                status="approved",
            ),
            SimpleNamespace(id=uuid4()),
            db,
        )
    )

    assert result.status == "approved"
    assert result.title == "Review this call"
    assert db.commits == 1

    assert "project_review_tasks.user_id" in db.queries[1]


def test_dismissed_task_cannot_be_approved():
    db = FakeDatabase(
        SimpleNamespace(status="ready"),
        SimpleNamespace(status="dismissed"),
    )

    with pytest.raises(HTTPException) as exception:
        asyncio.run(
            decide_project_review_task(
                uuid4(),
                uuid4(),
                ProjectReviewTaskDecision(
                    status="approved",
                ),
                SimpleNamespace(id=uuid4()),
                db,
            )
        )

    assert exception.value.status_code == 409
    assert db.commits == 0


def _approved_task(project_id, user_id, snapshot_id):
    return SimpleNamespace(
        id=uuid4(),
        user_id=user_id,
        project_id=project_id,
        snapshot_id=snapshot_id,
        status="approved",
        planning_status="not_started",
        planning_started_at=None,
        planning_token=None,
        planning_error=None,
        plan=None,
        planned_at=None,
        severity="high",
        title="Missing authorization check",
        path="app/routes/projects.py",
        explanation="The handler does not verify project ownership.",
        evidence="project = await db.get(Project, project_id)",
    )


def test_plan_rejects_stale_snapshot_before_ai(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    project_id = uuid4()
    task_snapshot_id = uuid4()
    task = _approved_task(project_id, user.id, task_snapshot_id)
    db = FakeDatabase(SimpleNamespace(status="ready"), task)

    monkeypatch.setattr(
        "app.api.v1.routes.project_review_tasks.read_snapshot_manifest",
        lambda *_: SimpleNamespace(snapshot_id=uuid4()),
    )

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("A stale task must not call the AI provider")

    monkeypatch.setattr(
        "app.api.v1.routes.project_review_tasks.generate_snapshot_task_plan",
        forbidden,
    )

    with pytest.raises(HTTPException) as exception:
        asyncio.run(
            create_project_review_task_plan(
                project_id,
                task.id,
                ProjectReviewTaskPlanCreate(
                    expected_snapshot_id=task_snapshot_id,
                ),
                user,
                db,
            )
        )

    assert exception.value.status_code == 409
    assert db.commits == 0


def test_ready_plan_is_idempotent_without_ai(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    project_id = uuid4()
    snapshot_id = uuid4()
    task = _approved_task(project_id, user.id, snapshot_id)
    task.planning_status = "ready"
    task.plan = {"plan_id": str(uuid4())}
    db = FakeDatabase(SimpleNamespace(status="ready"), task)

    monkeypatch.setattr(
        "app.api.v1.routes.project_review_tasks.read_snapshot_manifest",
        lambda *_: SimpleNamespace(snapshot_id=snapshot_id),
    )

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("An existing plan must be returned without AI")

    monkeypatch.setattr(
        "app.api.v1.routes.project_review_tasks.generate_snapshot_task_plan",
        forbidden,
    )

    result = asyncio.run(
        create_project_review_task_plan(
            project_id,
            task.id,
            ProjectReviewTaskPlanCreate(
                expected_snapshot_id=snapshot_id,
            ),
            user,
            db,
        )
    )

    assert result is task
    assert db.commits == 0


def test_active_planning_fence_rejects_duplicate_ai_call(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    project_id = uuid4()
    snapshot_id = uuid4()
    task = _approved_task(project_id, user.id, snapshot_id)
    task.planning_status = "planning"
    task.planning_started_at = datetime.now(timezone.utc)
    db = FakeDatabase(SimpleNamespace(status="ready"), task)

    monkeypatch.setattr(
        "app.api.v1.routes.project_review_tasks.read_snapshot_manifest",
        lambda *_: SimpleNamespace(snapshot_id=snapshot_id),
    )

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("A concurrent request must not call AI twice")

    monkeypatch.setattr(
        "app.api.v1.routes.project_review_tasks.generate_snapshot_task_plan",
        forbidden,
    )

    with pytest.raises(HTTPException) as exception:
        asyncio.run(
            create_project_review_task_plan(
                project_id,
                task.id,
                ProjectReviewTaskPlanCreate(
                    expected_snapshot_id=snapshot_id,
                ),
                user,
                db,
            )
        )

    assert exception.value.status_code == 409
    assert db.commits == 0


def test_successful_plan_is_saved_with_fencing_token(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    project_id = uuid4()
    snapshot_id = uuid4()
    task = _approved_task(project_id, user.id, snapshot_id)
    completed_task = _approved_task(project_id, user.id, snapshot_id)
    completed_task.id = task.id
    db = FakeDatabase(
        SimpleNamespace(status="ready"),
        task,
        completed_task,
    )

    monkeypatch.setattr(
        "app.api.v1.routes.project_review_tasks.read_snapshot_manifest",
        lambda *_: SimpleNamespace(snapshot_id=snapshot_id),
    )
    monkeypatch.setattr(
        "app.api.v1.routes.project_review_tasks.read_snapshot_sources",
        lambda *_: [("app/routes/projects.py", "def route(): pass")],
    )

    class Plan:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"plan_id": str(uuid4()), "steps": []}

    async def generate(request, sources):
        assert request.workspace_root == "private-project-snapshot"
        assert sources[0][0] == "app/routes/projects.py"
        completed_task.planning_token = task.planning_token
        return Plan()

    monkeypatch.setattr(
        "app.api.v1.routes.project_review_tasks.generate_snapshot_task_plan",
        generate,
    )

    result = asyncio.run(
        create_project_review_task_plan(
            project_id,
            task.id,
            ProjectReviewTaskPlanCreate(
                expected_snapshot_id=snapshot_id,
            ),
            user,
            db,
        )
    )

    assert result is completed_task
    assert result.planning_status == "ready"
    assert result.planning_token is None
    assert result.plan["plan_id"]
    assert result.planned_at is not None
    assert db.commits == 2
