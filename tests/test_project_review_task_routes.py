"""Task APIs require owned projects/reviews and never start agent work."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Response

from app.api.v1.routes.project_review_tasks import (
    ProjectReviewTaskCreate,
    ProjectReviewTaskDecision,
    create_project_review_task,
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