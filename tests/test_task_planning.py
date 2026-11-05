import asyncio
import json

import pytest

from app.schemas.task_planning import TaskPlanningRequest
from app.services.task_planning import planner
from app.services.workspace import WorkspaceAccessDenied
from app.services.workspace_ingestion import (
    IngestedWorkspaceDocument,
    WorkspaceIngestionResult,
)


def make_document(
    workspace_root: str,
    relative_path: str,
    content: str,
) -> IngestedWorkspaceDocument:
    import hashlib

    return IngestedWorkspaceDocument(
        workspace_root=workspace_root,
        relative_path=relative_path,
        language="python",
        content=content,
        sha256=hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest(),
        size_bytes=len(content.encode("utf-8")),
        line_count=len(content.splitlines()),
    )


def make_ingestion(workspace_root: str) -> WorkspaceIngestionResult:
    documents = (
        make_document(
            workspace_root,
            "README.md",
            "# Sample Project\nA small FastAPI project.",
        ),
        make_document(
            workspace_root,
            "app/main.py",
            "from fastapi import FastAPI\napp = FastAPI()",
        ),
        make_document(
            workspace_root,
            "app/services/users.py",
            "def get_user(user_id):\n    return None",
        ),
    )

    return WorkspaceIngestionResult(
        workspace_root=workspace_root,
        documents=documents,
        files_scanned=3,
        files_skipped=0,
        total_bytes=sum(doc.size_bytes for doc in documents),
        truncated=False,
        observed_relative_paths=tuple(
            doc.relative_path for doc in documents
        ),
    )


def make_valid_plan() -> dict:
    return {
        "task_summary": "Add a user export endpoint.",
        "objective": "Allow users to export their account data.",
        "complexity": "medium",
        "assumptions": [
            "The existing authentication system will be reused."
        ],
        "steps": [
            {
                "step_number": 1,
                "title": "Inspect existing user services",
                "description": "Understand the current user data access.",
                "dependencies": [],
                "candidate_files_to_inspect": [
                    "app/services/users.py",
                    "fake_file.py",
                ],
                "acceptance_criteria": [
                    "Existing user data access patterns are understood."
                ],
                "verification_actions": [
                    "Review the existing user service tests."
                ],
            },
            {
                "step_number": 2,
                "title": "Implement the export endpoint",
                "description": "Add the endpoint using existing conventions.",
                "dependencies": [1],
                "candidate_files_to_inspect": [
                    "app/main.py",
                ],
                "acceptance_criteria": [
                    "Authenticated users can request their data export."
                ],
                "verification_actions": [
                    "Add endpoint tests."
                ],
            },
        ],
        "risks": [],
        "open_questions": [],
        "overall_verification_strategy": [
            "Run the relevant API tests."
        ],
    }


def make_request(
    workspace_root: str,
    *,
    max_steps: int = 10,
) -> TaskPlanningRequest:
    return TaskPlanningRequest(
        workspace_root=workspace_root,
        task_description="Add a user data export endpoint.",
        planning_mode="feature",
        max_steps=max_steps,
    )


def test_generate_task_plan_returns_valid_plan(
    monkeypatch,
    tmp_path,
):
    workspace_root = str(tmp_path.resolve())

    ingestion = make_ingestion(workspace_root)

    def fake_ingest_workspace(
        workspace_root,
        **kwargs,
    ):
        assert kwargs["max_files"] == planner.MAX_INGESTION_FILES
        assert kwargs["max_total_bytes"] == planner.MAX_INGESTION_BYTES
        return ingestion

    captured = {}

    async def fake_generate_text(*, instructions, prompt):
        captured["instructions"] = instructions
        captured["prompt"] = prompt

        return json.dumps(make_valid_plan())

    monkeypatch.setattr(
        planner,
        "ingest_workspace",
        fake_ingest_workspace,
    )

    monkeypatch.setattr(
        planner,
        "generate_text",
        fake_generate_text,
    )

    result = asyncio.run(
        planner.generate_task_plan(
            make_request(workspace_root),
            project_instructions="Use existing FastAPI conventions.",
        )
    )

    assert result.workspace_root == workspace_root
    assert result.task_summary == "Add a user export endpoint."
    assert result.complexity == "medium"
    assert len(result.steps) == 2

    # Candidate paths absent from the snapshot must be removed.
    assert result.steps[0].candidate_files_to_inspect == [
        "app/services/users.py"
    ]

    assert result.steps[1].candidate_files_to_inspect == [
        "app/main.py"
    ]

    assert result.steps[1].dependencies == [1]

    assert "untrusted" in captured["instructions"].lower()
    assert "user data export" in captured["prompt"].lower()
    assert "app/main.py" in captured["prompt"]
    assert "Use existing FastAPI conventions." in captured["prompt"]


def test_build_project_snapshot_is_bounded(tmp_path):
    workspace_root = str(tmp_path.resolve())

    documents = tuple(
        make_document(
            workspace_root,
            f"module_{index}.py",
            f"# Module {index}\n" + ("x = 1\n" * 300),
        )
        for index in range(25)
    )

    ingestion = WorkspaceIngestionResult(
        workspace_root=workspace_root,
        documents=documents,
        files_scanned=25,
        files_skipped=0,
        total_bytes=sum(doc.size_bytes for doc in documents),
        truncated=False,
        observed_relative_paths=tuple(
            document.relative_path
            for document in documents
        ),
    )

    snapshot = planner._build_project_snapshot(ingestion)

    assert len(snapshot["project_files"]) == 25
    assert len(snapshot["project_file_excerpts"]) <= (
        planner.MAX_CONTEXT_FILES
    )

    total_excerpt_chars = sum(
        len(item["content_excerpt"])
        for item in snapshot["project_file_excerpts"]
    )

    assert total_excerpt_chars <= planner.MAX_TOTAL_EXCERPT_CHARS


def test_generate_task_plan_rejects_malformed_json(
    monkeypatch,
    tmp_path,
):
    workspace_root = str(tmp_path.resolve())

    monkeypatch.setattr(
        planner,
        "ingest_workspace",
        lambda workspace_root, **kwargs: make_ingestion(
            workspace_root
        ),
    )

    async def fake_generate_text(*, instructions, prompt):
        return '{"task_summary": invalid json}'

    monkeypatch.setattr(
        planner,
        "generate_text",
        fake_generate_text,
    )

    with pytest.raises(
        planner.TaskPlanningError,
        match="malformed JSON",
    ):
        asyncio.run(
            planner.generate_task_plan(
                make_request(workspace_root)
            )
        )


def test_generate_task_plan_rejects_invalid_dependencies(
    monkeypatch,
    tmp_path,
):
    workspace_root = str(tmp_path.resolve())

    invalid_plan = make_valid_plan()
    invalid_plan["steps"][0]["dependencies"] = [2]

    monkeypatch.setattr(
        planner,
        "ingest_workspace",
        lambda workspace_root, **kwargs: make_ingestion(
            workspace_root
        ),
    )

    async def fake_generate_text(*, instructions, prompt):
        return json.dumps(invalid_plan)

    monkeypatch.setattr(
        planner,
        "generate_text",
        fake_generate_text,
    )

    with pytest.raises(
        planner.TaskPlanningError,
        match="invalid task plan",
    ):
        asyncio.run(
            planner.generate_task_plan(
                make_request(workspace_root)
            )
        )


def test_generate_task_plan_enforces_maximum_steps(
    monkeypatch,
    tmp_path,
):
    workspace_root = str(tmp_path.resolve())

    monkeypatch.setattr(
        planner,
        "ingest_workspace",
        lambda workspace_root, **kwargs: make_ingestion(
            workspace_root
        ),
    )

    async def fake_generate_text(*, instructions, prompt):
        return json.dumps(make_valid_plan())

    monkeypatch.setattr(
        planner,
        "generate_text",
        fake_generate_text,
    )

    with pytest.raises(
        planner.TaskPlanningError,
        match="maximum number of steps",
    ):
        asyncio.run(
            planner.generate_task_plan(
                make_request(
                    workspace_root,
                    max_steps=1,
                )
            )
        )


def test_generate_task_plan_preserves_workspace_access_errors(
    monkeypatch,
    tmp_path,
):
    def fake_ingest_workspace(workspace_root, **kwargs):
        raise WorkspaceAccessDenied("Workspace not allowed.")

    monkeypatch.setattr(
        planner,
        "ingest_workspace",
        fake_ingest_workspace,
    )

    with pytest.raises(
        WorkspaceAccessDenied,
        match="not allowed",
    ):
        asyncio.run(
            planner.generate_task_plan(
                make_request(str(tmp_path))
            )
        )


def test_validate_candidate_files_removes_unknown_paths():
    plan_data = make_valid_plan()

    from app.schemas.task_planning import TaskPlanningResponse

    plan = TaskPlanningResponse(
        workspace_root="C:/Projects/Sample",
        **plan_data,
    )

    result = planner._validate_candidate_files(
        plan,
        {
            "app/main.py",
            "app/services/users.py",
        },
    )

    assert result.steps[0].candidate_files_to_inspect == [
        "app/services/users.py"
    ]

    assert result.steps[1].candidate_files_to_inspect == [
        "app/main.py"
    ]