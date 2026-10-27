import asyncio
from unittest.mock import AsyncMock, Mock

from app.api.v1.routes import code_review as code_review_routes
from app.schemas.code_review import (
    CodeReviewFileRequest,
    CodeReviewResponse,
)
from app.services.local_context import ProjectContext
from app.services.workspace import WorkspaceFile


def test_local_file_review_uses_saved_project_instructions(monkeypatch):
    source = WorkspaceFile(
        relative_path="app/example.py",
        language="python",
        content="def add(a, b):\n    return a + b\n",
        sha256="a" * 64,
        size_bytes=33,
        line_count=2,
    )

    saved_instructions = (
        "Use Python 3.13. Prefer small, maintainable changes."
    )

    context = ProjectContext(
        project_root="C:/projects/example",
        project_name="example",
        instructions=saved_instructions,
        created_at="2026-10-27T10:00:00+00:00",
        updated_at="2026-10-27T10:00:00+00:00",
    )

    monkeypatch.setattr(
        code_review_routes,
        "read_configured_workspace_file",
        Mock(return_value=source),
    )

    monkeypatch.setattr(
        code_review_routes,
        "get_project_instructions",
        AsyncMock(return_value=context),
    )

    review_spy = AsyncMock(
        return_value=CodeReviewResponse(
            summary="The function is simple and correct.",
            findings=[],
            strengths=["Straightforward implementation."],
            suggested_tests=["Test positive and negative inputs."],
        )
    )

    monkeypatch.setattr(
        code_review_routes,
        "review_code",
        review_spy,
    )

    request = CodeReviewFileRequest(
        workspace_root="C:/projects/example",
        relative_path="app/example.py",
        review_mode="general",
    )

    result = asyncio.run(
        code_review_routes.review_local_file(
            request,
            _current_user=object(),
        )
    )

    review_spy.assert_awaited_once()

    assert (
        review_spy.await_args.kwargs["project_instructions"]
        == saved_instructions
    )

    assert result.relative_path == "app/example.py"
    assert result.review.summary == (
        "The function is simple and correct."
    )
