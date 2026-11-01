from types import SimpleNamespace
from uuid import uuid4

import pytest

from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.main import app
from app.schemas.code_review import CodeReviewResponse

from app.api.v1.routes import code_review as code_review_route


REVIEW_URL = "/api/v1/code-review/file"


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()

    source = root / "main.py"
    source.write_bytes(
        b"def main():\n"
        b"    return authenticate_user()\n"
    )

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(root),
    )

    # Keep integration tests independent of real API clients.
    monkeypatch.setattr(
        settings,
        "embedding_provider",
        None,
    )

    async def override_db():
        yield SimpleNamespace()

    async def override_current_user():
        return SimpleNamespace(
            id=uuid4(),
            is_active=True,
        )

    app.dependency_overrides.clear()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = (
        override_current_user
    )

    try:
        with TestClient(app) as client:
            yield client, root
    finally:
        app.dependency_overrides.clear()


def make_review_result():
    return CodeReviewResponse(
        summary="A potential issue was identified.",
        findings=[],
        strengths=["The function is easy to understand."],
        suggested_tests=["Test authentication failure."],
    )


def install_fake_review(monkeypatch):
    captured = {}

    async def fake_review_code(
        request,
        *,
        project_instructions=None,
        related_workspace_context=None,
    ):
        captured["request"] = request
        captured["project_instructions"] = project_instructions
        captured["related_workspace_context"] = (
            related_workspace_context
        )

        return make_review_result()

    monkeypatch.setattr(
        code_review_route,
        "review_code",
        fake_review_code,
    )

    async def fake_project_instructions(**kwargs):
        return None

    monkeypatch.setattr(
        code_review_route,
        "get_project_instructions",
        fake_project_instructions,
    )

    return captured


def send_review(client, root):
    return client.post(
        REVIEW_URL,
        json={
            "workspace_root": str(root),
            "relative_path": "main.py",
            "review_mode": "general",
        },
    )


# ─────────────────────────────────────────────
# RAG ENABLED
# ─────────────────────────────────────────────

def test_code_review_passes_retrieved_context_to_ai(
    api_client,
    monkeypatch,
):
    client, root = api_client

    app.state.workspace_indexing_service = object()

    captured = install_fake_review(monkeypatch)

    expected_context = """
[
  {
    "relative_path": "app/auth.py",
    "language": "python",
    "start_line": 1,
    "end_line": 5,
    "content": "def authenticate_user(): return True"
  }
]
"""

    builder_calls = {}

    class FakeContext:
        is_empty = False
        chunk_count = 1
        total_chars = 150

        @staticmethod
        def to_prompt_text():
            return expected_context

    class FakeContextBuilder:
        def __init__(self, indexing_service):
            builder_calls["indexing_service"] = indexing_service

        async def build(self, **kwargs):
            builder_calls["arguments"] = kwargs
            return FakeContext()

    monkeypatch.setattr(
        code_review_route,
        "WorkspaceReviewContextBuilder",
        FakeContextBuilder,
    )

    response = send_review(client, root)

    assert response.status_code == 200, response.text

    assert response.json()["review"]["summary"] == (
        "A potential issue was identified."
    )

    assert captured["related_workspace_context"] == expected_context

    assert captured["project_instructions"] is None

    assert builder_calls["arguments"]["workspace_root"] == str(root)
    assert builder_calls["arguments"]["relative_path"] == "main.py"

    assert builder_calls["arguments"]["source_content"] == (
        "def main():\n"
        "    return authenticate_user()\n"
    )

    assert builder_calls["arguments"]["language"] == "python"

    assert builder_calls["indexing_service"] is (
        app.state.workspace_indexing_service
    )


# ─────────────────────────────────────────────
# RAG DISABLED
# ─────────────────────────────────────────────

def test_code_review_works_without_indexing_service(
    api_client,
    monkeypatch,
):
    client, root = api_client

    app.state.workspace_indexing_service = None

    captured = install_fake_review(monkeypatch)

    def unexpected_builder(*args, **kwargs):
        pytest.fail(
            "Context builder should not be initialized when indexing is disabled."
        )

    monkeypatch.setattr(
        code_review_route,
        "WorkspaceReviewContextBuilder",
        unexpected_builder,
    )

    response = send_review(client, root)

    assert response.status_code == 200, response.text

    assert captured["related_workspace_context"] is None

    assert response.json()["review"]["summary"] == (
        "A potential issue was identified."
    )


# ─────────────────────────────────────────────
# RAG RETRIEVAL FAILURE
# ─────────────────────────────────────────────

def test_code_review_falls_back_when_context_retrieval_fails(
    api_client,
    monkeypatch,
):
    client, root = api_client

    app.state.workspace_indexing_service = object()

    captured = install_fake_review(monkeypatch)

    class FailingContextBuilder:
        def __init__(self, indexing_service):
            pass

        async def build(self, **kwargs):
            raise RuntimeError(
                "Simulated semantic retrieval failure."
            )

    monkeypatch.setattr(
        code_review_route,
        "WorkspaceReviewContextBuilder",
        FailingContextBuilder,
    )

    response = send_review(client, root)

    assert response.status_code == 200, response.text

    assert captured["related_workspace_context"] is None

    assert response.json()["review"]["summary"] == (
        "A potential issue was identified."
    )


# ─────────────────────────────────────────────
# EMPTY VERIFIED CONTEXT
# ─────────────────────────────────────────────

def test_code_review_continues_when_retrieval_is_empty(
    api_client,
    monkeypatch,
):
    client, root = api_client

    app.state.workspace_indexing_service = object()

    captured = install_fake_review(monkeypatch)

    class EmptyContext:
        is_empty = True
        chunk_count = 0
        total_chars = 0

        @staticmethod
        def to_prompt_text():
            return "No verified context."

    class EmptyContextBuilder:
        def __init__(self, indexing_service):
            pass

        async def build(self, **kwargs):
            return EmptyContext()

    monkeypatch.setattr(
        code_review_route,
        "WorkspaceReviewContextBuilder",
        EmptyContextBuilder,
    )

    response = send_review(client, root)

    assert response.status_code == 200, response.text

    assert captured["related_workspace_context"] is None