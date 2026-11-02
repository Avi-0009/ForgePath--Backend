import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.main import app
from app.schemas.code_patch import CodePatchResponse

from app.api.v1.routes import code_patch as code_patch_route


PATCH_URL = "/api/v1/code-patch"

ORIGINAL_CODE = (
    "def authenticate_user():\n"
    "    return True\n"
)

PROPOSED_CODE = (
    "def authenticate_user():\n"
    "    return False\n"
)


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    source_file = workspace_root / "main.py"
    source_file.write_bytes(ORIGINAL_CODE.encode("utf-8"))

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(workspace_root),
    )

    # Keep the test independent of real embedding providers.
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
            yield client, workspace_root, source_file

    finally:
        app.dependency_overrides.clear()


def make_patch_result():
    return CodePatchResponse(
        relative_path="main.py",
        original_sha256="a" * 64,
        proposed_sha256="b" * 64,
        original_line_count=2,
        proposed_line_count=2,
        summary="Update the authentication result.",
        diff=(
            "--- a/main.py\n"
            "+++ b/main.py\n"
            "-    return True\n"
            "+    return False\n"
        ),
        proposed_code=PROPOSED_CODE,
        patch_applied=False,
    )


def install_fake_patch(monkeypatch):
    captured = {}

    async def fake_generate_code_patch(
        payload,
        *,
        related_workspace_context=None,
    ):
        captured["payload"] = payload
        captured["related_workspace_context"] = (
            related_workspace_context
        )

        return make_patch_result()

    monkeypatch.setattr(
        code_patch_route,
        "generate_code_patch",
        fake_generate_code_patch,
    )

    return captured


def send_patch(client, workspace_root, source_file):
    source_hash = hashlib.sha256(
        source_file.read_bytes()
    ).hexdigest()

    return client.post(
        PATCH_URL,
        json={
            "workspace_root": str(workspace_root),
            "relative_path": "main.py",
            "expected_sha256": source_hash,
            "finding_title": "Incorrect authentication result",
            "finding_evidence": "return True",
            "finding_explanation": (
                "The function should reject this authentication attempt."
            ),
            "suggested_fix": "Return False.",
        },
    )


# ─────────────────────────────────────────────
# RAG ENABLED
# ─────────────────────────────────────────────

def test_code_patch_passes_retrieved_context_to_ai(
    api_client,
    monkeypatch,
):
    client, workspace_root, source_file = api_client

    indexing_service = object()
    app.state.workspace_indexing_service = indexing_service

    captured = install_fake_patch(monkeypatch)

    expected_context = """
[
  {
    "relative_path": "app/auth.py",
    "language": "python",
    "start_line": 1,
    "end_line": 5,
    "content": "def authenticate_user(): return False"
  }
]
"""

    builder_calls = {}

    class FakeContext:
        is_empty = False
        chunk_count = 1
        total_chars = len(expected_context)

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
        code_patch_route,
        "WorkspaceReviewContextBuilder",
        FakeContextBuilder,
    )

    response = send_patch(
        client,
        workspace_root,
        source_file,
    )

    assert response.status_code == 200, response.text

    assert response.json()["patch_applied"] is False

    assert captured["related_workspace_context"] == expected_context

    assert builder_calls["indexing_service"] is indexing_service

    assert builder_calls["arguments"]["workspace_root"] == str(
        workspace_root
    )

    assert builder_calls["arguments"]["relative_path"] == "main.py"

    assert builder_calls["arguments"]["source_content"] == ORIGINAL_CODE

    assert builder_calls["arguments"]["language"] == "python"


# ─────────────────────────────────────────────
# RAG DISABLED
# ─────────────────────────────────────────────

def test_code_patch_works_without_indexing_service(
    api_client,
    monkeypatch,
):
    client, workspace_root, source_file = api_client

    app.state.workspace_indexing_service = None

    captured = install_fake_patch(monkeypatch)

    def unexpected_builder(*args, **kwargs):
        pytest.fail(
            "Context builder should not be initialized "
            "when indexing is disabled."
        )

    monkeypatch.setattr(
        code_patch_route,
        "WorkspaceReviewContextBuilder",
        unexpected_builder,
    )

    response = send_patch(
        client,
        workspace_root,
        source_file,
    )

    assert response.status_code == 200, response.text

    assert captured["related_workspace_context"] is None

    assert response.json()["patch_applied"] is False


# ─────────────────────────────────────────────
# RAG RETRIEVAL FAILURE
# ─────────────────────────────────────────────

def test_code_patch_falls_back_when_context_retrieval_fails(
    api_client,
    monkeypatch,
):
    client, workspace_root, source_file = api_client

    app.state.workspace_indexing_service = object()

    captured = install_fake_patch(monkeypatch)

    class FailingContextBuilder:
        def __init__(self, indexing_service):
            pass

        async def build(self, **kwargs):
            raise RuntimeError(
                "Simulated semantic retrieval failure."
            )

    monkeypatch.setattr(
        code_patch_route,
        "WorkspaceReviewContextBuilder",
        FailingContextBuilder,
    )

    response = send_patch(
        client,
        workspace_root,
        source_file,
    )

    assert response.status_code == 200, response.text

    assert captured["related_workspace_context"] is None

    assert response.json()["patch_applied"] is False


# ─────────────────────────────────────────────
# EMPTY VERIFIED CONTEXT
# ─────────────────────────────────────────────

def test_code_patch_continues_when_retrieval_is_empty(
    api_client,
    monkeypatch,
):
    client, workspace_root, source_file = api_client

    app.state.workspace_indexing_service = object()

    captured = install_fake_patch(monkeypatch)

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
        code_patch_route,
        "WorkspaceReviewContextBuilder",
        EmptyContextBuilder,
    )

    response = send_patch(
        client,
        workspace_root,
        source_file,
    )

    assert response.status_code == 200, response.text

    assert captured["related_workspace_context"] is None

    assert response.json()["patch_applied"] is False