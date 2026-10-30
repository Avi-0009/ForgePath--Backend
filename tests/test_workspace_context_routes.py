from types import SimpleNamespace
from uuid import uuid4

import pytest

from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.main import app


CONTEXT_URL = "/api/v1/workspace/context/prepare"


@pytest.fixture
def api_client(monkeypatch):
    async def override_db():
        yield SimpleNamespace()

    monkeypatch.setitem(
        app.dependency_overrides,
        get_db,
        override_db,
    )

    with TestClient(app) as client:
        yield client


def authenticate_user(monkeypatch):
    user = SimpleNamespace(
        id=uuid4(),
        is_active=True,
    )

    async def override_current_user():
        return user

    monkeypatch.setitem(
        app.dependency_overrides,
        get_current_user,
        override_current_user,
    )

    return user


@pytest.fixture
def allowed_workspace(tmp_path, monkeypatch):
    root = tmp_path / "forgepath"
    root.mkdir()

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(root),
    )

    return root


def test_prepare_context_requires_authentication(api_client):
    response = api_client.post(
        CONTEXT_URL,
        json={
            "workspace_root": "C:/projects/forgepath",
        },
    )

    assert response.status_code == 401


def test_prepare_context_returns_safe_document_metadata(
    api_client,
    allowed_workspace,
    monkeypatch,
):
    authenticate_user(monkeypatch)

    root = allowed_workspace

    source_content = (
        b"PRIVATE_SOURCE_MARKER\n"
        b"def calculate():\n"
        b"    return 42\n"
    )

    readme_content = b"# ForgePath\nLocal AI coding assistant.\n"

    (root / "main.py").write_bytes(source_content)
    (root / "README.md").write_bytes(readme_content)

    (root / "notes.txt").write_bytes(
        b"This format is not ingested."
    )

    response = api_client.post(
        CONTEXT_URL,
        json={
            "workspace_root": str(root),
        },
        headers={
            "Authorization": "Bearer test-token",
        },
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["document_count"] == 2
    assert data["chunk_count"] == 2

    assert data["files_scanned"] == 2
    assert data["files_skipped"] == 1

    assert data["total_bytes"] == (
        len(source_content) + len(readme_content)
    )

    assert data["truncated"] is False

    documents = {
        document["relative_path"]: document
        for document in data["documents"]
    }

    assert set(documents) == {
        "main.py",
        "README.md",
    }

    python_document = documents["main.py"]

    assert python_document["language"] == "python"
    assert python_document["sha256"]
    assert python_document["size_bytes"] == len(source_content)
    assert python_document["line_count"] == 3
    assert python_document["chunk_count"] == 1

    # Source content and chunk content must not be exposed.
    assert "PRIVATE_SOURCE_MARKER" not in response.text

    for document in data["documents"]:
        assert "content" not in document
        assert "source_code" not in document
        assert "chunks" not in document


def test_prepare_context_rejects_workspace_outside_allowlist(
    api_client,
    allowed_workspace,
    tmp_path,
    monkeypatch,
):
    authenticate_user(monkeypatch)

    outside_workspace = tmp_path / "outside"
    outside_workspace.mkdir()

    (outside_workspace / "main.py").write_text(
        "print('private')",
        encoding="utf-8",
    )

    response = api_client.post(
        CONTEXT_URL,
        json={
            "workspace_root": str(outside_workspace),
        },
        headers={
            "Authorization": "Bearer test-token",
        },
    )

    assert response.status_code == 403


def test_prepare_context_rejects_missing_workspace_configuration(
    api_client,
    allowed_workspace,
    monkeypatch,
):
    authenticate_user(monkeypatch)

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        "",
    )

    response = api_client.post(
        CONTEXT_URL,
        json={
            "workspace_root": str(allowed_workspace),
        },
        headers={
            "Authorization": "Bearer test-token",
        },
    )

    assert response.status_code == 500


def test_prepare_context_rejects_extra_request_fields(
    api_client,
    monkeypatch,
):
    authenticate_user(monkeypatch)

    response = api_client.post(
        CONTEXT_URL,
        json={
            "workspace_root": "C:/projects/forgepath",
            "include_source_content": True,
        },
        headers={
            "Authorization": "Bearer test-token",
        },
    )

    assert response.status_code == 422


def test_prepare_context_returns_empty_workspace_summary(
    api_client,
    allowed_workspace,
    monkeypatch,
):
    authenticate_user(monkeypatch)

    response = api_client.post(
        CONTEXT_URL,
        json={
            "workspace_root": str(allowed_workspace),
        },
        headers={
            "Authorization": "Bearer test-token",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_count"] == 0
    assert data["chunk_count"] == 0
    assert data["files_scanned"] == 0
    assert data["files_skipped"] == 0
    assert data["total_bytes"] == 0
    assert data["truncated"] is False
    assert data["documents"] == []