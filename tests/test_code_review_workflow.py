import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.main import app
from app.core.config import settings
from app.services import code_review as review_service
from app.services import code_patch as patch_service
from app.api.v1.routes import code_review as review_routes


REVIEW_URL = "/api/v1/code-review/file"
PATCH_URL = "/api/v1/code-patch"
APPLY_URL = "/api/v1/code-patch/apply"

ORIGINAL_CODE = "def add(a, b):\n    return a - b\n"
PROPOSED_CODE = "def add(a, b):\n    return a + b\n"


@pytest.fixture
def authenticated_client(monkeypatch):
    user = SimpleNamespace(
        id=uuid4(),
        is_active=True,
    )

    async def override_current_user():
        return user

    async def override_db():
        yield SimpleNamespace()

    monkeypatch.setitem(
        app.dependency_overrides,
        get_current_user,
        override_current_user,
    )
    monkeypatch.setitem(
        app.dependency_overrides,
        get_db,
        override_db,
    )

    with TestClient(app) as client:
        yield client


def test_review_patch_and_apply_workflow(
    authenticated_client,
    tmp_path,
    monkeypatch,
):
    client = authenticated_client

    # Create a real source file in an isolated temporary workspace.
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    source_file = workspace_root / "sample.py"
    source_file.write_text(ORIGINAL_CODE, encoding="utf-8")
    original_bytes = source_file.read_bytes()

    # Allowlist only this temporary workspace for the test.
    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(workspace_root),
    )

    # Keep patch backups out of the real user-data directory.
    local_app_data = tmp_path / "localappdata"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    # The test uses predictable AI responses, never a real AI provider.
    review_payload = {
        "summary": "The addition function subtracts its operands.",
        "findings": [
            {
                "severity": "medium",
                "category": "correctness",
                "title": "Incorrect arithmetic operator",
                "line_start": 2,
                "line_end": 2,
                "evidence": "return a - b",
                "explanation": (
                    "The function named add currently subtracts "
                    "the second operand."
                ),
                "suggested_fix": "Return a + b.",
            }
        ],
        "strengths": ["The function is concise."],
        "suggested_tests": [
            "Test the function with positive and negative operands."
        ],
    }

    async def fake_review_generation(*, instructions, prompt):
        assert "def add(a, b):" in prompt
        return json.dumps(review_payload)

    async def fake_patch_generation(*, instructions, prompt):
        assert "return a - b" in prompt
        return json.dumps(
            {
                "summary": "Correct the arithmetic operator.",
                "proposed_code": PROPOSED_CODE,
            }
        )

    monkeypatch.setattr(
        review_service,
        "generate_text",
        fake_review_generation,
    )
    monkeypatch.setattr(
        patch_service,
        "generate_text",
        fake_patch_generation,
    )

    # Avoid requiring project-context database setup for this test.
    monkeypatch.setattr(
        review_routes,
        "get_project_instructions",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        patch_service,
        "get_project_instructions",
        AsyncMock(return_value=None),
    )

    headers = {"Authorization": "Bearer test-token"}

    # 1. Review the actual workspace file.
    review_response = client.post(
        REVIEW_URL,
        headers=headers,
        json={
            "workspace_root": str(workspace_root),
            "relative_path": "sample.py",
            "review_mode": "general",
        },
    )

    assert review_response.status_code == 200, review_response.text

    review_data = review_response.json()

    assert review_data["relative_path"] == "sample.py"
    assert review_data["sha256"] == hashlib.sha256(
        original_bytes
    ).hexdigest()
    assert review_data["review"]["findings"][0]["title"] == (
        "Incorrect arithmetic operator"
    )
    assert source_file.read_bytes() == original_bytes

    # 2. Generate a patch using the exact reviewed source hash.
    finding = review_data["review"]["findings"][0]

    patch_response = client.post(
        PATCH_URL,
        headers=headers,
        json={
            "workspace_root": str(workspace_root),
            "relative_path": review_data["relative_path"],
            "expected_sha256": review_data["sha256"],
            "finding_title": finding["title"],
            "finding_evidence": finding["evidence"],
            "finding_explanation": finding["explanation"],
            "suggested_fix": finding["suggested_fix"],
        },
    )

    assert patch_response.status_code == 200, patch_response.text

    patch_data = patch_response.json()

    assert patch_data["original_sha256"] == review_data["sha256"]
    assert patch_data["proposed_sha256"] == hashlib.sha256(
        PROPOSED_CODE.encode("utf-8")
    ).hexdigest()
    assert patch_data["patch_applied"] is False
    assert patch_data["proposed_code"] == PROPOSED_CODE
    assert patch_data["diff"]
    assert source_file.read_bytes() == original_bytes

    # 3. Explicitly approve and apply the proposed content.
    apply_response = client.post(
        APPLY_URL,
        headers=headers,
        json={
            "workspace_root": str(workspace_root),
            "relative_path": patch_data["relative_path"],
            "expected_sha256": patch_data["original_sha256"],
            "proposed_sha256": patch_data["proposed_sha256"],
            "proposed_code": patch_data["proposed_code"],
            "approval": "apply",
        },
    )

    assert apply_response.status_code == 200, apply_response.text

    apply_data = apply_response.json()

    assert apply_data["patch_applied"] is True
    assert apply_data["original_sha256"] == review_data["sha256"]
    assert apply_data["applied_sha256"] == patch_data["proposed_sha256"]
    assert source_file.read_text(encoding="utf-8") == PROPOSED_CODE

    # The original source must still be recoverable from its backup.
    backup_path = (
        local_app_data
        / "ForgePath"
        / "backups"
        / f"{apply_data['backup_id']}.bak"
    )

    assert backup_path.is_file()
    assert backup_path.read_bytes() == original_bytes


def test_review_and_patch_endpoints_require_authentication(
    tmp_path,
    monkeypatch,
):
    async def override_db():
        yield SimpleNamespace()

    monkeypatch.setitem(
        app.dependency_overrides,
        get_db,
        override_db,
    )

    workspace_root = str(tmp_path)

    requests = [
        (
            REVIEW_URL,
            {
                "workspace_root": workspace_root,
                "relative_path": "sample.py",
            },
        ),
        (
            PATCH_URL,
            {
                "workspace_root": workspace_root,
                "relative_path": "sample.py",
                "expected_sha256": "a" * 64,
                "finding_title": "Test finding",
                "finding_evidence": "Test evidence",
                "finding_explanation": "Test explanation",
                "suggested_fix": "Test fix",
            },
        ),
        (
            APPLY_URL,
            {
                "workspace_root": workspace_root,
                "relative_path": "sample.py",
                "expected_sha256": "a" * 64,
                "proposed_sha256": "b" * 64,
                "proposed_code": "value = 1\n",
                "approval": "apply",
            },
        ),
    ]

    with TestClient(app) as client:
        for url, payload in requests:
            response = client.post(url, json=payload)
            assert response.status_code == 401, (
                f"{url} should require authentication; "
                f"got {response.status_code}: {response.text}"
            )



def test_file_review_rejects_parent_directory_traversal(
    authenticated_client,
    tmp_path,
    monkeypatch,
):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    outside_file = tmp_path / "outside.py"
    outside_file.write_text("print('private')\n", encoding="utf-8")

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(workspace_root),
    )

    response = authenticated_client.post(
        REVIEW_URL,
        headers={"Authorization": "Bearer test-token"},
        json={
            "workspace_root": str(workspace_root),
            "relative_path": "../outside.py",
            "review_mode": "general",
        },
    )

    assert response.status_code == 400
    assert outside_file.read_text(encoding="utf-8") == "print('private')\n"


def test_patch_generation_rejects_stale_review_hash(
    authenticated_client,
    tmp_path,
    monkeypatch,
):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    source_file = workspace_root / "sample.py"
    source_file.write_text(ORIGINAL_CODE, encoding="utf-8")

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(workspace_root),
    )

    response = authenticated_client.post(
        PATCH_URL,
        headers={"Authorization": "Bearer test-token"},
        json={
            "workspace_root": str(workspace_root),
            "relative_path": "sample.py",
            "expected_sha256": "0" * 64,
            "finding_title": "Incorrect arithmetic operator",
            "finding_evidence": "return a - b",
            "finding_explanation": "The operation is incorrect.",
            "suggested_fix": "Return a + b.",
        },
    )

    assert response.status_code == 409
    assert source_file.read_text(encoding="utf-8") == ORIGINAL_CODE

    backup_directory = (
        tmp_path / "localappdata" / "ForgePath" / "backups"
    )
    assert not backup_directory.exists()


def test_patch_application_rejects_stale_source(
    authenticated_client,
    tmp_path,
    monkeypatch,
):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    source_file = workspace_root / "sample.py"
    source_file.write_text(ORIGINAL_CODE, encoding="utf-8")

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(workspace_root),
    )
    monkeypatch.setenv(
        "LOCALAPPDATA",
        str(tmp_path / "localappdata"),
    )

    proposed_hash = hashlib.sha256(
        PROPOSED_CODE.encode("utf-8")
    ).hexdigest()

    response = authenticated_client.post(
        APPLY_URL,
        headers={"Authorization": "Bearer test-token"},
        json={
            "workspace_root": str(workspace_root),
            "relative_path": "sample.py",
            "expected_sha256": "0" * 64,
            "proposed_sha256": proposed_hash,
            "proposed_code": PROPOSED_CODE,
            "approval": "apply",
        },
    )

    assert response.status_code == 409
    assert source_file.read_text(encoding="utf-8") == ORIGINAL_CODE

    backup_directory = (
        tmp_path / "localappdata" / "ForgePath" / "backups"
    )
    assert not backup_directory.exists()
