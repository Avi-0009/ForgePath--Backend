import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, Mock

import pytest

from app.schemas.code_patch import CodePatchRequest
from app.services.code_patch import (
    CodePatchError,
    StaleSourceFileError,
    generate_code_patch,
)
from app.services.workspace import WorkspaceFile


SOURCE = "def add(a, b):\n    return a + b\n"
FIXED_SOURCE = "def add(a, b):\n    return a + b + 1\n"


def make_workspace_file(content: str) -> WorkspaceFile:
    raw = content.encode("utf-8")

    return WorkspaceFile(
        relative_path="example.py",
        language="python",
        content=content,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        line_count=len(content.splitlines()),
    )


def make_request(expected_sha256: str) -> CodePatchRequest:
    return CodePatchRequest(
        workspace_root="C:/projects/example",
        relative_path="example.py",
        expected_sha256=expected_sha256,
        finding_title="Incorrect addition result",
        finding_evidence="return a + b",
        finding_explanation="The return expression does not meet the stated requirement.",
        suggested_fix="Adjust the return expression to satisfy the requirement.",
    )


def make_proposal(code: str = FIXED_SOURCE) -> str:
    return json.dumps(
        {
            "summary": "Update the return expression.",
            "proposed_code": code,
        }
    )


def setup_service(monkeypatch, source_files, ai_response):
    reader = Mock(side_effect=source_files)

    monkeypatch.setattr(
        "app.services.code_patch.read_configured_workspace_file",
        reader,
    )

    monkeypatch.setattr(
        "app.services.code_patch.get_project_instructions",
        AsyncMock(return_value=None),
    )

    generator = AsyncMock(return_value=ai_response)

    monkeypatch.setattr(
        "app.services.code_patch.generate_text",
        generator,
    )

    return reader, generator


def test_generates_patch_without_modifying_original(monkeypatch):
    source_file = make_workspace_file(SOURCE)

    reader, generator = setup_service(
        monkeypatch,
        [source_file, source_file],
        make_proposal(),
    )

    result = asyncio.run(
        generate_code_patch(make_request(source_file.sha256))
    )

    assert result.relative_path == "example.py"
    assert result.original_sha256 == source_file.sha256
    assert result.proposed_sha256 == hashlib.sha256(
        FIXED_SOURCE.encode("utf-8")
    ).hexdigest()

    assert result.proposed_code == FIXED_SOURCE
    assert result.patch_applied is False
    assert result.original_line_count == 2
    assert result.proposed_line_count == 2
    assert "return a + b" in result.diff
    assert "return a + b + 1" in result.diff

    # Verify that the source was checked before and after generation.
    assert reader.call_count == 2
    generator.assert_awaited_once()


def test_rejects_stale_source_before_calling_ai(monkeypatch):
    source_file = make_workspace_file(SOURCE)

    _, generator = setup_service(
        monkeypatch,
        [source_file],
        make_proposal(),
    )

    request = make_request("0" * 64)

    with pytest.raises(StaleSourceFileError):
        asyncio.run(generate_code_patch(request))

    generator.assert_not_awaited()


def test_rejects_source_changed_during_generation(monkeypatch):
    original_file = make_workspace_file(SOURCE)
    changed_file = make_workspace_file(
        "def add(a, b):\n    return a - b\n"
    )

    setup_service(
        monkeypatch,
        [original_file, changed_file],
        make_proposal(),
    )

    with pytest.raises(StaleSourceFileError):
        asyncio.run(
            generate_code_patch(
                make_request(original_file.sha256)
            )
        )


def test_rejects_malformed_ai_json(monkeypatch):
    source_file = make_workspace_file(SOURCE)

    setup_service(
        monkeypatch,
        [source_file],
        '{"summary": invalid json}',
    )

    with pytest.raises(
        CodePatchError,
        match="malformed JSON",
    ):
        asyncio.run(
            generate_code_patch(
                make_request(source_file.sha256)
            )
        )


def test_rejects_proposal_with_no_changes(monkeypatch):
    source_file = make_workspace_file(SOURCE)

    setup_service(
        monkeypatch,
        [source_file],
        make_proposal(SOURCE),
    )

    with pytest.raises(
        CodePatchError,
        match="no changes",
    ):
        asyncio.run(
            generate_code_patch(
                make_request(source_file.sha256)
            )
        )