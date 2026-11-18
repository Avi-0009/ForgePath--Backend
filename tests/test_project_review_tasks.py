"""Project task inputs come only from valid saved findings."""

import pytest

from app.services.project_review_tasks import get_task_finding


VALID = {
    "path": "src/api.ts",
    "title": "Unbounded request",
    "severity": "medium",
    "evidence": "fetch(url)",
    "explanation": "The request has no timeout.",
    "verification_status": "unverified",
}


def test_valid_finding_is_copied_without_mutating_saved_review():
    findings = [VALID.copy()]

    task_data = get_task_finding(findings, 0)

    assert task_data is not None
    assert task_data["path"] == "src/api.ts"
    assert "verification_status" not in task_data

    task_data["title"] = "Changed"

    assert findings[0]["title"] == "Unbounded request"


@pytest.mark.parametrize(
    "findings,index",
    [
        (None, 0),
        ({"findings": [VALID]}, 0),
        ([], 0),
        ([VALID], -1),
        ([VALID], 1),
        (["unexpected"], 0),
    ],
)
def test_missing_or_malformed_finding_is_rejected(
    findings,
    index,
):
    assert get_task_finding(findings, index) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"title": "   "},
        {"path": None},
        {"evidence": ""},
        {"severity": "emergency"},
        {"verification_status": "verified"},
    ],
)
def test_untrusted_or_invalid_review_payload_is_rejected(
    changes,
):
    assert get_task_finding(
        [{**VALID, **changes}],
        0,
    ) is None