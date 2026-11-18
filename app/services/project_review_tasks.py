"""Defensive extraction of saved review findings for task creation."""

from collections.abc import Mapping

from app.services.project_review_findings import ALLOWED_SEVERITIES


def get_task_finding(
    findings: object,
    index: int,
) -> dict[str, str] | None:
    if (
        not isinstance(findings, list)
        or index < 0
        or index >= len(findings)
    ):
        return None

    item = findings[index]

    if not isinstance(item, Mapping):
        return None

    required = (
        "path",
        "title",
        "severity",
        "evidence",
        "explanation",
    )

    if any(
        not isinstance(item.get(name), str)
        or not item[name].strip()
        for name in required
    ):
        return None

    if item["severity"] not in ALLOWED_SEVERITIES:
        return None

    if item.get("verification_status") != "unverified":
        return None

    return {
        name: item[name]
        for name in required
    }