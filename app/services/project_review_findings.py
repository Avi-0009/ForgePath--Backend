"""Validate model-suggested findings against source excerpts."""

from __future__ import annotations

import json
from typing import Any


ALLOWED_SEVERITIES = {
    "critical",
    "high",
    "medium",
    "low",
    "info",
}

MAX_FINDINGS_PER_BATCH = 3


def parse_review_findings(
    response_text: str,
    excerpts: dict[str, str],
) -> tuple[str, list[dict[str, str]]]:
    """
    Parse structured model output.

    A finding is accepted only when its file path belongs to
    the reviewed batch and its evidence appears in that excerpt.

    Accepted findings are still unverified AI suggestions.
    """

    raw = response_text.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()

        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()

    try:
        payload: Any = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return response_text, []

    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("findings"), list)
    ):
        return response_text, []

    suggestions: list[dict[str, str]] = []

    for item in payload["findings"][:MAX_FINDINGS_PER_BATCH]:
        if not isinstance(item, dict):
            continue

        path = item.get("path")
        title = item.get("title")
        severity = item.get("severity")
        evidence = item.get("evidence")
        explanation = item.get("explanation")

        if not all(
            isinstance(value, str)
            for value in (
                path,
                title,
                severity,
                evidence,
                explanation,
            )
        ):
            continue

        if path not in excerpts:
            continue

        if severity not in ALLOWED_SEVERITIES:
            continue

        if not evidence.strip() or evidence not in excerpts[path]:
            continue

        if not title.strip() or not explanation.strip():
            continue

        suggestions.append(
            {
                "path": path,
                "title": title.strip()[:160],
                "severity": severity,
                "evidence": evidence[:500],
                "explanation": explanation.strip()[:1000],
                "verification_status": "unverified",
            }
        )

    summary = payload.get("summary")

    intro = (
        summary.strip()[:1500]
        if isinstance(summary, str)
        else ""
    )

    sections = [intro] if intro else []

    for finding in suggestions:
        sections.append(
            f"### {finding['severity'].capitalize()} suggestion: "
            f"{finding['title']}\n"
            f"File: {finding['path']}\n"
            f"Evidence: `{finding['evidence']}`\n"
            f"Explanation: {finding['explanation']}\n"
            "Status: Unverified AI suggestion"
        )

    if not suggestions:
        sections.append(
            "No source-supported structured suggestions "
            "were returned in this batch."
        )

    return "\n\n".join(sections), suggestions