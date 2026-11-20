"""Validate AI review output before storing or showing it to a user.

Evidence is checked against the exact snapshot excerpt supplied to the model.
Findings remain unverified AI suggestions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


MAX_FINDINGS_PER_BATCH = 3
MAX_CANDIDATES_PER_RESPONSE = 20

ALLOWED_SEVERITIES = {
    "critical",
    "high",
    "medium",
    "low",
    "info",
}


class ReviewFindingCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    path: str = Field(min_length=1)
    title: str = Field(min_length=1)
    severity: Literal["critical", "high", "medium", "low", "info"]
    evidence: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class ReviewResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    summary: str = Field(default="", max_length=20_000)
    findings: list[ReviewFindingCandidate] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ReviewParseResult:
    report: str
    findings: list[dict[str, str]]
    format_valid: bool


def parse_review_response(
    response_text: str,
    excerpts: dict[str, str],
) -> ReviewParseResult:
    """Return safe display text and source-supported structured suggestions."""
    raw = response_text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("findings"), list):
            raise ValueError("Missing findings list")
        candidates = data["findings"][:MAX_CANDIDATES_PER_RESPONSE]
        response = ReviewResponse.model_validate({**data, "findings": []})
    except (json.JSONDecodeError, ValueError, TypeError, ValidationError):
        return ReviewParseResult(
            report="The model did not return a valid structured review. "
            "No findings were accepted from this batch.",
            findings=[],
            format_valid=False,
        )

    suggestions: list[dict[str, str]] = []
    for item in candidates:
        try:
            candidate = ReviewFindingCandidate.model_validate(item)
        except (ValidationError, ValueError, TypeError):
            continue
        if candidate.path not in excerpts:
            continue
        if not candidate.evidence.strip() or candidate.evidence not in excerpts[candidate.path]:
            continue
        if not candidate.title.strip() or not candidate.explanation.strip():
            continue

        suggestions.append({
            "path": candidate.path,
            "title": candidate.title.strip()[:160],
            "severity": candidate.severity,
            "evidence": candidate.evidence[:500],
            "explanation": candidate.explanation.strip()[:1000],
            "verification_status": "unverified",
        })
        if len(suggestions) >= MAX_FINDINGS_PER_BATCH:
            break

    summary = response.summary.strip()[:1500]
    sections = [summary] if summary else []
    for finding in suggestions:
        sections.append(
            f"### {finding['severity'].capitalize()} suggestion: {finding['title']}\n"
            f"File: {finding['path']}\n"
            f"Evidence: `{finding['evidence']}`\n"
            f"Explanation: {finding['explanation']}\n"
            "Status: Unverified AI suggestion"
        )
    if not suggestions:
        sections.append("No source-supported structured suggestions were returned in this batch.")
    return ReviewParseResult("\n\n".join(sections), suggestions, True)


def parse_review_findings(
    response_text: str,
    excerpts: dict[str, str],
) -> tuple[str, list[dict[str, str]]]:
    """Backward-compatible interface used by existing review tests/callers."""
    result = parse_review_response(response_text, excerpts)
    return result.report, result.findings