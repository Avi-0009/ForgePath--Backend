"""Validate optional AI investigation citations against exact snapshot excerpts."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.project_review_findings import parse_review_response


class EvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    path: str = Field(min_length=1, max_length=1024)
    quote: str = Field(min_length=1, max_length=600)
    explanation: str = Field(min_length=1, max_length=600)


def _json_object(text: str) -> dict | None:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def validate_investigation_answer(text: str, context: dict) -> dict | None:
    """Never accept paths or quotes that were not supplied to the model."""
    excerpts = {str(item["path"]): str(item["excerpt"]) for item in context["files"]}
    starts = {str(item["path"]): int(item["start_line"]) for item in context["files"]}
    parsed = parse_review_response(text, excerpts)
    raw = _json_object(text)
    if raw is None or not parsed.format_valid:
        return None

    summary = raw.get("summary", "")
    if not isinstance(summary, str):
        return None
    summary = summary.strip()[:1500]
    evidence = []
    used: set[tuple[str, str]] = set()
    raw_links = raw.get("evidence_links", [])
    if isinstance(raw_links, list):
        for item in raw_links[:20]:
            try:
                candidate = EvidenceCandidate.model_validate(item)
            except (ValidationError, TypeError, ValueError):
                continue
            excerpt = excerpts.get(candidate.path)
            if excerpt is None or not candidate.quote.strip() or candidate.quote not in excerpt:
                continue
            key = candidate.path, candidate.quote
            if key in used:
                continue
            used.add(key)
            offset = excerpt.index(candidate.quote)
            start = starts[candidate.path] + excerpt[:offset].count("\n")
            evidence.append({
                "path": candidate.path,
                "start_line": start,
                "end_line": start + candidate.quote.rstrip("\r\n").count("\n"),
                "quote": candidate.quote,
                "explanation": candidate.explanation.strip(),
                "verification_status": "source_quote_verified",
            })
            if len(evidence) == 5:
                break

    def short_strings(name: str, *, limit: int) -> list[str]:
        values = raw.get(name, [])
        if not isinstance(values, list):
            return []
        return [item.strip()[:240] for item in values if isinstance(item, str) and item.strip()][:limit]

    return {
        "summary": summary,
        "findings": parsed.findings,
        "evidence_links": evidence,
        "gaps": short_strings("gaps", limit=4),
        "follow_up_questions": short_strings("follow_up_questions", limit=3),
    }