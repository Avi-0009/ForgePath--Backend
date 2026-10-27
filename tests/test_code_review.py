import asyncio
import json

import pytest

from app.schemas.code_review import CodeReviewRequest
from app.services.code_review import CodeReviewError, review_code

def make_valid_response():
    return {
        "summary": "A potential correctness issue was identified.",
        "findings": [
            {
                "severity": "medium",
                "category": "correctness",
                "title": "Potential incorrect result",
                "line_start": 2,
                "line_end": 2,
                "evidence": "return value * 2",
                "explanation": "Check whether doubling is intended.",
                "suggested_fix": "Verify the intended output.",
            }
        ],
        "strengths": ["The function is easy to read."],
        "suggested_tests": ["Test zero and negative inputs."],
    }

def test_review_code_returns_structured_findings(monkeypatch):
    async def fake_generate_text(*, instructions, prompt):
        assert "untrusted" in instructions.lower()
        assert "def calculate(value):" in prompt
        return json.dumps(make_valid_response())

    monkeypatch.setattr(
        "app.services.code_review.generate_text",
        fake_generate_text,
    )

    request = CodeReviewRequest(
        code="def calculate(value):\n    return value * 2",
        language="python",
        review_mode="general",
    )

    result = asyncio.run(review_code(request))

    assert result.summary
    assert len(result.findings) == 1
    assert result.findings[0].severity == "medium"
    assert result.findings[0].line_start == 2
    assert result.strengths
    assert result.suggested_tests


def test_review_code_clears_out_of_range_line_numbers(monkeypatch):
    response = make_valid_response()
    response["findings"][0]["line_start"] = 100
    response["findings"][0]["line_end"] = 101

    async def fake_generate_text(*, instructions, prompt):
        return json.dumps(response)

    monkeypatch.setattr(
        "app.services.code_review.generate_text",
        fake_generate_text,
    )

    request = CodeReviewRequest(
        code="print('hello')",
        language="python",
    )

    result = asyncio.run(review_code(request))

    assert result.findings[0].line_start is None
    assert result.findings[0].line_end is None


def test_review_code_rejects_malformed_json(monkeypatch):
    async def fake_generate_text(*, instructions, prompt):
        return '{"summary": invalid json}'

    monkeypatch.setattr(
        "app.services.code_review.generate_text",
        fake_generate_text,
    )

    request = CodeReviewRequest(code="print('hello')")

    with pytest.raises(CodeReviewError, match="malformed JSON"):
        asyncio.run(review_code(request))


def test_review_code_rejects_invalid_response_schema(monkeypatch):
    response = make_valid_response()
    response["findings"][0]["severity"] = "super-critical"

    async def fake_generate_text(*, instructions, prompt):
        return json.dumps(response)

    monkeypatch.setattr(
        "app.services.code_review.generate_text",
        fake_generate_text,
    )

    request = CodeReviewRequest(code="print('hello')")

    with pytest.raises(
        CodeReviewError,
        match="does not match",
    ):
        asyncio.run(review_code(request))