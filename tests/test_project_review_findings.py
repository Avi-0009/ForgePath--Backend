import json

from app.services.project_review_findings import (
    parse_review_findings,
    parse_review_response,
)


def test_retains_only_exact_source_supported_unverified_findings():
    response = json.dumps({
        "summary": "One potentially risky pattern.",
        "findings": [
            {
                "path": "src/auth.py",
                "title": "Unprotected access",
                "severity": "high",
                "evidence": "return data[request.user]",
                "explanation": "Check authorization at the caller.",
            },
            {
                "path": "src/imaginary.py",
                "title": "Invented",
                "severity": "critical",
                "evidence": "some code",
                "explanation": "Not in the batch.",
            },
        ],
    })

    report, findings = parse_review_findings(
        response,
        {
            "src/auth.py": (
                "def read(request):\n"
                "    return data[request.user]"
            ),
        },
    )

    assert len(findings) == 1
    assert findings[0]["path"] == "src/auth.py"
    assert findings[0]["verification_status"] == "unverified"
    assert "Unverified AI suggestion" in report


def test_rejects_invented_evidence_and_malformed_json():
    payload = json.dumps({
        "findings": [
            {
                "path": "src/app.py",
                "title": "Invented",
                "severity": "high",
                "evidence": "this line never existed",
                "explanation": "Unfounded",
            },
        ],
    })

    _, findings = parse_review_findings(
        payload,
        {"src/app.py": "print('safe')"},
    )

    assert findings == []

    raw, findings = parse_review_findings(
        "ordinary model reply",
        {},
    )

    assert "did not return a valid structured review" in raw
    assert findings == []


def test_capped_per_batch_and_unsupported_severity_ignored():
    candidate = {
        "path": "a.py",
        "title": "Check this",
        "severity": "low",
        "evidence": "return None",
        "explanation": "Potential issue",
    }

    invalid = {
        **candidate,
        "severity": "guaranteed",
    }

    _, findings = parse_review_findings(
        json.dumps({
            "findings": [
                invalid,
                *([candidate] * 10),
            ],
        }),
        {"a.py": "return None"},
    )

    assert len(findings) <= 3


def test_rejects_non_json_model_reasoning_without_echoing_it():
    private_text = "My internal reasoning about a secret configuration"
    result = parse_review_response(private_text, {"app.py": "print('ok')"})
    assert result.format_valid is False
    assert result.findings == []
    assert private_text not in result.report


def test_accepts_fenced_json_but_rejects_unsupported_evidence():
    result = parse_review_response(
        '```json\n{"summary":"Checked auth", "findings": ['
        '{"path":"auth.py","title":"Check permissions","severity":"high",'
        '"evidence":"return data","explanation":"Missing authorization"},'
        '{"path":"auth.py","title":"Invented","severity":"low",'
        '"evidence":"not in file","explanation":"Unsupported"}]}\n```',
        {"auth.py": "return data"},
    )
    assert result.format_valid is True
    assert len(result.findings) == 1
    assert result.findings[0]["verification_status"] == "unverified"


def test_invalid_summary_type_and_non_array_findings_fail_cleanly():
    for raw in ('{"summary": 123, "findings": []}',
                '{"summary":"ok", "findings": {}}'):
        result = parse_review_response(raw, {})
        assert result.format_valid is False
        assert result.findings == []