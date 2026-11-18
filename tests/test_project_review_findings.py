import json

from app.services.project_review_findings import (
    parse_review_findings,
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

    assert raw == "ordinary model reply"
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