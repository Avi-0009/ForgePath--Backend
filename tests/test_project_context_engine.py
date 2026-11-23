"""Deterministic context, import-neighbor retrieval and verified citations."""

import json

from app.services.project_investigation_answer import validate_investigation_answer
from app.services.project_investigation_engine import (
    MAX_CONTEXT_CHARS, build_investigation_context, public_context_preview,
)
from app.services.project_reference_integrity import resolved_source_graph


SOURCES = [
    ("app/api/v1/routes/auth.py", "from app.api.dependencies import get_current_user\n"
     "def login():\n    return create_access_token()\n"),
    ("app/api/dependencies.py", "def get_current_user():\n    return decode_jwt_and_verify_expiry()\n"),
    ("app/core/security.py", "def decode_jwt_and_verify_expiry():\n    return True\n"),
    ("tests/test_auth.py", "from app.api.dependencies import get_current_user\n"
     "def test_session():\n    assert get_current_user\n"),
    ("docs/notes.md", "Unrelated documentation\n"),
]


def test_graph_follows_python_absolute_imports_and_reverse_callers():
    graph = resolved_source_graph(SOURCES)
    assert "app/api/dependencies.py" in graph["app/api/v1/routes/auth.py"]
    assert "app/api/dependencies.py" in graph["tests/test_auth.py"]
    assert "docs/notes.md" not in graph["app/api/v1/routes/auth.py"]


def test_graph_resolves_relative_python_and_typescript_without_inventing_edges():
    sources = [
        ("app/auth/controller.py", "from .session import verify_token\n"),
        ("app/auth/session.py", "def verify_token():\n    return True\n"),
        ("src/pages/Login.tsx", "import { useToken } from '../store/session';\n"),
        ("src/store/session.ts", "export const useToken = () => true;\n"),
        ("src/store/other.ts", "export const other = 1;\n"),
    ]
    graph = resolved_source_graph(sources)
    assert graph["app/auth/controller.py"] == {"app/auth/session.py"}
    assert graph["src/pages/Login.tsx"] == {"src/store/session.ts"}
    assert graph["src/store/other.ts"] == set()


def test_context_includes_related_implementation_with_provenance():
    report = build_investigation_context(
        SOURCES, "Find authentication validating access tokens and expiry", 4,
    )
    paths = [file["path"] for file in report["files"]]
    assert "app/api/dependencies.py" in paths
    assert len(paths) <= 4
    assert report["graph_edges"] >= 2
    assert any("import" in item["reason"] for item in report["files"])
    for item in report["files"]:
        source = dict(SOURCES)[item["path"]]
        assert item["excerpt"] in source
        assert source[:source.index(item["excerpt"])].count("\n") + 1 == item["start_line"]


def test_semantic_hits_must_be_verbatim_and_in_snapshot():
    report = build_investigation_context(
        SOURCES, "Find session validation", 3,
        semantic_hits=[
            {"relative_path": "app/core/security.py", "content": "decode_jwt_and_verify_expiry"},
            {"relative_path": "../../private.py", "content": "secret"},
            {"relative_path": "app/core/security.py", "content": "hallucinated content"},
        ],
    )
    assert report["semantic_hits_verified"] == 1
    assert any(item["reason"] == "verified semantic match" for item in report["files"])


def test_context_is_bounded_and_preview_does_not_return_source():
    sources = [(f"src/file{i}.ts", "refresh_session()\n" * 3000) for i in range(40)]
    report = build_investigation_context(sources, "refresh session", 10)
    assert len(report["files"]) <= 10
    assert report["context_chars"] <= MAX_CONTEXT_CHARS
    assert all(file["excerpt_chars"] <= 3200 for file in report["files"])
    preview = public_context_preview(report)
    assert all("excerpt" not in entry for entry in preview["selected_files"])
    assert preview["estimated_tokens"] > 0


def test_invalid_or_invented_model_citations_are_discarded():
    context = build_investigation_context(SOURCES, "validate jwt expiry", 5)
    target = next(file for file in context["files"] if file["path"] == "app/api/dependencies.py")
    quote = "return decode_jwt_and_verify_expiry()"
    assert quote in target["excerpt"]
    text = json.dumps({
        "summary": "The route imports a JWT validation helper.",
        "findings": [],
        "evidence_links": [
            {"path": target["path"], "quote": quote, "explanation": "Validation occurs here."},
            {"path": "private/secret.py", "quote": "not in snapshot", "explanation": "Invented"},
        ],
        "gaps": ["No HTTP execution was performed"],
        "follow_up_questions": ["Which tests cover expiry?"],
    })
    result = validate_investigation_answer(text, context)
    assert result is not None
    assert len(result["evidence_links"]) == 1
    assert result["evidence_links"][0]["path"] == target["path"]
    assert result["evidence_links"][0]["start_line"] >= target["start_line"]
    assert result["gaps"] and result["follow_up_questions"]


def test_invalid_model_json_is_not_accepted():
    context = build_investigation_context(SOURCES, "auth session", 3)
    assert validate_investigation_answer("I think it is secure", context) is None