"""Past paths can guide selection but must never synthesize new source evidence."""
from app.services.project_investigation_engine import build_investigation_context, public_context_preview


def test_follow_up_retains_verified_project_file_path_only():
    sources = [
        ("src/auth.py", "def validate_token(token):\n    return bool(token)\n"),
        ("src/other.py", "def process_image(path):\n    return path\n"),
        ("tests/test_auth.py", "from src.auth import validate_token\ndef test_token():\n    assert validate_token('valid')\n"),
    ]
    result = build_investigation_context(
        sources, "What tests cover that implementation?", 3,
        continuity_paths=["src/auth.py", "private/never-uploaded.py"],
    )
    names = [item["path"] for item in result["files"]]
    assert "src/auth.py" in names
    assert "private/never-uploaded.py" not in names
    assert all(item["excerpt"] in dict(sources)[item["path"]] for item in result["files"])
    assert all("excerpt" not in item for item in public_context_preview(result)["selected_files"])


def test_source_file_limit_still_applies_to_long_conversation():
    sources = [(f"src/file_{i}.py", f"def function_{i}():\n    return {i}\n") for i in range(25)]
    result = build_investigation_context(
        sources, "Which functions exist in source files?", 10,
        continuity_paths=[f"src/file_{i}.py" for i in range(20)],
    )
    assert len(result["files"]) <= 10
    assert result["context_chars"] <= 26_000