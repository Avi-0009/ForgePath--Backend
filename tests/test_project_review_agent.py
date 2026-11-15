"""Bounded initial-review planning without calling external models."""
from app.services.project_review_agent import (
    MAX_AI_CALLS,
    MAX_BATCH_CHARS,
    _prepare_review_batches,
)


def test_review_never_exceeds_model_call_budget():
    sources = [(f"src/module_{number}.py", "x" * 3000) for number in range(100)]
    batches, reviewed = _prepare_review_batches(sources)
    assert 1 <= len(batches) <= MAX_AI_CALLS
    assert all(len(batch) <= MAX_BATCH_CHARS for batch in batches)
    assert reviewed < len(sources)


def test_review_prioritizes_security_related_files():
    batches, reviewed = _prepare_review_batches([
        ("src/z.py", "print('z')"),
        ("src/auth.py", "print('auth')"),
    ])
    assert reviewed == 2
    assert batches[0].index("src/auth.py") < batches[0].index("src/z.py")