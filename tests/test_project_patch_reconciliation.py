"""Approved patches are detected only by exact proposed-content hashes."""

import hashlib
from types import SimpleNamespace
from uuid import uuid4

from app.services.project_patch_reconciliation import find_applied_patch_task_ids


def _task(content: str, *, status: str = "approved", path: str = "app/main.py"):
    return SimpleNamespace(
        id=uuid4(),
        path=path,
        patch_status=status,
        patch={
            "relative_path": path,
            "proposed_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        },
    )


def test_exact_proposed_content_is_detected():
    proposed = "def main():\n    return 'fixed'\n"
    task = _task(proposed)

    assert find_applied_patch_task_ids(
        [task],
        [(task.path, proposed)],
    ) == {task.id}


def test_similar_rejected_or_wrong_path_content_is_not_detected():
    proposed = "def main():\n    return 'fixed'\n"
    approved = _task(proposed)
    rejected = _task(proposed, status="rejected")
    wrong_path = _task(proposed, path="app/other.py")

    detected = find_applied_patch_task_ids(
        [approved, rejected, wrong_path],
        [
            (approved.path, proposed + "# changed\n"),
            ("app/unrelated.py", proposed),
        ],
    )

    assert detected == set()
