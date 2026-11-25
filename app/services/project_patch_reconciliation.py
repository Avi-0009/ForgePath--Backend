"""Detect approved proposals present in a newly uploaded snapshot."""

import hashlib
from collections.abc import Iterable
from uuid import UUID

from app.models.project_review_task import ProjectReviewTask


def find_applied_patch_task_ids(
    tasks: Iterable[ProjectReviewTask],
    sources: Iterable[tuple[str, str]],
) -> set[UUID]:
    """Return approved task IDs whose exact proposed file now exists."""

    source_hashes = {
        path: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for path, content in sources
    }
    applied: set[UUID] = set()

    for task in tasks:
        patch = task.patch
        if task.patch_status != "approved" or not isinstance(patch, dict):
            continue

        proposed_sha256 = patch.get("proposed_sha256")
        if (
            patch.get("relative_path") == task.path
            and isinstance(proposed_sha256, str)
            and source_hashes.get(task.path) == proposed_sha256
        ):
            applied.add(task.id)

    return applied
