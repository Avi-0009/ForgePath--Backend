"""Focused safety checks for browser-supplied codebase snapshots."""
from uuid import uuid4

import pytest

from app.core.config import settings
from app.services.project_snapshots import (
    SnapshotValidationError,
    SourceSnapshot,
    read_snapshot_manifest,
    read_snapshot_sources,
    store_snapshot,
)


def _snapshot(*files: tuple[str, str]) -> SourceSnapshot:
    return SourceSnapshot(files=[{"relative_path": path, "content": content} for path, content in files])


@pytest.mark.parametrize("path", [
    "../outside.py", "/etc/passwd", "C:/Windows/system.ini", "app\\main.py",
    ".env", "config/.env.local", "node_modules/index.js", ".ssh/key.py",
    "app/secrets.json", "nested/private_key.py", "code.exe",
])
def test_rejects_unsafe_paths(tmp_path, monkeypatch, path):
    monkeypatch.setattr(settings, "project_snapshot_directory", str(tmp_path / "snapshots"))
    with pytest.raises(SnapshotValidationError):
        store_snapshot(uuid4(), uuid4(), _snapshot((path, "print('ok')")))


def test_rejects_embedded_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_snapshot_directory", str(tmp_path / "snapshots"))
    with pytest.raises(SnapshotValidationError):
        store_snapshot(uuid4(), uuid4(), _snapshot(("main.py", "-----BEGIN PRIVATE KEY-----")))


def test_rejects_duplicate_case_insensitive_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_snapshot_directory", str(tmp_path / "snapshots"))
    with pytest.raises(SnapshotValidationError):
        store_snapshot(uuid4(), uuid4(), _snapshot(("main.py", "a"), ("MAIN.py", "b")))


def test_isolates_snapshots_by_user_and_project(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_snapshot_directory", str(tmp_path / "snapshots"))
    user_a, user_b, project_a, project_b = uuid4(), uuid4(), uuid4(), uuid4()
    created = store_snapshot(user_a, project_a, _snapshot(("src/main.py", "print('hello')")))
    assert read_snapshot_manifest(user_a, project_a) == created
    assert read_snapshot_manifest(user_b, project_a) is None
    assert read_snapshot_manifest(user_a, project_b) is None
    assert not (tmp_path / "snapshots" / str(user_b) / str(project_a)).exists()


def test_snapshot_checksum_uses_identical_order_for_nested_and_mixed_case_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        settings,
        "project_snapshot_directory",
        str(tmp_path / "snapshots"),
    )

    user_id, project_id = uuid4(), uuid4()

    files = (
        ("a.py", "print('root')"),
        ("a/main.py", "print('nested')"),
        ("src/Zebra.py", "print('upper')"),
        ("src/alpha.py", "print('lower')"),
    )

    saved = store_snapshot(
        user_id,
        project_id,
        _snapshot(*reversed(files)),
    )

    assert read_snapshot_sources(
        user_id,
        project_id,
        saved.snapshot_id,
    ) == sorted(files)