import asyncio
import hashlib
import json

import pytest

from app.schemas.code_patch import CodePatchApplyRequest
from app.services import code_patch_apply as patch_apply
from app.services.workspace import read_workspace_file


@pytest.fixture
def patch_environment(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()

    local_app_data = tmp_path / "localappdata"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    # Keep this test isolated from the real workspace configuration.
    monkeypatch.setattr(
        patch_apply,
        "_resolve_allowlisted_root",
        lambda _: root,
    )

    # Use the real workspace reader with an isolated, temporary allowlist.
    def read_test_workspace_file(
        *,
        workspace_root,
        relative_path,
        max_bytes=1_000_000,
    ):
        return read_workspace_file(
            workspace_root=root,
            relative_path=relative_path,
            allowed_workspace_roots=[root],
            max_bytes=max_bytes,
        )

    monkeypatch.setattr(
        patch_apply,
        "read_configured_workspace_file",
        read_test_workspace_file,
    )

    return root, local_app_data


def make_request(root, original, proposed):
    return CodePatchApplyRequest(
        workspace_root=str(root),
        relative_path="sample.py",
        expected_sha256=hashlib.sha256(original).hexdigest(),
        proposed_sha256=hashlib.sha256(proposed).hexdigest(),
        proposed_code=proposed.decode("utf-8"),
        approval="apply",
    )


def test_applies_patch_and_preserves_original_backup(
    patch_environment,
):
    root, local_app_data = patch_environment
    target = root / "sample.py"

    original = b"value = 1\n"
    proposed = b"value = 2\n"
    target.write_bytes(original)

    response = asyncio.run(
        patch_apply.apply_code_patch(
            make_request(root, original, proposed)
        )
    )

    assert response.patch_applied is True
    assert response.original_sha256 == hashlib.sha256(
        original
    ).hexdigest()
    assert response.applied_sha256 == hashlib.sha256(
        proposed
    ).hexdigest()
    assert target.read_bytes() == proposed

    backup_directory = (
        local_app_data / "ForgePath" / "backups"
    )
    backup_path = backup_directory / f"{response.backup_id}.bak"
    manifest_path = backup_directory / f"{response.backup_id}.json"

    assert backup_path.read_bytes() == original
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["original_sha256"] == response.original_sha256
    assert manifest["proposed_sha256"] == response.applied_sha256


def test_rejects_stale_source_without_modifying_file(
    patch_environment,
):
    root, local_app_data = patch_environment
    target = root / "sample.py"

    original = b"value = 1\n"
    proposed = b"value = 2\n"
    target.write_bytes(original)

    request = make_request(root, original, proposed)
    request.expected_sha256 = "0" * 64

    with pytest.raises(patch_apply.StalePatchSourceError):
        asyncio.run(patch_apply.apply_code_patch(request))

    assert target.read_bytes() == original
    assert not (
        local_app_data / "ForgePath" / "backups"
    ).exists()


def test_rejects_mismatched_proposed_hash(
    patch_environment,
):
    root, _ = patch_environment
    target = root / "sample.py"

    original = b"value = 1\n"
    proposed = b"value = 2\n"
    target.write_bytes(original)

    request = make_request(root, original, proposed)
    request.proposed_sha256 = "0" * 64

    with pytest.raises(patch_apply.ProposedPatchHashMismatch):
        asyncio.run(patch_apply.apply_code_patch(request))

    assert target.read_bytes() == original


def test_rechecks_source_before_replacement(
    patch_environment,
    monkeypatch,
):
    root, local_app_data = patch_environment
    target = root / "sample.py"

    original = b"value = 1\n"
    proposed = b"value = 2\n"
    concurrent_change = b"value = 99\n"
    target.write_bytes(original)

    request = make_request(root, original, proposed)
    real_write_backup = patch_apply._write_backup

    # Simulate another writer changing the file after the backup
    # is created but before the patch is replaced.
    def backup_then_change_source(**kwargs):
        real_write_backup(**kwargs)
        target.write_bytes(concurrent_change)

    monkeypatch.setattr(
        patch_apply,
        "_write_backup",
        backup_then_change_source,
    )

    with pytest.raises(patch_apply.StalePatchSourceError):
        asyncio.run(patch_apply.apply_code_patch(request))

    assert target.read_bytes() == concurrent_change

    backup_directory = (
        local_app_data / "ForgePath" / "backups"
    )
    backups = list(backup_directory.glob("*.bak"))

    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def test_backup_failure_does_not_modify_source(
    patch_environment,
    monkeypatch,
):
    root, _ = patch_environment
    target = root / "sample.py"

    original = b"value = 1\n"
    proposed = b"value = 2\n"
    target.write_bytes(original)

    def fail_backup(**kwargs):
        raise patch_apply.PatchApplyError(
            "Simulated backup failure"
        )

    monkeypatch.setattr(patch_apply, "_write_backup", fail_backup)

    with pytest.raises(patch_apply.PatchApplyError):
        asyncio.run(
            patch_apply.apply_code_patch(
                make_request(root, original, proposed)
            )
        )

    assert target.read_bytes() == original