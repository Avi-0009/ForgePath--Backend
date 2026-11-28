import asyncio
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1.routes import project_execution as routes
from app.core.config import settings
from app.services.project_snapshots import SourceSnapshot, store_snapshot
from app.services.test_execution import project_runner, sandbox


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_snapshot_directory", str(tmp_path / "snapshots"))
    user_id, project_id = uuid4(), uuid4()
    return user_id, project_id


def save(project, files):
    return store_snapshot(*project, SourceSnapshot(files=[
        {"relative_path": name, "content": content} for name, content in files.items()
    ]))


def run(project, manifest, runner="pytest"):
    return project_runner.execute_project_tests(
        user_id=project[0], project_id=project[1],
        expected_snapshot_id=manifest.snapshot_id, runner=runner,
        timeout_seconds=180,
    )


def test_project_copy_is_disposable_and_result_has_no_host_path(project, monkeypatch):
    manifest = save(project, {"test_sample.py": "def test_ok(): assert True\n"})
    paths = []

    def fake_run(*, snapshot, runner, timeout_seconds):
        paths.append(snapshot)
        assert (snapshot / "test_sample.py").read_text() == "def test_ok(): assert True\n"
        assert not (snapshot / "manifest.json").exists()
        (snapshot / "generated.txt").write_text("temporary")
        return sandbox.SandboxResult(0, False, "passed", "", False, False)

    monkeypatch.setattr(sandbox, "_run_prepared_snapshot", fake_run)
    result = run(project, manifest)
    assert result.status == "passed"
    assert result.snapshot_id == manifest.snapshot_id
    assert result.project_id == project[1]
    assert not hasattr(result, "workspace_root")
    assert not paths[0].exists()


def test_stale_snapshot_never_runs(project, monkeypatch):
    old = save(project, {"test_sample.py": "a"})
    save(project, {"test_sample.py": "b"})
    monkeypatch.setattr(sandbox, "_run_prepared_snapshot", lambda **_: pytest.fail("stale snapshot executed"))
    with pytest.raises(project_runner.ProjectSnapshotChanged):
        run(project, old)


def test_foreign_snapshot_never_runs(project, monkeypatch):
    manifest = save(project, {"test_sample.py": "a"})
    monkeypatch.setattr(sandbox, "_run_prepared_snapshot", lambda **_: pytest.fail("foreign snapshot executed"))
    with pytest.raises(project_runner.ProjectSnapshotChanged):
        run((uuid4(), project[1]), manifest)


@pytest.mark.parametrize("runner", ["npm", "go"])
def test_missing_runner_metadata_never_runs(project, monkeypatch, runner):
    manifest = save(project, {"test_sample.py": "a"})
    monkeypatch.setattr(sandbox, "_run_prepared_snapshot", lambda **_: pytest.fail("invalid snapshot executed"))
    with pytest.raises(project_runner.ProjectTestConfigurationError):
        run(project, manifest, runner)


def test_tampered_snapshot_never_runs(project, monkeypatch):
    manifest = save(project, {"test_sample.py": "a"})
    from pathlib import Path
    target = Path(settings.project_snapshot_directory) / str(project[0]) / str(project[1]) / str(manifest.snapshot_id) / "test_sample.py"
    target.write_text("tampered")
    monkeypatch.setattr(sandbox, "_run_prepared_snapshot", lambda **_: pytest.fail("tampered snapshot executed"))
    with pytest.raises(project_runner.ProjectTestUnavailable):
        run(project, manifest)


def test_go_metadata_survives_snapshot(project):
    from app.services.project_snapshots import read_snapshot_sources
    files = {"go.mod": "module example.com/test\n\ngo 1.23\n", "go.sum": "", "main.go": "package example\n"}
    manifest = save(project, files)
    assert dict(read_snapshot_sources(*project, manifest.snapshot_id)) == files


def test_route_rejects_foreign_project_before_execution(monkeypatch):
    user_id, project_id = uuid4(), uuid4()

    class DB:
        async def execute(self, query):
            compiled = query.compile()
            assert user_id in compiled.params.values()
            assert project_id in compiled.params.values()
            return SimpleNamespace(scalar_one_or_none=lambda: None)

    monkeypatch.setattr(routes, "execute_project_tests", lambda **_: pytest.fail("foreign project executed"))
    payload = routes.ProjectTestRequest(expected_snapshot_id=uuid4(), runner="pytest", approval="run")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes.run_project_tests(project_id, payload, SimpleNamespace(id=user_id), DB()))
    assert exc.value.status_code == 404


def test_route_requires_approval_and_rejects_host_paths():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        routes.ProjectTestRequest(expected_snapshot_id=uuid4(), runner="pytest")
    with pytest.raises(ValidationError):
        routes.ProjectTestRequest(expected_snapshot_id=uuid4(), runner="pytest", approval="run", workspace_root="C:/private")


@pytest.mark.skipif(os.getenv("FORGEPATH_DOCKER_TESTS") != "1", reason="Real Docker tests require explicit opt-in")
@pytest.mark.parametrize("runner,files", [
    ("pytest", {"test_sample.py": "def test_ok(): assert True\n"}),
    ("npm", {"package.json": '{"scripts":{"test":"node --test"}}', "example.test.js": 'require("node:test")("ok", () => {});\n'}),
    ("go", {"go.mod": "module example.com/test\n\ngo 1.23\n", "example_test.go": 'package example\nimport "testing"\nfunc TestOK(t *testing.T) {}\n'}),
])
def test_real_project_snapshot_execution(project, runner, files):
    result = run(project, save(project, files), runner)
    assert result.status == "passed", result.stdout + "\n" + result.stderr
