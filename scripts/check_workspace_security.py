r"""Run the workspace regression suite without loading .env or contacting services.

Usage (activated environment): python.exe -B scripts/check_workspace_security.py
Uses existing dependencies only; all generated test files belong to pytest temp dirs.
"""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True

    # Retain only OS process basics, never inherited application credentials.
    process_environment = {
        key: value for key, value in os.environ.items()
        if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "COMSPEC", "PATHEXT"}
    }
    os.environ.clear()
    os.environ.update(process_environment)
    os.environ.update({
        "DATABASE_URL": "postgresql+asyncpg://test:test@127.0.0.1:1/unused",
        "JWT_SECRET_KEY": "workspace-security-tests-only-000000000000",
        "BREVO_API_KEY": "unused-test-value",
        "BREVO_SENDER_EMAIL": "tests@example.com",
        "PASSWORD_RESET_SECRET_KEY": "workspace-security-tests-only-000000000000",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    from pydantic_settings import DotEnvSettingsSource
    DotEnvSettingsSource._read_env_files = lambda self: {}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Network/process execution is prohibited in this isolated suite")

    # Windows asyncio may implement its wake-up socketpair using loopback TCP.
    # Permit only connections made inside that stdlib function, never API/DB calls.
    socketpair_code = getattr(socket.socketpair, "__code__", None)
    def socketpair_only(original):
        def connect(sock, address):
            frame = sys._getframe(1)
            while frame is not None:
                if socketpair_code is not None and frame.f_code is socketpair_code:
                    return original(sock, address)
                frame = frame.f_back
            return forbidden()
        return connect
    socket.socket.connect = socketpair_only(socket.socket.connect)
    socket.socket.connect_ex = socketpair_only(socket.socket.connect_ex)
    socket.create_connection = forbidden
    subprocess.Popen = forbidden

    import pytest
    with tempfile.TemporaryDirectory(prefix="forgepath-security-") as temporary:
        for key in ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME"):
            os.environ[key] = temporary
        return pytest.main([
            "-q", "-p", "no:cacheprovider", "-p", "anyio.pytest_plugin",
            "tests/test_workspace_ownership.py",
            "tests/test_workspace.py", "tests/test_workspace_tree.py",
            "tests/test_workspace_context_routes.py", "tests/test_workspace_indexing.py",
            "tests/test_workspace_review_context.py", "tests/test_local_context.py",
            "tests/test_code_patch_apply.py", "tests/test_code_review_workflow.py",
            "tests/test_code_patch.py",
            "tests/test_code_review_rag_routes.py", "tests/test_code_patch_rag_routes.py",
            "tests/test_task_planning_routes.py", "tests/test_test_execution_routes.py",
            "tests/test_agent_orchestration_routes.py",
            "tests/test_ai_configuration_routes.py",
            "tests/test_auth_cookie_security.py", "tests/test_auth_session_revocation.py",
            "tests/test_profile_update.py",
            "tests/test_project_review_jobs.py", "tests/test_project_review_agent.py",
            "tests/test_project_review_findings.py",
            "tests/test_project_review_tasks.py", "tests/test_project_review_task_routes.py",
            "tests/test_project_workflows.py",
            "tests/test_project_patch_reconciliation.py",
            "tests/test_project_snapshot_reconciliation_routes.py",
            "tests/test_task_planning.py",
            "tests/test_project_deletion.py", "tests/test_project_index_lifecycle.py",
            "tests/test_project_conversations.py", "tests/test_project_conversations_routes.py",
        ])


if __name__ == "__main__":
    raise SystemExit(main())
