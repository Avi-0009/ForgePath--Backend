from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compose_fails_closed_and_binds_services_to_loopback():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?" in compose
    assert "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-password}" not in compose
    assert '127.0.0.1:${POSTGRES_PORT:-5433}:5432' in compose
    assert '127.0.0.1:${BACKEND_PORT:-8000}:8000' in compose
    assert '127.0.0.1:${FRONTEND_PORT:-5173}:8080' in compose
    assert '127.0.0.1:${PGADMIN_PORT:-5050}:80' in compose
    assert "PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD:-}" in compose
    assert "PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD:-admin}" not in compose


def test_compose_keeps_migrations_explicit_and_private_data_persistent():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "PROJECT_SNAPSHOT_DIRECTORY: /var/lib/forgepath/snapshots" in compose
    assert "FORGEPATH_CONVERSATION_DB: /var/lib/forgepath/conversations/conversations.sqlite3" in compose
    assert "forgepath_private_data:/var/lib/forgepath" in compose
    assert "command:\n      - python\n      - -B\n      - -m\n      - alembic" in compose
    assert "profiles:\n      - tools" in compose
    assert "no-new-privileges:true" in compose
    assert "read_only: true" in compose
    assert "p.stat().st_mode & 0o077" in compose


def test_backend_container_runs_as_an_unprivileged_user():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "USER forgepath" in dockerfile
    assert "--chown=forgepath:forgepath" in dockerfile
    assert "chmod 0700 /var/lib/forgepath" in dockerfile
    assert "/var/lib/forgepath/snapshots" in dockerfile
    assert "/var/lib/forgepath/conversations" in dockerfile
    assert "--reload" not in dockerfile


def test_secret_files_are_excluded_from_the_build_context():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert ".env\n" in dockerignore
    assert ".env.*" in dockerignore
    assert ".git\n" in dockerignore
