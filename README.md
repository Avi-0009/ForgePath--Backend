# ForgePath Backend

Private, user-scoped backend services for ForgePath, an AI-assisted software engineering application. The API accepts bounded browser-uploaded source snapshots, performs codebase analysis, preserves review and approval state, and keeps consequential actions behind explicit user decisions.

Built with FastAPI, SQLAlchemy, PostgreSQL, Alembic, SQLite, Chroma, and LangGraph.

## What is implemented

- Account registration, login, refresh-session rotation/revocation, profile updates, email verification, and password reset.
- User-owned projects and immutable private source snapshots.
- Initial codebase reviews with durable job state, leases, retry handling, and AI-usage tracking.
- File investigation, code search, reference-integrity checks, and context-aware reports.
- Opt-in semantic indexing with snapshot-pinned vector namespaces.
- Snapshot-pinned project conversations.
- Review findings promoted into explicit tasks.
- Task planning, patch generation, proposal approval/rejection, and verified `.diff` export.
- Explicitly confirmed browser-local patch application with full-snapshot and file-hash validation.
- Detection of approved patches when their exact proposed content appears in a later upload.
- Legacy administrator-mapped local workspace APIs for planning, patching, rollback, and tests.
- Encrypted per-user NVIDIA, OpenAI, and Hugging Face credentials.

AI-backed operations may incur provider charges and should be initiated deliberately.

## Architecture

```text
React client
    |
    v
FastAPI
    +-- PostgreSQL: users, projects, reviews, tasks, settings, usage, workflows
    +-- Private filesystem: immutable user/project/snapshot source trees
    +-- SQLite: owner-scoped local instructions and project conversations
    +-- Chroma: optional snapshot-scoped semantic indexes
    +-- AI providers: platform configuration or encrypted per-user credentials
```

PostgreSQL is the source of truth for durable application state. Source snapshots are stored outside PostgreSQL in an operator-selected private directory. Enabled SQLite and Chroma data must also be included in the deployment backup strategy.

## Requirements

- Python 3.13 recommended (the container image uses Python 3.13).
- PostgreSQL 16 recommended.
- Docker with Compose, or an independently managed PostgreSQL instance.
- A Brevo API key and verified sender for verification/reset emails.
- The sibling `ForgePath--Frontend` repository for the browser application.

## Local setup

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -B -m pip install --upgrade pip
python -B -m pip install -r requirements.txt
```

The production requirements do not include pytest. Install it explicitly in development environments that run the test suite:

```powershell
python -B -m pip install pytest
```

POSIX:

```bash
python -m venv .venv
source .venv/bin/activate
python -B -m pip install --upgrade pip
python -B -m pip install -r requirements.txt
```

### 2. Start PostgreSQL

```powershell
docker compose up -d db
```

The default database is exposed on `localhost:5433`. To use another server, set `DATABASE_URL` accordingly.

> The Compose `backend` service does not currently forward every required mail, reset, encryption, and snapshot variable. For local development, start PostgreSQL with Compose and run FastAPI from the activated environment.

### 3. Configure the environment

Copy `.env.example` to `.env`, replace its placeholders, and add the required application settings below. Never commit `.env`.

```dotenv
DATABASE_URL=postgresql+asyncpg://forgepath_user:change_me@localhost:5433/forgepath

JWT_SECRET_KEY=<at-least-32-random-characters>
PASSWORD_RESET_SECRET_KEY=<independent-random-secret>

BREVO_API_KEY=<brevo-api-key>
BREVO_SENDER_EMAIL=<verified-sender@example.com>
BREVO_SENDER_NAME=ForgePath

CORS_ORIGINS=http://localhost:5173
REFRESH_COOKIE_SECURE=false

PROJECT_SNAPSHOT_DIRECTORY=<absolute-private-directory>
AI_CREDENTIALS_ENCRYPTION_KEY=<fernet-key>
```

Generate secrets locally:

```powershell
python -B -c "import secrets; print(secrets.token_urlsafe(48))"
python -B -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Use separate values for JWT signing, password-reset signing, and Fernet encryption. `REFRESH_COOKIE_SECURE=false` is only appropriate for trusted local HTTP development.

### 4. Apply migrations

```powershell
python -B -m alembic heads
python -B -m alembic upgrade head
python -B -m alembic current
```

There should be exactly one Alembic head. Apply migrations before starting code that depends on a new schema revision.

### 5. Start FastAPI

```powershell
python -B -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- API root: `http://127.0.0.1:8000/`
- OpenAPI UI: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Database health: `http://127.0.0.1:8000/health/db`

## Configuration reference

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | Yes | Async SQLAlchemy PostgreSQL URL. |
| `JWT_SECRET_KEY` | Yes | Access-token signing key; minimum 32 characters. |
| `JWT_ALGORITHM` | No | JWT algorithm; currently restricted to `HS256`. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | Access-token lifetime; default `30`. |
| `REFRESH_TOKEN_EXPIRE_DAYS` | No | Refresh-token lifetime; default `7`. |
| `REFRESH_COOKIE_SECURE` | No | Require HTTPS refresh cookies; defaults to `true`. |
| `CORS_ORIGINS` | No | Comma-separated allowed frontend origins. |
| `BREVO_API_KEY` | Yes | Transactional email credential. |
| `BREVO_SENDER_EMAIL` | Yes | Verified sender address. |
| `BREVO_SENDER_NAME` | No | Sender display name. |
| `PASSWORD_RESET_SECRET_KEY` | Yes | Independent password-reset signing secret. |
| `PROJECT_SNAPSHOT_DIRECTORY` | For uploads | Absolute private directory for immutable snapshots. |
| `AI_CREDENTIALS_ENCRYPTION_KEY` | For personal providers | Fernet key used to encrypt user API keys. |
| `EMBEDDING_PROVIDER` | No | Set to `openai` to enable semantic indexing. |
| `OPENAI_API_KEY` | With OpenAI embeddings | Platform embedding credential. |
| `EMBEDDING_MODEL` | No | Defaults to `text-embedding-3-small`. |
| `EMBEDDING_DIMENSIONS` | No | Vector dimensions, up to `1536`. |
| `EMBEDDING_BATCH_SIZE` | No | Embedding batch size, `1–128`. |
| `FORGEPATH_CONVERSATION_DB` | No | Absolute SQLite path for project conversations. |
| `WORKSPACE_ALLOWED_ROOTS` | For legacy APIs | Semicolon-separated server-local roots. |
| `WORKSPACE_ROOT_OWNERS` | For legacy APIs | JSON mapping of exact roots to owner UUIDs. |

### Storage permissions

`PROJECT_SNAPSHOT_DIRECTORY` must be absolute, private to the backend service account, and not a symlink or junction. On POSIX systems ForgePath rejects snapshot roots accessible by group or other users.

The conversation SQLite database contains questions, reports, and verified source quotations in plaintext. Restrict its permissions and back it up only when history must be retained.

## Primary API areas

All application endpoints are under `/api/v1`.

| Area | Prefix | Notes |
| --- | --- | --- |
| Authentication | `/auth` | Signup, login, refresh, logout, profile update, verification, reset. |
| Projects | `/projects` | User-owned lifecycle and deletion. |
| Snapshots | `/projects/{id}/snapshot` | Bounded browser source upload and metadata. |
| Reviews | `/projects/{id}/reviews` | Initial review jobs and results. |
| Tasks | `/projects/{id}/tasks` | Findings, plans, patches, decisions, exports. |
| Investigation | `/projects/{id}/investigation*` | Snapshot file and context-aware analysis. |
| Conversations | `/projects/{id}/investigation/conversations` | Snapshot-pinned project questions. |
| Semantic index | `/projects/{id}/semantic-index*` | Estimate, build, status, search, removal. |
| AI configuration | `/ai/configuration` | Encrypted per-user provider settings. |
| AI usage | `/ai/usage` | User-scoped usage and test metrics. |
| Legacy workspace | `/workspace`, `/task-planning`, `/code-patch` | Administrator-mapped local workflows. |
| Agent orchestration | `/agent/workflows` | Approval-gated local workflow state. |

Use `/docs` as the generated request/response reference for the running revision.

## Browser-project workflow

```text
Select folder
  -> preview permitted files
  -> upload immutable snapshot
  -> initial review
  -> save finding as task
  -> approve and create plan
  -> create patch proposal
  -> approve or reject proposal
  -> explicitly confirm local application
  -> verify the connected folder and exact source hash
  -> replace one existing file in the browser
  -> upload the updated immutable snapshot
  -> reconcile the exact proposal as applied
```

Approval of a plan or patch never edits a file by itself. Local application requires a separate confirmation and browser write permission. Browser-uploaded projects are not arbitrary server filesystem paths, and the backend never receives a client filesystem path.

## Security boundaries

- Every project, review, task, snapshot, conversation, and semantic namespace is user-scoped.
- Sign-out revokes the hashed refresh-session record and removes the HTTP-only refresh cookie.
- Snapshot paths are relative, bounded, extension-checked, and screened for sensitive filenames and credential-like content.
- Snapshots are immutable and checksum-verified when read.
- Plans, patches, decisions, and exports are pinned to exact snapshots and hashes.
- Local patch application reauthorizes the approved proposal immediately before the browser write and binds the replacement upload to its expected base snapshot.
- Duplicate paid planning/patch calls are fenced with durable tokens and timeouts.
- Personal provider keys are encrypted with Fernet and never serialized in API responses.
- Legacy workspace APIs require both an allowlisted root and explicit root-to-user ownership.
- Patch application, rollback, test execution, connection tests, and other consequential actions require explicit authorization.
- Do not expose PostgreSQL, pgAdmin, snapshot storage, Chroma, SQLite, or development servers to untrusted networks.

## Tests and verification

Run the isolated security/regression suite without loading `.env`, contacting providers, or executing subprocesses:

```powershell
python -B scripts/check_workspace_security.py
```

Run the complete suite when the test environment is configured:

```powershell
python -B -m pytest -q
```

Read-only migration checks:

```powershell
python -B -m alembic heads
python -B -m alembic current
```

The isolated runner supplies test-only settings, blocks network/process access, and disables automatic third-party pytest plugin loading.

## Repository layout

```text
app/
  api/                 FastAPI routers and dependencies
  core/                settings, auth, provider and request contexts
  db/                  SQLAlchemy engine and declarative base
  models/              PostgreSQL ORM models
  schemas/             validated API/service contracts
  services/            snapshots, reviews, indexing, planning, patches, agents
alembic/versions/       ordered PostgreSQL migrations
scripts/                isolated checks and maintenance helpers
tests/                  unit, route, security, and workflow regressions
```

## Troubleshooting

### Import fails before the server starts

Settings are validated at import time. Confirm the database URL, JWT/reset secrets, and Brevo values are present. Do not weaken validation or place real credentials in source control.

### Login works, but refresh fails locally

For trusted HTTP development, set `REFRESH_COOKIE_SECURE=false`. Restore secure cookies before HTTPS deployment.

### Snapshot upload returns 503

Confirm `PROJECT_SNAPSHOT_DIRECTORY` is absolute, writable by the service account, not a link/junction, and privately permissioned.

### Semantic indexing returns 503

Indexing is optional. Configure `EMBEDDING_PROVIDER=openai` and the platform embedding key, then restart. Indexing may incur provider charges.

### Personal AI settings return 503

Set a valid Fernet `AI_CREDENTIALS_ENCRYPTION_KEY`. Existing ciphertext requires the same key.

### A task becomes historical after upload

Tasks are pinned to their source snapshot. Create new work from the current review. Approved proposals whose exact content appears in a new snapshot are marked applied.

## Operational notes

- Back up PostgreSQL and any enabled snapshot, SQLite, and Chroma storage.
- Apply migrations once per deployment before starting the new revision.
- Use a reverse proxy with HTTPS, request-size limits, and security headers.
- Coordinate local Chroma access; multi-worker indexing needs a durable distributed lock/job queue.

## Related repository

The React client lives in `ForgePath--Frontend`. Its Vite server proxies `/api` to `http://127.0.0.1:8000` by default.
