# Legacy workspace ownership

The server filesystem allowlist is a location policy, not a user permission.
Legacy HTTP APIs now require an explicit administrator assignment in addition to
`WORKSPACE_ALLOWED_ROOTS`. Uploaded `/projects` snapshots retain their existing
database ownership checks and require no new setting.

## Configuration

Set `WORKSPACE_ROOT_OWNERS` to a JSON object mapping absolute folder paths to
existing ForgePath user UUIDs. Obtain the UUID from the authenticated `/auth/me`
response or the administrator's user database. Do not use an email or username.

Example deployment environment values (replace both UUIDs and paths):

```text
WORKSPACE_ALLOWED_ROOTS=C:/ForgePath/workspaces
WORKSPACE_ROOT_OWNERS={"C:/ForgePath/workspaces/alice":"11111111-1111-4111-8111-111111111111","C:/ForgePath/workspaces/bob":"22222222-2222-4222-8222-222222222222"}
```

Use the environment syntax appropriate to your deployment; shell quoting is not
part of the JSON value. Restart all API processes after changing assignments.
No schema migration or new dependency is needed. No environment files are
modified by this change.

- Each assignment must be inside the global allowlist and use an absolute path.
- Cross-user parent/child assignments, including aliases resolving to overlapping
  directories, fail closed. Existing symlink/junction restrictions still apply.
- An unassigned account gets an empty workspace tree and 403 for filesystem
  targets, even if those targets are globally allowlisted.
- Invalid ownership configuration returns a generic 503 without folder details.
- Only administrators may edit assignments. The API cannot claim a folder.
- Put only one owner's source beneath a granted root. Filesystem hard links,
  privileged host access, and operator-controlled mounts are not tenant sandboxes.

## HTTP contract

Routes and successful response schemas are unchanged. The mounted legacy routers
for workspace browsing/indexing/search, code review, patches/backups/rollback,
local instructions, task planning, test execution and agent workflows share an
authenticated request scope. Direct foreign targets return 403 before service
execution; authentication still returns 401. Ownership is checked before normal
payload validation when a request specifies a filesystem target.

Workflow checkpoint reads and resumes revalidate the target root, including after
an assignment has been revoked. Workflow history remains filtered by its original
user ID. Ownership changes are not a retroactive purge of a user's historical
reports; review retention separately before transferring a real project.

The service context follows `asyncio.to_thread` and is reset after every request.
Direct, non-HTTP service calls without a scope remain trusted internal operations;
any future worker must explicitly bind the authenticated owner and grants before
using legacy services. Do not expose a legacy router without the dependency
installed by `app/api/v1/router.py`.

## Existing data

- Backups now live under `backups/<user UUID>/`.
- Instructions use `users/<user UUID>/context.sqlite3`.
- Legacy vectors use owner-specific namespaces. Reindex explicitly after rollout;
  reindexing can incur provider charges and is never started automatically.
- Previous shared backups, instructions and vectors are neither exposed through
  the new owner scopes nor automatically migrated/deleted. Administrator review
  is required to establish ownership before any separate migration.
- Revoking access does not cancel an already authorized in-flight operation.
  Drain/restart processes when changing grants.

## Verification

From the backend repository with its Python environment activated:

```powershell
python.exe -B scripts/check_workspace_security.py
```

The runner disables dotenv loading, substitutes test-only settings, blocks network
connections and subprocess execution, and selects the workspace regression suite.
It does not run migrations, Docker, or paid provider calls. Tests use temporary
source files and mock authentication/database/provider boundaries. This verifies
API authorization and service isolation, not deployed PostgreSQL or Docker behavior.
