# Initial review recovery and storage

## Storage ownership

PostgreSQL (`project_reviews`) is the only authority for review attempts, leases,
progress, findings and errors. Each attempt retains its user, project and immutable
snapshot UUID. The source files remain in private local snapshot storage. No source
files are copied into another job store or modified during retry.

Local conversation SQLite, legacy instruction SQLite, Chroma indexes and local
semantic-index job JSON records have different lifecycles. They are not review-job
queues and need no schema change for this feature. Missing local snapshot files
fail a review; database recovery does not reconstruct or silently replace sources.
Multiple API processes must see the same snapshot storage to review the same data.

## Behavior

- A new attempt gets a random lease token and a ten-minute deadline using
  PostgreSQL time, not individual API-process clocks.
- Only one worker can atomically change that attempt from queued to running.
- Workers renew before each model call and persist progress after it. Each call
  has a four-minute wall-clock timeout. Attempted calls are counted before sending.
- Every progress/terminal write requires the correct token, running status and an
  unexpired deadline. A delayed worker cannot revive an interrupted attempt.
- Reading review status or explicitly starting a review reconciles expired jobs
  for the authenticated user. Recovery therefore occurs on polling/request, within
  ten minutes of the last renewal, rather than immediately on server startup.
- A PostgreSQL user-row lock serializes starts across projects/processes. A partial
  unique index also prevents two queued/running attempts for one user.
- Retry is an explicit POST, after frontend confirmation of possible charges.
  Completed/partial reviews are reused. No restart/reaper automatically calls AI.
- Graceful worker cancellation marks interruption when the database is available;
  a hard crash or database outage falls back to lease expiration.

This is durable failure detection with manual retry, not a durable execution queue.
An external provider may finish a request despite cancellation; exact-once billing
cannot be guaranteed. An interrupted attempt preserves already saved findings.

## Deployment and rollback

Migration `f734bc81a902` follows `ee7a1b6d4c25`. It adds nullable lease fields,
extends the status check constraint, interrupts pre-upgrade active attempts, and
creates the active-user unique index. Stop old workers before applying it; old
code does not honor leases. Apply to each intended PostgreSQL deployment using
your normal reviewed migration procedure, then start the new backend and frontend.
Do not run old and new review workers concurrently. No migration was run by Codex.

Rollback also requires stopped workers. Downgrade converts interrupted to failed,
drops the index and lease columns, and restores the previous status constraint.
Do not use downgrade as a routine retry mechanism.

## Verification limits

`python.exe -B scripts/check_workspace_security.py` includes offline lease SQL,
worker fencing, timeout and evidence-count tests plus existing regressions. It
does not connect to local or hosted PostgreSQL or apply migrations. The new tests
use mocked sessions, not proof of actual PostgreSQL locking/concurrency behavior.

Before rollout, verify the migration on an approved disposable PostgreSQL database:
upgrade from the preceding revision with existing queued/running/completed rows;
confirm interrupted legacy rows, preserved results, and one-active-attempt unique
enforcement; run two concurrent starts for one user; terminate a worker; expire its
lease; retry; confirm a delayed old token cannot write. Test upgrade/downgrade
separately from production. Live database schema and deployment storage mounts
must be checked by the operator; configuration/source inspection is not that check.
