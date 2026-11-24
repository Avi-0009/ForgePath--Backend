# Project deletion across PostgreSQL and local storage

The owner-authenticated DELETE endpoint locks the PostgreSQL project row, then
cleans local stores before deleting the row and committing. The existing foreign
keys cascade project reviews and review tasks in PostgreSQL.

Cleanup includes every saved snapshot version's vectors and index receipts,
index-job JSON records, conversation threads/turns in SQLite, and private snapshot
files. Conversation cleanup spans old snapshot versions as well as the current
one. It matches both owner and project; other accounts/projects remain intact.

When embeddings are disabled, cleanup opens the existing Chroma collection with
no embedding function. It does not require an API key, generate embeddings, or
create a new collection. A never-indexed project with no vector database can be
deleted. Unreadable storage or vector artifacts with a missing database fail
closed rather than being silently discarded or replaced.

SQLite lazily creates an additional table,
`forgepath_deleted_conversation_projects`, using the conversation store's existing
schema initialization mechanism. Its content-free owner/project marker prevents
late conversation requests from recreating history after cleanup. Turn inserts
from in-flight investigations fail because their thread no longer exists. There
is no PostgreSQL migration for this change; no live database is changed by merely
editing these source files.

These stores do not share a transaction. Failures retain the PostgreSQL project
row, return 503, and explicitly say deletion is incomplete. Some local content
may already be gone. Retry DELETE to finish; this is not an undo operation. A
failed PostgreSQL commit can likewise leave a row whose local data was removed.

SQLite secure_delete is enabled for the purge, but this is application-level
deletion, not a forensic-erasure guarantee for WAL files, filesystem snapshots,
external backups, provider logs or storage hardware. Backup retention remains an
operator responsibility. The content-free deletion marker is deliberately kept.

Legacy local-workspace instructions/backups and historical agent workflows are
not attached to uploaded project UUIDs and are not deleted by this API.

The existing semantic-index lifecycle lock is process-local. This change does
not establish safe concurrent indexing/deletion across multiple API workers or
hosts. Use the supported single-process/shared-storage deployment until durable
index-job coordination is implemented; do not treat this as distributed deletion.

Verification: `python.exe -B scripts/check_workspace_security.py` now includes
real temporary SQLite/source-storage deletion, provider-free mocked vector cleanup,
owner isolation, late-write rejection, and failure-retention checks. PostgreSQL
route sessions are mocked; this does not verify live cascades, Chroma persistence,
or production storage mounts. No paid providers, migrations or Docker are invoked.
