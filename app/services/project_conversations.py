"""Bounded, owner-scoped investigation conversations stored outside source snapshots.

Uses only Python's sqlite3. No migrations, database credentials, vector queries,
or AI calls are required for history. The server-side database contains questions,
model reports, and verified evidence quotes in plaintext; operators must secure
and back up the configured FORGEPATH_CONVERSATION_DB directory accordingly.

Never accept a filesystem path from an HTTP request. All scopes are server-side
UUID values and every operation scopes by authenticated owner AND project.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

MAX_THREADS_PER_PROJECT = 40
MAX_TURNS_PER_THREAD = 20
MAX_STORED_REPORT_BYTES = 48_000
PENDING_LEASE_SECONDS = 600


class ConversationNotFound(Exception):
    pass


class ConversationConflict(Exception):
    pass


class ConversationLimit(Exception):
    pass


class ConversationStorageError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db_file() -> Path:
    configured = os.getenv("FORGEPATH_CONVERSATION_DB", "").strip()
    path = Path(configured).expanduser() if configured else Path.home() / ".forgepath" / "conversations.sqlite3"
    if not path.is_absolute():
        raise ConversationStorageError("FORGEPATH_CONVERSATION_DB must be an absolute path")
    return path


@contextmanager
def _connection():
    path = _db_file()
    conn: sqlite3.Connection | None = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        first_creation = not path.exists()
        conn = sqlite3.connect(path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 15000")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS forgepath_conversation_threads (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                snapshot_sha TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                pending_id TEXT,
                pending_until INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_forgepath_threads_owner_project
                ON forgepath_conversation_threads(owner_id, project_id, updated_at);
            CREATE TABLE IF NOT EXISTS forgepath_conversation_turns (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES forgepath_conversation_threads(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                question TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(thread_id, ordinal)
            );
            CREATE INDEX IF NOT EXISTS idx_forgepath_turns_thread
                ON forgepath_conversation_turns(thread_id, ordinal);
        """)
        if first_creation and os.name == "posix":
            path.chmod(0o600)
        yield conn
    except (sqlite3.Error, OSError) as exc:
        raise ConversationStorageError("Conversation history storage is unavailable") from exc
    finally:
        if conn is not None:
            conn.close()


def _row(conn: sqlite3.Connection, owner: UUID, project: UUID, thread: UUID) -> sqlite3.Row:
    result = conn.execute(
        "SELECT * FROM forgepath_conversation_threads WHERE id = ? AND owner_id = ? AND project_id = ?",
        (str(thread), str(owner), str(project)),
    ).fetchone()
    if result is None:
        raise ConversationNotFound("Conversation not found")
    return result


def _turns(conn: sqlite3.Connection, thread_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT ordinal, question, report_json, created_at FROM forgepath_conversation_turns "
        "WHERE thread_id = ? ORDER BY ordinal", (thread_id,),
    ).fetchall()
    return [{"ordinal": row["ordinal"], "question": row["question"],
             "report": json.loads(row["report_json"]), "created_at": row["created_at"]}
            for row in rows]


def _public_thread(row: sqlite3.Row, snapshot_id: UUID, snapshot_sha: str) -> dict:
    return {
        "id": row["id"], "project_id": row["project_id"],
        "snapshot_id": row["snapshot_id"], "title": row["title"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
        "revision": row["revision"],
        "stale": row["snapshot_id"] != str(snapshot_id) or row["snapshot_sha"] != snapshot_sha,
    }


def list_threads(owner: UUID, project: UUID, snapshot_id: UUID, snapshot_sha: str) -> list[dict]:
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM forgepath_conversation_threads WHERE owner_id = ? AND project_id = ? "
            "ORDER BY updated_at DESC, id LIMIT ?",
            (str(owner), str(project), MAX_THREADS_PER_PROJECT),
        ).fetchall()
        return [_public_thread(row, snapshot_id, snapshot_sha) for row in rows]


def create_thread(owner: UUID, project: UUID, snapshot_id: UUID, snapshot_sha: str) -> dict:
    with _connection() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            count = conn.execute(
                "SELECT COUNT(*) FROM forgepath_conversation_threads WHERE owner_id = ? AND project_id = ?",
                (str(owner), str(project)),
            ).fetchone()[0]
            if count >= MAX_THREADS_PER_PROJECT:
                raise ConversationLimit("Maximum conversations reached for this project; delete an old conversation")
            thread_id = str(uuid4())
            now = _now()
            conn.execute("INSERT INTO forgepath_conversation_threads "
                         "(id, owner_id, project_id, snapshot_id, snapshot_sha, title, created_at, updated_at) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (thread_id, str(owner), str(project), str(snapshot_id), snapshot_sha,
                          "New investigation", now, now))
            conn.commit()
            return {"id": thread_id, "project_id": str(project), "snapshot_id": str(snapshot_id),
                    "title": "New investigation", "created_at": now, "updated_at": now,
                    "revision": 0, "stale": False, "turns": []}
        except Exception:
            conn.rollback()
            raise


def get_thread(owner: UUID, project: UUID, thread: UUID,
               snapshot_id: UUID, snapshot_sha: str) -> dict:
    with _connection() as conn:
        row = _row(conn, owner, project, thread)
        return {**_public_thread(row, snapshot_id, snapshot_sha), "turns": _turns(conn, row["id"])}


def delete_thread(owner: UUID, project: UUID, thread: UUID) -> None:
    with _connection() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            _row(conn, owner, project, thread)
            conn.execute("DELETE FROM forgepath_conversation_threads WHERE id = ? AND owner_id = ? AND project_id = ?",
                         (str(thread), str(owner), str(project)))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def reserve_turn(owner: UUID, project: UUID, thread: UUID, snapshot_id: UUID,
                 snapshot_sha: str, expected_revision: int) -> tuple[str, list[dict]]:
    """Atomically prevent two charged model requests using the same thread revision."""
    with _connection() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = _row(conn, owner, project, thread)
            if row["snapshot_id"] != str(snapshot_id) or row["snapshot_sha"] != snapshot_sha:
                raise ConversationConflict("Snapshot changed. Start a new conversation for the current snapshot")
            if row["revision"] != expected_revision:
                raise ConversationConflict("Conversation changed. Refresh its history and preview again")
            if row["revision"] >= MAX_TURNS_PER_THREAD:
                raise ConversationLimit("This conversation reached 20 messages; start a new conversation")
            if row["pending_id"] and (row["pending_until"] or 0) > int(time.time()):
                raise ConversationConflict("An investigation is already running in this conversation")
            history = _turns(conn, row["id"])
            reservation = str(uuid4())
            conn.execute("UPDATE forgepath_conversation_threads SET pending_id = ?, pending_until = ? "
                         "WHERE id = ? AND owner_id = ? AND project_id = ?",
                         (reservation, int(time.time()) + PENDING_LEASE_SECONDS,
                          str(thread), str(owner), str(project)))
            conn.commit()
            return reservation, history
        except Exception:
            conn.rollback()
            raise


def cancel_turn(owner: UUID, project: UUID, thread: UUID, reservation: str) -> None:
    with _connection() as conn:
        conn.execute("UPDATE forgepath_conversation_threads SET pending_id = NULL, pending_until = NULL "
                     "WHERE id = ? AND owner_id = ? AND project_id = ? AND pending_id = ?",
                     (str(thread), str(owner), str(project), reservation))


def finish_turn(owner: UUID, project: UUID, thread: UUID, reservation: str,
                expected_revision: int, question: str, report: dict,
                snapshot_id: UUID, snapshot_sha: str) -> dict:
    report_json = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    if len(report_json.encode("utf-8")) > MAX_STORED_REPORT_BYTES:
        raise ConversationLimit("Investigation result exceeds the conversation storage limit")
    with _connection() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = _row(conn, owner, project, thread)
            if (row["pending_id"] != reservation or row["revision"] != expected_revision
                    or row["snapshot_id"] != str(snapshot_id) or row["snapshot_sha"] != snapshot_sha):
                raise ConversationConflict("Conversation was changed before this answer could be saved")
            ordinal = expected_revision + 1
            now = _now()
            conn.execute("INSERT INTO forgepath_conversation_turns "
                         "(id, thread_id, ordinal, question, report_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                         (str(uuid4()), str(thread), ordinal, question, report_json, now))
            conn.execute("UPDATE forgepath_conversation_threads SET revision = ?, updated_at = ?, "
                         "title = CASE WHEN revision = 0 THEN ? ELSE title END, "
                         "pending_id = NULL, pending_until = NULL WHERE id = ? AND owner_id = ? AND project_id = ?",
                         (ordinal, now, question.strip()[:90], str(thread), str(owner), str(project)))
            conn.commit()
            return get_thread(owner, project, thread, snapshot_id, snapshot_sha)
        except Exception:
            conn.rollback()
            raise


def prior_retrieval_query(question: str, turns: list[dict]) -> str:
    """Short history helps resolve follow-ups such as 'what calls that function?'"""
    recent = [item["question"].strip()[:260] for item in turns[-2:] if item.get("question")]
    result = question.strip()
    if recent:
        result += "\nPrevious investigation topics: " + " | ".join(recent)
    return result[:2000]


def continuity_paths(turns: list[dict]) -> list[str]:
    result: list[str] = []
    for item in reversed(turns[-3:]):
        paths = item.get("report", {}).get("selected_files", [])
        for path in paths:
            if isinstance(path, str) and path not in result:
                result.append(path)
            if len(result) >= 6:
                return result
    return result


def bounded_history_prompt(turns: list[dict]) -> str:
    """Only bounded summaries/questions, not old code, enter the next AI request."""
    segments = []
    for item in turns[-4:]:
        report = item.get("report", {})
        segments.append(
            f"User asked: {str(item.get('question', ''))[:400]}\n"
            f"Previous model summary (not verified evidence): {str(report.get('summary', ''))[:850]}\n"
            f"Previously selected paths (not proof): {', '.join(continuity_paths([item]))[:350]}"
        )
    return "\n\n".join(segments)[:5600] or "No earlier turns"