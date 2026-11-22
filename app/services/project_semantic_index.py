"""Version-pinned, opt-in vector indexing of private project snapshots.

Never reads arbitrary client paths. A completion receipt is written only after
all provider embeddings and vector upserts have succeeded.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

from app.services.workspace_ingestion import IngestedWorkspaceDocument
from app.services.workspace_chunking import chunk_document
from app.services.workspace_indexing import WorkspaceIndexingService


def snapshot_vector_namespace(
    service: WorkspaceIndexingService,
    user_id: UUID, project_id: UUID, snapshot_id: UUID,
) -> str:
    """An internal namespace that cannot be selected by an API caller."""
    return str(
        service.vector_store.persist_directory
        / "project-snapshots" / str(user_id) / str(project_id) / str(snapshot_id)
    )


def _receipt_path(
    service: WorkspaceIndexingService, user_id: UUID,
    project_id: UUID, snapshot_id: UUID,
) -> Path:
    return (
        service.vector_store.persist_directory / "project-index-receipts"
        / str(user_id) / str(project_id) / f"{snapshot_id}.json"
    )


def _expected_receipt(
    service: WorkspaceIndexingService, snapshot_id: UUID,
    manifest_sha256: str, file_count: int,
) -> dict:
    return {
        "snapshot_id": str(snapshot_id),
        "snapshot_sha256": manifest_sha256,
        "file_count": file_count,
        "embedding_model": service.vector_store.embedding_model,
    }


def is_snapshot_index_ready(
    service: WorkspaceIndexingService, user_id: UUID, project_id: UUID,
    snapshot_id: UUID, manifest_sha256: str, file_count: int,
) -> bool:
    """Reject stale, incomplete or mismatched model indexes."""
    path = _receipt_path(service, user_id, project_id, snapshot_id)
    if not path.is_file() or path.is_symlink():
        return False
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return False
    if recorded != _expected_receipt(service, snapshot_id, manifest_sha256, file_count):
        return False
    namespace = snapshot_vector_namespace(service, user_id, project_id, snapshot_id)
    try:
        return bool(service.vector_store.get_workspace_file_paths(workspace_root=namespace))
    except Exception:
        return False


def estimate_snapshot_chunks(
    service: WorkspaceIndexingService,
    user_id: UUID, project_id: UUID, snapshot_id: UUID,
    sources: list[tuple[str, str]],
) -> int:
    namespace = snapshot_vector_namespace(service, user_id, project_id, snapshot_id)
    total = 0
    for path, content in sources:
        if not content.strip():
            continue
        encoded = content.encode("utf-8")
        document = IngestedWorkspaceDocument(
            namespace, path, path.rsplit(".", 1)[-1].lower() if "." in path else "",
            content, hashlib.sha256(encoded).hexdigest(), len(encoded), len(content.splitlines()),
        )
        total += len(chunk_document(document))
    return total


async def index_project_snapshot(
    service: WorkspaceIndexingService,
    user_id: UUID, project_id: UUID, snapshot_id: UUID,
    manifest_sha256: str, sources: list[tuple[str, str]],
) -> dict:
    """Explicitly embed verified snapshot sources, then atomically mark ready."""
    namespace = snapshot_vector_namespace(service, user_id, project_id, snapshot_id)
    documents = []
    for path, content in sources:
        if not content.strip():
            continue
        encoded = content.encode("utf-8")
        documents.append(IngestedWorkspaceDocument(
            workspace_root=namespace,
            relative_path=path,
            language=path.rsplit(".", 1)[-1].lower() if "." in path else "",
            content=content,
            sha256=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
            line_count=len(content.splitlines()),
        ))
    if not documents:
        raise ValueError("No nonempty source files available for indexing")

    result = await service.index_documents(documents)
    receipt = _receipt_path(service, user_id, project_id, snapshot_id)
    receipt.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if receipt.is_symlink():
        raise OSError("Unsafe index receipt")
    temporary = receipt.parent / f".{uuid4().hex[:12]}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as output:
            os.chmod(temporary, 0o600)
            json.dump(_expected_receipt(service, snapshot_id, manifest_sha256, len(sources)), output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, receipt)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "files_indexed": len(documents),
        "chunks_embedded": result.chunks_processed,
        "vectors_upserted": result.vectors_upserted,
        "unchanged_files_skipped": result.unchanged_files_skipped,
    }
