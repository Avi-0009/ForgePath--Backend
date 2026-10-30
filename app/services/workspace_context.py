from dataclasses import dataclass

from app.services.workspace_ingestion import IngestedWorkspaceDocument, ingest_workspace
from app.services.workspace_chunking import (
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_MAX_CHUNK_CHARS,
    WorkspaceChunk,
    chunk_document,
)


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    """Structured, in-memory context prepared from a workspace."""

    workspace_root: str

    documents: tuple[IngestedWorkspaceDocument, ...]
    chunks: tuple[WorkspaceChunk, ...]

    files_scanned: int
    files_skipped: int
    total_bytes: int

    truncated: bool

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def build_workspace_context(
    workspace_root: str,
    *,
    max_files: int = 1_000,
    max_total_bytes: int = 20_000_000,
    max_file_bytes: int = 1_000_000,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> WorkspaceContext:
    """
    Build structured context from an allowlisted workspace.

    Pipeline:
    1. Discover and safely read supported workspace files.
    2. Split each document into bounded, line-aware chunks.
    3. Preserve source paths, hashes, line ranges, and resource metadata.

    This function performs no database writes, embedding generation,
    network requests, or modifications to workspace files.
    """

    if max_chunk_chars < 1:
        raise ValueError(
            "max_chunk_chars must be greater than zero."
        )

    if overlap_chars < 0 or overlap_chars >= max_chunk_chars:
        raise ValueError(
            "overlap_chars must be non-negative and smaller "
            "than max_chunk_chars."
        )

    ingestion = ingest_workspace(
        workspace_root,
        max_files=max_files,
        max_total_bytes=max_total_bytes,
        max_file_bytes=max_file_bytes,
    )

    # prepare chunks while preserving document provenance.
    prepared_chunks: list[WorkspaceChunk] = []

    for document in ingestion.documents:
        document_chunks = chunk_document(
            document,
            max_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )

        prepared_chunks.extend(document_chunks)

    # return one structured context result.
    return WorkspaceContext(
        workspace_root=ingestion.workspace_root,
        documents=ingestion.documents,
        chunks=tuple(prepared_chunks),
        files_scanned=ingestion.files_scanned,
        files_skipped=ingestion.files_skipped,
        total_bytes=ingestion.total_bytes,
        truncated=ingestion.truncated,
    )