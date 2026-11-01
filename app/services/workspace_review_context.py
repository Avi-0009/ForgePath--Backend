import asyncio
import json

from dataclasses import dataclass

from app.services.workspace_indexing import WorkspaceIndexingService
from app.services.workspace import (
    WorkspaceReadError,
    read_configured_workspace_file,
)


DEFAULT_REVIEW_CONTEXT_TOP_K = 5
DEFAULT_MAX_REVIEW_CONTEXT_CHARS = 12_000
MAX_RETRIEVAL_QUERY_CHARS = 8_000


@dataclass(frozen=True, slots=True)
class RetrievedWorkspaceChunk:
    """A verified code chunk retrieved from the workspace index."""

    relative_path: str
    language: str
    start_line: int
    end_line: int
    content: str
    distance: float | None


@dataclass(frozen=True, slots=True)
class WorkspaceReviewContext:
    """Verified reference context for an AI code review."""

    chunks: tuple[RetrievedWorkspaceChunk, ...]
    stale_chunks_skipped: int
    unavailable_chunks_skipped: int
    total_chars: int

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def to_prompt_text(self) -> str:
        """
        Serialize retrieved context as JSON.

        JSON keeps file metadata and source content clearly separated.
        Retrieved source code must still be treated as untrusted data.
        """

        if not self.chunks:
            return "No verified related workspace context was found."

        documents = [
            {
                "relative_path": chunk.relative_path,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content,
            }
            for chunk in self.chunks
        ]

        return json.dumps(
            documents,
            ensure_ascii=False,
            indent=2,
        )


class WorkspaceReviewContextBuilder:
    """
    Retrieve and validate related workspace code for code reviews.

    Pipeline:
        Selected source file
            -> Semantic search
            -> Exclude selected file
            -> Validate workspace and file hashes
            -> Enforce context size limit
            -> Return verified reference chunks

    This service never modifies source files or indexed vectors.
    """

    def __init__(
        self,
        indexing_service: WorkspaceIndexingService | None,
        *,
        top_k: int = DEFAULT_REVIEW_CONTEXT_TOP_K,
        max_context_chars: int = DEFAULT_MAX_REVIEW_CONTEXT_CHARS,
    ) -> None:

        if top_k < 1 or top_k > 20:
            raise ValueError(
                "top_k must be between 1 and 20."
            )

        if max_context_chars < 1:
            raise ValueError(
                "max_context_chars must be greater than zero."
            )

        self.indexing_service = indexing_service
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    @staticmethod
    def _build_retrieval_query(
        *,
        relative_path: str,
        language: str | None,
        source_content: str,
    ) -> str:
        """Build a bounded semantic query from the selected source."""

        source_excerpt = source_content[
            :MAX_RETRIEVAL_QUERY_CHARS
        ]

        return (
            "Find code related to the following source file. "
            "Retrieve relevant implementations, dependencies, "
            "definitions, interfaces, and related functionality.\n\n"
            f"File: {relative_path}\n"
            f"Language: {language or 'unknown'}\n\n"
            "Selected source code:\n"
            f"{source_excerpt}"
        )

    async def build(
        self,
        *,
        workspace_root: str,
        relative_path: str,
        source_content: str,
        language: str | None = None,
    ) -> WorkspaceReviewContext:
        """
        Build verified reference context for one selected source file.

        Missing indexing configuration or an empty index results in
        an empty context rather than preventing ordinary code review.

        Retrieved chunks are only accepted when their indexed document
        hash matches the current source file on disk.
        """

        if not workspace_root.strip():
            raise ValueError("workspace_root cannot be empty.")

        if not relative_path.strip():
            raise ValueError("relative_path cannot be empty.")

        if not source_content.strip():
            return WorkspaceReviewContext(
                chunks=(),
                stale_chunks_skipped=0,
                unavailable_chunks_skipped=0,
                total_chars=0,
            )

        if self.indexing_service is None:
            return WorkspaceReviewContext(
                chunks=(),
                stale_chunks_skipped=0,
                unavailable_chunks_skipped=0,
                total_chars=0,
            )

        query = self._build_retrieval_query(
            relative_path=relative_path,
            language=language,
            source_content=source_content,
        )

        hits = await self.indexing_service.search(
            query,
            top_k=self.top_k,
            workspace_root=workspace_root,
            exclude_relative_path=relative_path,
        )

        verified_chunks: list[RetrievedWorkspaceChunk] = []

        stale_chunks_skipped = 0
        unavailable_chunks_skipped = 0
        total_chars = 0

        # Cache file validation so multiple chunks from one file
        # do not trigger repeated disk reads.
        file_validation: dict[
            tuple[str, str],
            bool | None,
        ] = {}

        seen_chunks: set[tuple[str, int]] = set()

        for hit in hits:

            metadata = hit.metadata

            hit_workspace = metadata.get("workspace_root")
            hit_path = metadata.get("relative_path")
            document_sha256 = metadata.get("document_sha256")

            # Never trust vector metadata outside the requested workspace.
            if hit_workspace != workspace_root:
                unavailable_chunks_skipped += 1
                continue

            if (
                not isinstance(hit_path, str)
                or not hit_path.strip()
                or hit_path == relative_path
            ):
                unavailable_chunks_skipped += 1
                continue

            if (
                not isinstance(document_sha256, str)
                or len(document_sha256) != 64
            ):
                unavailable_chunks_skipped += 1
                continue

            chunk_index = metadata.get("chunk_index")

            if (
                isinstance(chunk_index, bool)
                or not isinstance(chunk_index, int)
                or chunk_index < 0
            ):
                unavailable_chunks_skipped += 1
                continue

            chunk_identity = (
                hit_path,
                chunk_index,
            )

            if chunk_identity in seen_chunks:
                continue

            seen_chunks.add(chunk_identity)

            validation_key = (
                hit_path,
                document_sha256,
            )

            if validation_key not in file_validation:
                try:
                    current_source = await asyncio.to_thread(
                        read_configured_workspace_file,
                        workspace_root=workspace_root,
                        relative_path=hit_path,
                    )

                    file_validation[validation_key] = (
                        current_source.sha256 == document_sha256
                    )

                except WorkspaceReadError:
                    file_validation[validation_key] = None

            is_current = file_validation[validation_key]

            if is_current is None:
                unavailable_chunks_skipped += 1
                continue

            if not is_current:
                stale_chunks_skipped += 1
                continue

            content = hit.document

            if not isinstance(content, str) or not content.strip():
                unavailable_chunks_skipped += 1
                continue

            if len(content) > (
                self.max_context_chars - total_chars
            ):
                continue

            start_line = metadata.get("start_line")
            end_line = metadata.get("end_line")
            chunk_language = metadata.get("language")

            if (
                isinstance(start_line, bool)
                or not isinstance(start_line, int)
                or start_line < 1
            ):
                unavailable_chunks_skipped += 1
                continue

            if (
                isinstance(end_line, bool)
                or not isinstance(end_line, int)
                or end_line < start_line
            ):
                unavailable_chunks_skipped += 1
                continue

            if not isinstance(chunk_language, str):
                chunk_language = "unknown"

            verified_chunks.append(
                RetrievedWorkspaceChunk(
                    relative_path=hit_path,
                    language=chunk_language,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    distance=hit.distance,
                )
            )

            total_chars += len(content)

        return WorkspaceReviewContext(
            chunks=tuple(verified_chunks),
            stale_chunks_skipped=stale_chunks_skipped,
            unavailable_chunks_skipped=unavailable_chunks_skipped,
            total_chars=total_chars,
        )