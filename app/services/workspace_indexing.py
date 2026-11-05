from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import (
    ChromaVectorStore,
    VectorSearchHit,
    create_chunk_id
)
from app.services.workspace_chunking import (
    WorkspaceChunk,
    chunk_document,
)
from app.services.workspace_ingestion import (
    IngestedWorkspaceDocument,
    WorkspaceIngestionResult,
)


class WorkspaceIndexingError(RuntimeError):
    """Base error for workspace indexing operations."""


class WorkspaceIndexingConfigurationError(WorkspaceIndexingError):
    """Invalid workspace indexing configuration."""


@dataclass(frozen=True, slots=True)
class WorkspaceIndexingResult:
    """Summary of one workspace indexing operation."""

    files_processed: int
    chunks_processed: int
    vectors_upserted: int
    stale_vectors_deleted: int
    empty_files: int
    unchanged_files_skipped: int = 0
    deleted_files: int = 0


def get_embedding_model_identifier(
    embedding_service: EmbeddingService,
) -> str:
    """
    Generate a stable identifier for the configured embedding model.

    The provider name is included to prevent accidentally mixing
    vectors from different providers using similarly named models.
    """

    return (
        f"{embedding_service.provider_name}:"
        f"{embedding_service.model_name}"
    )


class WorkspaceIndexingService:
    """
    Coordinates workspace chunking, embedding generation and storage.

    Pipeline:
        Ingested documents
            -> Chunking
            -> EmbeddingService
            -> ChromaVectorStore

    Embedding generation is delegated entirely to EmbeddingService.
    This class does not call any external API directly.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
    ) -> None:

        expected_model = get_embedding_model_identifier(
            embedding_service
        )

        if vector_store.dimensions != embedding_service.dimensions:
            raise WorkspaceIndexingConfigurationError(
                "Embedding dimensions do not match the vector store."
            )

        if vector_store.embedding_model != expected_model:
            raise WorkspaceIndexingConfigurationError(
                "The embedding provider/model does not match "
                "the vector store configuration. Expected "
                f"'{expected_model}', received "
                f"'{vector_store.embedding_model}'."
            )

        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def index_document(
        self,
        document: IngestedWorkspaceDocument,
    ) -> WorkspaceIndexingResult:
        """Index a single ingested workspace document."""

        return await self.index_documents([document])

    
    async def index_documents(
        self,
        documents: Sequence[IngestedWorkspaceDocument],
    ) -> WorkspaceIndexingResult:
        """
        Index changed documents and reuse vectors for unchanged files.

        Existing vectors are preserved until replacement embeddings
        have been generated successfully.

        Obsolete chunks are removed after successful upsert.
        """

        if not documents:
            return WorkspaceIndexingResult(
                files_processed=0,
                chunks_processed=0,
                vectors_upserted=0,
                stale_vectors_deleted=0,
                empty_files=0,
                unchanged_files_skipped=0,
            )

        prepared_documents: list[
            tuple[
                IngestedWorkspaceDocument,
                tuple[WorkspaceChunk, ...],
                list[str],
            ]
        ] = []

        all_chunks: list[WorkspaceChunk] = []
        seen_files: set[tuple[str, str]] = set()
        unchanged_files_skipped = 0

        # Check each document before generating chunks or embeddings.
        for document in documents:

            file_identity = (
                document.workspace_root,
                document.relative_path,
            )

            if file_identity in seen_files:
                raise ValueError(
                    "Duplicate workspace document supplied: "
                    f"{document.relative_path}"
                )

            seen_files.add(file_identity)

            existing_document_sha256 = (
                self.vector_store.get_file_document_sha256(
                    workspace_root=document.workspace_root,
                    relative_path=document.relative_path,
                )
            )

            # Identical document: reuse its existing vectors.
            if existing_document_sha256 == document.sha256:
                unchanged_files_skipped += 1
                continue

            chunks = chunk_document(document)

            existing_ids = self.vector_store.get_file_chunk_ids(
                workspace_root=document.workspace_root,
                relative_path=document.relative_path,
            )

            prepared_documents.append(
                (
                    document,
                    chunks,
                    existing_ids,
                )
            )

            all_chunks.extend(chunks)

        # Generate embeddings only for changed documents.
        if all_chunks:
            texts = [
                chunk.content
                for chunk in all_chunks
            ]

            embeddings = await self.embedding_service.embed_documents(
                texts
            )

            vectors_upserted = self.vector_store.upsert_chunks(
                all_chunks,
                embeddings,
            )
        else:
            vectors_upserted = 0

        stale_vectors_deleted = 0

        # Remove obsolete chunks only after successful embedding/upsert.
        for document, chunks, existing_ids in prepared_documents:

            expected_ids = {
                create_chunk_id(chunk)
                for chunk in chunks
            }

            stale_ids = [
                record_id
                for record_id in existing_ids
                if record_id not in expected_ids
            ]

            if stale_ids:
                stale_vectors_deleted += (
                    self.vector_store.delete_ids(stale_ids)
                )

        return WorkspaceIndexingResult(
            files_processed=len(prepared_documents),
            chunks_processed=len(all_chunks),
            vectors_upserted=vectors_upserted,
            stale_vectors_deleted=stale_vectors_deleted,
            empty_files=sum(
                1
                for _, chunks, _ in prepared_documents
                if not chunks
            ),
            unchanged_files_skipped=unchanged_files_skipped,
        )

    
    async def index_workspace(
        self,
        ingestion: WorkspaceIngestionResult,
    ) -> WorkspaceIndexingResult:
        """
        Index a workspace snapshot and safely remove deleted-file vectors.
        """

        # Validate that all documents belong to this workspace.
        if any(
            document.workspace_root != ingestion.workspace_root
            for document in ingestion.documents
        ):
            raise ValueError(
                "All documents must belong to the ingested workspace."
            )

        # First index all new and modified documents.
        result = await self.index_documents(
            ingestion.documents
        )

        # Never delete vectors if the scan was incomplete.
        if ingestion.truncated:
            return result

        # Get all files currently stored in ChromaDB.
        indexed_paths = self.vector_store.get_workspace_file_paths(
            workspace_root=ingestion.workspace_root,
        )

        # Get all files observed during the latest filesystem scan.
        observed_paths = set(
            ingestion.observed_relative_paths
        )

        empty_paths = set(
            ingestion.empty_relative_paths
        )

        # Remove files that disappeared or became empty.
        missing_paths = indexed_paths - observed_paths

        empty_indexed_paths = indexed_paths & empty_paths

        paths_to_delete = missing_paths | empty_indexed_paths

        deleted_files = 0
        deleted_vectors = 0

        for relative_path in sorted(paths_to_delete):
            deleted = self.vector_store.delete_file_chunks(
                workspace_root=ingestion.workspace_root,
                relative_path=relative_path,
            )

            if deleted:
                deleted_files += 1
                deleted_vectors += deleted

        return WorkspaceIndexingResult(
            files_processed=result.files_processed,
            chunks_processed=result.chunks_processed,
            vectors_upserted=result.vectors_upserted,
            stale_vectors_deleted=(
                result.stale_vectors_deleted + deleted_vectors
            ),
            empty_files=result.empty_files,
            unchanged_files_skipped=result.unchanged_files_skipped,
            deleted_files=deleted_files,
        )


    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        workspace_root: str | None = None,
        relative_path: str | None = None,
        language: str | None = None,
        exclude_relative_path: str | None = None,
    ) -> list[VectorSearchHit]:
        """
        Perform semantic search against indexed workspace chunks.

        Optional filters restrict retrieval to a workspace, file,
        or programming language.

        A selected source file can be excluded to retrieve only
        additional project context.
        """

        if (
            exclude_relative_path is not None
            and workspace_root is None
        ):
            raise ValueError(
                "workspace_root is required when excluding a relative_path."
            )

        query_embedding = await self.embedding_service.embed_query(
            query
        )

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
            workspace_root=workspace_root,
            relative_path=relative_path,
            language=language,
            exclude_relative_path=exclude_relative_path,
        )
