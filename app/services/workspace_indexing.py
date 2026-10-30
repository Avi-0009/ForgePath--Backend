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
        Index multiple ingested documents.

        Existing vectors for each file are preserved until embedding
        generation succeeds and replacement vectors are upserted.

        Obsolete chunks from previously indexed versions of supplied
        files are deleted after successful upsert.

        Files absent from this input are not automatically deleted.
        """

        if not documents:
            return WorkspaceIndexingResult(
                files_processed=0,
                chunks_processed=0,
                vectors_upserted=0,
                stale_vectors_deleted=0,
                empty_files=0,
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

        # Prepare chunks and collect the previously indexed IDs.
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

        # Generate all embeddings before modifying stored vectors.
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
        )

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        workspace_root: str | None = None,
        relative_path: str | None = None,
        language: str | None = None,
    ) -> list[VectorSearchHit]:
        """
        Perform semantic search against indexed workspace chunks.

        The query is embedded using the same EmbeddingService
        configuration used when indexing documents.
        """

        query_embedding = await self.embedding_service.embed_query(
            query
        )

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
            workspace_root=workspace_root,
            relative_path=relative_path,
            language=language,
        )