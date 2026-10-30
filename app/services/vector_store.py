from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb

from app.services.workspace_chunking import WorkspaceChunk


COLLECTION_NAME = "forgepath_code_v1"
DEFAULT_BATCH_SIZE = 128


class VectorStoreError(RuntimeError):
    """Base error for vector-store operations."""


class VectorStoreConfigurationError(VectorStoreError):
    """The vector-store configuration is invalid."""


class VectorStoreOperationError(VectorStoreError):
    """A vector-store operation failed."""


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """One retrieved workspace chunk."""

    id: str
    document: str
    metadata: dict[str, Any]
    distance: float | None


def get_vector_store_path() -> Path:
    """Return ForgePath's persistent local vector-store directory."""

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")

        if not base:
            base = str(Path.home() / "AppData" / "Local")

    else:
        base = os.environ.get("XDG_DATA_HOME")

        if not base:
            base = str(Path.home() / ".local" / "share")

    return Path(base) / "ForgePath" / "chroma"


def create_chunk_id(chunk: WorkspaceChunk) -> str:
    """
    Create a deterministic ID for one workspace chunk.

    The same workspace, file and chunk position always produce
    the same ID. Re-indexing can therefore update the existing record.
    """

    workspace = str(
        Path(chunk.workspace_root).expanduser().resolve()
    )

    identity = (
        f"{workspace}\0"
        f"{chunk.relative_path}\0"
        f"{chunk.chunk_index}"
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


class ChromaVectorStore:
    """
    Persistent embedded Chroma vector store.

    This class accepts externally generated embeddings.
    It never creates embeddings from source text.

    Create one application-level instance and reuse it.
    Do not create multiple clients for the same persistence path.
    """

    def __init__(
        self,
        *,
        dimensions: int,
        embedding_model: str,
        persist_directory: str | Path | None = None,
        collection_name: str = COLLECTION_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:

        if dimensions < 1:
            raise VectorStoreConfigurationError(
                "Embedding dimensions must be greater than zero."
            )

        if not embedding_model.strip():
            raise VectorStoreConfigurationError(
                "An embedding model name is required."
            )

        if not collection_name.strip():
            raise VectorStoreConfigurationError(
                "A collection name is required."
            )

        if batch_size < 1:
            raise VectorStoreConfigurationError(
                "Batch size must be greater than zero."
            )

        self.dimensions = dimensions
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.batch_size = batch_size
        self._closed = False

        self.persist_directory = Path(
            persist_directory
            if persist_directory is not None
            else get_vector_store_path()
        ).expanduser().resolve()

        try:
            self.persist_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                ),
            )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "forgepath_dimensions": self.dimensions,
                    "forgepath_embedding_model": self.embedding_model,
                },
                embedding_function=None,
            )

            self._validate_collection_configuration()

        except Exception as exc:
            # Close the client if initialization fails after it was created.
            client = getattr(self, "_client", None)

            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

            if isinstance(exc, VectorStoreError):
                raise

            raise VectorStoreConfigurationError(
                "Could not initialize the persistent Chroma collection."
            ) from exc

    def _validate_collection_configuration(self) -> None:
        """Prevent accidentally mixing incompatible embedding models."""

        metadata = self._collection.metadata or {}

        stored_dimensions = metadata.get(
            "forgepath_dimensions"
        )

        stored_model = metadata.get(
            "forgepath_embedding_model"
        )

        if stored_dimensions != self.dimensions:
            raise VectorStoreConfigurationError(
                "The existing collection uses different embedding "
                "dimensions. Create a new collection before switching "
                "embedding configurations."
            )

        if stored_model != self.embedding_model:
            raise VectorStoreConfigurationError(
                "The existing collection uses a different embedding "
                "model. Re-index the workspace into a compatible "
                "collection before switching models."
            )

    def _validate_vector(
        self,
        embedding: Sequence[float],
    ) -> list[float]:
        """Validate embedding dimensions and numeric values."""

        if len(embedding) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} dimensions, "
                f"received {len(embedding)}."
            )

        vector: list[float] = []

        for value in embedding:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise ValueError(
                    "Embedding vectors must contain numeric values."
                )

            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise ValueError(
                    "Embedding vectors cannot contain NaN or infinity."
                )

            vector.append(numeric_value)

        return vector

    def upsert_chunks(
        self,
        chunks: Sequence[WorkspaceChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """
        Insert or update workspace chunks and their embeddings.

        The chunk and embedding sequences must have identical lengths.
        Existing deterministic IDs are updated rather than duplicated.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        vectors: list[list[float]] = []

        # Validate the complete input before writing any records.
        for chunk, embedding in zip(chunks, embeddings):

            if not chunk.content.strip():
                raise ValueError(
                    "Empty chunks cannot be stored."
                )

            if chunk.chunk_index < 0:
                raise ValueError(
                    "Chunk index cannot be negative."
                )

            if (
                chunk.start_line < 1
                or chunk.end_line < chunk.start_line
            ):
                raise ValueError(
                    "Invalid chunk line range."
                )

            ids.append(create_chunk_id(chunk))

            documents.append(chunk.content)

            vectors.append(
                self._validate_vector(embedding)
            )

            metadatas.append(
                {
                    "workspace_root": chunk.workspace_root,
                    "relative_path": chunk.relative_path,
                    "language": chunk.language,
                    "chunk_index": chunk.chunk_index,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "document_sha256": chunk.document_sha256,
                    "chunk_sha256": chunk.chunk_sha256,
                    "embedding_model": self.embedding_model,
                }
            )

        try:
            for start in range(
                0,
                len(ids),
                self.batch_size,
            ):
                end = start + self.batch_size

                self._collection.upsert(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    embeddings=vectors[start:end],
                    metadatas=metadatas[start:end],
                )

        except Exception as exc:
            raise VectorStoreOperationError(
                "Could not insert workspace vectors into Chroma."
            ) from exc

        return len(ids)

    def get_file_chunk_ids(
        self,
        *,
        workspace_root: str,
        relative_path: str,
    ) -> list[str]:
        """Retrieve all vector IDs belonging to one source file."""

        where = self._file_filter(
            workspace_root,
            relative_path,
        )

        try:
            result = self._collection.get(
                where=where,
                include=[],
            )

            return result["ids"]

        except Exception as exc:
            raise VectorStoreOperationError(
                "Could not retrieve file vector IDs."
            ) from exc

    def close(self) -> None:
        """Close the Chroma client and release its resources."""

        if self._closed:
            return

        self._client.close()

        # Mark the store as closed, without overwriting the client.
        self._closed = True

    def __enter__(self) -> ChromaVectorStore:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    @staticmethod
    def _file_filter(
        workspace_root: str,
        relative_path: str,
    ) -> dict[str, Any]:
        """Build an AND filter for one file inside a workspace."""

        return {
            "$and": [
                {"workspace_root": workspace_root},
                {"relative_path": relative_path},
            ]
        }

    @staticmethod
    def _combine_filters(
        filters: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Combine multiple metadata conditions."""

        if not filters:
            return None

        if len(filters) == 1:
            return filters[0]

        return {"$and": filters}

    def delete_ids(
        self,
        ids: Sequence[str],
    ) -> int:
        """Delete explicitly identified vector records."""

        unique_ids = list(dict.fromkeys(ids))

        if not unique_ids:
            return 0

        try:
            existing = self._collection.get(
                ids=unique_ids,
                include=[],
            )

            existing_ids = existing["ids"]

            if existing_ids:
                self._collection.delete(
                    ids=existing_ids,
                )

            return len(existing_ids)

        except Exception as exc:
            raise VectorStoreOperationError(
                "Could not delete vector records."
            ) from exc

    def delete_file_chunks(
        self,
        *,
        workspace_root: str,
        relative_path: str,
    ) -> int:
        """Delete all indexed chunks for one workspace file."""

        ids = self.get_file_chunk_ids(
            workspace_root=workspace_root,
            relative_path=relative_path,
        )

        return self.delete_ids(ids)

    def delete_workspace(
        self,
        workspace_root: str,
    ) -> int:
        """Delete every indexed chunk belonging to one workspace."""

        try:
            existing = self._collection.get(
                where={"workspace_root": workspace_root},
                include=[],
            )

            ids = existing["ids"]

            if ids:
                self._collection.delete(ids=ids)

            return len(ids)

        except Exception as exc:
            raise VectorStoreOperationError(
                "Could not delete workspace vectors."
            ) from exc

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        workspace_root: str | None = None,
        relative_path: str | None = None,
        language: str | None = None,
    ) -> list[VectorSearchHit]:
        """
        Search by an externally generated query embedding.

        Optional metadata filters restrict the search to a workspace,
        file, or programming language.
        """

        if top_k < 1:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if (
            relative_path is not None
            and workspace_root is None
        ):
            raise ValueError(
                "workspace_root is required when filtering by relative_path."
            )

        vector = self._validate_vector(query_embedding)

        filters: list[dict[str, Any]] = []

        if workspace_root is not None:
            filters.append(
                {"workspace_root": workspace_root}
            )

        if relative_path is not None:
            filters.append(
                {"relative_path": relative_path}
            )

        if language is not None:
            filters.append(
                {"language": language}
            )

        where = self._combine_filters(filters)

        try:
            if self._collection.count() == 0:
                return []

            result = self._collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            ids = result["ids"][0]
            documents = result["documents"][0]
            metadatas = result["metadatas"][0]
            distances = result["distances"][0]

            return [
                VectorSearchHit(
                    id=record_id,
                    document=document or "",
                    metadata=metadata or {},
                    distance=(
                        float(distance)
                        if distance is not None
                        else None
                    ),
                )
                for record_id, document, metadata, distance in zip(
                    ids,
                    documents,
                    metadatas,
                    distances,
                )
            ]

        except Exception as exc:
            raise VectorStoreOperationError(
                "Vector similarity search failed."
            ) from exc

    def count(self) -> int:
        """Return the total number of indexed vector records."""

        try:
            return self._collection.count()

        except Exception as exc:
            raise VectorStoreOperationError(
                "Could not count stored vectors."
            ) from exc