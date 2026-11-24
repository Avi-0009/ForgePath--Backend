"""Provider-free access to an existing vector store for project deletion."""
from contextlib import contextmanager
from types import SimpleNamespace

import chromadb
from chromadb.errors import NotFoundError

from app.services.vector_store import COLLECTION_NAME, VectorStoreError, get_vector_store_path


@contextmanager
def deletion_index_service(active_service):
    if active_service is not None:
        yield active_service
        return

    directory = get_vector_store_path()
    # Never follow a local storage link or silently create a replacement DB.
    for part in (directory, *directory.parents):
        if part.is_symlink() or (getattr(part, "is_junction", None) and part.is_junction()):
            raise VectorStoreError("Unsafe vector storage path")
    database = directory / "chroma.sqlite3"
    if database.is_symlink():
        raise VectorStoreError("Unsafe vector database path")
    client = None
    try:
        collection = None
        if database.is_file():
            client = chromadb.PersistentClient(
                path=str(directory), settings=chromadb.config.Settings(anonymized_telemetry=False),
            )
            try:
                collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
            except NotFoundError:
                pass
        elif directory.exists() and any(directory.iterdir()):
            # Artifacts without their DB may indicate missing/corrupt storage.
            raise VectorStoreError("Vector database is missing but storage artifacts remain")

        def delete_workspace(namespace):
            if collection is None:
                return 0
            ids = collection.get(where={"workspace_root": namespace}, include=[])["ids"]
            if ids:
                collection.delete(ids=ids)
            return len(ids)

        yield SimpleNamespace(vector_store=SimpleNamespace(
            persist_directory=directory, delete_workspace=delete_workspace,
        ))
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError("Project index cleanup failed") from exc
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                raise VectorStoreError("Could not release vector cleanup resources") from exc
