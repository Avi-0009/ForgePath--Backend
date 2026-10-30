import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.health import router as health_router
from app.api.v1.router import router as api_v1_router
from app.core.config import settings
from app.db.session import get_db

from app.services.embedding_service import EmbeddingService
from app.services.openai_embedding_provider import OpenAIEmbeddingProvider
from app.services.vector_store import ChromaVectorStore
from app.services.workspace_indexing import (
    WorkspaceIndexingService,
    get_embedding_model_identifier,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize and clean up workspace indexing resources.

    When no provider is configured, the API starts normally,
    but semantic indexing and search remain unavailable.
    """

    embedding_provider = None
    vector_store = None

    app.state.workspace_indexing_service = None

    try:
        if settings.embedding_provider is None:
            logger.warning(
                "Workspace semantic indexing is disabled. "
                "Configure EMBEDDING_PROVIDER to enable it."
            )

        elif settings.embedding_provider == "openai":

            if (
                settings.openai_api_key is None
                or not settings.openai_api_key.get_secret_value().strip()
            ):
                raise RuntimeError(
                    "OPENAI_API_KEY is required when "
                    "EMBEDDING_PROVIDER=openai."
                )

            # Initialize the real embedding provider.
            embedding_provider = OpenAIEmbeddingProvider(
                api_key=settings.openai_api_key.get_secret_value(),
                model_name=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            )

            # Wrap the provider with common validation and batching.
            embedding_service = EmbeddingService(
                provider=embedding_provider,
                batch_size=settings.embedding_batch_size,
            )

            # Initialize persistent local ChromaDB storage.
            vector_store = ChromaVectorStore(
                dimensions=embedding_service.dimensions,
                embedding_model=get_embedding_model_identifier(
                    embedding_service
                ),
            )

            # Expose one shared indexing service to FastAPI routes.
            app.state.workspace_indexing_service = (
                WorkspaceIndexingService(
                    embedding_service=embedding_service,
                    vector_store=vector_store,
                )
            )

            logger.info(
                "Workspace semantic indexing initialized. "
                "Provider=%s, model=%s, dimensions=%s",
                embedding_service.provider_name,
                embedding_service.model_name,
                embedding_service.dimensions,
            )

        yield

    finally:
        # Prevent new requests from obtaining a shutting-down service.
        app.state.workspace_indexing_service = None

        try:
            if vector_store is not None:
                vector_store.close()

        finally:
            if embedding_provider is not None:
                await embedding_provider.close()

        logger.info("Workspace indexing resources released.")


app: FastAPI = FastAPI(
    title="ForgePath API",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
    ],
)


app.include_router(health_router)
app.include_router(api_v1_router)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    return {
        "message": "ForgePath API is running",
        "docs": "/docs",
        "health": "/health",
        "db_health": "/health/db",
    }


@app.get("/health/db", tags=["Health"])
async def database_health_check(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, str]:
    await db.execute(text("SELECT 1"))

    return {
        "status": "ok",
        "database": "connected",
    }