from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.health import router as health_router
from app.api.v1.router import router as api_v1_router
from app.core.config import settings
from app.db.session import get_db


app: FastAPI = FastAPI(
    title="ForgePath API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
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