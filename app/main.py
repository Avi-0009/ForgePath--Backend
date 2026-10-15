from typing import Annotated

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.health import router as health_router
from app.db.session import get_db


app: FastAPI = FastAPI(
    title="ForgePath API",
    version="0.1.0",
)

app.include_router(health_router)


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