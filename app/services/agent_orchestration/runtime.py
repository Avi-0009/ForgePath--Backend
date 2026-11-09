import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.engine import make_url

from app.core.config import settings
from app.services.agent_orchestration.graph import build_agent_graph


logger = logging.getLogger(__name__)


class AgentRuntimeError(RuntimeError):
    """Raised when the persistent agent runtime cannot initialize."""


@dataclass(slots=True)
class AgentRuntime:
    """Resources shared by all agent workflows in one application."""

    pool: AsyncConnectionPool
    checkpointer: AsyncPostgresSaver
    graph: Any

    async def close(self) -> None:
        """Close the PostgreSQL connection pool."""

        await self.pool.close()


class AgentRuntimeManager:
    """
    Lazily initialize and reuse the persistent LangGraph runtime.

    Only one runtime is created per application instance.
    Concurrent requests share the same graph and connection pool.
    """

    def __init__(self) -> None:
        self._runtime: AgentRuntime | None = None
        self._lock = asyncio.Lock()

    async def get_runtime(self) -> AgentRuntime:
        """Return the initialized runtime, creating it when necessary."""

        if self._runtime is not None:
            return self._runtime

        async with self._lock:

            if self._runtime is not None:
                return self._runtime

            self._runtime = await self._create_runtime()

            return self._runtime

    async def _create_runtime(self) -> AgentRuntime:
        """Create the PostgreSQL pool, checkpointer, and compiled graph."""

        pool: AsyncConnectionPool | None = None

        try:
            # Convert SQLAlchemy's asyncpg URL into a Psycopg URL.
            database_url = make_url(
                settings.database_url
            ).set(
                drivername="postgresql",
            ).render_as_string(
                hide_password=False,
            )

            pool = AsyncConnectionPool(
                conninfo=database_url,
                min_size=1,
                max_size=5,
                timeout=15,
                open=False,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                },
            )

            # Establish connections before setting up the checkpointer.
            await pool.open(wait=True)

            checkpointer = AsyncPostgresSaver(
                conn=pool,
            )

            # Create or migrate LangGraph checkpoint tables.
            await checkpointer.setup()

            # Compile once and reuse the graph.
            graph = build_agent_graph(
                checkpointer=checkpointer,
            )

            logger.info(
                "ForgePath LangGraph runtime initialized successfully."
            )

            return AgentRuntime(
                pool=pool,
                checkpointer=checkpointer,
                graph=graph,
            )

        except Exception as exc:
            if pool is not None:
                try:
                    await pool.close()
                except Exception:
                    logger.exception(
                        "Could not close PostgreSQL pool after "
                        "agent runtime initialization failed."
                    )

            logger.exception(
                "Could not initialize the ForgePath agent runtime."
            )

            raise AgentRuntimeError(
                "The persistent agent runtime could not be initialized."
            ) from exc

    async def close(self) -> None:
        """Release runtime resources during application shutdown."""

        async with self._lock:

            if self._runtime is None:
                return

            await self._runtime.close()
            self._runtime = None

            logger.info(
                "ForgePath LangGraph runtime resources released."
            )