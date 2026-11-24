"""Request-local identity for legacy filesystem APIs (also copied by to_thread)."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True)
class WorkspaceAccess:
    user_id: UUID
    roots: tuple[Path, ...]


workspace_access: ContextVar[WorkspaceAccess | None] = ContextVar(
    "workspace_access", default=None,
)


@contextmanager
def owned_workspaces(user_id: UUID, roots: tuple[Path, ...]):
    token = workspace_access.set(WorkspaceAccess(user_id, roots))
    try:
        yield
    finally:
        workspace_access.reset(token)
