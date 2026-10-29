from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.schemas.workspace import WorkspaceNode
from app.services.workspace import (
    WorkspaceConfigurationError,
    WorkspaceReadError,
    _validate_relative_path,
)


MAX_TREE_DEPTH = 8
MAX_TREE_NODES = 2_000


@dataclass
class _TreeBudget:
    """Limits the amount of filesystem data returned in one response."""

    nodes: int = 1


def _is_link_or_junction(path: Path) -> bool:
    """Avoid traversing symbolic links and Windows directory junctions."""
    try:
        if path.is_symlink():
            return True

        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction and is_junction())

    except OSError:
        return True


def _resolve_configured_roots() -> list[Path]:
    """Resolve only the exact roots approved in backend settings."""
    configured_roots = settings.workspace_root_list

    if not configured_roots:
        raise WorkspaceConfigurationError(
            "No workspace roots are configured. "
            "Set WORKSPACE_ALLOWED_ROOTS in the backend environment."
        )

    roots: list[Path] = []
    seen: set[Path] = set()

    for configured_root in configured_roots:
        try:
            root = Path(configured_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise WorkspaceConfigurationError(
                "A configured workspace root is invalid or unavailable."
            ) from None

        if not root.is_dir():
            raise WorkspaceConfigurationError(
                "A configured workspace root is not a directory."
            )

        if root not in seen:
            roots.append(root)
            seen.add(root)

    return roots


def _build_node(
    *,
    path: Path,
    workspace_root: Path,
    relative_path: str,
    depth: int,
    budget: _TreeBudget,
) -> WorkspaceNode:
    """Build a bounded tree node without following links."""
    is_dir = path.is_dir()

    budget.nodes += 1

    node = WorkspaceNode(
        name=path.name or str(path),
        path=str(path),
        is_dir=is_dir,
        children=[],
        workspace_root=str(workspace_root),
        relative_path=relative_path,
    )

    if not is_dir:
        return node

    if depth >= MAX_TREE_DEPTH:
        node.truncated = True
        return node

    try:
        entries = list(path.iterdir())
    except OSError:
        node.truncated = True
        return node

    entries.sort(
        key=lambda item: (
            not item.is_dir() if not _is_link_or_junction(item) else True,
            item.name.casefold(),
        )
    )

    for entry in entries:
        if _is_link_or_junction(entry):
            continue

        try:
            child_relative_path = (
                entry.relative_to(workspace_root).as_posix()
            )

            _validate_relative_path(child_relative_path)

            if entry.is_dir():
                child_is_dir = True
            elif entry.is_file():
                child_is_dir = False
            else:
                continue

        except WorkspaceReadError:
            continue
        except (OSError, RuntimeError, ValueError):
            continue

        if budget.nodes >= MAX_TREE_NODES:
            node.truncated = True
            break

        child_node = _build_node(
            path=entry,
            workspace_root=workspace_root,
            relative_path=child_relative_path,
            depth=depth + 1,
            budget=budget,
        )

        child_node.is_dir = child_is_dir
        node.children.append(child_node)

    return node


def get_workspace_tree() -> WorkspaceNode:
    """
    Return a bounded tree of explicitly configured workspaces.

    Paths are returned as metadata for workspace navigation. Files are
    never opened or modified by this service.
    """
    roots = _resolve_configured_roots()
    budget = _TreeBudget()

    virtual_root = WorkspaceNode(
        name="workspaces",
        path="workspaces",
        is_dir=True,
        children=[],
        workspace_root=None,
        relative_path="",
    )

    for root in roots:
        if budget.nodes >= MAX_TREE_NODES:
            virtual_root.truncated = True
            break

        relative_to_root = ""

        workspace_node = _build_node(
            path=root,
            workspace_root=root,
            relative_path=relative_to_root,
            depth=0,
            budget=budget,
        )

        virtual_root.children.append(workspace_node)

    return virtual_root