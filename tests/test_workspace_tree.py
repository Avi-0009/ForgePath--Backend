
import pytest

from app.core.config import settings
from app.services.workspace import WorkspaceConfigurationError
from app.services import workspace_tree


@pytest.fixture
def configured_workspace(tmp_path, monkeypatch):
    root = tmp_path / "forgepath"
    root.mkdir()

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(root),
    )

    return root


def find_node(node, name):
    """Find a node by name in a returned workspace tree."""
    if node.name == name:
        return node

    for child in node.children:
        result = find_node(child, name)
        if result is not None:
            return result

    return None


def test_lists_configured_workspace_and_files(configured_workspace):
    root = configured_workspace
    (root / "main.py").write_text("print('hello')", encoding="utf-8")
    (root / "README.md").write_text("# ForgePath", encoding="utf-8")

    tree = workspace_tree.get_workspace_tree()

    assert tree.name == "workspaces"
    assert tree.is_dir is True
    assert len(tree.children) == 1

    workspace = tree.children[0]
    assert workspace.name == "forgepath"
    assert workspace.path == str(root)
    assert workspace.workspace_root == str(root)

    main_file = find_node(workspace, "main.py")
    assert main_file is not None
    assert main_file.is_dir is False
    assert main_file.relative_path == "main.py"


def test_excludes_sensitive_files_and_directories(
    configured_workspace,
):
    root = configured_workspace

    (root / "main.py").write_text("print('ok')", encoding="utf-8")
    (root / ".env").write_text("SECRET=value", encoding="utf-8")
    (root / ".env.local").write_text("SECRET=value", encoding="utf-8")
    (root / "credentials.json").write_text("{}", encoding="utf-8")
    (root / "secrets.json").write_text("{}", encoding="utf-8")
    (root / "private.pem").write_text("private", encoding="utf-8")
    (root / "pgadmin-servers.json").write_text('{"Servers": {}}',encoding="utf-8",)

    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config", encoding="utf-8")

    node_modules = root / "node_modules"
    node_modules.mkdir()
    (node_modules / "package.json").write_text("{}", encoding="utf-8")

    tree = workspace_tree.get_workspace_tree()
    workspace = tree.children[0]

    visible_names = {
        node.name for node in workspace.children
    }

    assert "main.py" in visible_names
    assert ".env" not in visible_names
    assert ".env.local" not in visible_names
    assert "credentials.json" not in visible_names
    assert "secrets.json" not in visible_names
    assert "private.pem" not in visible_names
    assert ".git" not in visible_names
    assert "node_modules" not in visible_names
    assert "pgadmin-servers.json" not in visible_names


def test_nested_files_have_relative_paths(configured_workspace):
    root = configured_workspace
    source_dir = root / "src"
    source_dir.mkdir()
    (source_dir / "main.py").write_text(
        "def main():\n    pass\n",
        encoding="utf-8",
    )

    tree = workspace_tree.get_workspace_tree()
    workspace = tree.children[0]

    source_node = find_node(workspace, "src")
    file_node = find_node(workspace, "main.py")

    assert source_node is not None
    assert source_node.is_dir is True
    assert source_node.relative_path == "src"

    assert file_node is not None
    assert file_node.relative_path == "src/main.py"
    assert file_node.workspace_root == str(root)


def test_missing_configured_root_is_rejected(tmp_path, monkeypatch):
    missing_root = tmp_path / "does-not-exist"

    monkeypatch.setattr(
        settings,
        "workspace_allowed_roots",
        str(missing_root),
    )

    with pytest.raises(WorkspaceConfigurationError):
        workspace_tree.get_workspace_tree()


def test_depth_limit_marks_directory_as_truncated(
    configured_workspace,
    monkeypatch,
):
    root = configured_workspace

    child_dir = root / "src"
    child_dir.mkdir()

    deeper_dir = child_dir / "nested"
    deeper_dir.mkdir()
    (deeper_dir / "main.py").write_text(
        "print('hello')",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        workspace_tree,
        "MAX_TREE_DEPTH",
        1,
    )

    tree = workspace_tree.get_workspace_tree()
    workspace = tree.children[0]
    source_dir = find_node(workspace, "src")

    assert source_dir is not None
    assert source_dir.truncated is True
    assert source_dir.children == []


def test_node_limit_marks_incomplete_tree(
    configured_workspace,
    monkeypatch,
):
    root = configured_workspace

    folder = root / "src"
    folder.mkdir()
    (folder / "main.py").write_text("pass", encoding="utf-8")

    (root / "one.py").write_text("pass", encoding="utf-8")
    (root / "two.py").write_text("pass", encoding="utf-8")

    monkeypatch.setattr(
        workspace_tree,
        "MAX_TREE_NODES",
        3,
    )

    tree = workspace_tree.get_workspace_tree()
    workspace = tree.children[0]

    assert workspace.truncated is True
