from pydantic import BaseModel, ConfigDict, Field


class WorkspaceNode(BaseModel) :
    """A file or directory inside an allowlisted workspace."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    path: str = Field(min_length=1, max_length=2048)
    is_dir: bool

    children: list["WorkspaceNode"] = Field(default_factory=list)

    workspace_root: str | None = None
    relative_path: str = ""

    truncated: bool = False