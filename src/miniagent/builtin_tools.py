from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from miniagent.tools import BaseTool


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace. Paths are relative to the workspace root."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        path: str = Field(
            description="File path relative to the workspace, e.g. 'docs/a.md'"
        )
        max_chars: int = Field(default=10_000, ge=1, le=50_000)

    args_schema = Args

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def execute(self, path: str, max_chars: int = 10_000) -> str:
        target = (self.workspace / path).resolve()
        if not target.is_relative_to(self.workspace):
            raise PermissionError(f"Access outside the workspace is denied: {path}")
        if not target.is_file():
            raise FileNotFoundError(f"No such file in the workspace: {path}")
        return target.read_text(encoding="utf-8", errors="replace")[:max_chars]
