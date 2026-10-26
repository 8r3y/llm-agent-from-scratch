import pytest
from pydantic import ValidationError

from miniagent.builtin_tools import ReadFileTool


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A", encoding="utf-8")
    (tmp_path.parent / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    return tmp_path


@pytest.fixture
def read_file(workspace):
    return ReadFileTool(workspace)


def test_reads_files_inside_workspace(read_file):
    assert read_file.run({"path": "notes.txt"}) == "hello"
    assert read_file.run({"path": "docs/a.md"}) == "# A"


@pytest.mark.parametrize(
    "path",
    ["../secret.txt", "docs/../../secret.txt", "/etc/passwd"],
    ids=["parent-dir", "nested-escape", "absolute-path"],
)
def test_path_traversal_is_blocked(read_file, path):
    with pytest.raises(PermissionError):
        read_file.run({"path": path})


def test_missing_file(read_file):
    with pytest.raises(FileNotFoundError):
        read_file.run({"path": "nope.txt"})


def test_max_chars_limit_is_validated(read_file):
    with pytest.raises(ValidationError):
        read_file.run({"path": "notes.txt", "max_chars": 10_000_000})
