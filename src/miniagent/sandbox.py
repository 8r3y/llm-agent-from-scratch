import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from miniagent.tools import BaseTool, tool

IMAGE = "miniagent-sandbox"
MAX_OUTPUT_CHARS = 10_000


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_llm_text(self) -> str:
        if self.timed_out:
            return "Error: execution timed out. Optimize the code or process less data."
        text = f"exit code: {self.exit_code}\n--- stdout ---\n{self.stdout}"
        if self.stderr:
            text += f"\n--- stderr ---\n{self.stderr}"
        return text[:MAX_OUTPUT_CHARS]


def run_python(
    code: str, data_dir: Path | None = None, timeout_s: float = 30
) -> SandboxResult:
    name = f"sandbox-{uuid.uuid4().hex[:12]}"
    # name = "sandbox-1"
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        name,
        "--network",
        "none",
        "--memory",
        "512m",
        "--memory-swap",
        "512m",
        "--cpus",
        "1",
        "--pids-limit",
        "64",
        "--read-only",
        "--tmpfs",
        "/tmp:size=64m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
    ]
    if data_dir is not None:
        cmd += ["-v", f"{data_dir.resolve()}:/data:ro"]
    cmd += [IMAGE, "python", "-"]

    try:
        proc = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return SandboxResult(exit_code=None, stdout="", stderr="", timed_out=True)

    return SandboxResult(proc.returncode, proc.stdout, proc.stderr)


def make_execute_python(data_dir: Path) -> BaseTool:
    @tool
    def execute_python(code: str) -> str:
        """Execute Python 3.11 code in an isolated sandbox and return stdout/stderr.

        Input files are in /data (read-only). pandas, numpy and scikit-learn are installed.
        No network access. Each call is a fresh process: variables do NOT persist
        between calls, so re-read the data every time. Always print() what you need to see.
        """
        return run_python(code, data_dir=data_dir).to_llm_text()

    return execute_python
