# llm-agent-from-scratch

**English** · [Русский](README.ru.md)

Python package: `miniagent`.

A framework-agnostic LLM agent core in plain Python — no LangChain, no LlamaIndex.
Built from scratch to show the moving parts: tool schemas, the ReAct loop, a code sandbox and guardrails.

Give the agent an Excel file and a question, and it will explore the data, train a logistic
regression and report the key features. All the code it generates runs in an isolated Docker container.

## Features

- **`@tool` decorator.** Builds a JSON Schema from a plain function: signature, type hints, docstring
  and `Annotated[..., Field(description=...)]`, via Pydantic `create_model`.
- **Class-based tools.** `BaseTool` subclasses with an explicit `args_schema`, for tools that hold
  state (DB connections, workspace roots). Both styles share one interface.
- **Async ReAct loop.** Parallel tool calls with `asyncio.gather`, sync tools offloaded with
  `asyncio.to_thread`, a `max_steps` guard.
- **Self-correction.** Validation errors and tool exceptions go back to the model as `tool` messages
  instead of crashing the agent. Unknown arguments are rejected (`extra="forbid"`).
- **Fail fast.** Functions without type hints or with `*args` / `**kwargs` / positional-only
  parameters are rejected when they are decorated, not during a conversation.
- **Docker sandbox.** `execute_python` runs model-generated code with no network, a read-only file
  system, memory / CPU / process limits, a non-root user, all capabilities dropped and a hard timeout.
- **Guardrails.** `read_file` blocks path traversal and caps output size.
- **Provider-agnostic.** Any OpenAI-compatible API: OpenAI, Gemini, vLLM, Ollama.

## Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker.

```bash
uv sync
cp .env.sample .env                       # put your GEMINI_API_KEY there
docker build -t miniagent-sandbox sandbox/   # sandbox image with pandas and scikit-learn

uv run python examples/demo.py            # parallel tool calls

# Excel case: generate a synthetic churn dataset, then let the agent analyse it
uv run --with pandas --with openpyxl python examples/make_excel.py
uv run python examples/excel_demo.py
```

The examples use Gemini through its OpenAI-compatible endpoint. To use another provider, change
`base_url` and `model` in the example.

> **Docker Desktop on Windows:** add the project folder and `%LOCALAPPDATA%\Temp` to
> *Settings → Resources → File sharing*, otherwise mounting `/data` asks for permission on every run.

## Usage

```python
import asyncio
import os
from typing import Annotated, Literal

from openai import AsyncOpenAI
from pydantic import Field

from miniagent.agent import CustomAgent
from miniagent.tools import tool


@tool
def get_weather(
    city: Annotated[str, Field(description="City name, e.g. 'Paris'")],
    unit: Literal["c", "f"] = "c",
) -> str:
    """Get current weather for a city."""
    return f"{city}: +20{unit}"


async def main() -> None:
    client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GEMINI_API_KEY"],
    )
    agent = CustomAgent(client, model="gemini-3.5-flash-lite", tools=[get_weather])
    print(await agent.run("What's the weather in Paris and Moscow?"))


asyncio.run(main())
```

A tool with state is a class:

```python
from pathlib import Path
from miniagent.builtin_tools import ReadFileTool

read_file = ReadFileTool(workspace=Path("docs"))   # the model never sees or changes the root
```

## How it works

```
messages = [system, user]
repeat up to max_steps:
    reply = LLM(messages, tool_schemas)        → appended as "assistant"
    no tool_calls in reply?  → return the final answer
    for each tool call, concurrently:
        validate arguments (Pydantic) → run the tool → append a "tool" message with tool_call_id
        on any error → the error text becomes the tool result, so the model can fix the call
```

Model-generated code never runs inside the agent process. `execute_python` is a thin tool adapter
over `run_python`, which starts a fresh container for every call. The data directory is fixed in a
closure, so it is not part of the tool's schema and the model cannot point it elsewhere.

| Threat | Mitigation |
| --- | --- |
| Data exfiltration, SSRF | `--network none` |
| Memory / CPU exhaustion, fork bombs | `--memory`, `--cpus`, `--pids-limit` |
| Tampering with the system | `--read-only` (writable `tmpfs` only), non-root user, `--cap-drop ALL`, `no-new-privileges` |
| Changing the input data | data mounted `:ro` |
| Infinite loops | timeout, then `docker kill` by container name |
| Flooding the model's context | output truncated to 10,000 characters |

## Testing

The project was built test-first.

```bash
uv run pytest -m "not integration"   # 24 unit tests: no Docker, no LLM, under a second
uv run pytest -m integration         # 8 sandbox tests against real Docker
```

Covered: schema generation and descriptions, `required` and defaults, invalid arguments (missing,
wrong type, out-of-enum, broken JSON, unknown field), rejected signatures, class-based tools, path
traversal attempts, and sandbox isolation (stdout/stderr capture, timeout, no network, read-only
file system, non-root user, read-only data mount).

## Project structure

```
src/miniagent/
  tools.py           BaseTool, FunctionTool, @tool
  agent.py           async ReAct loop
  sandbox.py         run_python and the execute_python tool
  builtin_tools.py   ReadFileTool with path guardrails
sandbox/Dockerfile   sandbox image
examples/            runnable demos
tests/               unit and integration tests
```

## Limitations

This is a learning prototype, not a production framework.

- Docker shares the host kernel. For untrusted users, use gVisor or a microVM (Firecracker, Kata).
- The sandbox is stateless: every call is a fresh process, with no persistent Jupyter kernel.
- Tools are synchronous; async tools and streaming are not supported yet.
- No conversation memory, retries with backoff, or tracing.
- The agent loop itself is not yet covered by tests with a mocked LLM.

## License

[MIT](LICENSE)
