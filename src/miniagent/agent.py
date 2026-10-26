import asyncio
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from miniagent.tools import BaseTool


class CustomAgent:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        tools: list[BaseTool],
        max_steps: int = 10,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    async def run(self, user_input: str) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use tools when needed.",
            },
            {"role": "user", "content": user_input},
        ]
        tool_schemas = [t.to_openai_schema() for t in self.tools.values()]

        for _ in range(self.max_steps):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_schemas,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                return message.content or ""

            results = await asyncio.gather(
                *(self._call_tool(tc) for tc in message.tool_calls)
            )
            messages.extend(results)

        raise RuntimeError(f"Agent did not finish in {self.max_steps} steps")

    async def _call_tool(self, tool_call: Any) -> dict[str, Any]:
        name = tool_call.function.name
        args = tool_call.function.arguments
        print(f"→ tool call: {name}({args})")

        tool = self.tools.get(name)
        if tool is None:
            content = f"Error: unknown tool '{name}'. Available: {list(self.tools)}"
        else:
            try:
                result = await asyncio.to_thread(tool.run, args)
                content = str(result)
            except ValidationError as e:
                content = f"Error: invalid arguments, fix and retry: {e}"
            except Exception as e:
                content = f"Error: {type(e).__name__}: {e}"

        return {"role": "tool", "tool_call_id": tool_call.id, "content": content}
