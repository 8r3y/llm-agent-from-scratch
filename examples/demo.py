import asyncio
import os
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI

from miniagent.agent import CustomAgent
from miniagent.tools import tool


@tool
def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Get current weather for a city."""
    return f"{city}: +20{unit}"


async def main() -> None:
    load_dotenv()
    client = AsyncOpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
    )
    agent = CustomAgent(client, model=os.environ["LLM_MODEL"], tools=[get_weather])
    print(await agent.run("Какая погода в Москве и в Париже, в Цельс?"))


asyncio.run(main())
