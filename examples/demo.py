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
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GEMINI_API_KEY"],
    )
    agent = CustomAgent(client, model="gemini-3.5-flash-lite", tools=[get_weather])
    print(await agent.run("Какая погода в Москве и в Париже, в Цельс?"))


asyncio.run(main())
