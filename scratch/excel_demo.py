import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from miniagent.agent import CustomAgent
from miniagent.sandbox import make_execute_python


async def main() -> None:
    load_dotenv()
    client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GEMINI_API_KEY"],
    )
    agent = CustomAgent(
        client,
        model="gemini-3.5-flash-lite",
        tools=[make_execute_python(Path("data"))],
        max_steps=8,
    )
    answer = await agent.run(
        "В файле /data/churn.xlsx данные клиентов, колонка churned — ушёл ли клиент. "
        "Сначала изучи данные, затем обучи логистическую регрессию, оцени качество "
        "(ROC AUC на отложенной выборке) и назови признаки, сильнее всего влияющие на отток."
    )
    print("\n=== ОТВЕТ ===\n" + answer)


asyncio.run(main())