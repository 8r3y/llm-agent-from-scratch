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
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
    )
    agent = CustomAgent(
        client,
        model=os.environ["LLM_MODEL"],
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
