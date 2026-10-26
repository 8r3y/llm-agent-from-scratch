import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

print("Старт запроса")
start = time.time()

chat = client.chat.completions.create(
    messages=[{"role": "user", "content": "Ты кто?"}], model=LLM_MODEL
)

print(
    "Результат",
    f"Время выполнения: {time.time() - start}",
    chat.choices[0].message.content,
    sep="\n--\n",
)
