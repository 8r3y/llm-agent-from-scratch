import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = OpenAI(base_url=BASE_URL, api_key=GEMINI_API_KEY)

chat = client.chat.completions.create(
    messages=[{"role": "user", "content": "Ты кто?"}], model="gemini-3.5-flash-lite"
)

print("Проверка", chat.choices[0].message.content, sep="\n--\n")
