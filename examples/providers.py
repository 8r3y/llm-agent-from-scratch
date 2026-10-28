import os

from dotenv import load_dotenv

# одни и те же 10 моделей: имя у plusvibe -> id у Selectel (стиль OpenRouter: vendor/model)
SELECTEL_IDS = {
    "kimi-k3.1-pro": "moonshotai/kimi-k3.1-pro",
    "gemini-4-argon": "google/gemini-4-argon",
    "gpt-6.1-sol": "openai/gpt-6.1-sol",
    "grok-4.7": "x-ai/grok-4.7",
    "mimo-v2.6-pro": "xiaomi/mimo-v2.6-pro",
    # у Selectel нет qwen3.8-max без даты, только снапшот 0902
    "qwen3.8-max": "qwen/qwen3.8-max-0902",
    "deepseek-v4.1-flash-1017": "deepseek/deepseek-v4.1-flash-1017",
    "glm-5.3": "z-ai/glm-5.3",
    "claude-haiku-5.5": "anthropic/claude-haiku-5.5",
    "minimax-m3.1-flash": "minimax/minimax-m3.1-flash",
}


def get_provider() -> tuple[str, str, str, list[str]]:
    """Провайдер из PROVIDER=plusvibe|selectel. Вернёт (имя, base_url, api_key, модели)."""
    load_dotenv()
    name = os.environ.get("PROVIDER", "plusvibe")
    if name == "selectel":
        return name, os.environ["SELECTEL_BASE_URL"], os.environ["SELECTEL_API_KEY"], list(SELECTEL_IDS.values())
    if name == "plusvibe":
        return name, os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], list(SELECTEL_IDS)
    raise ValueError(f"неизвестный PROVIDER={name}")
