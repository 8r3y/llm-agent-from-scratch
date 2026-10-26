# llm-agent-from-scratch

[English](README.md) · **Русский**

Python-пакет: `miniagent`.

Ядро LLM-агента на чистом Python, без фреймворков: без LangChain и LlamaIndex.
Собрано с нуля, чтобы показать, из чего агент состоит: схемы инструментов, цикл ReAct, песочница для кода и guardrails.

Агенту можно дать Excel-файл и вопрос: он сам изучит данные, обучит логистическую регрессию и назовёт
ключевые признаки. Весь код, который он пишет, выполняется в изолированном Docker-контейнере.

## Возможности

- **Декоратор `@tool`.** Строит JSON-схему из обычной функции: сигнатура, подсказки типов, docstring
  и `Annotated[..., Field(description=...)]`, через `create_model` из Pydantic.
- **Инструменты-классы.** Наследники `BaseTool` с явной `args_schema` — для инструментов с состоянием
  (подключение к БД, корневая папка). Оба способа приводятся к одному интерфейсу.
- **Асинхронный цикл ReAct.** Параллельные вызовы инструментов через `asyncio.gather`, синхронные
  инструменты выносятся в поток через `asyncio.to_thread`, ограничение `max_steps`.
- **Самокоррекция.** Ошибки валидации и исключения инструментов возвращаются модели сообщением `tool`
  и не роняют агента. Неизвестные аргументы запрещены (`extra="forbid"`).
- **Fail fast.** Функции без подсказок типов или с `*args` / `**kwargs` / positional-only параметрами
  отклоняются в момент декорирования, а не посреди диалога.
- **Песочница на Docker.** `execute_python` выполняет код модели без сети, на файловой системе только
  для чтения, с лимитами памяти, CPU и числа процессов, не от root, без capabilities и с жёстким таймаутом.
- **Guardrails.** `read_file` блокирует выход за пределы рабочей папки (path traversal) и ограничивает объём вывода.
- **Независимость от провайдера.** Любой OpenAI-совместимый API: OpenAI, Gemini, vLLM, Ollama.

## Быстрый старт

Нужны Python 3.11+, [uv](https://docs.astral.sh/uv/) и Docker.

```bash
uv sync
cp .env.sample .env                       # впишите свой GEMINI_API_KEY
docker build -t miniagent-sandbox sandbox/   # образ песочницы с pandas и scikit-learn

uv run python examples/demo.py            # параллельные вызовы инструментов

# Кейс с Excel: сгенерировать синтетические данные об оттоке, затем дать их агенту
uv run --with pandas --with openpyxl python examples/make_excel.py
uv run python examples/excel_demo.py
```

Примеры работают с Gemini через его OpenAI-совместимый endpoint. Для другого провайдера поменяйте
`base_url` и `model` в примере.

> **Docker Desktop на Windows:** добавьте папку проекта и `%LOCALAPPDATA%\Temp` в
> *Settings → Resources → File sharing*, иначе при монтировании `/data` Docker будет каждый раз спрашивать разрешение.

## Использование

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
    print(await agent.run("Какая погода в Париже и в Москве?"))


asyncio.run(main())
```

Инструмент с состоянием оформляется классом:

```python
from pathlib import Path
from miniagent.builtin_tools import ReadFileTool

read_file = ReadFileTool(workspace=Path("docs"))   # корень модель не видит и поменять не может
```

## Как это устроено

```
messages = [system, user]
повторять до max_steps:
    ответ = LLM(messages, схемы инструментов)        → добавляется как "assistant"
    в ответе нет tool_calls?  → это финальный ответ, вернуть его
    для каждого вызова, параллельно:
        валидировать аргументы (Pydantic) → выполнить инструмент → добавить сообщение "tool" с tool_call_id
        при любой ошибке → текст ошибки становится результатом, и модель может исправить вызов
```

Код, сгенерированный моделью, никогда не выполняется в процессе агента. `execute_python` — тонкий
адаптер над `run_python`, которая на каждый вызов запускает новый контейнер. Папка с данными
зафиксирована в замыкании: её нет в схеме инструмента, и модель не может указать другую.

| Угроза | Защита |
| --- | --- |
| Утечка данных, SSRF | `--network none` |
| Исчерпание памяти и CPU, fork-бомба | `--memory`, `--cpus`, `--pids-limit` |
| Изменение системы | `--read-only` (запись только в `tmpfs`), не root, `--cap-drop ALL`, `no-new-privileges` |
| Изменение входных данных | данные смонтированы `:ro` |
| Бесконечный цикл | таймаут, затем `docker kill` по имени контейнера |
| Переполнение контекста модели | вывод обрезается до 10 000 символов |

## Тесты

Проект написан через TDD.

```bash
uv run pytest -m "not integration"   # 24 юнит-теста: без Docker и LLM, меньше секунды
uv run pytest -m integration         # 8 тестов песочницы на настоящем Docker
```

Покрыто: генерация схемы и описаний, `required` и значения по умолчанию, невалидные аргументы
(нет обязательного, неверный тип, значение вне перечисления, битый JSON, лишнее поле), отклонение
неподходящих сигнатур, инструменты-классы, попытки path traversal и изоляция песочницы (перехват
stdout/stderr, таймаут, нет сети, ФС только для чтения, не root, данные только для чтения).

## Структура проекта

```
src/miniagent/
  tools.py           BaseTool, FunctionTool, @tool
  agent.py           асинхронный цикл ReAct
  sandbox.py         run_python и инструмент execute_python
  builtin_tools.py   ReadFileTool с защитой путей
sandbox/Dockerfile   образ песочницы
examples/            запускаемые примеры
tests/               юнит- и интеграционные тесты
```

## Ограничения

Это учебный прототип, а не продакшен-фреймворк.

- Docker разделяет ядро с хостом. Для недоверенных пользователей нужен gVisor или microVM (Firecracker, Kata).
- Песочница без состояния: каждый вызов — новый процесс, постоянного Jupyter-ядра нет.
- Инструменты только синхронные; асинхронные инструменты и стриминг пока не поддерживаются.
- Нет памяти диалога, повторов с backoff и трассировки.
- Сам цикл агента пока не покрыт тестами с подменённой LLM.

## Лицензия

[MIT](LICENSE)
