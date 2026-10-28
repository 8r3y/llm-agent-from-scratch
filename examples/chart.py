import json
import statistics
import time
from typing import Literal

from openai import OpenAI
from providers import get_provider

from miniagent.tools import tool

PROVIDER, BASE_URL, API_KEY, MODELS = get_provider()
PROMPT = "Объясни в 3 коротких абзацах, что такое ReAct-агент и чем он отличается от обычного чат-бота."
TOOL_PROMPT = "Какая погода в Москве и в Париже, в Цельсиях?"
RUNS = 3
TIMEOUT = 180


@tool
def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Get current weather for a city."""
    return f"{city}: +20{unit}"


def measure(client: OpenAI, model: str) -> dict:
    start = time.perf_counter()
    first = None
    text = ""
    tokens = None

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.usage:
            tokens = chunk.usage.completion_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            # первый видимый текст: до этого момента пользователь смотрит в пустой экран
            if first is None:
                first = time.perf_counter()
            text += chunk.choices[0].delta.content

    end = time.perf_counter()
    if first is None:
        raise RuntimeError("пустой ответ")
    return {
        "ttft": first - start,
        "total": end - start,
        # символы, а не токены: у моделей разные токенизаторы, а пользователь видит текст
        "cps": len(text) / (end - first) if end > first else 0.0,
        "chars": len(text),
        "tokens": tokens,
    }


def check_calls(tool_calls: list) -> str | None:
    """Вернёт причину провала или None, если вызовы корректны."""
    if len(tool_calls) != 2:
        return f"вызовов {len(tool_calls)}, ждали 2 параллельных"
    cities = []
    for tc in tool_calls:
        if tc.function.name != get_weather.name:
            return f"неизвестный инструмент {tc.function.name}"
        try:
            args = json.loads(tc.function.arguments)
            # та же pydantic-валидация, что и в агенте: лишние поля и неверный unit не пройдут
            get_weather.args_schema.model_validate(args)
        except ValueError:
            return f"аргументы не по схеме: {tc.function.arguments}"
        if args.get("unit", "c") != "c":
            return f"unit={args['unit']}, просили Цельсии"
        cities.append(args["city"].lower())
    joined = " ".join(cities)
    if not any(m in joined for m in ("моск", "moscow")) or not any(p in joined for p in ("пари", "paris")):
        return f"не те города: {cities}"
    return None


def measure_tools(client: OpenAI, model: str) -> dict:
    messages = [{"role": "user", "content": TOOL_PROMPT}]
    tools = [get_weather.to_openai_schema()]

    # шаг 1: модель должна запросить оба города
    start = time.perf_counter()
    response = client.chat.completions.create(model=model, messages=messages, tools=tools)
    step1 = time.perf_counter() - start
    message = response.choices[0].message

    result = {"step1": step1, "step2": None, "total": None, "calls": 0, "error": None}
    if not message.tool_calls:
        result["error"] = "нет tool_calls, ответ текстом"
        return result
    result["calls"] = len(message.tool_calls)
    result["error"] = check_calls(message.tool_calls)
    if result["error"]:
        return result

    # шаг 2: результаты инструментов -> итоговый ответ, как во втором витке агента
    messages.append(message.model_dump(exclude_none=True))
    for tc in message.tool_calls:
        content = get_weather.run(tc.function.arguments)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    start = time.perf_counter()
    response = client.chat.completions.create(model=model, messages=messages, tools=tools)
    result["step2"] = time.perf_counter() - start
    result["total"] = step1 + result["step2"]
    if not response.choices[0].message.content:
        result["error"] = "пустой итоговый ответ"
    return result


def median(results: list[dict], key: str) -> float | None:
    values = [r[key] for r in results if r[key] is not None]
    return statistics.median(values) if values else None


def fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def print_table(header: list[str], rows: list[list[str]]) -> None:
    table = [header] + rows
    widths = [max(len(line[i]) for line in table) for i in range(len(header))]
    for i, line in enumerate(table):
        # текстовые колонки (первая и "Проблема") — влево, числа — вправо
        print("  ".join(cell.ljust(w) if j in (0, len(header) - 1) and not cell[:1].isdigit() else cell.rjust(w)
                        for j, (cell, w) in enumerate(zip(line, widths))))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def run_rounds(name: str, fn, client: OpenAI) -> tuple[dict, dict]:
    results = {model: [] for model in MODELS}
    errors = {model: [] for model in MODELS}
    # раунды по кругу, а не все прогоны модели подряд: колебания сети делятся между моделями поровну
    for run in range(1, RUNS + 1):
        for model in MODELS:
            print(f"[{name} {run}/{RUNS}] {model} ...", end=" ", flush=True)
            try:
                r = fn(client, model)
                results[model].append(r)
                print(r.get("error") or "ok")
            except Exception as e:
                errors[model].append(str(e))
                print(f"ошибка: {e}")
    return results, errors


def print_errors(errors: dict) -> None:
    for model, errs in errors.items():
        if errs:
            print(f"{model}: {errs[0]}")


def main() -> None:
    print(f"Провайдер: {PROVIDER}")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)

    results, errors = run_rounds("текст", measure, client)
    rows = []
    for model in MODELS:
        rs = results[model]
        rows.append(
            [
                model,
                median(rs, "ttft"),
                median(rs, "total"),
                median(rs, "cps"),
                median(rs, "chars"),
                median(rs, "tokens"),
                f"{len(rs)}/{RUNS}",
            ]
        )
    # сортировка по TTFT: именно ожидание первого текста сильнее всего ощущается в чате
    rows.sort(key=lambda row: (row[1] is None, row[1] or 0))
    print(f"\nТекст, медиана по {RUNS} прогонам. Промпт: {PROMPT}\n")
    print_table(
        ["Модель", "TTFT, с", "Всего, с", "Симв/с", "Символов", "Токенов", "Успешно"],
        [[r[0], fmt(r[1], 2), fmt(r[2], 2), fmt(r[3], 0), fmt(r[4], 0), fmt(r[5], 0), r[6]] for r in rows],
    )
    print_errors(errors)

    results, errors = run_rounds("tools", measure_tools, client)
    rows = []
    for model in MODELS:
        rs = results[model]
        passed = [r for r in rs if not r["error"]]
        reasons = [r["error"] for r in rs if r["error"]] + errors[model]
        rows.append(
            [
                model,
                median(passed, "step1"),
                median(passed, "step2"),
                median(passed, "total"),
                f"{len(passed)}/{RUNS}",
                reasons[0][:60] if reasons else "",
            ]
        )
    # для агента важно полное время цикла, а сломанные вызовы — в конец
    rows.sort(key=lambda row: (row[3] is None, row[3] or 0))
    print(f"\nTool calling, медиана по {RUNS} прогонам. Промпт: {TOOL_PROMPT}\n")
    print_table(
        ["Модель", "Вызов, с", "Ответ, с", "Всего, с", "Корректно", "Проблема"],
        [[r[0], fmt(r[1], 2), fmt(r[2], 2), fmt(r[3], 2), r[4], r[5]] for r in rows],
    )


main()
