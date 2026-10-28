import asyncio
import json
import random
import statistics
import string
import time
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI, BadRequestError
from providers import get_provider

PROVIDER, BASE_URL, API_KEY, MODELS = get_provider()
QUESTION = "Сколько ног у сороканожки?"
# от самого сильного к слабому: не каждая модель принимает "max", берём первый принятый
EFFORTS = ["max", "xhigh", "high", None]
TIMEOUT = 600
RESULTS_DIR = Path(__file__).parent / "results"

GRADE_PROMPT = """Вопрос: {question}

Ниже анонимные ответы разных участников. Оцени каждый по шкале от 1 до 10:
фактическая точность важнее всего, затем полнота и ясность.

{answers}

Верни только JSON без пояснений вокруг, ключи — буквы ответов:
{{"A": {{"score": 7, "comment": "коротко, за что такая оценка"}}, ...}}"""

REVISE_PROMPT = """Другие участники анонимно оценили твой ответ в среднем на {score:.1f} из 10.
Если считаешь нужным, исправь ответ. Напиши итоговый ответ целиком: исправленный или прежний."""


async def ask(client: AsyncOpenAI, model: str, effort: str | None, messages: list[dict]) -> str:
    kwargs = {"reasoning_effort": effort} if effort else {}
    response = await client.chat.completions.create(model=model, messages=messages, **kwargs)
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("пустой ответ")
    return content.strip()


async def pick_effort(client: AsyncOpenAI, model: str, messages: list[dict]) -> tuple[str | None, str]:
    """Раунд 1: пробуем уровни effort сверху вниз, возвращаем принятый и ответ."""
    for effort in EFFORTS:
        try:
            return effort, await ask(client, model, effort, messages)
        except BadRequestError:
            if effort is None:
                raise
    raise AssertionError("недостижимо")


def parse_grades(text: str, labels: dict[str, str]) -> dict[str, dict]:
    """JSON вида {"A": {"score": 7, "comment": "..."}} -> {модель: {...}}."""
    # модели любят оборачивать JSON в ```json ... ``` или добавлять текст вокруг
    data = json.loads(text[text.index("{") : text.rindex("}") + 1])
    grades = {}
    for label, model in labels.items():
        item = data[label]
        score = float(item["score"])
        if not 1 <= score <= 10:
            raise ValueError(f"оценка {score} вне шкалы 1–10")
        grades[model] = {"score": score, "comment": str(item.get("comment", ""))}
    return grades


async def grade(client: AsyncOpenAI, grader: str, effort: str | None, answers: dict[str, str]) -> dict:
    others = [m for m in answers if m != grader]
    # своя перестановка для каждого оценщика: позиция ответа не должна влиять на оценку
    random.shuffle(others)
    labels = dict(zip(string.ascii_uppercase, others))
    blocks = "\n\n".join(f"--- Ответ {label} ---\n{answers[m]}" for label, m in labels.items())
    messages = [{"role": "user", "content": GRADE_PROMPT.format(question=QUESTION, answers=blocks)}]

    last_error = None
    for _ in range(2):
        text = await ask(client, grader, effort, messages)
        try:
            return {"labels": labels, "grades": parse_grades(text, labels), "raw": text}
        except (ValueError, KeyError, TypeError) as e:
            last_error = f"{e}: {text[:200]}"
    raise RuntimeError(f"не удалось разобрать оценки: {last_error}")


def peer_scores(grading: dict) -> dict[str, float]:
    """Интегральная оценка: среднее по всем оценщикам, кроме самой модели."""
    collected = {}
    for result in grading.values():
        for model, g in result.get("grades", {}).items():
            collected.setdefault(model, []).append(g["score"])
    return {model: statistics.mean(scores) for model, scores in collected.items()}


async def run_round(name: str, coros: dict) -> dict:
    print(f"\n=== {name} ===", flush=True)
    start = time.perf_counter()

    # печатаем каждую модель по готовности, а не весь раунд разом: max effort думает минутами
    async def tracked(model: str, coro) -> dict:
        try:
            r = await coro
            print(f"  {model}: ok, {time.perf_counter() - start:.0f} с", flush=True)
            return r
        except Exception as e:
            print(f"  {model}: ошибка: {str(e)[:120]}", flush=True)
            return {"error": str(e)}

    results = await asyncio.gather(*(tracked(m, c) for m, c in coros.items()))
    print(f"  раунд занял {time.perf_counter() - start:.0f} с", flush=True)
    return dict(zip(coros, results))


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> None:
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"intellect_{PROVIDER}_{datetime.now():%Y%m%d_%H%M%S}.json"
    data = {"provider": PROVIDER, "question": QUESTION, "models": MODELS, "started_at": datetime.now().isoformat()}
    question = [{"role": "user", "content": QUESTION}]

    # раунд 1: ответ в максимальном effort, который примет модель
    async def answer(model: str) -> dict:
        effort, text = await pick_effort(client, model, question)
        return {"effort": effort, "answer": text}

    data["round1"] = await run_round("Раунд 1: ответы", {m: answer(m) for m in MODELS})
    save(path, data)
    efforts = {m: r.get("effort") for m, r in data["round1"].items()}
    # в следующих раундах модель работает в том же effort, что ответила в первом
    alive = [m for m in MODELS if "answer" in data["round1"][m]]
    answers1 = {m: data["round1"][m]["answer"] for m in alive}

    # раунд 2: каждая оценивает чужие ответы
    data["round2"] = await run_round(
        "Раунд 2: оценка ответов", {m: grade(client, m, efforts[m], answers1) for m in alive}
    )
    save(path, data)
    scores2 = peer_scores(data["round2"])

    # раунд 3: модель видит свою среднюю оценку и может исправиться
    async def revise(model: str) -> dict:
        messages = question + [
            {"role": "assistant", "content": answers1[model]},
            {"role": "user", "content": REVISE_PROMPT.format(score=scores2[model])},
        ]
        text = await ask(client, model, efforts[model], messages)
        return {"peer_score": scores2[model], "answer": text}

    data["round3"] = await run_round(
        "Раунд 3: исправления", {m: revise(m) for m in alive if m in scores2}
    )
    save(path, data)
    answers3 = {m: r["answer"] for m, r in data["round3"].items() if "answer" in r}

    # раунд 4: оценка исправленных ответов
    data["round4"] = await run_round(
        "Раунд 4: оценка исправлений", {m: grade(client, m, efforts[m], answers3) for m in alive}
    )
    scores4 = peer_scores(data["round4"])

    # насколько строг оценщик: средняя оценка, которую он выставил другим
    strictness = {}
    for grader in alive:
        given = [
            g["score"]
            for rnd in ("round2", "round4")
            for g in data[rnd].get(grader, {}).get("grades", {}).values()
        ]
        strictness[grader] = statistics.mean(given) if given else None

    data["summary"] = {
        m: {"effort": efforts[m], "round2": scores2.get(m), "round4": scores4.get(m), "gives": strictness.get(m)}
        for m in MODELS
    }
    data["finished_at"] = datetime.now().isoformat()
    save(path, data)

    rows = sorted(
        data["summary"].items(),
        key=lambda item: -(item[1]["round4"] or item[1]["round2"] or 0),
    )
    print(f"\nВопрос: {QUESTION}\n")
    header = ["Модель", "Effort", "Раунд 2", "Раунд 4", "Δ", "Ставит"]
    table = [header]
    for model, s in rows:
        r2, r4 = s["round2"], s["round4"]
        delta = r4 - r2 if r2 is not None and r4 is not None else None
        table.append(
            [
                model,
                s["effort"] or "по умолч.",
                "—" if r2 is None else f"{r2:.1f}",
                "—" if r4 is None else f"{r4:.1f}",
                "—" if delta is None else f"{delta:+.1f}",
                "—" if s["gives"] is None else f"{s['gives']:.1f}",
            ]
        )
    widths = [max(len(line[i]) for line in table) for i in range(len(header))]
    for i, line in enumerate(table):
        print("  ".join(cell.ljust(w) if j < 2 else cell.rjust(w) for j, (cell, w) in enumerate(zip(line, widths))))
        if i == 0:
            print("  ".join("-" * w for w in widths))
    print(f"\nВсе ответы и оценки: {path}")


asyncio.run(main())
