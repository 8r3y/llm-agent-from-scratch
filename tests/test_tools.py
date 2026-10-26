import json
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from miniagent.tools import BaseTool, tool


@pytest.fixture
def weather_tool():
    @tool
    def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
        """Get current weather"""
        return f"{city}: +20{unit}"

    return get_weather


def test_tool_decorator_creates_base_tool(weather_tool):
    assert isinstance(weather_tool, BaseTool)
    assert weather_tool.name == "get_weather"
    assert weather_tool.description == "Get current weather"


def test_schema_has_openai_format(weather_tool):
    schema = weather_tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_weather"
    assert schema["function"]["description"] == "Get current weather"


def test_schema_parameters(weather_tool):
    params = weather_tool.to_openai_schema()["function"]["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["city"]
    assert params["properties"]["city"]["type"] == "string"
    assert params["properties"]["unit"]["enum"] == ["c", "f"]
    assert params["properties"]["unit"]["default"] == "c"
    assert params["additionalProperties"] is False
    print(json.dumps(params, indent=2))


def test_param_without_type_hint_is_rejected():
    def search(query: str, limit=10):
        return query

    with pytest.raises(TypeError, match="limit"):
        tool(search)


@pytest.mark.parametrize(
    "raw_args",
    ['{"city": "Moscow"}', {"city": "Moscow"}],
    ids=["json-string", "dict"],
)
def test_run_calls_function(weather_tool, raw_args):
    assert weather_tool.run(raw_args) == "Moscow: +20c"


@pytest.mark.parametrize(
    "raw_args",
    [
        {},  # нет обязательного city
        {"city": "Moscow", "unit": "k"},  # unit не из Literal
        {"city": 123},  # число вместо строки
        '{"city": ',  # битый JSON
        {
            "city": "Moscow",
            "units": "f",
        },  # лишнее поле (опечатка LLM: units вместо unit)
    ],
    ids=[
        "missing-required",
        "bad-literal",
        "int-for-str",
        "broken-json",
        "extra-field",
    ],
)
def test_run_rejects_invalid_args(weather_tool, raw_args):
    with pytest.raises(ValidationError):
        weather_tool.run(raw_args)


def test_param_description_from_annotated():
    @tool
    def search_logs(
        service: Annotated[str, Field(description="Service name, e.g. 'billing'")],
        limit: int = 10,
    ) -> list[str]:
        """Search recent error logs."""
        return []

    params = search_logs.to_openai_schema()["function"]["parameters"]
    assert (
        params["properties"]["service"]["description"] == "Service name, e.g. 'billing'"
    )


def _positional_only(city: str, /) -> str:
    return city


def _var_args(*cities: str) -> str:
    return ""


def _var_kwargs(**options: str) -> str:
    return ""


@pytest.mark.parametrize(
    "fn",
    [_positional_only, _var_args, _var_kwargs],
    ids=["positional-only", "var-args", "var-kwargs"],
)
def test_unsupported_param_kinds_are_rejected(fn):
    with pytest.raises(TypeError, match="unsupported"):
        tool(fn)


def test_keyword_only_params_are_supported():
    @tool
    def get_city(*, city: str) -> str:
        """Return city."""
        return city

    assert get_city.run({"city": "Moscow"}) == "Moscow"


class PrefixEcho(BaseTool):
    name = "prefix_echo"
    description = "Echo text with a configured prefix."

    class Args(BaseModel):
        text: str

    args_schema = Args

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def execute(self, text: str) -> str:
        return self.prefix + text


def test_class_based_tool_runs_with_state():
    echo = PrefixEcho(prefix=">> ")
    assert echo.run({"text": "hi"}) == ">> hi"


def test_base_tool_cannot_be_created_without_execute():
    class Broken(BaseTool):
        name = "broken"
        description = "No execute."

    with pytest.raises(TypeError):
        Broken()
