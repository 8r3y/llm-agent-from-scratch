import inspect
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

_UNSUPPORTED_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
)


class BaseTool(ABC):
    name: str
    description: str
    args_schema: type[BaseModel]

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any: ...

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }

    def run(self, raw_args: str | dict[str, Any]) -> Any:
        if isinstance(raw_args, str):
            args = self.args_schema.model_validate_json(raw_args)
        else:
            args = self.args_schema.model_validate(raw_args)
        return self.execute(**dict(args))


class FunctionTool(BaseTool):
    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.name = fn.__name__
        self.description = inspect.getdoc(fn) or ""
        hints = typing.get_type_hints(fn, include_extras=True)
        fields = {}
        for param_name, param in inspect.signature(fn).parameters.items():
            if param.kind in _UNSUPPORTED_KINDS:
                raise TypeError(
                    f"Tool '{fn.__name__}': unsupported parameter '{param_name}'"
                    f"({param.kind.description})"
                )

            if param_name not in hints:
                raise TypeError(
                    f"Tool '{fn.__name__}': parameter '{param_name}' has no type hint"
                )
            annotation = hints[param_name]
            if param.default is inspect.Parameter.empty:
                default = ...
            else:
                default = param.default

            fields[param_name] = (annotation, default)

        self.args_schema = create_model(
            fn.__name__,
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )

    def execute(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def tool(fn: Callable[..., Any]) -> FunctionTool:
    return FunctionTool(fn)
