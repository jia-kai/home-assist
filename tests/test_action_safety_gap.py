"""Known action-authorization gap: generated calls are not user intent."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from hoast.agent import LocalAgent
from hoast.lights import LightArguments
from hoast.llm import FunctionGemma, LLMConfig, Tool, ToolArguments, ToolCall, ToolRegistry
from hoast.session import Session
from hoast.weather import WeatherArguments
from test_agent import Routing, WeatherAgent, agent_runtime  # noqa: F401


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="LocalAgent dispatches schema-valid model proposals without authorization",
)
@pytest.mark.parametrize("text", ["hello", "don't turn on the light", "你好"])
def test_generated_light_call_requires_user_authorization(
    agent_runtime: tuple[WeatherAgent, Routing], tmp_path: Path, text: str
) -> None:
    """Unrelated or negated input must not authorize an invented light action.

    Args:
        agent_runtime:
            Controlled inference using the real session and tool validation.

        tmp_path:
            Unused model path for the controlled backend.

        text:
            User input that does not request switching on the light.

    """
    _, state = agent_runtime
    dispatched: list[str] = []

    def handler(name: str, args: ToolArguments) -> JsonValue:
        """Record an offline dispatch and return a renderer-compatible result.

        Args:
            name:
                Tool identifier under test.

            args:
                Validated handler arguments.

        """
        dispatched.append(name)
        if name == "set_light":
            return {"status": "confirmed", "on": True}
        return {"status": "unavailable"}

    def tool(name: str, arguments: type[ToolArguments]) -> Tool[Any]:
        """Construct a recording tool using its actual argument schema.

        Args:
            name:
                Tool identifier under test.

            arguments:
                Validated argument model.

        """
        return Tool(name, "Offline fixture", arguments, lambda args: handler(name, args))

    tools = ToolRegistry(
        [tool("get_weather", WeatherArguments), tool("set_light", LightArguments)]
    )
    agent = LocalAgent(Session(FunctionGemma(LLMConfig(tmp_path), tools)))
    state.calls = (ToolCall("set_light", {"on": True}),)
    list(agent.stream(text))
    assert dispatched == []
