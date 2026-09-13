"""Offline checks for LFM's non-executing call parser and shared tool contracts."""

from pathlib import Path

import pytest
from pydantic import Field

from hoast.lfm import LFM2, LFMConfig, LFMParseError, parse_response, tool_declarations
from hoast.llm import Tool, ToolArguments, ToolCall, ToolRegistry


class LocationArguments(ToolArguments):
    """Minimal location contract for native declaration validation."""

    location: str = Field(description="City name")
    """City whose temperature is requested."""


def temperature(arguments: LocationArguments) -> str:
    """Return the fixture location without accessing a weather provider.

    Args:
        arguments:
            Strictly validated location fixture.

    """
    return arguments.location


def test_native_calls_and_literal_markers() -> None:
    """Preserve quoted delimiters, Unicode, escaped quotes, and JSON-like values."""
    text, calls = parse_response(
        "Before. <|tool_call_start|>[run(payload={'text': '<|tool_call_end|>雪', "
        "'values': [true, False, null, -1.5]}, name='Miles\\'s'), finish()]"
        "<|tool_call_end|> After.<|im_end|>"
    )
    assert text == "Before.  After."
    assert calls == (
        ToolCall(
            "run",
            {
                "payload": {
                    "text": "<|tool_call_end|>雪",
                    "values": [True, False, None, -1.5],
                },
                "name": "Miles's",
            },
        ),
        ToolCall("finish", {}),
    )


@pytest.mark.parametrize(
    "expression",
    [
        "[run(x=__import__('os').getcwd())]",
        "[os.system(command='anything')]",
        "[run(**{'x': 1})]",
        "[run('positional')]",
        "[run(x=1, x=2)]",
        "[run(x={'a': 1, 'a': 2})]",
        "[run(x=[i for i in range(3)])]",
        "[run(x=1+2)]",
        "[run(x=1e999)]",
        "[run(x=b'bytes')]",
    ],
)
def test_reject_expressions(expression: str) -> None:
    """Reject executable expressions and ambiguous arguments at the wire boundary.

    Args:
        expression:
            Invalid native call list, never evaluated or executed.

    """
    with pytest.raises(LFMParseError):
        parse_response(f"<|tool_call_start|>{expression}<|tool_call_end|>")


def test_incomplete_batch_and_plain_text() -> None:
    """Require complete batches while allowing normal assistant text."""
    assert parse_response("Hello!<|im_end|>") == ("Hello!", ())
    with pytest.raises(LFMParseError):
        parse_response(
            "<|tool_call_start|>[first()]<|tool_call_end|><|tool_call_start|>[second("
        )
    with pytest.raises(LFMParseError):
        parse_response("<|tool_call_start|>[first()]")
    with pytest.raises(LFMParseError):
        parse_response("x" * 65537)


def test_unloaded_owner_and_config(tmp_path: Path) -> None:
    """Reject generation without ownership and invalid model configuration.

    Args:
        tmp_path:
            Isolated placeholder model directory.

    """
    owner = LFM2(LFMConfig(tmp_path), ToolRegistry([]))
    with pytest.raises(RuntimeError, match="Load"):
        owner.generate("Hello")
    owner.close()
    owner.close()
    with pytest.raises(ValueError):
        LFMConfig(tmp_path, repetition_penalty=float("nan"))
    with pytest.raises(ValueError):
        LFMConfig(tmp_path, threads=0)


def test_native_declarations_keep_validation_separate() -> None:
    """Remove schema bookkeeping from prompts without permitting extra arguments."""
    tools = ToolRegistry(
        [
            Tool(
                "get_current_temperature",
                "Read a city's temperature.",
                LocationArguments,
                temperature,
            )
        ]
    )
    declarations = tool_declarations(tools)
    parameters = declarations[0]["parameters"]
    assert "additionalProperties" not in parameters and "title" not in parameters
    assert parameters["required"] == ["location"]
    assert parameters["properties"]["location"]["type"] == "string"
    assert "description" in parameters["properties"]["location"]
    with pytest.raises(ValueError):
        tools.validate(
            [
                ToolCall(
                    "get_current_temperature",
                    {"location": "London", "additionalProperties": False},
                )
            ]
        )
