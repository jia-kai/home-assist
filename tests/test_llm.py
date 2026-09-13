"""Offline tests for protocol boundaries, dispatch, and model ownership."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import Field, JsonValue, ValidationError

from hoast.llm import (
    FunctionGemma,
    LLMConfig,
    Tool,
    ToolArguments,
    ToolCall,
    ToolRegistry,
    parse_response,
)


class LightArgs(ToolArguments):
    """Minimal bounded fixture contract."""

    room: str = Field(description="Room name")
    """Room containing the light."""

    brightness: int = Field(ge=0, le=100, description="Brightness in percent")
    """Target light brightness, from 0 through 100 percent."""


def test_nested_protocol_and_literal_delimiters() -> None:
    """Preserve punctuation, Unicode, and protocol-looking text in strings."""
    text, calls = parse_response(
        "Working on it. <start_function_call>call:run{"
        "payload:{ <escape>key<escape>:<escape>雪, } <end_function_call><escape>},"
        "values:[true,false,null,-1.2e3]}<end_function_call>"
        "<start_function_call>call:finish{}<end_function_call><start_function_response>"
    )
    assert text == "Working on it."
    assert calls == (
        ToolCall(
            "run",
            {
                "payload": {"key": "雪, } <end_function_call>"},
                "values": [True, False, None, -1200.0],
            },
        ),
        ToolCall("finish", {}),
    )


@pytest.mark.parametrize(
    "raw",
    [
        "<start_function_call>call:run{x:1,x:2}<end_function_call>",
        "<start_function_call>call:run{x:<escape>broken}",
        "<start_function_call>call:run{x:1}",
        "<start_function_call>call:run{x:NaN}<end_function_call>",
        "<start_function_call>call:run{x:1e999}<end_function_call>",
        "<start_function_call>call:run[]<end_function_call>",
        "<start_function_call>call:run{x:trueish}<end_function_call>",
        "<start_function_call>call:run{x:1,}<end_function_call>",
        "<end_function_call>",
        "<start_function_call>call:run{}<end_function_call><start_function_call>",
    ],
)
def test_reject_malformed_protocol(raw: str) -> None:
    """Reject malformed or truncated batches before they reach dispatch.

    Args:
        raw:
            Invalid model response fixture.

    """
    with pytest.raises(ValueError):
        parse_response(raw)


def test_plain_text_and_limits() -> None:
    """Preserve text spacing and reject excessive output size or depth."""
    assert parse_response("Hello, how are you?<end_of_turn>") == (
        "Hello, how are you?",
        (),
    )
    with pytest.raises(ValueError, match="size limit"):
        parse_response("x" * 65537)
    with pytest.raises(ValueError, match="nesting"):
        parse_response("<start_function_call>call:run{x:" + "[" * 40)


def test_dispatch_validates_entire_batch() -> None:
    """An invalid later call prevents all handlers from running."""
    seen: list[LightArgs] = []

    def handler(args: LightArgs) -> bool:
        """Record a successful fixture invocation.

        Args:
            args:
                Validated lighting command.

        """
        seen.append(args)
        return True

    registry = ToolRegistry([Tool("light", "Set light brightness", LightArgs, handler)])
    valid = ToolCall("light", {"room": "kitchen", "brightness": 50})
    invalid = ToolCall("light", {"room": "kitchen", "brightness": "50"})
    with pytest.raises(ValidationError):
        registry.dispatch([valid, invalid])
    assert seen == []
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.dispatch([valid, ToolCall("missing", {})])
    assert seen == []
    assert registry.dispatch([valid]) == [True]
    assert seen == [LightArgs(room="kitchen", brightness=50)]
    invalid_arguments: list[dict[str, JsonValue]] = [
        {"room": "kitchen"},
        {"room": "kitchen", "brightness": True},
        {"room": "kitchen", "brightness": 101},
        {"room": "kitchen", "brightness": 50, "extra": 1},
    ]
    for arguments in invalid_arguments:
        with pytest.raises(ValidationError):
            registry.validate([ToolCall("light", arguments)])


def test_schema_and_duplicate_registration() -> None:
    """Schemas expose descriptions and constraints and require unique names."""

    def handler(args: LightArgs) -> bool:
        """Accept a fixture request.

        Args:
            args:
                Validated lighting command.

        """
        return args.brightness > 0

    tool = Tool("light", "Set brightness", LightArgs, handler)
    schema = tool.schema()["function"]["parameters"]
    assert schema["properties"]["brightness"]["maximum"] == 100
    assert schema["properties"]["room"]["description"] == "Room name"
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry([tool, tool])


def test_lifecycle_failure_and_close(tmp_path: Path) -> None:
    """Require loading, leave failures unloaded, and allow repeated close.

    Args:
        tmp_path:
            Isolated model path without external fixture dependencies.

    """
    llm = FunctionGemma(LLMConfig(tmp_path / "missing"), ToolRegistry([]))
    with pytest.raises(RuntimeError, match="Load"):
        llm.generate("Hello")
    with pytest.raises(FileNotFoundError):
        llm.load()
    llm.close()
    llm.close()
    assert llm._model is None
    assert llm._tokenizer is None


def test_context_releases_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handler/application exception escapes while model resources release.

    Args:
        tmp_path:
            Isolated model directory.

        monkeypatch:
            Replaces model loading with an offline stand-in.

    """

    def load(self: FunctionGemma) -> FunctionGemma:
        """Install a model stand-in without downloading weights.

        Args:
            self:
                Lifecycle owner under test.

        """
        self._model = object()
        self._tokenizer = object()
        return self

    monkeypatch.setattr(FunctionGemma, "load", load)
    llm = FunctionGemma(LLMConfig(tmp_path), ToolRegistry([]))
    with pytest.raises(RuntimeError, match="application failure"), llm:
        raise RuntimeError("application failure")
    assert llm._model is None and llm._tokenizer is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"threads": 0},
        {"max_new_tokens": 0},
        {"max_context_tokens": 40000},
        {"backend": "unknown"},
        {"system_prompt": "missing activation"},
    ],
)
def test_config_validation(overrides: dict[str, Any]) -> None:
    """Reject invalid runtime settings without allocating models.

    Args:
        overrides:
            Invalid configuration overrides.

    """
    with pytest.raises(ValueError):
        LLMConfig(Path("unused"), **overrides)


class SceneArgs(ToolArguments):
    """Fixture with referenced nested contracts and a string enum."""

    lights: list[LightArgs] = Field(description="Lights to set")
    """Ordered light commands."""

    mode: Literal["immediate", "fade"] = Field(description="Transition mode")
    """How brightness should change."""


def test_nested_tool_schema_expansion() -> None:
    """Expand references for the official template and validate nested inputs."""

    def handler(args: SceneArgs) -> bool:
        """Accept an embedded scene fixture.

        Args:
            args:
                Validated scene and nested light settings.

        """
        return bool(args.lights)

    tool = Tool("scene", "Set a lighting scene", SceneArgs, handler)
    parameters = tool.schema()["function"]["parameters"]
    nested = parameters["properties"]["lights"]["items"]
    assert nested["properties"]["room"]["type"] == "string"
    assert parameters["properties"]["mode"]["enum"] == ["immediate", "fade"]
    registry = ToolRegistry([tool])
    with pytest.raises(ValidationError):
        registry.validate(
            [
                ToolCall(
                    "scene",
                    {
                        "mode": "fade",
                        "lights": [{"room": "kitchen", "brightness": "50"}],
                    },
                )
            ]
        )


def test_dispatch_handler_failure_does_not_retry() -> None:
    """Handler failures preserve earlier effects and stop later calls."""
    seen: list[str] = []

    def handler(args: LightArgs) -> bool:
        """Record the call and deliberately fail the second room.

        Args:
            args:
                Validated lighting request.

        """
        seen.append(args.room)
        if args.room == "bedroom":
            raise RuntimeError("device unavailable")
        return True

    registry = ToolRegistry([Tool("light", "Set brightness", LightArgs, handler)])
    with pytest.raises(RuntimeError, match="device unavailable"):
        registry.dispatch(
            [
                ToolCall("light", {"room": room, "brightness": 50})
                for room in ("kitchen", "bedroom", "hallway")
            ]
        )
    assert seen == ["kitchen", "bedroom"]


class Mode(StrEnum):
    """String enum used by a strict tool contract."""

    ON = "on"
    OFF = "off"


class ModeArgs(ToolArguments):
    """Fixture testing JSON enum validation under strict typing."""

    mode: Mode = Field(description="Requested mode")
    """Allowed power mode."""


def test_strict_string_enum_arguments() -> None:
    """Accept declared JSON enum values while rejecting undeclared values."""

    def handler(args: ModeArgs) -> bool:
        """Return the typed enum's power state.

        Args:
            args:
                Validated enum argument.

        """
        assert isinstance(args.mode, Mode)
        return args.mode is Mode.ON

    registry = ToolRegistry([Tool("mode", "Select mode", ModeArgs, handler)])
    assert registry.dispatch([ToolCall("mode", {"mode": "on"})]) == [True]
    with pytest.raises(ValidationError):
        registry.dispatch([ToolCall("mode", {"mode": "unknown"})])
