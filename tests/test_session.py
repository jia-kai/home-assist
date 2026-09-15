"""Offline runtime integration with real decoding adapters and synthetic token IDs."""

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
from pydantic import JsonValue

from hoast.lfm import LFM2, LFMConfig
from hoast.llm import (
    FunctionGemma,
    GeneratedCallError,
    LLMConfig,
    Tool,
    ToolArguments,
    ToolCall,
    ToolRegistry,
)
from hoast.session import Session


class Arguments(ToolArguments):
    """Strict fixture input."""

    value: str
    """String to record verbatim."""


@dataclass(slots=True)
class Tokenizer:
    """Minimal character vocabulary and native-message capture."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    """Most recently formatted history, including the configured prompt."""

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Capture messages passed to the template boundary.

        Args:
            messages:
                Native history supplied by the runtime.

            kwargs:
                Template controls and tool declarations.

        """
        self.messages = messages
        assert kwargs["add_generation_prompt"]
        return "prompt"

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        """Return a batch-one int64 prompt tensor of shape (1, 1).

        Args:
            prompt:
                Rendered fixture prompt.

            kwargs:
                Tokenization controls.

        """
        return SimpleNamespace(input_ids=np.array([[1]], dtype=np.int64))

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        """Decode character IDs while asserting protocol is preserved.

        Args:
            tokens:
                One-dimensional integer sequence containing Unicode codepoints.

            kwargs:
                Decoder controls, including special-token preservation.

        """
        assert kwargs["skip_special_tokens"] is False
        return "".join(chr(int(token)) for token in tokens)


@dataclass(slots=True)
class Pipeline:
    """Synthetic OpenVINO token producer exercising the actual streamer adapter."""

    raw: str = "Hello world."
    """Next response, including protocol delimiters."""

    failure: bool = False
    """Whether to fail after emitting the response."""

    release: threading.Event | None = None
    """Optional barrier after the first word, proving incremental delivery."""

    def generate(self, inputs: Any, **kwargs: Any) -> Any:
        """Emit tokens and return a batch-one encoded generation.

        Args:
            inputs:
                OpenVINO input tensor.

            kwargs:
                Generation controls and optional token streamer.

        """
        streamer = kwargs.get("streamer")
        for index, char in enumerate(self.raw):
            if streamer is not None:
                streamer.write(ord(char))
            if index == 5 and self.release is not None:
                assert self.release.wait(5), "Consumer did not receive incremental text"
        if self.failure:
            error = RuntimeError("backend exploded")
            error.add_note("fixture diagnostics")
            raise error
        if streamer is not None:
            streamer.end()
        return SimpleNamespace(
            tokens=[[ord(char) for char in self.raw]],
            perf_metrics=SimpleNamespace(
                get_num_generated_tokens=lambda: len(self.raw),
                get_ttft=lambda: SimpleNamespace(mean=1),
            ),
        )


@pytest.fixture(params=["gemma", "lfm"])
def runtime(
    request: pytest.FixtureRequest,
) -> tuple[Session, Pipeline, Tokenizer, list[str]]:
    """Construct both runtime families with isolated, weight-free backends.

    Args:
        request:
            Parametrized model family.

    """
    seen: list[str] = []

    def handler(args: Arguments) -> JsonValue:
        """Record calls and exercise a handler failure with preceding side effects.

        Args:
            args:
                Strictly validated fixture input.

        """
        seen.append(args.value)
        if args.value == "fail":
            raise LookupError("handler exploded")
        return False

    tools = ToolRegistry([Tool("record", "Record a value", Arguments, handler)])
    model = (
        FunctionGemma(LLMConfig(Path("unused"), max_new_tokens=1024), tools)
        if request.param == "gemma"
        else LFM2(LFMConfig(Path("unused"), max_new_tokens=1024), tools)
    )
    pipeline, tokenizer = Pipeline(), Tokenizer()
    model._model, model._tokenizer = pipeline, tokenizer
    return Session(model), pipeline, tokenizer, seen


def call(session: Session, value: str) -> str:
    """Format a native call fixture for the session's family.

    Args:
        session:
            Session identifying the model family.

        value:
            Literal string without quotes or escape delimiters.

    """
    if isinstance(session.model, FunctionGemma):
        return f"<start_function_call>call:record{{value:<escape>{value}<escape>}}<end_function_call>"
    return f"<|tool_call_start|>[record(value='{value}')]<|tool_call_end|>"


@pytest.mark.parametrize("failure", ["syntax", "argument", "name"])
def test_backend_call_repair(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Repair real backend parsing/validation failures without leaking rejected prose.

    Args:
        runtime:
            Synthetic pipeline with production backend adapters for both families.

        monkeypatch:
            Supplies a rejected response followed by a corrected tool call.

        failure:
            Syntax, unsupported argument, or unknown tool failure to generate.

    """
    session, _, tokenizer, seen = runtime
    valid = call(session, "fixed")
    invalid = (
        valid[:-10]
        if failure == "syntax"
        else valid.replace("value", "country")
        if failure == "argument"
        else valid.replace("record", "unknown")
    )
    outputs = ["Untrusted prose. " + invalid, valid]
    original = Pipeline.generate

    def generate(owner: Pipeline, inputs: Any, **kwargs: Any) -> Any:
        """Emit a controlled native response, checking retry feedback reaches template.

        Args:
            owner:
                Synthetic token pipeline.

            inputs:
                Encoded prompt tensor.

            kwargs:
                Backend generation controls and cancellation-aware streamer.

        """
        if len(outputs) == 1:
            assert "Validation error:" in tokenizer.messages[-1]["content"]
            assert "Rejected output:" in tokenizer.messages[-1]["content"]
            assert tokenizer.messages[1]["content"] == "Record a value"
            assert not seen
        owner.raw = outputs.pop(0)
        return original(owner, inputs, **kwargs)

    monkeypatch.setattr(Pipeline, "generate", generate)
    assert list(session.stream("Record a value", max_repair_attempts=2)) == []
    assert not outputs
    assert len(session.pending_calls) == 1
    assert "Untrusted prose" not in str(session.history)
    assert "Validation error:" not in str(session.history)
    assert session.invoke_tools(lambda tool, result: "Recorded.") == (False,)
    assert seen == ["fixed"]
    assert session.pending_calls == ()
    assert session.history[-1]["content"]["result"] == "Recorded."
    session.complete("Recorded.")
    session.request_tools("Next", [ToolCall("record", {"value": "next"})])
    assert session.invoke_tools() == (False,)
    assert seen == ["fixed", "next"]


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_repair_budget_and_rollback(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
) -> None:
    """Bound regeneration and retain a pending continuation after exhausted repair.

    Args:
        runtime:
            Both model families with a side-effect recording handler.

        monkeypatch:
            Counts backend generations.

        budget:
            Permitted additional attempts beyond the first generation.

    """
    session, pipeline, _, seen = runtime
    pipeline.raw = call(session, "once")
    list(session.stream("Record"))
    session.invoke_tools()
    history = session.history
    pipeline.raw = call(session, "bad").replace("value", "country")
    original = Pipeline.generate
    attempts = 0

    def generate(owner: Pipeline, inputs: Any, **kwargs: Any) -> Any:
        """Count generations while retaining the production validation path.

        Args:
            owner:
                Synthetic pipeline.

            inputs:
                Encoded prompt.

            kwargs:
                Generation controls.

        """
        nonlocal attempts
        attempts += 1
        return original(owner, inputs, **kwargs)

    monkeypatch.setattr(Pipeline, "generate", generate)
    with pytest.raises(GeneratedCallError) as raised:
        list(session.stream(max_repair_attempts=budget))
    assert raised.value.__cause__ is not None
    assert attempts == budget + 1
    assert session.history == history
    assert seen == ["once"]
    pipeline.raw = "Done."
    assert "".join(session.stream()) == "Done."


@pytest.mark.parametrize("budget", [-1, 3, True, 1.5, "2", None])
def test_invalid_repair_budget(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]], budget: Any
) -> None:
    """Reject invalid budgets before acquiring the session lock or generating.

    Args:
        runtime:
            Session used to verify validation leaves it ready for another request.

        budget:
            Out-of-range or non-integer repair budget.

    """
    session, _, _, _ = runtime
    with pytest.raises(ValueError, match="max_repair_attempts"):
        list(session.stream("Hi", max_repair_attempts=budget))
    assert "".join(session.stream("Hi")) == "Hello world."


def test_incremental_multiround(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]],
) -> None:
    """Deliver text before inference finishes and preserve independent histories.

    Args:
        runtime:
            Parametrized synthetic model and observation state.

    """
    session, pipeline, tokenizer, _ = runtime
    pipeline.release = threading.Event()
    stream = session.stream("Hi")
    assert next(stream) == "Hello"
    assert session.history == ()
    with pytest.raises(RuntimeError, match="active"):
        session.reset()
    pipeline.release.set()
    assert "Hello" + "".join(stream) == "Hello world."
    pipeline.release = None
    assert "".join(session.stream("Again")) == "Hello world."
    assert [message["role"] for message in tokenizer.messages] == [
        "developer" if isinstance(session.model, FunctionGemma) else "system",
        "user",
        "assistant",
        "user",
    ]
    assert tokenizer.messages[0]["content"] == session.model.config.system_prompt
    other = Session(session.model)
    assert other.history == ()
    session.reset()
    assert session.history == ()


def test_tools_and_native_continuation(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]],
) -> None:
    """Filter split markers and quoted delimiters; execute only when asked.

    Args:
        runtime:
            Parametrized synthetic model and observation state.

    """
    session, pipeline, tokenizer, seen = runtime
    literal = (
        "<end_function_call>"
        if isinstance(session.model, FunctionGemma)
        else "<|tool_call_end|>"
    )
    pipeline.raw = "Checking. " + call(session, literal) + " Done."
    assert "".join(session.stream("Check")) == "Checking.  Done."
    assert seen == []
    calls = session.pending_calls
    calls[0].arguments["value"] = "tampered"
    with pytest.raises(RuntimeError, match="pending"):
        list(session.stream("Another"))
    assert session.invoke_tools() == (False,)
    assert seen == [literal]
    assert session.pending_calls == ()
    with pytest.raises(RuntimeError, match="Continue"):
        list(session.stream("Another"))
    pipeline.raw = "The result is false."
    assert "".join(session.stream()) == pipeline.raw
    result = tokenizer.messages[-1]
    assert result["role"] == "tool" and result["name"] == "record"
    assert result["content"]["result"] is False
    if isinstance(session.model, LFM2):
        assert result["content"]["name"] == "record"
    assert tokenizer.messages[-2]["tool_calls"][0]["function"]["arguments"] == {
        "value": literal
    }


def test_errors_and_abandonment(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]],
) -> None:
    """Preserve history on worker errors, truncation, and early iterator close.

    Args:
        runtime:
            Parametrized synthetic model and observation state.

    """
    session, pipeline, _, _ = runtime
    list(session.stream("Committed"))
    history = session.history
    pipeline.failure = True
    with pytest.raises(RuntimeError, match="backend exploded") as info:
        list(session.stream("Failure"))
    assert info.value.__notes__ == ["fixture diagnostics"]
    assert session.history == history
    pipeline.failure = False
    stream = session.stream("Abandoned")
    next(stream)
    stream.close()
    assert session.history == history
    pipeline.raw = "x" * 1024
    with pytest.raises(ValueError, match="limit"):
        list(session.stream("Truncated"))
    assert session.history == history


def test_batch_validation_and_handler_failure(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]],
) -> None:
    """Reject an invalid later call before dispatch and prevent side-effect retries.

    Args:
        runtime:
            Parametrized synthetic model and observation state.

    """
    session, pipeline, _, seen = runtime
    pipeline.raw = call(session, "ok") + call(session, "bad").replace(
        "record", "unknown"
    )
    with pytest.raises(ValueError, match="Unknown"):
        list(session.stream("Invalid batch"))
    assert session.history == () and seen == [] and session.pending_calls == ()
    pipeline.raw = call(session, "ok") + call(session, "fail")
    list(session.stream("Valid batch"))
    with pytest.raises(LookupError, match="handler exploded"):
        session.invoke_tools()
    assert seen == ["ok", "fail"]
    with pytest.raises(RuntimeError, match="reset"):
        session.invoke_tools()
    with pytest.raises(RuntimeError, match="reset"):
        list(session.stream())
    assert session.history[-1]["role"] == "assistant"
    session.reset()
    assert session.pending_calls == ()


@pytest.mark.parametrize("device", ["CPU", "GPU"])
def test_explicit_device_config(device: Literal["CPU", "GPU"]) -> None:
    """Both model families accept explicit devices; torch rejects GPU execution.

    Args:
        device:
            Explicit supported OpenVINO device.

    """
    assert LLMConfig(Path("unused"), device=device).device == device
    assert LFMConfig(Path("unused"), device=device).device == device
    if device == "GPU":
        with pytest.raises(ValueError, match="CPU only"):
            LLMConfig(Path("unused"), backend="torch", device=device)


@pytest.mark.parametrize("device", ["CPU", "GPU"])
@pytest.mark.parametrize("fail", [False, True])
def test_device_load_without_fallback(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: Literal["CPU", "GPU"],
    fail: bool,
) -> None:
    """Send exactly the selected device and avoid CPU-only properties on GPU.

    Args:
        runtime:
            Parametrized synthetic model family.

        tmp_path:
            Isolated synthetic model artifact directory.

        monkeypatch:
            Replaces expensive loading while exercising runtime property selection.

        device:
            Explicit target device.

        fail:
            Whether compilation should raise without a fallback attempt.

    """
    # Loading libraries here keeps the test module cheap to collect.
    import openvino_genai
    import transformers

    attempts: list[str] = []
    failure = RuntimeError("unsupported device")

    def pipeline(
        path: str, target: str, properties: Any = None, **kwargs: Any
    ) -> Pipeline:
        """Capture compile properties and optionally fail.

        Args:
            path:
                Local model artifact path.

            target:
                Requested OpenVINO device.

            properties:
                Positional compile properties used by FunctionGemma.

            kwargs:
                Keyword compile properties used by LFM.

        """
        attempts.append(target)
        settings = properties if properties is not None else kwargs
        assert ("INFERENCE_NUM_THREADS" in settings) == (device == "CPU")
        if device == "CPU":
            assert settings["INFERENCE_PRECISION_HINT"] == "f32"
        elif isinstance(runtime[0].model, LFM2):
            assert settings["INFERENCE_PRECISION_HINT"] == "f16"
            assert settings["GPU_QUEUE_THROTTLE"] == "LOW"
            assert settings["COMPILATION_NUM_THREADS"] == 1
        else:
            assert "INFERENCE_PRECISION_HINT" not in settings
        if fail:
            raise failure
        return Pipeline()

    monkeypatch.setattr(openvino_genai, "LLMPipeline", pipeline)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer()
    )
    monkeypatch.setattr(
        transformers, "PreTrainedTokenizerFast", lambda **kw: Tokenizer()
    )
    (tmp_path / "openvino_model.xml").touch()
    (tmp_path / "chat_template.jinja").write_text("fixture")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "backend": "tokenizers",
                "extra_special_tokens": [],
                "model_specific_special_tokens": {},
            }
        )
    )
    tools = runtime[0].model.tools
    model = (
        FunctionGemma(LLMConfig(tmp_path, cache_dir=tmp_path, device=device), tools)
        if isinstance(runtime[0].model, FunctionGemma)
        else LFM2(LFMConfig(tmp_path, cache_dir=tmp_path, device=device), tools)
    )
    if fail:
        with pytest.raises(RuntimeError) as info:
            model.load()
        assert info.value is failure
        assert model._model is None and model._tokenizer is None
    else:
        model.load()
        model.close()
    assert attempts == [device]


def test_partial_protocol_never_leaks(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]],
) -> None:
    """Reject every truncated opening-marker boundary without yielding protocol.

    Args:
        runtime:
            Parametrized synthetic model and observation state.

    """
    session, pipeline, _, _ = runtime
    marker = (
        "<start_function_call>"
        if isinstance(session.model, FunctionGemma)
        else "<|tool_call_start|>"
    )
    for end in range(1, len(marker) + 1):
        pipeline.raw = "Safe text. " + marker[:end]
        pieces: list[str] = []
        with pytest.raises(ValueError):
            pieces.extend(session.stream("Broken"))
        assert "".join(pieces) == "Safe text."
    assert session.history == () and session.pending_calls == ()


@pytest.mark.parametrize(
    "invalid", [ToolCall("unknown", {}), ToolCall("record", {"value": 1})]
)
def test_explicit_batch_validation(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]], invalid: ToolCall
) -> None:
    """Validate all explicit calls before committing history or running handlers.

    Args:
        runtime:
            Both native model families with synthetic backends.

        invalid:
            Invalid second call after a valid side-effecting call.

    """
    session, _, tokenizer, seen = runtime
    list(session.stream("Prior context"))
    history = session.history
    messages = tokenizer.messages
    with pytest.raises(ValueError):
        session.request_tools(
            "Invalid", [ToolCall("record", {"value": "first"}), invalid]
        )
    assert session.history == history
    assert not session.pending_calls and not seen
    assert tokenizer.messages is messages
    session.request_tools("Valid", [ToolCall("record", {"value": "valid"})])
    assert session.invoke_tools() == (False,)
    session.complete("Done.")


@pytest.mark.parametrize(
    "text,calls", [(" ", [ToolCall("record", {"value": "a"})]), ("Request", [])]
)
def test_explicit_empty_request(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]],
    text: str,
    calls: list[ToolCall],
) -> None:
    """Reject empty explicit inputs while leaving the session available.

    Args:
        runtime:
            Both native model families.

        text:
            Original user text.

        calls:
            Proposed batch, possibly empty.

    """
    session, _, _, seen = runtime
    with pytest.raises(ValueError):
        session.request_tools(text, calls)
    assert session.history == () and not session.pending_calls and not seen
    assert "".join(session.stream("Still ready")) == "Hello world."


def test_explicit_request_lifecycle(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]],
) -> None:
    """Isolate routed calls and enforce pending, continuation, and reset boundaries.

    Args:
        runtime:
            Both native model families with recording handlers.

    """
    session, pipeline, tokenizer, seen = runtime
    pipeline.failure = True
    calls = [ToolCall("record", {"value": "original"})]
    session.request_tools("Record explicitly", calls)
    calls[0].arguments["value"] = "tampered"
    session.pending_calls[0].arguments["value"] = "also tampered"
    session.history[-1]["tool_calls"][0]["function"]["arguments"]["value"] = (
        "history tamper"
    )
    assert tokenizer.messages == [] and seen == []
    history = session.history
    with pytest.raises(RuntimeError):
        session.request_tools("Again", calls)
    with pytest.raises(RuntimeError, match="pending"):
        list(session.stream("Again"))
    assert session.history == history
    assert session.invoke_tools() == (False,)
    assert seen == ["original"]
    with pytest.raises(RuntimeError):
        session.request_tools("Again", calls)
    with pytest.raises(RuntimeError, match="No pending"):
        session.invoke_tools()
    session.complete("Recorded.")
    session.request_tools("Next", [ToolCall("record", {"value": "next"})])
    session.reset()
    assert session.history == () and session.pending_calls == ()
    pipeline.failure = False
    assert "".join(session.stream("Fresh")) == "Hello world."


@pytest.mark.parametrize("failure", ["exception", "invalid_json"])
def test_transform_failure_requires_reset(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]], failure: str
) -> None:
    """Fail projection atomically after dispatch and forbid retries until reset.

    Args:
        runtime:
            Both model families with side-effect recording handlers.

        failure:
            Exception or nonfinite JSON returned by the second projection.

    """
    session, _, _, seen = runtime
    calls = [ToolCall("record", {"value": value}) for value in ("first", "second")]
    session.request_tools("Batch", calls)
    history = session.history
    projected: list[str] = []
    error = GeneratedCallError("projection failed", "not generated")
    error.add_note("External handlers already completed")

    def transform(tool: ToolCall, result: JsonValue) -> JsonValue:
        """Fail the second result after one successful local projection.

        Args:
            tool:
                Isolated pending call.

            result:
                Isolated raw handler result.

        """
        assert result is False
        projected.append(str(tool.arguments["value"]))
        if len(projected) == 2:
            if failure == "exception":
                raise error
            return float("nan")
        return "Recorded."

    with pytest.raises(
        GeneratedCallError if failure == "exception" else ValueError
    ) as raised:
        session.invoke_tools(transform)
    if failure == "exception":
        assert raised.value is error
        assert error.__notes__ == [
            "Generated output: 'not generated'",
            "External handlers already completed",
        ]
    assert session.history == history
    assert session.pending_calls == tuple(calls)
    assert seen == projected == ["first", "second"]
    with pytest.raises(RuntimeError, match="reset"):
        session.invoke_tools(transform)
    with pytest.raises(RuntimeError, match="reset"):
        list(session.stream(max_repair_attempts=2))
    with pytest.raises(RuntimeError):
        session.complete("Invented success")
    with pytest.raises(RuntimeError):
        session.request_tools("Retry", calls)
    assert seen == projected == ["first", "second"]
    session.reset()
    session.request_tools("Fresh", [ToolCall("record", {"value": "fresh"})])
    assert session.invoke_tools() == (False,)
    session.complete("Done.")
    assert seen == ["first", "second", "fresh"]


@pytest.mark.parametrize("project", [False, True])
def test_result_mutation_isolation(
    runtime: tuple[Session, Pipeline, Tokenizer, list[str]], project: bool
) -> None:
    """Isolate provider objects, projection inputs/outputs, returns, and model history.

    Args:
        runtime:
            Both native model families with synthetic generation.

        project:
            Whether to replace the full JSON result with a projection.

    """
    session, pipeline, tokenizer, _ = runtime
    source: dict[str, JsonValue] = {"details": ["private"]}
    compact: dict[str, JsonValue] = {"answer": ["Recorded."]}

    def handler(args: Arguments) -> JsonValue:
        """Return a retained mutable provider object.

        Args:
            args:
                Validated record arguments.

        """
        assert args.value == "original"
        return source

    def transform(tool: ToolCall, result: JsonValue) -> JsonValue:
        """Mutate isolated projection inputs and return a retained mutable output.

        Args:
            tool:
                Isolated call snapshot.

            result:
                Isolated provider snapshot.

        """
        tool.arguments["value"] = "projection tamper"
        assert isinstance(result, dict)
        result["details"] = ["projection tamper"]
        return compact

    session.model.tools = ToolRegistry([Tool("record", "Record", Arguments, handler)])
    session.request_tools("Record", [ToolCall("record", {"value": "original"})])
    results = session.invoke_tools(transform if project else None)
    assert results == ({"details": ["private"]},)
    expected = {"answer": ["Recorded."]} if project else {"details": ["private"]}
    assert session.history[-1]["content"]["result"] == expected
    assert source == {"details": ["private"]}
    assert isinstance(results[0], dict)
    results[0]["details"] = ["return tamper"]
    source["details"] = ["provider tamper"]
    compact["answer"] = ["output tamper"]
    session.history[-1]["content"]["result"].clear()
    pipeline.raw = "Done."
    assert "".join(session.stream()) == "Done."
    assert tokenizer.messages[-1]["content"]["result"] == expected
    assert tokenizer.messages[-2]["tool_calls"][0]["function"]["arguments"] == {
        "value": "original"
    }
