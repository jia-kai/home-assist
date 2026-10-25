"""Model-free size, ownership, streaming, and reclamation boundary coverage."""

import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from weakref import ReferenceType, ref

import numpy as np
import pytest

import hoast.lfm as lfm_module
import hoast.llm as llm_module
from hoast.lfm import LFM2, LFMConfig
from hoast.llm import FunctionGemma, GeneratedCallError, LLMConfig, ToolRegistry


@dataclass(slots=True)
class Scenario:
    """Small native/tokenizer stand-in controls without model dependencies."""

    kind: str
    """LFM, FunctionGemma/OpenVINO, or FunctionGemma/torch execution path."""

    events: list[str] = field(default_factory=list)
    """Observed generation and reclamation order."""

    arrays: list[ReferenceType[Any]] = field(default_factory=list)
    """Weak references proving temporary arrays/results are released."""

    prompt: str = "prompt"
    """Formatted prompt returned without loading a real template."""

    raw: str = "hello"
    """Decoded raw response including any protocol markers."""

    input_tokens: int = 2
    """Mock prompt token count."""

    output_tokens: int = 2
    """Mock generated token count."""

    failure: str = ""
    """Stage that should raise instead of returning normally."""

    error: BaseException | None = None
    """Optional exact failure instance used to verify exception identity."""


class Inputs(dict[str, Any]):
    """Dictionary with the attribute interface of a tokenizer batch."""

    @property
    def input_ids(self) -> Any:
        """Return the batch-one token array."""
        return self["input_ids"]


@dataclass(slots=True, weakref_slot=True)
class NativeOutput:
    """Minimal owned native generation result."""

    tokens: list[list[int]]
    """Batch-one generated token IDs, shape (1, generated_tokens)."""

    perf_metrics: Any
    """Native token count and millisecond first-token timing interface."""


@dataclass(slots=True)
class FakeTokenizer:
    """Record expensive tokenizer stages and return embedded fixture data."""

    scenario: Scenario
    """Shared controls and event history."""

    pad_token_id: int = 0
    """Torch generation padding ID."""

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        """Return the configured prompt without tokenizing.

        Args:
            args:
                Chat history supplied by the wrapper.

            kwargs:
                Template controls and tool declarations.

        """
        self.scenario.events.append("format")
        return self.scenario.prompt

    def __call__(self, *args: Any, **kwargs: Any) -> Inputs:
        """Allocate a tiny batch-one int64 prompt token array.

        Args:
            args:
                Formatted prompt.

            kwargs:
                Tokenization options including the requested tensor backend.

        """
        self.scenario.events.append("tokenize")
        array = np.zeros((1, self.scenario.input_tokens), dtype=np.int64)
        self.scenario.arrays.append(ref(array))
        return Inputs(input_ids=array)

    def decode(self, *args: Any, **kwargs: Any) -> str:
        """Return fixture text or raise a decode-stage failure.

        Args:
            args:
                Generated token sequence.

            kwargs:
                Protocol-preserving decoder options.

        """
        self.scenario.events.append("decode")
        if self.scenario.failure == "decode":
            if self.scenario.error is not None:
                raise self.scenario.error
            raise RuntimeError("decode failed")
        return self.scenario.raw


@dataclass(slots=True)
class FakeModel:
    """Return tiny outputs using the public native generation interfaces."""

    scenario: Scenario
    """Shared controls and event history."""

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Record native token limits and simulate success or backend failure.

        Args:
            args:
                OpenVINO prompt tensor, if applicable.

            kwargs:
                Native generation controls and torch token arrays.

        """
        scenario = self.scenario
        scenario.events.append("generate")
        assert kwargs["max_new_tokens"] == 8
        if scenario.kind == "torch":
            output = np.zeros(
                (1, scenario.input_tokens + scenario.output_tokens), dtype=np.int64
            )
        else:
            output = NativeOutput(
                [list(range(scenario.output_tokens))],
                SimpleNamespace(
                    get_ttft=lambda: SimpleNamespace(mean=25.0),
                    get_num_generated_tokens=lambda: scenario.output_tokens,
                ),
            )
        scenario.arrays.append(ref(output))
        if scenario.failure == "generate":
            if scenario.error is not None:
                raise scenario.error
            raise RuntimeError("generate failed")
        streamer = kwargs.get("streamer")
        if streamer is not None:
            if scenario.kind == "torch":
                if streamer.decoder is not None:
                    streamer.decoder.callback(scenario.raw)
            else:
                streamer(scenario.raw)
        return output


@pytest.fixture(params=["lfm", "openvino", "torch"])
def harness(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> tuple[Scenario, Any]:
    """Create a loaded model owner using only tiny arrays and fake modules.

    Args:
        request:
            Selects one of the three supported execution paths.

        monkeypatch:
            Replaces optional inference modules, streamers, and reclamation.

    """
    scenario = Scenario(request.param)
    ov = ModuleType("openvino")
    monkeypatch.setattr(ov, "Tensor", lambda array: array, raising=False)
    monkeypatch.setitem(sys.modules, "openvino", ov)
    torch = ModuleType("torch")
    monkeypatch.setattr(torch, "inference_mode", nullcontext, raising=False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    module = lfm_module if scenario.kind == "lfm" else llm_module

    def reclaim(boundary: str) -> None:
        """Record reclamation after ensuring all native temporaries have dropped.

        Args:
            boundary:
                Public model boundary label.

        """
        lock: Any = owner._lock
        assert lock._is_owned()
        assert all(item() is None for item in scenario.arrays)
        scenario.events.append(boundary)

    monkeypatch.setattr(module, "reclaim_memory", reclaim)
    monkeypatch.setattr(module, "_ov_streamer", lambda tokenizer, callback: callback)
    monkeypatch.setattr(
        llm_module,
        "_text_streamer",
        lambda tokenizer, callback, skip_prompt: SimpleNamespace(callback=callback),
    )
    if scenario.kind == "lfm":
        owner = LFM2(
            LFMConfig(Path("unused"), max_new_tokens=8, reclaim_after_call=True),
            ToolRegistry([]),
        )
    else:
        owner = FunctionGemma(
            LLMConfig(
                Path("unused"),
                backend="torch" if scenario.kind == "torch" else "openvino",
                max_new_tokens=8,
                reclaim_after_call=True,
            ),
            ToolRegistry([]),
        )
    owner._model = FakeModel(scenario)
    owner._tokenizer = FakeTokenizer(scenario)
    return scenario, owner


def test_success_and_owned_result(harness: tuple[Scenario, Any]) -> None:
    """Release temporaries before post-call reclamation and preserve the result.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

    """
    scenario, owner = harness
    result = owner.generate("hello")
    prefix = "lfm" if scenario.kind == "lfm" else "functiongemma"
    assert scenario.events == [
        f"{prefix}.before",
        "format",
        "tokenize",
        "generate",
        "decode",
        f"{prefix}.after",
    ]
    assert result.raw == result.text == "hello"
    assert result.calls == ()
    assert result.input_tokens == result.output_tokens == 2
    assert result.elapsed_seconds >= 0
    if scenario.kind != "torch":
        assert result.first_token_seconds == 0.025
    assert isinstance(owner._model, FakeModel)
    assert isinstance(owner._tokenizer, FakeTokenizer)


@pytest.mark.parametrize("limit,accepted", [(6, True), (5, False)])
def test_prompt_character_boundary(
    harness: tuple[Scenario, Any], limit: int, accepted: bool
) -> None:
    """Accept exact formatted character limits and reject before tokenization.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

        limit:
            Configured formatted prompt character budget.

        accepted:
            Whether the six-character prompt fits the budget.

    """
    scenario, owner = harness
    object.__setattr__(owner.config, "max_prompt_chars", limit)
    if accepted:
        assert owner.generate("hello").text == "hello"
    else:
        with pytest.raises(ValueError, match="prompt character"):
            owner.generate("hello")
        assert "tokenize" not in scenario.events
        assert "generate" not in scenario.events
    assert scenario.events[-1].endswith(".after")


@pytest.mark.parametrize("limit,accepted", [(5, True), (4, False)])
def test_raw_character_boundary(
    harness: tuple[Scenario, Any],
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
    accepted: bool,
) -> None:
    """Reject excessive decoded raw text before protocol parsing or validation.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

        monkeypatch:
            Replaces parsing to prove oversized text never reaches it.

        limit:
            Configured decoded raw character budget.

        accepted:
            Whether the five-character response fits the budget.

    """
    scenario, owner = harness
    object.__setattr__(owner.config, "max_output_chars", limit)
    if accepted:
        assert owner.generate("hello").text == "hello"
    else:
        module = lfm_module if scenario.kind == "lfm" else llm_module
        monkeypatch.setattr(module, "parse_response", lambda raw: pytest.fail("parsed"))
        with pytest.raises(ValueError, match="output character"):
            owner.generate("hello")
    assert scenario.events[-1].endswith(".after")


@pytest.mark.parametrize("count", [7, 8, 9])
def test_output_token_boundary(harness: tuple[Scenario, Any], count: int) -> None:
    """Reject reaching or exceeding native token budgets before decoding.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

        count:
            Native generated token count relative to the eight-token budget.

    """
    scenario, owner = harness
    scenario.output_tokens = count
    if count < 8:
        assert owner.generate("hello").output_tokens == count
    else:
        with pytest.raises(ValueError, match="limit"):
            owner.generate("hello")
        assert "decode" not in scenario.events
    assert scenario.events[-1].endswith(".after")


@pytest.mark.parametrize("count,accepted", [(4088, True), (4089, False)])
def test_context_boundary(
    harness: tuple[Scenario, Any], count: int, accepted: bool
) -> None:
    """Reserve the output budget and reject excessive context before generation.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

        count:
            Prompt token count with an eight-token output reservation.

        accepted:
            Whether prompt plus reserved output fits the 4096-token context.

    """
    scenario, owner = harness
    scenario.input_tokens = count
    if accepted:
        assert owner.generate("hello").input_tokens == count
    else:
        with pytest.raises(ValueError, match="context budget"):
            owner.generate("hello")
        assert "generate" not in scenario.events
    assert scenario.events[-1].endswith(".after")


@pytest.mark.parametrize("stage", ["generate", "decode", "parse", "callback"])
def test_failure_reclaims_and_preserves_exception(
    harness: tuple[Scenario, Any], stage: str
) -> None:
    """Clear traceback-held native temporaries and reclaim after every failure.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

        stage:
            Failure stage, including malformed protocol and streaming errors.

    """
    scenario, owner = harness
    scenario.failure = stage
    if stage == "parse":
        scenario.raw = (
            "<|tool_call_start|>" if scenario.kind == "lfm" else "<start_function_call>"
        )

    def callback(text: str) -> None:
        """Abort raw streaming to exercise callback exception cleanup.

        Args:
            text:
                Raw incremental text emitted by the fake backend.

        """
        raise RuntimeError("callback failed")

    expected = GeneratedCallError if stage == "parse" else RuntimeError
    with pytest.raises(expected) as failure:
        owner.generate_messages(
            [{"role": "user", "content": "hello"}],
            on_text=callback if stage == "callback" else None,
        )
    assert failure.value.__traceback__ is not None
    assert scenario.events[0].endswith(".before")
    assert scenario.events[-1].endswith(".after")
    assert all(item() is None for item in scenario.arrays)
    assert owner._model is not None


def _retained_failure(scenario: Scenario, label: str) -> BaseException:
    """Return an exception whose unwound frame retains a tiny native array.

    Args:
        scenario:
            Shared weak references observed by after-generation reclamation.

        label:
            Distinguishing exception message and diagnostic note.

    """
    array = np.zeros(1, dtype=np.int64)
    scenario.arrays.append(ref(array))
    try:
        failure = RuntimeError(label)
        failure.add_note(f"note for {label}")
        raise failure
    except RuntimeError as error:
        return error


@pytest.mark.parametrize("graph", ["cause", "context", "suppressed", "cycle", "group"])
def test_exception_graph_releases_all_native_frames(
    harness: tuple[Scenario, Any], graph: str
) -> None:
    """Clear every chained/group traceback while preserving diagnostic structure.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

        graph:
            Exception topology, including cycles and nested base exception groups.

    """
    scenario, owner = harness
    failures: list[BaseException] = []
    tracebacks: list[Any] = []
    links: list[tuple[BaseException | None, BaseException | None, bool]] = []

    def callback(text: str) -> None:
        """Raise a graph with native allocations retained in leaf tracebacks.

        Args:
            text:
                Raw streaming fragment triggering the synthetic callback failure.

        """
        first = _retained_failure(scenario, "first")
        second = _retained_failure(scenario, "second")
        root: BaseException = RuntimeError("root")
        if graph == "group":
            root = BaseExceptionGroup(
                "root",
                [first, BaseExceptionGroup("nested", [second, KeyboardInterrupt()])],
            )
        elif graph in ("context", "suppressed"):
            root.__context__ = first
            root.__suppress_context__ = graph == "suppressed"
            first.__cause__ = second
        else:
            root.__cause__ = first
            first.__context__ = second
            if graph == "cycle":
                second.__cause__ = root
        root.add_note("original root note")
        failures.extend([root, first, second])
        tracebacks.extend([first.__traceback__, second.__traceback__])
        links.extend(
            (error.__cause__, error.__context__, error.__suppress_context__)
            for error in failures
        )
        raise root

    with pytest.raises(BaseException) as caught:
        owner.generate_messages(
            [{"role": "user", "content": "hello"}], on_text=callback
        )
    assert caught.value is failures[0]
    assert caught.value.__notes__ == ["original root note"]
    for error, traceback, label in zip(
        failures[1:], tracebacks, ("first", "second"), strict=True
    ):
        assert error.__traceback__ is traceback
        assert traceback.tb_frame.f_code.co_name == "_retained_failure"
        assert traceback.tb_lineno > 0
        assert error.__notes__ == [f"note for {label}"]
    assert links == [
        (error.__cause__, error.__context__, error.__suppress_context__)
        for error in failures
    ]
    if isinstance(caught.value, BaseExceptionGroup):
        assert caught.value.exceptions[0] is failures[1]
        nested = caught.value.exceptions[1]
        assert isinstance(nested, BaseExceptionGroup)
        assert nested.exceptions[0] is failures[2]
        assert isinstance(nested.exceptions[1], KeyboardInterrupt)
    assert all(item() is None for item in scenario.arrays)
    assert scenario.events[-1].endswith(".after")


@pytest.mark.parametrize("stage", ["before", "generate", "decode", "callback"])
def test_cleanup_failure_preserves_primary_exception(
    harness: tuple[Scenario, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stage: str,
) -> None:
    """Keep the exact inference failure primary when final reclamation also fails.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

        monkeypatch:
            Installs failing boundary reclamation without altering model execution.

        caplog:
            Captures the cleanup failure's complete diagnostic traceback.

        stage:
            Initial failure stage, including pre-call cleanup and raw callbacks.

    """
    scenario, owner = harness
    module = lfm_module if scenario.kind == "lfm" else llm_module
    original = RuntimeError("primary failure")
    cause = ValueError("primary cause")
    original.__cause__ = cause
    original.add_note("original note")
    scenario.failure = stage
    scenario.error = original
    cleanup = RuntimeError("after cleanup failure")
    original_reclaim = module.reclaim_memory

    def reclaim(boundary: str) -> None:
        """Fail after ownership checks, optionally failing before generation too.

        Args:
            boundary:
                Boundary used to select the primary or secondary failure.

        """
        original_reclaim(boundary)
        if boundary.endswith(".after"):
            raise cleanup
        if stage == "before":
            raise original

    def callback(text: str) -> None:
        """Raise the caller's exact exception instance from raw streaming.

        Args:
            text:
                Raw fragment emitted before final validation.

        """
        raise original

    monkeypatch.setattr(module, "reclaim_memory", reclaim)
    with pytest.raises(RuntimeError) as caught:
        owner.generate_messages(
            [{"role": "user", "content": "hello"}],
            on_text=callback if stage == "callback" else None,
        )
    assert caught.value is original
    assert original.__cause__ is cause
    assert original.__notes__[0] == "original note"
    assert "after cleanup failure" in original.__notes__[1]
    assert any(
        record.exc_info is not None and record.exc_info[1] is cleanup
        for record in caplog.records
    )
    assert all(item() is None for item in scenario.arrays)
    assert scenario.events[-1].endswith(".after")


def test_cleanup_failure_after_success_propagates(
    harness: tuple[Scenario, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Propagate unexpected cleanup failures when there is no primary failure.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

        monkeypatch:
            Installs a failing after-generation reclamation hook.

    """
    scenario, owner = harness
    module = lfm_module if scenario.kind == "lfm" else llm_module
    cleanup = RuntimeError("cleanup after success")
    original_reclaim = module.reclaim_memory

    def reclaim(boundary: str) -> None:
        """Raise after verifying the successful call released native temporaries.

        Args:
            boundary:
                Before/after boundary forwarded to the original assertions.

        """
        original_reclaim(boundary)
        if boundary.endswith(".after"):
            raise cleanup

    monkeypatch.setattr(module, "reclaim_memory", reclaim)
    with pytest.raises(RuntimeError) as caught:
        owner.generate("hello")
    assert caught.value is cleanup
    assert scenario.events[-2] == "decode"
    assert all(item() is None for item in scenario.arrays)


def test_streaming_contract(harness: tuple[Scenario, Any]) -> None:
    """Continue forwarding raw protocol text synchronously before result return.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

    """
    scenario, owner = harness
    fragments: list[str] = []
    result = owner.generate_messages(
        [{"role": "user", "content": "hello"}], on_text=fragments.append
    )
    assert fragments == [scenario.raw] == [result.raw]


def test_default_disabled_and_unloaded(harness: tuple[Scenario, Any]) -> None:
    """Standalone inference skips cleanup by default; unloaded calls do no work.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

    """
    scenario, owner = harness
    assert type(owner.config)(Path("unused")).reclaim_after_call is False
    object.__setattr__(owner.config, "reclaim_after_call", False)
    owner.generate("hello")
    assert scenario.events == ["format", "tokenize", "generate", "decode"]
    scenario.events.clear()
    object.__setattr__(owner.config, "reclaim_after_call", True)
    owner._model = None
    with pytest.raises(RuntimeError, match="Load"):
        owner.generate("hello")
    assert scenario.events == []


def test_validation_failure_reclaims(harness: tuple[Scenario, Any]) -> None:
    """Reclaim on rejected history without reaching formatting or tokenization.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

    """
    scenario, owner = harness
    with pytest.raises(ValueError, match="history"):
        owner.generate_messages([])
    prefix = "lfm" if scenario.kind == "lfm" else "functiongemma"
    assert scenario.events == [f"{prefix}.before", f"{prefix}.after"]


def test_cleanup_excluded_from_result_timing(
    harness: tuple[Scenario, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep externally measured cleanup latency outside generation timings.

    Args:
        harness:
            Loaded offline owner and native fixture controls.

        monkeypatch:
            Installs a deterministic clock and two-second cleanup stand-in.

    """
    scenario, owner = harness
    module = lfm_module if scenario.kind == "lfm" else llm_module
    clock = [0.0]
    original_reclaim = module.reclaim_memory

    def reclaim(boundary: str) -> None:
        """Advance the deterministic wall clock for each reclamation boundary.

        Args:
            boundary:
                Boundary label forwarded to ownership and lock assertions.

        """
        original_reclaim(boundary)
        clock[0] += 2.0

    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(module, "reclaim_memory", reclaim)
    result = owner.generate("hello")
    assert clock[0] == 4.0
    assert result.elapsed_seconds == 0.0


@pytest.mark.parametrize("config_type", [LFMConfig, LLMConfig])
@pytest.mark.parametrize(
    "name",
    [
        "threads",
        "max_new_tokens",
        "max_context_tokens",
        "max_prompt_chars",
        "max_output_chars",
    ],
)
@pytest.mark.parametrize("value", [True, False, 0, -1, 2.5, "8", None])
def test_invalid_integer_budgets(config_type: Any, name: str, value: Any) -> None:
    """Reject coercible values, booleans, and nonpositive integer budgets.

    Args:
        config_type:
            Model configuration class under test.

        name:
            Integer setting requiring strict positive validation.

        value:
            Invalid runtime setting supplied without type coercion.

    """
    with pytest.raises(ValueError, match="positive integer"):
        config_type(Path("unused"), **{name: value})


@pytest.mark.parametrize("config_type", [LFMConfig, LLMConfig])
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_invalid_reclamation_setting(config_type: Any, value: Any) -> None:
    """Require an actual boolean for boundary reclamation.

    Args:
        config_type:
            Model configuration class under test.

        value:
            Invalid reclamation control supplied at runtime.

    """
    with pytest.raises(ValueError, match="must be a bool"):
        config_type(Path("unused"), reclaim_after_call=value)
