"""CPU-only generation contract tests with deterministic runner/tokenizer fixtures."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from transformers import RepetitionPenaltyLogitsProcessor

from hoast.lfm_gpu.generation import VOCAB_SIZE, GenerationConfig, GPUGeneration
from hoast.llm import GeneratedCallError, Tool, ToolArguments, ToolRegistry


class Tokenizer:
    """Small native-tokenizer stand-in recording the rendered history."""

    history: Any = None
    options: Any = None
    raw: str = "Hello<|im_end|>"
    eos_token_id: int = 7
    token_ids: tuple[int, ...] = (1, 2)

    def apply_chat_template(self, history: Any, **options: Any) -> str:
        """Record template arguments and return a fixed prompt.

        Args:
            history: Native structured messages.
            **options: Template options and declarations.
        """
        self.history, self.options = history, options
        return "prompt"

    def __call__(self, prompt: str, **options: Any) -> Any:
        """Return one small prompt token sequence.

        Args:
            prompt: Formatted prompt.
            **options: Tokenization options.
        """
        assert prompt == "prompt"
        assert options == {"add_special_tokens": False, "return_tensors": "np"}
        return SimpleNamespace(input_ids=np.array([self.token_ids], dtype=np.int64))

    def decode(self, tokens: list[int], **options: Any) -> str:
        """Return configured raw output while checking decoding options.

        Args:
            tokens: Generated token IDs including EOS.
            **options: Special-token and spacing options.
        """
        assert tokens[-1] == 7
        assert options == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return self.raw


def scores(**values: float) -> np.ndarray:
    """Construct finite vocabulary logits from stringified token IDs.

    Args:
        **values: Token IDs mapped to their scores.
    """
    result = np.full(VOCAB_SIZE, -100.0, dtype=np.float32)
    for token, value in values.items():
        result[int(token)] = value
    return result


class FakeRunner:
    """Record each fresh sequence and yield scripted logits."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        """Initialize fixed output steps and a small capacity.

        Args:
            outputs: Final-position logit arrays for prefill and decode.
        """
        self.outputs = outputs
        self.workspace = SimpleNamespace(
            spec=SimpleNamespace(capacity=32, prefix_cache=True)
        )
        self.sequences: list[list[int]] = []
        self.step = 0
        self.fail = False
        self.prefix: tuple[int, ...] | None = None
        self.cache_calls = 0
        self.restore_calls = 0
        self.fail_restore = False
        self.fail_reset = False
        self.fail_decode = False
        self.execution_failure: BaseException | None = None
        self.failure_operation: str | None = None

    def reset(self) -> None:
        """Start an independent recorded sequence."""
        if self.fail_reset:
            self.fail_reset = False
            self.prefix = None
            raise RuntimeError("reset failure")
        self.sequences.append([])
        self.step = 0

    def _maybe_fail(self, operation: str) -> None:
        """Simulate runner execution failure and snapshot invalidation.

        Args:
            operation: Runner operation currently executing.
        """
        if self.failure_operation == operation:
            error = self.execution_failure
            assert error is not None
            self.failure_operation = None
            self.prefix = None
            raise error

    def prefill(self, token_ids: list[int]) -> np.ndarray:
        """Record a prompt and return prefill logits.

        Args:
            token_ids: Complete prompt IDs.
        """
        self._maybe_fail("prefill")
        self.sequences[-1].extend(token_ids)
        if self.fail:
            self.fail = False
            raise RuntimeError("fixture failure")
        return self.outputs[0]

    def cache_prefix(self, token_ids: list[int]) -> np.ndarray:
        """Save a simulated snapshot.

        Args:
            token_ids: Stable prefix IDs.
        """
        self.cache_calls += 1
        self.prefix = tuple(token_ids)
        self.sequences[-1].extend(token_ids)
        return self.outputs[0]

    def prefill_cached(self, token_ids: list[int]) -> np.ndarray:
        """Restore an exact prefix and record the full resulting prompt.

        Args:
            token_ids: Complete prompt IDs.
        """
        self.restore_calls += 1
        self._maybe_fail("prefill_cached")
        if self.fail_restore:
            self.fail_restore = False
            self.prefix = None
            raise RuntimeError("restore failure")
        assert (
            self.prefix is not None
            and tuple(token_ids[: len(self.prefix)]) == self.prefix
        )
        self.sequences[-1].extend(token_ids)
        return self.outputs[0]

    def decode(self, token_id: int) -> np.ndarray:
        """Record one selected token and return next logits.

        Args:
            token_id: Selected generated token.
        """
        self._maybe_fail("decode")
        if self.fail_decode:
            self.fail_decode = False
            self.prefix = None
            raise RuntimeError("decode failure")
        self.sequences[-1].append(token_id)
        self.step += 1
        return self.outputs[self.step]


class Args(ToolArguments):
    """Strict fixture tool argument."""

    value: int
    """Required integer value."""


def handler(arguments: Args) -> str:
    """Fail if generation ever dispatches a tool.

    Args:
        arguments: Validated arguments, not used in generation.
    """
    raise AssertionError("tool dispatched")


def make_adapter(
    outputs: list[np.ndarray], **config: Any
) -> tuple[GPUGeneration, FakeRunner, Tokenizer]:
    """Build an adapter with isolated fixture dependencies.

    Args:
        outputs: Scripted logits.
        **config: Overrides to the generation limits.
    """
    runner = FakeRunner(outputs)
    tokenizer = Tokenizer()
    tools = ToolRegistry([Tool("run", "Fixture.", Args, handler)])
    return (
        GPUGeneration(
            runner,
            tokenizer,
            tools,
            GenerationConfig(max_context_tokens=32, max_new_tokens=4, **config),
        ),
        runner,
        tokenizer,
    )


def test_penalty_template_fresh_sequences_and_validation() -> None:
    """Penalize prompt and output IDs once, retain EOS, and never dispatch."""
    adapter, runner, tokenizer = make_adapter(
        [
            scores(**{"1": 10.0, "3": 9.8}),
            scores(**{"3": 10.0, "4": 9.8}),
            scores(**{"7": 10.0}),
        ]
    )
    tokenizer.raw = "<|tool_call_start|>[run(value=2)]<|tool_call_end|><|im_end|>"
    for _ in range(2):
        result = adapter.generate("Hello — world!")
        assert result.calls[0].arguments == {"value": 2}
        assert result.input_tokens == 2 and result.output_tokens == 3
    assert runner.sequences == [[1, 2, 3, 4], [1, 2, 3, 4]]
    assert tokenizer.history[1]["content"] == "Hello world!"
    assert tokenizer.history[0]["role"] == "system"
    assert tokenizer.options["tools"][0]["name"] == "run"
    assert tokenizer.options["add_generation_prompt"] is True


def test_repetition_matches_transformers_oracle() -> None:
    """Match the CPU oracle for negative scores and repeated prompt/output IDs."""
    first = scores(**{"1": -1.0, "3": -1.04})
    second = scores(**{"3": -1.0, "4": -1.04})
    final = scores(**{"7": 1.0})
    adapter, runner, _ = make_adapter([first, second, final])
    processor = RepetitionPenaltyLogitsProcessor(penalty=1.05)
    history = [1, 2]
    expected = []
    for logits in (first, second):
        adjusted = processor(
            torch.tensor([history]),  # pyright: ignore[reportArgumentType]
            torch.from_numpy(logits.copy()).unsqueeze(0),  # pyright: ignore[reportArgumentType]
        )
        token = int(adjusted.argmax(dim=-1).item())
        history.append(token)
        expected.append(token)
    adapter.generate("hello")
    assert runner.sequences == [[1, 2, *expected]]
    assert expected == [3, 4]


def test_failures_and_recovery() -> None:
    """Reject malformed calls, invalid logits, and length exhaustion; reset after failure."""
    adapter, runner, tokenizer = make_adapter([scores(**{"7": 1})])
    runner.fail = True
    with pytest.raises(RuntimeError, match="fixture failure"):
        adapter.generate("hello")
    tokenizer.raw = "<|tool_call_start|>[unknown()]<|tool_call_end|>"
    with pytest.raises(GeneratedCallError) as error:
        adapter.generate("hello")
    assert error.value.raw == tokenizer.raw
    tokenizer.raw = "Hello<|im_end|>"
    assert adapter.generate("hello").text == "Hello"
    runner.outputs = [scores(**{"3": 1})] * 4
    with pytest.raises(ValueError, match="output limit"):
        adapter.generate("hello")
    runner.outputs = [np.full(VOCAB_SIZE, np.nan, dtype=np.float32)]
    with pytest.raises(ValueError, match="invalid logits"):
        adapter.generate("hello")


def test_limits_and_invalid_history() -> None:
    """Reject context, prompt, output and unsupported history before accepting results."""
    adapter, runner, _ = make_adapter(
        [scores(**{"7": 1})], max_prompt_chars=5, max_output_chars=3
    )
    with pytest.raises(ValueError, match="prompt character"):
        adapter.generate("hello")
    adapter.config = GenerationConfig(
        max_context_tokens=32, max_new_tokens=4, max_output_chars=3
    )
    with pytest.raises(ValueError, match="output character"):
        adapter.generate("hello")
    with pytest.raises(ValueError, match="history"):
        adapter.generate_messages([{"role": "system", "content": "x"}])
    runner.workspace.spec.capacity = 5
    with pytest.raises(ValueError, match="context budget"):
        adapter.generate("hello")
    assert len(runner.sequences) == 1


def test_explicit_prefix_reuse_mismatch_and_restore_failure() -> None:
    """Reuse only exact prefix matches; mismatch preserves it and failure rebuilds it."""
    plain, runner, tokenizer = make_adapter(
        [scores(**{"1": 10, "3": 9.8}), scores(**{"7": 1})]
    )
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,))
    first = adapter.generate("hello")
    assert first.output_tokens == 2
    assert runner.sequences[-1] == [1, 2, 3]
    assert (runner.cache_calls, runner.restore_calls) == (1, 1)
    adapter.generate("hello again")
    assert (runner.cache_calls, runner.restore_calls) == (1, 2)
    tokenizer.token_ids = (2, 1)
    adapter.generate("mismatch")
    assert (runner.cache_calls, runner.restore_calls) == (1, 2)
    assert runner.prefix == (1,)
    tokenizer.token_ids = (1,)
    adapter.generate("exact")
    assert (runner.cache_calls, runner.restore_calls) == (1, 3)
    runner.fail_restore = True
    with pytest.raises(RuntimeError, match="restore failure"):
        adapter.generate("failure")
    adapter.generate("recovery")
    assert (runner.cache_calls, runner.restore_calls) == (2, 5)


def test_snapshot_rebuilds_after_decode_or_reset_failure() -> None:
    """Rebuild snapshots invalidated by failed decode or reset."""
    plain, runner, tokenizer = make_adapter([scores(**{"3": 1}), scores(**{"7": 1})])
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,))
    adapter.generate("hello")
    runner.fail_decode = True
    with pytest.raises(RuntimeError, match="decode failure"):
        adapter.generate("hello")
    assert runner.prefix is None
    adapter.generate("hello")
    assert runner.cache_calls == 2
    runner.fail_reset = True
    with pytest.raises(RuntimeError, match="reset failure"):
        adapter.generate("hello")
    adapter.generate("hello")
    assert runner.cache_calls == 3


@pytest.mark.parametrize("operation", ["prefill", "prefill_cached", "decode"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_execution_baseexception_rebuilds_snapshot(
    operation: str, error_type: type[BaseException]
) -> None:
    """Forget a snapshot invalidated by any runner execution exception.

    Args:
        operation: Runner operation that invalidates the snapshot.
        error_type: Ordinary or control-flow exception propagated unchanged.
    """
    plain, runner, tokenizer = make_adapter([scores(**{"3": 1}), scores(**{"7": 1})])
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,))
    adapter.generate("initial")
    if operation == "prefill":
        tokenizer.token_ids = (2, 1)
    error = error_type("execution failure")
    runner.failure_operation = operation
    runner.execution_failure = error
    with pytest.raises(error_type) as raised:
        adapter.generate("failure")
    assert raised.value is error
    assert runner.prefix is None
    tokenizer.token_ids = (1, 2)
    adapter.generate("recovery")
    assert runner.prefix == (1,)
    assert runner.cache_calls == 2


def test_validation_failure_keeps_snapshot() -> None:
    """Non-runner parse and input errors leave an established snapshot reusable."""
    plain, runner, tokenizer = make_adapter([scores(**{"7": 1})])
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,))
    adapter.generate("initial")
    tokenizer.raw = "<|tool_call_start|>[unknown()]<|tool_call_end|>"
    with pytest.raises(GeneratedCallError):
        adapter.generate("invalid output")
    with pytest.raises(ValueError, match="history"):
        adapter.generate_messages([{"role": "system", "content": "x"}])
    tokenizer.raw = "Hello<|im_end|>"
    adapter.generate("recovery")
    assert runner.cache_calls == 1
    assert runner.prefix == (1,)


def test_strict_prefix_requires_initialization_and_rolls_back() -> None:
    """Strict requests use only the initialized snapshot and restore it on exit."""
    plain, runner, tokenizer = make_adapter([scores(**{"3": 1}), scores(**{"7": 1})])
    adapter = GPUGeneration(
        runner, tokenizer, plain.tools, plain.config, (1,), strict_prefix_cache=True
    )
    with pytest.raises(RuntimeError, match="not initialized"):
        adapter.generate("miss")
    assert not runner.sequences
    adapter.initialize_prefix()
    assert runner.sequences[-1] == [1]
    with pytest.raises(RuntimeError, match="already initialized"):
        adapter.initialize_prefix()
    for _ in range(2):
        adapter.generate("hit")
        assert runner.sequences[-1] == [1]
    assert runner.cache_calls == 1
    tokenizer.token_ids = (2, 1)
    before = len(runner.sequences)
    with pytest.raises(ValueError, match="does not match"):
        adapter.generate("miss")
    assert len(runner.sequences) == before
    assert runner.sequences[-1] == [1]
    tokenizer.token_ids = (1, 2)
    tokenizer.raw = "<|tool_call_start|>[unknown()]<|tool_call_end|>"
    with pytest.raises(GeneratedCallError):
        adapter.generate("invalid output")
    assert runner.sequences[-1] == [1]
    tokenizer.raw = "Hello<|im_end|>"
    assert adapter.generate("next").text == "Hello"
    assert runner.cache_calls == 1


@pytest.mark.parametrize("nested", ["generate", "messages", "initialize"])
@pytest.mark.parametrize("caught", [True, False])
def test_strict_reentrant_decode_preserves_rollback(nested: str, caught: bool) -> None:
    """Reject nested operations without suppressing the outer request rollback.

    Args:
        nested: Public operation attempted by the tokenizer callback.

        caught: Whether the callback catches the rejection.
    """
    plain, runner, tokenizer = make_adapter([scores(**{"3": 1}), scores(**{"7": 1})])
    adapter = GPUGeneration(
        runner, tokenizer, plain.tools, plain.config, (1,), strict_prefix_cache=True
    )
    adapter.initialize_prefix()
    original = tokenizer.decode

    def reentrant_decode(tokens: list[int], **options: Any) -> str:
        """Invoke a nested operation after the outer request modifies runner state.

        Args:
            tokens: Generated token IDs.

            **options: Native tokenizer decoding options.
        """

        def invoke() -> None:
            """Attempt an operation, including invalid request preflight inputs."""
            if nested == "generate":
                adapter.generate("")
            elif nested == "messages":
                adapter.generate_messages([])
            else:
                adapter.initialize_prefix()

        if caught:
            with pytest.raises(RuntimeError, match="Reentrant"):
                invoke()
        else:
            invoke()
        return original(tokens, **options)

    tokenizer.decode = reentrant_decode  # type: ignore[method-assign]
    if caught:
        assert adapter.generate("outer").text == "Hello"
    else:
        with pytest.raises(RuntimeError, match="Reentrant"):
            adapter.generate("outer")
    assert runner.sequences[-1] == [1]
    assert not adapter._strict_faulted
    tokenizer.decode = original  # type: ignore[method-assign]
    assert adapter.generate("recovery").text == "Hello"
    assert runner.sequences[-1] == [1]


def test_reentrant_initialization_and_preflight() -> None:
    """Reject callbacks during initialization and preflight without corrupting state."""
    plain, runner, tokenizer = make_adapter([scores(**{"7": 1})])
    adapter = GPUGeneration(
        runner, tokenizer, plain.tools, plain.config, (1,), strict_prefix_cache=True
    )
    original_cache = runner.cache_prefix

    def reentrant_cache(token_ids: list[int]) -> np.ndarray:
        """Attempt a request and initialization while saving the snapshot.

        Args:
            token_ids: Prefix token IDs.
        """
        with pytest.raises(RuntimeError, match="Reentrant"):
            adapter.generate("nested")
        with pytest.raises(RuntimeError, match="Reentrant"):
            adapter.initialize_prefix()
        return original_cache(token_ids)

    runner.cache_prefix = reentrant_cache  # type: ignore[method-assign]
    adapter.initialize_prefix()
    original_template = tokenizer.apply_chat_template

    def reentrant_template(history: Any, **options: Any) -> str:
        """Attempt initialization while rendering the request.

        Args:
            history: Template messages.

            **options: Template options.
        """
        with pytest.raises(RuntimeError, match="Reentrant"):
            adapter.initialize_prefix()
        return original_template(history, **options)

    tokenizer.apply_chat_template = reentrant_template  # type: ignore[method-assign]
    assert adapter.generate("outer").text == "Hello"
    assert runner.sequences[-1] == [1]


def test_reentrant_rollback_callback() -> None:
    """Keep the operation guard active until strict rollback completes."""
    plain, runner, tokenizer = make_adapter([scores(**{"7": 1})])
    adapter = GPUGeneration(
        runner, tokenizer, plain.tools, plain.config, (1,), strict_prefix_cache=True
    )
    adapter.initialize_prefix()
    original_restore = runner.prefill_cached

    def reentrant_restore(token_ids: list[int]) -> np.ndarray:
        """Reject nested requests specifically during prefix restoration.

        Args:
            token_ids: Full prompt or rollback prefix IDs.
        """
        if token_ids == [1]:
            with pytest.raises(RuntimeError, match="Reentrant"):
                adapter.generate("nested")
        return original_restore(token_ids)

    runner.prefill_cached = reentrant_restore  # type: ignore[method-assign]
    assert adapter.generate("outer").text == "Hello"
    assert runner.sequences[-1] == [1]


@pytest.mark.parametrize("operation", ["reset", "prefill_cached", "decode"])
def test_strict_runner_failure_faults_without_rebuild(operation: str) -> None:
    """Device failures propagate and cannot trigger implicit snapshot rebuilds.

    Args:
        operation: Runner operation to fail during the request.
    """
    plain, runner, tokenizer = make_adapter([scores(**{"3": 1}), scores(**{"7": 1})])
    adapter = GPUGeneration(
        runner, tokenizer, plain.tools, plain.config, (1,), strict_prefix_cache=True
    )
    adapter.initialize_prefix()
    if operation == "reset":
        runner.fail_reset = True
    else:
        runner.failure_operation = operation
        runner.execution_failure = RuntimeError("device failure")
    with pytest.raises(RuntimeError, match="failure"):
        adapter.generate("fail")
    with pytest.raises(RuntimeError, match="invalid after runner failure"):
        adapter.generate("retry")
    with pytest.raises(RuntimeError, match="invalid after runner failure"):
        adapter.initialize_prefix()
    assert runner.cache_calls == 1


def test_strict_initialization_and_nondevice_failure() -> None:
    """Initialization faults loudly; invalid output logits roll back safely."""
    plain, runner, tokenizer = make_adapter([scores(**{"7": 1})])
    with pytest.raises(ValueError, match="requires a stable prefix"):
        GPUGeneration(
            runner, tokenizer, plain.tools, plain.config, strict_prefix_cache=True
        )
    adapter = GPUGeneration(
        runner, tokenizer, plain.tools, plain.config, (1,), strict_prefix_cache=True
    )
    runner.fail_reset = True
    with pytest.raises(RuntimeError, match="reset failure"):
        adapter.initialize_prefix()
    with pytest.raises(RuntimeError, match="invalid after runner failure"):
        adapter.initialize_prefix()
    assert runner.cache_calls == 0

    plain, runner, tokenizer = make_adapter(
        [np.full(VOCAB_SIZE, np.nan, dtype=np.float32)]
    )
    adapter = GPUGeneration(
        runner, tokenizer, plain.tools, plain.config, (1,), strict_prefix_cache=True
    )
    adapter.initialize_prefix()
    with pytest.raises(ValueError, match="invalid logits"):
        adapter.generate("invalid")
    assert runner.sequences[-1] == [1]
    runner.outputs = [scores(**{"7": 1})]
    assert adapter.generate("valid").text == "Hello"
    assert runner.cache_calls == 1


def test_strict_rollback_failure_and_mutation() -> None:
    """Rollback failure faults the adapter; configuration changes are rejected."""
    plain, runner, tokenizer = make_adapter([scores(**{"7": 1})])
    adapter = GPUGeneration(
        runner, tokenizer, plain.tools, plain.config, (1,), strict_prefix_cache=True
    )
    adapter.initialize_prefix()
    adapter.stable_prefix = (2,)
    with pytest.raises(RuntimeError, match="mutated"):
        adapter.generate("mutated")
    adapter.stable_prefix = (1,)
    adapter.strict_prefix_cache = False
    with pytest.raises(RuntimeError, match="mode was mutated"):
        adapter.generate("mutated mode")
    adapter.strict_prefix_cache = True
    original = runner.prefill_cached
    calls = 0

    def failing_rollback(token_ids: list[int]) -> np.ndarray:
        """Fail the second restore, after successful request prefill.

        Args:
            token_ids: Full prompt or rollback prefix.
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            runner.prefix = None
            raise RuntimeError("rollback failure")
        return original(token_ids)

    runner.prefill_cached = failing_rollback  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="rollback failure"):
        adapter.generate("fail rollback")
    with pytest.raises(RuntimeError, match="invalid after runner failure"):
        adapter.generate("retry")
    assert runner.cache_calls == 1


def test_prefix_validation_and_budget() -> None:
    """Require enabled storage, legal immutable IDs, and the normal full context budget."""
    plain, runner, tokenizer = make_adapter([scores(**{"7": 1})])
    for prefix in ((), (-1,), (True,), (VOCAB_SIZE,)):
        with pytest.raises(ValueError, match="Stable prefix"):
            GPUGeneration(runner, tokenizer, plain.tools, plain.config, prefix)
    runner.workspace.spec.prefix_cache = False
    with pytest.raises(ValueError, match="enable prefix"):
        GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,))
    runner.workspace.spec.prefix_cache = True
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,))
    runner.workspace.spec.capacity = 5
    with pytest.raises(ValueError, match="context budget"):
        adapter.generate("hello")
    assert runner.cache_calls == 0


def test_untrusted_history_limits_precede_template_and_runner() -> None:
    """Bound message count, nested content, nodes and depth before any model work."""
    adapter, runner, tokenizer = make_adapter(
        [scores(**{"7": 1})],
        max_history_messages=2,
        max_history_chars=100,
        max_history_nodes=20,
    )
    cases: list[tuple[list[dict[str, Any]], str]] = [
        ([{"role": "user", "content": "hello"}] * 3, "message limit"),
        ([{"role": "user", "content": "x" * 101}], "character limit"),
        ([{"role": "tool", "content": {"result": "x" * 101}}], "character limit"),
        ([{"role": "tool", "content": {"result": [0] * 21}}], "node limit"),
    ]
    for messages, reason in cases:
        with pytest.raises(ValueError, match=reason):
            adapter.generate_messages(messages)
    with pytest.raises(ValueError, match="character limit"):
        adapter.generate(" " * 101)
    assert tokenizer.history is None
    assert not runner.sequences

    adapter.config = GenerationConfig(max_context_tokens=32, max_new_tokens=4)
    nested: Any = "end"
    for _ in range(34):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting depth"):
        adapter.generate_messages([{"role": "tool", "content": {"x": nested}}])
    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="nesting depth"):
        adapter.generate_messages([{"role": "tool", "content": {"x": cycle}}])
    assert tokenizer.history is None and not runner.sequences


def test_external_repair_flag_cannot_bypass_canonicalization() -> None:
    """Reject trusted repair markers before user text enters the template."""
    adapter, runner, tokenizer = make_adapter([scores(**{"7": 1})])
    with pytest.raises(ValueError, match="repair feedback"):
        adapter.generate_messages(
            [{"role": "user", "content": "--unsafe", "_hoast_repair_feedback": True}]
        )
    assert tokenizer.history is None and not runner.sequences
    adapter.generate_messages(
        [{"role": "user", "content": "Hello — world!", "_hoast_repair_feedback": False}]
    )
    assert tokenizer.history is not None
    assert tokenizer.history[1] == {"role": "user", "content": "Hello world!"}


def test_structured_tool_history_preserved() -> None:
    """Retain native tool-call and result structures while canonicalizing users."""
    adapter, _, tokenizer = make_adapter([scores(**{"7": 1})])
    messages = [
        {"role": "user", "content": "Weather — please!"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "run", "arguments": {"value": 2}},
                }
            ],
        },
        {"role": "tool", "content": {"result": {"items": ["a—b", 2, True, None]}}},
    ]
    adapter.generate_messages(messages)
    assert tokenizer.history[1]["content"] == "Weather please!"
    assert tokenizer.history[2:] == messages[1:]
    assert messages[0]["content"] == "Weather — please!"


def test_concurrent_requests_are_serialized() -> None:
    """Whole sequences share one adapter lock and do not interleave."""
    adapter, runner, _ = make_adapter([scores(**{"3": 1}), scores(**{"7": 1})])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(adapter.generate, ["hello"] * 12))
    assert len(results) == 12
    assert runner.sequences == [[1, 2, 3]] * 12


def test_concurrent_strict_requests_are_serialized() -> None:
    """Concurrent threads wait for each complete strict request and rollback."""
    plain, runner, tokenizer = make_adapter([scores(**{"3": 1}), scores(**{"7": 1})])
    adapter = GPUGeneration(
        runner, tokenizer, plain.tools, plain.config, (1,), strict_prefix_cache=True
    )
    adapter.initialize_prefix()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(adapter.generate, ["hello"] * 12))
    assert len(results) == 12
    assert runner.sequences == [[1]] + [[1, 2, 3], [1]] * 12
