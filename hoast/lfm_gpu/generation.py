"""Generation over an exclusively owned LFM GPU runner.

The caller supplies the native checkpoint tokenizer and a configured Runner; this
module does not load artifacts, create a GPU queue, dispatch tools, or integrate
with the service. The caller must not use the runner or its queue concurrently,
including through another adapter. GPU arithmetic may differ from OpenVINO.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..input_text import REPAIR_FEEDBACK_KEY, canonical_user_messages
from ..lfm import SYSTEM_PROMPT, parse_response, tool_declarations
from ..llm import GeneratedCallError, Generation, ToolRegistry, _text_streamer

EOS_ID = 7
VOCAB_SIZE = 65536


class RunnerProtocol(Protocol):
    """The Runner operations needed by the adapter, also usable by fixtures."""

    workspace: Any
    """Runner workspace exposing its fixed specification and capacity."""

    def reset(self) -> None:
        """Reset the live sequence before a new prompt."""
        ...

    def prefill(self, token_ids: list[int]) -> np.ndarray:
        """Return last-position logits for a fresh prompt.

        Args:
            token_ids:
                Prompt token IDs.
        """
        ...

    def cache_prefix(self, token_ids: list[int], *, slot: int = 0) -> np.ndarray:
        """Compute and snapshot a prefix from empty state.

        Args:
            token_ids:
                Nonempty stable prefix IDs.

            slot:
                Snapshot index.
        """
        ...

    def prefill_cached(self, token_ids: list[int], *, slot: int = 0) -> np.ndarray:
        """Restore a snapshot and prefill a matching complete prompt.

        Args:
            token_ids:
                Complete prompt starting with the saved prefix.

            slot:
                Snapshot index.
        """
        ...

    def decode(self, token_id: int) -> np.ndarray:
        """Append a generated token and return last-position logits.

        Args:
            token_id:
                Generated token ID to append.
        """
        ...


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Fixed deterministic output and input limits for the standalone adapter."""

    max_new_tokens: int = 128
    """Output token limit, including EOS; reaching the limit is rejected."""

    max_context_tokens: int = 4096
    """Maximum prompt plus reserved output tokens, at most 32768."""

    max_prompt_chars: int = 32768
    """Maximum formatted prompt length in characters."""

    max_output_chars: int = 8192
    """Maximum decoded output length in characters."""

    system_prompt: str = SYSTEM_PROMPT
    """System instruction inserted ahead of the supplied history."""

    max_history_messages: int = 128
    """Maximum number of caller-supplied history messages."""

    max_history_chars: int = 32768
    """Maximum total characters in history strings and mapping keys."""

    max_history_nodes: int = 8192
    """Maximum history values and mapping keys visited during validation."""

    repetition_penalty: float = 1.05
    """Positive repetition penalty applied to previously seen tokens."""

    def __post_init__(self) -> None:
        """Reject invalid limits before any runner operations."""
        for name in (
            "max_new_tokens",
            "max_context_tokens",
            "max_prompt_chars",
            "max_output_chars",
            "max_history_messages",
            "max_history_chars",
            "max_history_nodes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.max_new_tokens < self.max_context_tokens <= 32768:
            raise ValueError("Invalid context/output budget")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("System prompt must not be empty")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("Repetition penalty must be finite and positive")


def _validate_history(
    messages: Sequence[Mapping[str, Any]],
    config: GenerationConfig,
    allow_repair: bool = False,
) -> None:
    """Bound and validate untrusted structured history before rendering.

    Traverses at most the configured node budget, including mapping keys; string
    lengths are counted without copying. JSON-like lists, tuples, and mappings
    are accepted up to 32 nested levels. Integers are limited to 4096 bits and
    floats must be finite. Cycles are rejected by the nesting bound.

    Args:
        messages:
            Caller-owned history, treated as untrusted data.

        config:
            Per-request history limits.

        allow_repair:
            Whether to accept internally constructed repair metadata.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TypeError("History must be a sequence of message mappings")
    if not messages or len(messages) > config.max_history_messages:
        raise ValueError("History must be nonempty and within the message limit")
    nodes = 0
    chars = 0

    def visit(value: Any, depth: int) -> None:
        """Count one bounded JSON-like value and recursively inspect containers.

        Args:
            value:
                Untrusted history value or mapping key.

            depth:
                Nesting depth below the message mapping.
        """
        nonlocal nodes, chars
        nodes += 1
        if nodes > config.max_history_nodes:
            raise ValueError("History exceeds configured node limit")
        if depth > 32:
            raise ValueError("History exceeds maximum nesting depth")
        if isinstance(value, str):
            chars += len(value)
            if chars > config.max_history_chars:
                raise ValueError("History exceeds configured character limit")
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("History mapping keys must be strings")
                visit(key, depth + 1)
                visit(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, depth + 1)
        elif value is None or isinstance(value, bool):
            pass
        elif isinstance(value, int):
            if value.bit_length() > 4096:
                raise ValueError("History integer exceeds size limit")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("History numbers must be finite")
        else:
            raise TypeError("History values must be JSON-compatible")

    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("History messages must be mappings")
        if not allow_repair and message.get(REPAIR_FEEDBACK_KEY) is True:
            raise ValueError("Internal repair feedback is not accepted from callers")
        visit(message, 0)


@dataclass(slots=True)
class GPUGeneration:
    """Serialize complete requests to a caller-owned runner and native tokenizer.

    Use the checkpoint's original fast tokenizer and chat template, with EOS ID 7.
    Each request starts a fresh runner sequence. By default, an optional exact
    token prefix is cached lazily on its first match; nonmatches use ordinary
    prefill. Strict mode requires explicit initialization, rejects nonmatches,
    and restores the saved prefix after each request or non-device failure.
    Runner failures permanently fault strict mode because the snapshot may be
    invalid. The adapter exclusively owns the runner and its snapshot. No
    sampling, tool execution, or conversation state is provided. Optional raw
    text callbacks run synchronously and may abort generation by raising.
    Reentrant operations on the same adapter are rejected.
    """

    runner: RunnerProtocol
    """Exclusively owned, already initialized runner."""

    tokenizer: Any
    """Native fast tokenizer with its checkpoint chat template."""

    tools: ToolRegistry
    """Registry used for declarations and validation, never dispatch."""

    config: GenerationConfig = field(default_factory=GenerationConfig)
    """Deterministic generation limits."""

    stable_prefix: tuple[int, ...] | None = None
    """Optional exact prompt-token prefix to snapshot on the first matching request."""

    strict_prefix_cache: bool = False
    """Require explicit prefix initialization, exact cache hits, and rollback."""

    second_prefix: tuple[int, ...] | None = None
    """Optional second strict prefix stored in an independent runner slot."""

    _initial_second_prefix: tuple[int, ...] | None = field(
        default=None, init=False, repr=False
    )
    """Immutable second prefix selected at construction."""

    _active_slot: int = field(default=0, init=False, repr=False)
    """Snapshot selected for the active strict request."""

    _strict_configured: bool = field(default=False, init=False, repr=False)
    """Strict mode selected at construction and immutable thereafter."""

    _initial_prefix: tuple[int, ...] | None = field(
        default=None, init=False, repr=False
    )
    """Strict mode's immutable prefix captured at construction."""

    _strict_faulted: bool = field(default=False, init=False, repr=False)
    """Whether a runner failure made the strict snapshot untrustworthy."""

    _strict_active: bool = field(default=False, init=False, repr=False)
    """Whether the current request has started modifying runner state."""

    _operation_active: bool = field(default=False, init=False, repr=False)
    """Whether a public operation owns the lock, including its callbacks."""

    _cached_prefix: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    """Prefix successfully snapshotted by this adapter, if any."""

    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False)
    """Protects the entire reset, prefill, and decode sequence."""

    def __post_init__(self) -> None:
        """Validate the optional immutable prefix and workspace cache capability."""
        if type(self.strict_prefix_cache) is not bool:
            raise TypeError("strict_prefix_cache must be a boolean")
        prefix = self.stable_prefix
        if self.strict_prefix_cache and prefix is None:
            raise ValueError("Strict prefix caching requires a stable prefix")
        self._initial_prefix = prefix
        self._initial_second_prefix = self.second_prefix
        self._strict_configured = self.strict_prefix_cache
        if self.second_prefix is not None:
            if not self.strict_prefix_cache or prefix is None:
                raise ValueError("Second prefix requires strict prefix caching")
            if getattr(self.runner.workspace.spec, "prefix_slots", 1) != 2:
                raise ValueError("Runner workspace must enable two prefix slots")
            if (
                not isinstance(self.second_prefix, tuple)
                or not self.second_prefix
                or any(
                    type(token) is not int or not 0 <= token < VOCAB_SIZE
                    for token in self.second_prefix
                )
                or len(self.second_prefix) > self.runner.workspace.spec.capacity
                or self.second_prefix[: len(prefix)] == prefix
                or prefix[: len(self.second_prefix)] == self.second_prefix
            ):
                raise ValueError("Invalid second stable prefix")
        if prefix is not None:
            if (
                not isinstance(prefix, tuple)
                or not prefix
                or any(
                    type(token) is not int or not 0 <= token < VOCAB_SIZE
                    for token in prefix
                )
            ):
                raise ValueError(
                    "Stable prefix must be a nonempty tuple of vocabulary IDs"
                )
            if not self.runner.workspace.spec.prefix_cache:
                raise ValueError("Runner workspace must enable prefix caching")
            if len(prefix) > self.runner.workspace.spec.capacity:
                raise ValueError("Stable prefix exceeds workspace capacity")

    def _reset(self) -> None:
        """Reset live state and forget a snapshot if the device reset fails."""
        try:
            self.runner.reset()
        except BaseException:
            self._cached_prefix = None
            raise

    @contextmanager
    def _operation(self) -> Iterator[None]:
        """Serialize operations and reject recursion without disturbing the owner."""
        with self._lock:
            if self._operation_active:
                raise RuntimeError("Reentrant GPUGeneration operation is not allowed")
            self._operation_active = True
            try:
                yield
            finally:
                self._operation_active = False

    def initialize_prefix(self) -> None:
        """Explicitly capture the strict prefix once, leaving live state at the prefix.

        A device failure permanently faults this adapter; create a new runner and
        adapter rather than assuming that a failed device snapshot is recoverable.
        """
        with self._operation():
            if not self.strict_prefix_cache:
                raise RuntimeError(
                    "Explicit initialization requires strict prefix caching"
                )
            self._check_strict_prefix()
            if self._cached_prefix is not None:
                raise RuntimeError("Strict prefix is already initialized")
            prefix = self._initial_prefix
            assert prefix is not None
            try:
                self._reset()
                self.runner.cache_prefix(list(prefix))
                if self._initial_second_prefix is not None:
                    self._reset()
                    self.runner.cache_prefix(list(self._initial_second_prefix), slot=1)
                self._cached_prefix = prefix
            except BaseException:
                self._strict_faulted = True
                self._cached_prefix = None
                raise

    def _check_strict_prefix(self) -> None:
        """Reject mutated configuration and snapshots invalidated by runner failures."""
        if self.strict_prefix_cache != self._strict_configured:
            raise RuntimeError("Strict prefix mode was mutated")
        if self.stable_prefix != self._initial_prefix:
            raise RuntimeError("Strict stable prefix was mutated")
        if self.second_prefix != self._initial_second_prefix:
            raise RuntimeError("Strict second prefix was mutated")
        if self._strict_faulted:
            raise RuntimeError("Strict prefix snapshot is invalid after runner failure")

    def generate(self, user_text: str) -> Generation:
        """Generate a fresh user turn without executing tools.

        Args:
            user_text:
                Nonempty user request, canonicalized before templating.
        """
        with self._operation():
            if len(user_text) > self.config.max_history_chars:
                raise ValueError("History exceeds configured character limit")
            if not user_text.strip():
                raise ValueError("User request must not be empty")
            return self._run_messages([{"role": "user", "content": user_text}])

    def generate_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> Generation:
        """Generate a response and restore the strict prefix after runner use.

        Args:
            messages:
                Caller-owned conversation history.

            on_text:
                Optional raw, prevalidation text callback.
        """
        with self._operation():
            return self._run_messages(messages, on_text)

    def _generate_repair(
        self,
        messages: Sequence[Mapping[str, Any]],
        feedback: str,
        on_text: Callable[[str], None] | None = None,
    ) -> Generation:
        """Generate using Session-created feedback after validating the base history.

        Args:
            messages:
                Base history without internal metadata.

            feedback:
                Session-generated diagnostic feedback, not caller history.

            on_text:
                Optional raw, prevalidation text callback.
        """
        with self._operation():
            _validate_history(messages, self.config)
            repair = {"role": "user", "content": feedback, REPAIR_FEEDBACK_KEY: True}
            return self._run_messages([*messages, repair], on_text, True)

    def _run_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        on_text: Callable[[str], None] | None = None,
        allow_repair: bool = False,
    ) -> Generation:
        """Run a guarded request and restore any modified strict runner state.

        Args:
            messages:
                Caller-owned conversation history.

            on_text:
                Optional raw, prevalidation text callback.

            allow_repair:
                Whether the internally constructed feedback is present.
        """
        if self.strict_prefix_cache != self._strict_configured:
            raise RuntimeError("Strict prefix mode was mutated")
        if not self._strict_configured:
            return self._generate_messages(messages, on_text, allow_repair)
        self._check_strict_prefix()
        if self._cached_prefix is None:
            raise RuntimeError("Strict prefix cache is not initialized")
        self._strict_active = False
        failure: BaseException | None = None
        try:
            return self._generate_messages(messages, on_text, allow_repair)
        except BaseException as error:
            failure = error
            raise
        finally:
            if self._strict_active:
                if self._cached_prefix is None:
                    self._strict_faulted = True
                else:
                    try:
                        self._reset()
                        prefix = (
                            self._initial_prefix
                            if self._active_slot == 0
                            else self._initial_second_prefix
                        )
                        assert prefix is not None
                        if self._active_slot == 0:
                            self.runner.prefill_cached(list(prefix))
                        else:
                            self.runner.prefill_cached(list(prefix), slot=1)
                    except BaseException as error:
                        self._strict_faulted = True
                        self._cached_prefix = None
                        if failure is None:
                            raise
                        failure.add_note(f"Strict prefix rollback failed: {error!r}")
                self._strict_active = False

    def _generate_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        on_text: Callable[[str], None] | None,
        allow_repair: bool,
    ) -> Generation:
        """Generate and validate a complete response from caller-owned history.

        History is an untrusted boundary: it is bounded before canonicalization
        and template rendering, and trusted internal repair flags cannot be set
        by callers. EOS counts as an output token and is decoded with special
        tokens preserved. Any failure propagates; the next request resets the
        runner before prefill. Timing includes formatting in elapsed time and
        prefill in first-token time.

        Args:
            messages:
                Nonempty user/assistant/tool history; external user text is
                canonicalized, and structured assistant/tool content is preserved.

            on_text:
                Optional raw, prevalidation text callback.

            allow_repair:
                Whether history contains internally constructed repair feedback.
        """
        with self._lock:
            _validate_history(messages, self.config, allow_repair)
            if self.tokenizer.eos_token_id != EOS_ID:
                raise ValueError("Tokenizer EOS ID must be 7")
            if not messages or any(
                m["role"] not in ("user", "assistant", "tool") for m in messages
            ):
                raise ValueError("Expected nonempty user/assistant/tool history")
            for message in messages:
                content = message.get("content")
                if content is None and message["role"] != "assistant":
                    raise ValueError("User and tool messages require content")
                if content is not None and not isinstance(content, (str, Mapping)):
                    raise TypeError(
                        "History content must be a string or mapping; wrap scalar tool results"
                    )
            started = time.perf_counter()
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.config.system_prompt},
                    *canonical_user_messages(messages),
                ],
                tools=tool_declarations(self.tools),
                tokenize=False,
                add_generation_prompt=True,
            )
            if len(prompt) > self.config.max_prompt_chars:
                raise ValueError("Request exceeds configured prompt character limit")
            inputs = self.tokenizer(
                prompt, add_special_tokens=False, return_tensors="np"
            )
            ids = inputs.input_ids
            if (
                ids.ndim != 2
                or ids.shape[0] != 1
                or ids.shape[1] == 0
                or not np.issubdtype(ids.dtype, np.integer)
            ):
                raise ValueError(
                    "Tokenizer must return one nonempty integer token sequence"
                )
            tokens = [int(token) for token in ids[0]]
            if any(token < 0 or token >= VOCAB_SIZE for token in tokens):
                raise ValueError("Tokenizer returned an out-of-vocabulary ID")
            input_count = len(tokens)
            if input_count + self.config.max_new_tokens > min(
                self.config.max_context_tokens, self.runner.workspace.spec.capacity
            ):
                raise ValueError("Request exceeds configured context budget")
            prefix = self.stable_prefix
            if self._strict_configured:
                second = self._initial_second_prefix
                if second is not None and tuple(tokens[: len(second)]) == second:
                    prefix = second
                    self._active_slot = 1
                elif prefix is not None and tuple(tokens[: len(prefix)]) == prefix:
                    self._active_slot = 0
                else:
                    raise ValueError("Prompt does not match strict stable prefix")
                self._strict_active = True
            self._reset()
            inference_start = time.perf_counter()
            if prefix is not None and tuple(tokens[: len(prefix)]) == prefix:
                if self._cached_prefix != prefix and not (
                    self._strict_configured and self._active_slot == 1
                ):
                    if self._strict_configured:
                        raise RuntimeError("Strict prefix snapshot is invalid")
                    self._cached_prefix = None
                    self.runner.cache_prefix(list(prefix))
                    self._cached_prefix = prefix
                    self._reset()
                try:
                    logits = (
                        self.runner.prefill_cached(tokens, slot=1)
                        if self._strict_configured and self._active_slot == 1
                        else self.runner.prefill_cached(tokens)
                    )
                except BaseException:
                    # A failed restore invalidates the runner's snapshot.
                    self._cached_prefix = None
                    raise
            else:
                try:
                    logits = self.runner.prefill(tokens)
                except BaseException:
                    self._cached_prefix = None
                    raise
            generated: list[int] = []
            streamer = (
                _text_streamer(self.tokenizer, on_text, False) if on_text else None
            )
            first_token_seconds = 0.0
            for index in range(self.config.max_new_tokens):
                # Repetition penalty applies once per distinct prompt or generated ID.
                scores = np.asarray(logits)
                if (
                    scores.shape != (VOCAB_SIZE,)
                    or not np.issubdtype(scores.dtype, np.floating)
                    or not np.isfinite(scores).all()
                ):
                    raise ValueError("Runner returned invalid logits")
                scores = scores.copy()
                seen = np.fromiter(set(tokens + generated), dtype=np.intp)
                scores[seen] = np.where(
                    scores[seen] < 0,
                    scores[seen] * self.config.repetition_penalty,
                    scores[seen] / self.config.repetition_penalty,
                )
                token = int(np.argmax(scores))
                if index == 0:
                    first_token_seconds = time.perf_counter() - inference_start
                generated.append(token)
                if streamer is not None:
                    streamer.put(np.asarray([token]))
                if token == EOS_ID:
                    break
                if index + 1 < self.config.max_new_tokens:
                    try:
                        logits = self.runner.decode(token)
                    except BaseException:
                        self._cached_prefix = None
                        raise
            if len(generated) >= self.config.max_new_tokens:
                raise ValueError("Generation reached output limit")
            if streamer is not None:
                streamer.end()
            raw = self.tokenizer.decode(
                generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if len(raw) > self.config.max_output_chars:
                raise ValueError("Generation exceeds configured output character limit")
            try:
                text, calls = parse_response(raw)
                self.tools.validate(calls)
            except ValueError as error:
                raise GeneratedCallError(str(error), raw) from error
            return Generation(
                text,
                calls,
                raw,
                input_count,
                len(generated),
                time.perf_counter() - started,
                first_token_seconds,
            )
