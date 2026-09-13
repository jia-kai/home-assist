"""Transactional native conversations with explicit tools and incremental speech text."""

import json
import logging
import threading
from collections.abc import Callable, Generator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from queue import Queue
from typing import Any

from pydantic import JsonValue

from .lfm import LFM2
from .llm import FunctionGemma, GeneratedCallError, Generation, ToolCall

logger = logging.getLogger(__name__)


class _Cancelled(Exception):
    """Internal callback cancellation when a consumer closes its iterator."""


@dataclass(slots=True)
class Session:
    """Own conversation state independently of a loaded model and its registry.

    Exhaust each stream before inspecting calls. Close an abandoned iterator to
    cancel and join its worker. Failed or abandoned generation leaves history
    unchanged, although text already delivered cannot be retracted. A handler
    failure requires reset because preceding external side effects may exist.
    Concurrent operations on one session raise rather than interleave turns.
    """

    model: FunctionGemma | LFM2
    """Externally loaded model, with registered tools and configured system prompt."""

    _messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    """Committed native user/assistant/tool messages, excluding the system prompt."""

    _pending: tuple[ToolCall, ...] = field(default=(), init=False)
    """Validated assistant calls awaiting explicit dispatch."""

    _continue: bool = field(default=False, init=False)
    """Whether successful tool results await assistant continuation."""

    _failed: bool = field(default=False, init=False)
    """Whether tool execution failed and reset is required."""

    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    """Non-reentrant operation guard held through stream consumption."""

    @property
    def pending_calls(self) -> tuple[ToolCall, ...]:
        """Return isolated copies of pending calls; mutation cannot alter dispatch."""
        return deepcopy(self._pending)

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        """Return an isolated snapshot of committed native history."""
        return tuple(deepcopy(self._messages))

    def _acquire(self) -> None:
        """Reject overlapping session operations without blocking the caller."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "Session has an active operation; exhaust or close stream"
            )

    def reset(self) -> None:
        """Clear history, calls, and failure state while retaining the loaded model."""
        self._acquire()
        try:
            self._messages.clear()
            self._pending = ()
            self._continue = self._failed = False
        finally:
            self._lock.release()

    def request_tools(self, user_text: str, calls: Sequence[ToolCall]) -> None:
        """Record explicit application-routed calls without running inference or handlers.

        Validate the whole batch before committing native history. Dispatch and
        continuation follow the same contract as generated calls.

        Args:
            user_text:
                Nonempty original user request retained in conversation history.

            calls:
                Nonempty application-selected calls to registered tools.

        """
        self._acquire()
        try:
            if self._failed or self._pending or self._continue:
                raise RuntimeError("Finish or reset the session before a new request")
            if not user_text.strip() or not calls:
                raise ValueError("Explicit routing requires a request and tool calls")
            pending = tuple(deepcopy(calls))
            self.model.tools.validate(pending)
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": deepcopy(call.arguments),
                        },
                    }
                    for call in pending
                ],
            }
            self._messages.extend([{"role": "user", "content": user_text}, assistant])
            self._pending = pending
        finally:
            self._lock.release()

    def stream(
        self, user_text: str | None = None, *, max_repair_attempts: int = 0
    ) -> Generator[str]:
        """Generate a transactional reply, optionally repairing invalid tool calls.

        Plain text is emitted at backend word boundaries. From the first possible
        protocol delimiter onward, text is withheld until complete strict parsing
        and batch validation succeed; quoted delimiters cannot leak tool arguments.
        Whitespace at the response edges is stripped. Errors propagate with their
        traceback. History commits only when this iterator is fully exhausted.
        Repair-enabled routing buffers prose until validation succeeds. Diagnostic
        prompts remain transient; only the original request and valid reply commit.
        Inference limits and tool execution failures are not repaired.

        Args:
            user_text:
                Nonempty new user turn, or None to continue after invoke_tools().
                A new user turn is forbidden while calls or continuation are pending.

            max_repair_attempts:
                Zero through two extra generations after generated syntax or
                argument failures. Zero retains fail-fast incremental streaming.
                Repair feedback includes at most 400 characters each of the error
                and rejected output; full diagnostics remain in logs.

        """
        if type(max_repair_attempts) is not int or not 0 <= max_repair_attempts <= 2:
            raise ValueError("max_repair_attempts must be an integer from zero to two")
        self._acquire()
        worker: threading.Thread | None = None
        cancelled = threading.Event()
        events: Queue[str | Generation | BaseException] = Queue()
        try:
            if self._failed:
                raise RuntimeError("Tool execution failed; reset the session")
            if self._pending:
                raise RuntimeError("Invoke pending tools before continuing")
            messages = deepcopy(self._messages)
            if user_text is None:
                if not self._continue:
                    raise RuntimeError("No tool results await continuation")
            else:
                if self._continue:
                    raise RuntimeError("Continue after tools before a new user turn")
                if not user_text.strip():
                    raise ValueError("User request must not be empty")
                messages.append({"role": "user", "content": user_text})

            def receive(text: str) -> None:
                """Queue incremental backend text or abort an abandoned generation.

                Args:
                    text:
                        Newly decoded raw fragment, including special tokens.

                """
                if cancelled.is_set():
                    raise _Cancelled()
                if not max_repair_attempts:
                    events.put(text)

            def generate() -> None:
                """Repair only generated-call failures and transfer the final outcome."""
                try:
                    attempt_messages = messages
                    for attempt in range(max_repair_attempts + 1):
                        if cancelled.is_set():
                            raise _Cancelled()
                        try:
                            result = self.model.generate_messages(
                                attempt_messages, on_text=receive
                            )
                        except GeneratedCallError as error:
                            logger.warning(
                                "tool routing status=invalid attempt=%d repair_limit=%d reason=%s",
                                attempt + 1,
                                max_repair_attempts,
                                error,
                            )
                            logger.debug(
                                "Rejected tool output: %r", error.raw, exc_info=True
                            )
                            if attempt == max_repair_attempts:
                                logger.exception(
                                    "tool routing status=repair_exhausted attempts=%d",
                                    attempt + 1,
                                )
                                raise
                            feedback = (
                                "No tools ran. Retry the original request with valid tools, "
                                "fields and types. Put location qualifiers inside city "
                                "(e.g. 'Ottawa, Canada'). Return complete calls only.\n"
                                f"Validation error: {str(error)[:400]}\n"
                                f"Rejected output: {error.raw[:400]!r}"
                            )
                            attempt_messages = [
                                *messages,
                                {"role": "user", "content": feedback},
                            ]
                            continue
                        events.put(result)
                        break
                except BaseException as error:  # noqa: BLE001 - re-raised by consumer
                    # The consumer thread must observe every worker failure.
                    events.put(error)

            worker = threading.Thread(target=generate, name="hoast-generation")
            worker.start()
            raw = ""
            emitted = ""
            while True:
                event = events.get()
                if isinstance(event, BaseException):
                    raise event
                if isinstance(event, Generation):
                    if not event.text.startswith(emitted):
                        raise RuntimeError(
                            "Streamed text disagrees with parsed response"
                        )
                    remaining = event.text[len(emitted) :]
                    if remaining:
                        yield remaining
                    assistant: dict[str, Any] = {
                        "role": "assistant",
                        "content": event.text,
                    }
                    if event.calls:
                        assistant["tool_calls"] = [
                            {
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": deepcopy(call.arguments),
                                },
                            }
                            for call in event.calls
                        ]
                    messages.append(assistant)
                    self._messages = messages
                    self._pending = deepcopy(event.calls)
                    self._continue = False
                    break
                raw += event
                if len(raw) > 65536:
                    raise ValueError("Generated response exceeds parser size limit")
                safe = raw.partition("<")[0].strip()
                if len(safe) > len(emitted):
                    yield safe[len(emitted) :]
                    emitted = safe
        finally:
            cancelled.set()
            try:
                if worker is not None and worker.ident is not None:
                    worker.join()
            finally:
                self._lock.release()

    def complete(self, text: str) -> None:
        """Commit an application-rendered answer after successful tool invocation.

        This provides a grounded alternative to model continuation while keeping
        native history available for subsequent user turns.

        Args:
            text:
                Nonempty assistant answer derived by the calling application.

        """
        self._acquire()
        try:
            if self._failed or self._pending or not self._continue:
                raise RuntimeError("Complete requires successful tool results")
            if not text.strip():
                raise ValueError("Assistant answer must not be empty")
            self._messages.append({"role": "assistant", "content": text})
            self._continue = False
        finally:
            self._lock.release()

    def invoke_tools(
        self,
        result_transform: Callable[[ToolCall, JsonValue], JsonValue] | None = None,
    ) -> tuple[JsonValue, ...]:
        """Validate the entire pending batch, dispatch once, and append native results.

        Return results in call order. Invalid result serialization and handler
        errors propagate and require reset, preventing retries of partial side
        effects. Results are JSON snapshots; scalar and empty values are wrapped
        under ``result`` so both native templates preserve them.

        Args:
            result_transform:
                Optional application projection for model-visible tool results.
                Receives isolated call/result snapshots; returned values must be
                JSON-compatible. Full results are still returned to the caller.
                Projection failures require reset and never retry handlers.

        """
        self._acquire()
        try:
            if self._failed:
                raise RuntimeError("Tool execution failed; reset the session")
            if not self._pending:
                raise RuntimeError("No pending tool calls")
            self.model.tools.validate(self._pending)
            self._failed = True
            results = self.model.tools.dispatch(self._pending)
            snapshots: list[JsonValue] = json.loads(
                json.dumps(results, allow_nan=False)
            )
            messages: list[dict[str, Any]] = []
            for call, result in zip(self._pending, snapshots, strict=True):
                model_result = (
                    result
                    if result_transform is None
                    else result_transform(deepcopy(call), deepcopy(result))
                )
                content: dict[str, JsonValue] = {
                    "result": json.loads(json.dumps(model_result, allow_nan=False))
                }
                if isinstance(self.model, LFM2):
                    content["name"] = call.name
                messages.append({"role": "tool", "name": call.name, "content": content})
            self._messages.extend(messages)
            self._pending = ()
            self._continue = True
            self._failed = False
            return tuple(snapshots)
        finally:
            self._lock.release()
