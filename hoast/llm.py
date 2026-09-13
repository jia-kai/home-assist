"""Lifecycle-managed FunctionGemma inference and validated tool interception."""

import gc
import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, JsonValue

from .logging import get_logger

logger = get_logger(__name__)
DEFAULT_CACHE = Path(".cache/hoast")
SYSTEM_PROMPT = (
    "You are a model that can do function calling with the following functions"
)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _text_streamer(
    tokenizer: Any, callback: Callable[[str], None], skip_prompt: bool
) -> Any:
    """Build a lazy word-boundary decoder that preserves native protocol tokens.

    Args:
        tokenizer:
            Loaded model tokenizer used to decode generated token IDs.

        callback:
            Receives incremental decoded text; exceptions abort generation.

        skip_prompt:
            Whether the backend sends the prompt as its first token callback.

    """
    # Transformers is optional until inference is requested.
    from transformers import TextStreamer

    class CallbackStreamer(TextStreamer):
        """Deliver decoded words to the caller instead of standard output."""

        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            """Forward a decoded fragment, including the final partial word.

            Args:
                text:
                    Newly decoded text, with special tokens preserved.

                stream_end:
                    Whether this fragment ends decoding.

            """
            if text:
                callback(text)

    return CallbackStreamer(
        tokenizer,
        skip_prompt=skip_prompt,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _ov_streamer(tokenizer: Any, callback: Callable[[str], None]) -> Any:
    """Adapt OpenVINO token callbacks without losing special-token delimiters.

    Args:
        tokenizer:
            Loaded native tokenizer.

        callback:
            Receives decoded fragments synchronously in the inference thread.

    """
    # These optional libraries are needed only for OpenVINO generation.
    import numpy as np
    import openvino_genai

    decoder = _text_streamer(tokenizer, callback, False)

    class TokenStreamer(openvino_genai.StreamerBase):
        """Decode generated IDs, rather than the backend's special-token-free text."""

        def write(self, token: int | list[int]) -> Any:
            """Decode one backend token or token batch and continue generation.

            Args:
                token:
                    Generated vocabulary ID or ordered batch of IDs.

            """
            decoder.put(np.asarray([token] if isinstance(token, int) else token))
            return openvino_genai.StreamingStatus.RUNNING

        def end(self) -> None:
            """Flush the decoder's final incomplete word."""
            decoder.end()

    return TokenStreamer()


class ToolArguments(BaseModel):
    """Base for tool arguments: reject extra fields and implicit type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


@dataclass(slots=True, frozen=True)
class ToolCall:
    """An intercepted function call; execution always revalidates arguments."""

    name: str
    """Registered function name."""

    arguments: dict[str, JsonValue]
    """JSON-compatible arguments from the generated call."""


@dataclass(slots=True, frozen=True)
class Tool[Args: ToolArguments]:
    """A named synchronous handler with a Pydantic argument contract."""

    name: str
    """Identifier exposed to the model."""

    description: str
    """Purpose and appropriate use of the tool."""

    arguments: type[Args]
    """Strict model defining handler inputs; describe fields with `Field`."""

    handler: Callable[[Args], JsonValue]
    """Handler invoked only by explicit dispatch; exceptions propagate."""

    def __post_init__(self) -> None:
        """Reject invalid identifiers and argument contracts at registration."""
        if not _NAME.fullmatch(self.name) or not self.description.strip():
            raise ValueError("Tools require a valid identifier and a description")
        if self.arguments.model_config.get("extra") != "forbid":
            raise ValueError("Tool argument models must forbid extra fields")
        self.schema()

    def schema(self) -> dict[str, Any]:
        """Return a chat-template schema with local references expanded.

        Unsupported unions and recursive models raise rather than silently losing
        their type information in FunctionGemma's template. Constraints are also
        enforced by Pydantic when calls are intercepted and dispatched.
        """
        root = self.arguments.model_json_schema()
        definitions = root.get("$defs", {})

        def expand(node: dict[str, Any], seen: frozenset[str]) -> dict[str, Any]:
            """Resolve one schema node and validate template-compatible types.

            Args:
                node:
                    JSON Schema node to normalize.

                seen:
                    References on the current expansion path.

            """
            node = dict(node)
            reference = node.pop("$ref", None)
            if reference is not None:
                if reference in seen or not reference.startswith("#/$defs/"):
                    raise ValueError(
                        f"Unsupported recursive/external schema: {reference}"
                    )
                node = {**definitions[reference.removeprefix("#/$defs/")], **node}
                seen = seen | {reference}
            node.pop("$defs", None)
            if "anyOf" in node or "oneOf" in node or "allOf" in node:
                raise ValueError("FunctionGemma tool schemas require concrete types")
            if node.get("type") not in {
                "object",
                "array",
                "string",
                "integer",
                "number",
                "boolean",
            }:
                raise ValueError(f"Unsupported tool schema type: {node.get('type')}")
            node.setdefault("description", node.get("title", "Tool argument"))
            if "const" in node:
                node["enum"] = [node.pop("const")]
            if "enum" in node and node["type"] != "string":
                raise ValueError("FunctionGemma's template supports string enums only")
            if node["type"] == "object":
                if node.get("additionalProperties") is not False:
                    raise ValueError("Use ToolArguments for nested objects")
                for key in node.get("properties", {}):
                    if not _NAME.fullmatch(key) or key in {
                        "description",
                        "type",
                        "properties",
                        "required",
                        "nullable",
                    }:
                        raise ValueError(
                            f"Argument name unsupported by template: {key}"
                        )
                node["properties"] = {
                    key: expand(value, seen)
                    for key, value in node.get("properties", {}).items()
                }
                node.setdefault("required", [])
            elif node["type"] == "array":
                node["items"] = expand(node["items"], seen)
            return node

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": expand(root, frozenset()),
            },
        }


class GeneratedCallError(ValueError):
    """Repairable generated syntax or argument failure, before any tool dispatch."""

    raw: str
    """Rejected model output, retained for repair prompts and diagnostics."""

    def __init__(self, reason: str, raw: str) -> None:
        """Store the validation reason and rejected model output.

        Args:
            reason:
                Parser or argument validation message for the model to correct.

            raw:
                Model-generated response that failed validation.

        """
        super().__init__(reason)
        self.raw = raw
        self.add_note(f"Generated output: {raw!r}")


class ToolRegistry:
    """Immutable registration set with validate-all-before-execute dispatch."""

    _tools: dict[str, Tool[Any]]
    """Handlers indexed by unique function identifier."""

    def __init__(self, tools: Sequence[Tool[Any]]) -> None:
        """Register tools, rejecting duplicate names.

        Args:
            tools:
                Typed synchronous tools available to the model.

        """
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Duplicate tool name")

    def schemas(self) -> list[dict[str, Any]]:
        """Return fresh chat-template declarations for the registered tools."""
        return [tool.schema() for tool in self._tools.values()]

    def validate(self, calls: Sequence[ToolCall]) -> list[ToolArguments]:
        """Validate all names and inputs without executing handlers.

        Args:
            calls:
                Ordered intercepted calls; unknown names raise `ValueError`.

        """
        result: list[ToolArguments] = []
        for call in calls:
            if call.name not in self._tools:
                raise ValueError(f"Unknown tool: {call.name}")
            result.append(
                self._tools[call.name].arguments.model_validate_json(
                    json.dumps(call.arguments, allow_nan=False), strict=True
                )
            )
        return result

    def dispatch(self, calls: Sequence[ToolCall]) -> list[JsonValue]:
        """Validate the complete batch, then execute handlers in order.

        Handler exceptions propagate immediately; preceding side effects are not
        rolled back. This method neither retries nor asks the model to continue.

        Args:
            calls:
                Calls to execute exactly once, in sequence.

        """
        calls = tuple(calls)
        arguments = self.validate(calls)
        return [
            self._tools[call.name].handler(args)
            for call, args in zip(calls, arguments, strict=True)
        ]


@dataclass(slots=True)
class _Parser:
    """Bounded recursive parser for FunctionGemma's non-JSON wire format."""

    source: str
    """Generated response with turn termination removed."""

    position: int = 0
    """Current character offset."""

    def take(self, token: str) -> bool:
        """Consume a token after whitespace if present.

        Args:
            token:
                Exact wire-format token to match.

        """
        while self.position < len(self.source) and self.source[self.position].isspace():
            self.position += 1
        if self.source.startswith(token, self.position):
            self.position += len(token)
            return True
        return False

    def require(self, token: str) -> None:
        """Consume a required token or raise with its location.

        Args:
            token:
                Exact expected token.

        """
        if not self.take(token):
            raise ValueError(f"Expected {token!r} at offset {self.position}")

    def name(self) -> str:
        """Consume an unquoted function or argument identifier."""
        self.take("")
        match = _NAME.match(self.source, self.position)
        if match is None:
            raise ValueError(f"Expected identifier at offset {self.position}")
        self.position = match.end()
        return match.group()

    def value(self, depth: int = 0) -> JsonValue:
        """Parse a value, preserving literal content inside escape delimiters.

        Args:
            depth:
                Nesting depth, limited to 32 at this untrusted boundary.

        """
        if depth > 32:
            raise ValueError("Tool arguments exceed maximum nesting depth")
        if self.take("<escape>"):
            end = self.source.find("<escape>", self.position)
            if end < 0:
                raise ValueError("Unterminated escaped string")
            value = self.source[self.position : end]
            self.position = end + len("<escape>")
            return value
        if self.take("{"):
            obj: dict[str, JsonValue] = {}
            if self.take("}"):
                return obj
            while True:
                self.take("")
                if self.source.startswith("<escape>", self.position):
                    key = self.value(depth + 1)
                    assert isinstance(key, str)
                else:
                    key = self.name()
                if key in obj:
                    raise ValueError(f"Duplicate argument: {key}")
                self.require(":")
                obj[key] = self.value(depth + 1)
                if self.take("}"):
                    return obj
                self.require(",")
        if self.take("["):
            items: list[JsonValue] = []
            if self.take("]"):
                return items
            while True:
                items.append(self.value(depth + 1))
                if self.take("]"):
                    return items
                self.require(",")
        for token, literal in (("true", True), ("false", False), ("null", None)):
            if self.take(token):
                return literal
        match = _NUMBER.match(self.source, self.position)
        if match is None:
            raise ValueError(f"Invalid value at offset {self.position}")
        self.position = match.end()
        number = json.loads(match.group())
        if isinstance(number, float) and not math.isfinite(number):
            raise ValueError("Non-finite tool argument")
        return number


def parse_response(raw: str) -> tuple[str, tuple[ToolCall, ...]]:
    """Split model text from complete calls; reject malformed protocol output.

    Args:
        raw:
            Decoded generated tokens with special tokens preserved. A terminal
            turn/EOS or start-function-response token is accepted. At most 65,536 characters
            and 32 calls are accepted; strings can contain literal call markers.
            Truncated delimiter prefixes and unexpected protocol tokens raise.

    """
    if len(raw) > 65536:
        raise ValueError("Generated response exceeds parser size limit")
    source = raw.strip()
    for suffix in ("<pad>", "<eos>", "<end_of_turn>", "<start_function_response>"):
        source = source.removesuffix(suffix).rstrip()
    parser = _Parser(source)
    text: list[str] = []
    calls: list[ToolCall] = []
    while parser.position < len(source):
        if source.startswith("<start_function_call>", parser.position):
            parser.require("<start_function_call>")
            parser.require("call:")
            name = parser.name()
            arguments = parser.value()
            if not isinstance(arguments, dict):
                raise ValueError("Tool call arguments must be an object")
            parser.require("<end_function_call>")
            calls.append(ToolCall(name, arguments))
            if len(calls) > 32:
                raise ValueError("Too many tool calls")
        else:
            if parser.position >= len(source):
                break
            if source[parser.position] == "<" and re.match(
                r"<(?:/?(?:start|end)_|escape|eos|bos|pad|unk|/?think|\|)",
                source[parser.position :],
            ):
                raise ValueError(f"Unexpected protocol token at {parser.position}")
            if source[parser.position] == "<" and any(
                token.startswith(source[parser.position :])
                for token in (
                    "<start_function_call>",
                    "<end_of_turn>",
                    "<escape>",
                    "<eos>",
                    "<pad>",
                )
            ):
                raise ValueError("Truncated protocol token")
            text.append(source[parser.position])
            parser.position += 1
    return "".join(text).strip(), tuple(calls)


@dataclass(slots=True, frozen=True)
class LLMConfig:
    """Local prepared model and batch-one generation settings."""

    model_path: Path
    """Downloaded checkpoint (torch) or exported IR directory (openvino)."""

    backend: Literal["torch", "openvino"] = "openvino"
    """Execution engine; loading never downloads or exports models."""

    cache_dir: Path = DEFAULT_CACHE
    """Working-directory-relative persistent compilation cache root."""

    threads: int = 4
    """CPU inference threads; torch changes its process-global thread count."""

    attention_backend: Literal["PA", "SDPA"] = "PA"
    """OpenVINO attention: PA favors inference speed; SDPA reuses CPU blobs."""

    max_new_tokens: int = 128
    """Maximum generated tokens; length-limited responses are rejected."""

    max_context_tokens: int = 4096
    """Serving limit for prompt plus generation, bounded by model context."""

    system_prompt: str = SYSTEM_PROMPT
    """Developer instruction; preserve the function-calling activation sentence."""

    device: Literal["CPU", "GPU"] = "CPU"
    """Explicit OpenVINO device; GPU selects the Intel GPU without fallback."""

    @classmethod
    def from_cache(
        cls,
        cache_dir: Path = DEFAULT_CACHE,
        precision: Literal["fp32", "int8", "int4"] = "int8",
        threads: int = 4,
        attention_backend: Literal["PA", "SDPA"] = "PA",
        device: Literal["CPU", "GPU"] = "CPU",
    ) -> Self:
        """Resolve a prepared OpenVINO artifact without network access.

        Args:
            cache_dir:
                Working-directory-relative cache used by `prepare_llm.py`.

            precision:
                Prepared weight format whose manifest selects the export.

            threads:
                CPU inference thread count.

            attention_backend:
                OpenVINO attention implementation, PA for inference throughput or
                SDPA for reusable device compilation on the tested runtime.

            device:
                Explicit OpenVINO execution device, without fallback.

        """
        manifest = json.loads((cache_dir / f"{precision}.json").read_text())
        return cls(
            Path(manifest["path"]),
            cache_dir=cache_dir,
            threads=threads,
            attention_backend=attention_backend,
            device=device,
        )

    def __post_init__(self) -> None:
        """Reject invalid settings before allocating model resources."""
        if self.backend not in ("torch", "openvino"):
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self.device not in ("CPU", "GPU"):
            raise ValueError("Device must be CPU or GPU")
        if self.backend == "torch" and self.device != "CPU":
            raise ValueError("The torch backend supports CPU only")
        if self.attention_backend not in ("PA", "SDPA"):
            raise ValueError("Attention backend must be PA or SDPA")
        if self.threads < 1 or self.max_new_tokens < 1:
            raise ValueError("Thread and output-token counts must be positive")
        if not self.max_new_tokens < self.max_context_tokens <= 32768:
            raise ValueError("Context must exceed output budget and be <= 32768")
        if SYSTEM_PROMPT not in self.system_prompt:
            raise ValueError("System prompt must include function-calling activation")


@dataclass(slots=True, frozen=True)
class Generation:
    """Completed single-turn inference with intercepted, validated calls."""

    text: str
    """Natural-language content outside the tool-call protocol."""

    calls: tuple[ToolCall, ...]
    """Validated calls; no handlers have been run."""

    raw: str
    """Generated content, preserving protocol delimiters for diagnostics."""

    input_tokens: int
    """Number of prompt tokens."""

    output_tokens: int
    """Number of generated tokens including termination tokens."""

    elapsed_seconds: float
    """Wall time for prompt formatting, generation, parsing, and validation."""

    first_token_seconds: float
    """Backend time to first generated token, excluding prompt formatting."""


@dataclass(slots=True)
class _TokenTimer:
    """Transformers streamer measuring first-token latency without decoding."""

    started: float = field(default_factory=time.perf_counter)
    """Generation start time in monotonic seconds."""

    first_token_seconds: float = 0.0
    """Seconds to first generated token."""

    _prompt_seen: bool = False
    """Whether the initial prompt callback has been consumed."""

    decoder: Any = None
    """Optional raw-text decoder receiving the same prompt and generated tokens."""

    def put(self, value: Any) -> None:
        """Record first-token time and forward tokens to an optional decoder.

        Args:
            value:
                CPU int64 token tensor, shape (1, prompt_length) for the prompt
                or (1,) for a generated token; values are not inspected.

        """
        if not self._prompt_seen:
            self._prompt_seen = True
        elif not self.first_token_seconds:
            self.first_token_seconds = time.perf_counter() - self.started
        if self.decoder is not None:
            self.decoder.put(value)

    def end(self) -> None:
        """Flush the optional decoder on stream completion."""
        if self.decoder is not None:
            self.decoder.end()


@dataclass(slots=True)
class FunctionGemma:
    """Own one model; context-manager loading and serialized requests.

    `generate` is stateless across requests and intercepts calls without running
    them. `tools.dispatch(result.calls)` executes them explicitly. OpenVINO owns
    its thread pool; torch thread settings are process-global. Close waits for
    inference, drops owned references, and allows subsequent explicit reloading.
    """

    config: LLMConfig
    """Immutable model path and execution settings."""

    tools: ToolRegistry
    """Registered tool contracts and explicit dispatch interface."""

    _model: Any = field(default=None, init=False, repr=False)
    """Owned backend model, or None while unloaded."""

    _tokenizer: Any = field(default=None, init=False, repr=False)
    """Owned Hugging Face tokenizer using the checkpoint's chat template."""

    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    """Serializes lifecycle transitions and batch-one model inference."""

    def load(self) -> Self:
        """Load local artifacts once; failure leaves this instance unloaded."""
        # Heavy inference dependencies remain lazy so tool contracts work alone.
        from transformers import AutoTokenizer

        with self._lock:
            if self._model is not None:
                return self
            path = self.config.model_path.resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"Prepare model artifacts first: {path}")
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            if self.config.backend == "torch":
                # Eager model loading is unnecessary for the compiled backend.
                import torch
                from transformers import AutoModelForCausalLM

                torch.set_num_threads(self.config.threads)
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    local_files_only=True,
                    dtype=torch.float32,
                    attn_implementation="sdpa",
                ).eval()
            else:
                # OpenVINO is an optional execution path until a model is loaded.
                import openvino_genai

                cache = self.config.cache_dir.resolve() / "compiled"
                cache.mkdir(parents=True, exist_ok=True)
                model = openvino_genai.LLMPipeline(
                    str(path),
                    self.config.device,
                    {
                        "CACHE_DIR": str(cache),
                        "CACHE_MODE": "OPTIMIZE_SPEED",
                        "ATTENTION_BACKEND": self.config.attention_backend,
                        "PERFORMANCE_HINT": "LATENCY",
                        "NUM_STREAMS": "1",
                        **(
                            {
                                "INFERENCE_NUM_THREADS": self.config.threads,
                                "INFERENCE_PRECISION_HINT": "f32",
                            }
                            if self.config.device == "CPU"
                            else {}
                        ),
                    },
                )
            self._tokenizer, self._model = tokenizer, model
            logger.info(
                "Loaded %s %s model from %s",
                self.config.backend,
                self.config.device,
                path,
            )
            return self

    def close(self) -> None:
        """Wait for inference and release owned model/tokenizer references."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            gc.collect()

    def __enter__(self) -> Self:
        """Load local artifacts on context entry."""
        return self.load()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release resources while preserving any exception.

        Args:
            exc_type:
                Active exception class, if any.

            exc_value:
                Active exception instance, if any.

            traceback:
                Active traceback, if any.

        """
        self.close()

    def generate(self, user_text: str) -> Generation:
        """Generate one independent turn and intercept all tool calls.

        Args:
            user_text:
                Nonempty user request. Prompts beyond the configured context are
                rejected, never truncated. Call `load` before inference.

        """
        if not user_text.strip():
            raise ValueError("User request must not be empty")
        return self.generate_messages([{"role": "user", "content": user_text}])

    def generate_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> Generation:
        """Generate native history, optionally delivering incremental raw text.

        Args:
            messages:
                Nonempty user/assistant/tool history; the developer prompt is inserted.

            on_text:
                Optional decoded-token callback including protocol. Exceptions propagate.

        """
        with self._lock:
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Load FunctionGemma before generating")
            if not messages or any(
                m["role"] not in ("user", "assistant", "tool") for m in messages
            ):
                raise ValueError("Expected nonempty user/assistant/tool history")
            started = time.perf_counter()
            prompt = self._tokenizer.apply_chat_template(
                [
                    {"role": "developer", "content": self.config.system_prompt},
                    *messages,
                ],
                tools=self.tools.schemas(),
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt" if self.config.backend == "torch" else "np",
            )
            input_count = inputs.input_ids.shape[1]
            if (
                input_count + self.config.max_new_tokens
                > self.config.max_context_tokens
            ):
                raise ValueError("Request exceeds configured context budget")
            if self.config.backend == "torch":
                # Keep PyTorch optional in a compiled-only serving process.
                import torch

                timer = _TokenTimer()
                if on_text:
                    timer.decoder = _text_streamer(self._tokenizer, on_text, True)
                with torch.inference_mode():
                    output = self._model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=self.config.max_new_tokens,
                        pad_token_id=self._tokenizer.pad_token_id,
                        streamer=timer,
                    )[0, input_count:]
                output_count = len(output)
                raw = self._tokenizer.decode(
                    output,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                first_token = timer.first_token_seconds
            else:
                # Import this backend only when executing an OpenVINO request.
                import openvino as ov

                output = self._model.generate(
                    ov.Tensor(inputs.input_ids),
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    **(
                        {"streamer": _ov_streamer(self._tokenizer, on_text)}
                        if on_text
                        else {}
                    ),
                )
                raw = self._tokenizer.decode(
                    output.tokens[0],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                output_count = len(output.tokens[0])
                assert output_count == output.perf_metrics.get_num_generated_tokens()
                first_token = output.perf_metrics.get_ttft().mean / 1000.0
            if output_count >= self.config.max_new_tokens:
                raise ValueError(
                    f"Generation reached token limit; refusing partial calls: {raw}"
                )
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
                output_count,
                time.perf_counter() - started,
                first_token,
            )
