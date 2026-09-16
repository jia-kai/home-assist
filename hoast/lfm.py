"""LFM2.5 OpenVINO serving with native history and non-executing tool parsing."""

import ast
import gc
import io
import json
import math
import threading
import time
import tokenize
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from pydantic import JsonValue

from .input_text import canonical_user_messages
from .llm import (
    DEFAULT_CACHE,
    GeneratedCallError,
    Generation,
    ToolCall,
    ToolRegistry,
    _ov_streamer,
)
from .logging import get_logger

logger = get_logger(__name__)
SYSTEM_PROMPT = "You are a helpful assistant."
_START = "<|tool_call_start|>"
_END = "<|tool_call_end|>"


class LFMParseError(ValueError):
    """Malformed or unsupported native tool-call syntax in a decoded response."""


def tool_declarations(tools: ToolRegistry) -> list[dict[str, Any]]:
    """Render native declarations without schema bookkeeping becoming arguments.

    Runtime validation remains strict. Parameter names, descriptions, required
    fields, types, enums, and constraints remain in the prompt; titles and object
    bookkeeping omitted by FunctionGemma's template are not transmitted to LFM.

    Args:
        tools:
            Shared typed registry defining the actual callable contracts.

    """

    def parameters(node: dict[str, Any], root: bool = False) -> dict[str, Any]:
        """Normalize one schema node while retaining argument constraints.

        Args:
            node:
                Expanded JSON Schema node supplied by the registry.

            root:
                Whether this is a function's outer parameter object.

        """
        result: dict[str, Any] = {"type": node["type"]}
        if not root and "description" in node:
            result["description"] = node["description"]
        if "properties" in node:
            result["properties"] = {
                name: parameters(value) for name, value in node["properties"].items()
            }
        if "items" in node:
            result["items"] = parameters(node["items"])
        if "required" in node:
            result["required"] = node["required"]
        excluded = {
            "type",
            "description",
            "properties",
            "items",
            "required",
            "title",
            "additionalProperties",
            "$schema",
        }
        result.update(
            {name: value for name, value in node.items() if name not in excluded}
        )
        return result

    return [
        {
            "name": schema["function"]["name"],
            "description": schema["function"]["description"],
            "parameters": parameters(schema["function"]["parameters"], root=True),
        }
        for schema in tools.schemas()
    ]


def _call_expression(source: str) -> tuple[str, str]:
    """Locate a complete outer call list, ignoring delimiters inside quoted values.

    Args:
        source:
            Decoded text immediately following the native tool-call start marker.

    """
    source = source.lstrip()
    if not source.startswith("["):
        raise LFMParseError("LFM tool calls must form a list")
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.OP:
                continue
            if token.string in ("[", "(", "{"):
                depth += 1
            elif token.string in ("]", ")", "}"):
                depth -= 1
                if depth == 0:
                    row, column = token.end
                    end = (
                        sum(
                            len(line)
                            for line in source.splitlines(keepends=True)[: row - 1]
                        )
                        + column
                    )
                    remainder = source[end:].lstrip()
                    if not remainder.startswith(_END):
                        raise LFMParseError("Missing LFM tool-call end marker")
                    return source[:end], remainder[len(_END) :]
    except (tokenize.TokenError, IndentationError) as error:
        raise LFMParseError("Malformed LFM call-list tokens") from error
    raise LFMParseError("Unterminated LFM tool-call list")


def _literal(node: ast.AST, depth: int = 0) -> JsonValue:
    """Read a bounded JSON-like literal from an AST without evaluating code.

    Args:
        node:
            Literal argument expression. Calls, attributes, comprehensions,
            indexing, arithmetic, unpacking, and arbitrary names are rejected.

        depth:
            Current nesting level, limited to 32.

    """
    if depth > 32:
        raise LFMParseError("LFM arguments exceed maximum nesting depth")
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
    elif isinstance(node, ast.Name) and node.id in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[node.id]
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal(node.operand, depth + 1)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return -value
    elif isinstance(node, ast.List):
        return [_literal(item, depth + 1) for item in node.elts]
    elif isinstance(node, ast.Dict):
        result: dict[str, JsonValue] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise LFMParseError("Tool object keys must be literal strings")
            if key.value in result:
                raise LFMParseError(f"Duplicate object key: {key.value}")
            result[key.value] = _literal(value, depth + 1)
        return result
    raise LFMParseError(f"Unsupported tool argument expression: {type(node).__name__}")


def parse_response(raw: str) -> tuple[str, tuple[ToolCall, ...]]:
    """Parse LFM's delimited Python-like call lists without executing expressions.

    Args:
        raw:
            Generated tokens decoded with special tokens preserved. At most
            65,536 characters and 32 calls are accepted. Ordinary surrounding
            assistant text is returned separately; malformed batches raise.
            Truncated delimiters and unsupported reasoning protocol also raise.

    """
    if len(raw) > 65536:
        raise LFMParseError("LFM response exceeds parser size limit")
    source = raw.strip().removesuffix("<|im_end|>").rstrip()
    texts: list[str] = []
    calls: list[ToolCall] = []
    while _START in source:
        text, _, remainder = source.partition(_START)
        if any(marker in text for marker in ("<|", "<think>", "</think>")):
            raise LFMParseError("Unexpected protocol token before LFM tool call")
        texts.append(text)
        expression, source = _call_expression(remainder)
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except (SyntaxError, RecursionError) as error:
            raise LFMParseError("Malformed LFM tool-call expression") from error
        if not isinstance(tree.body, ast.List):
            raise LFMParseError("LFM tool calls must form a list")
        for node in tree.body.elts:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                raise LFMParseError("Only direct named tool calls are accepted")
            if node.args:
                raise LFMParseError("Tool arguments must be named")
            arguments: dict[str, JsonValue] = {}
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg in arguments:
                    raise LFMParseError("Unpacked or duplicate tool arguments")
                arguments[keyword.arg] = _literal(keyword.value)
            calls.append(ToolCall(node.func.id, arguments))
            if len(calls) > 32:
                raise LFMParseError("Too many LFM tool calls")
    if any(marker in source for marker in ("<|", "<think>", "</think>")):
        raise LFMParseError("Unexpected or truncated LFM protocol token")
    if source.endswith("<"):
        raise LFMParseError("Truncated LFM protocol token")
    texts.append(source)
    return "".join(texts).strip(), tuple(calls)


@dataclass(slots=True, frozen=True)
class LFMConfig:
    """Local LFM2.5 model and deterministic OpenVINO serving configuration."""

    model_path: Path
    """Prepared OpenVINO artifact directory."""

    cache_dir: Path = DEFAULT_CACHE
    """Working-directory-relative compiled-model cache root."""

    threads: int = 1
    """One/two CPU workers; the GPU default uses a one-core host budget."""

    max_new_tokens: int = 128
    """Maximum output tokens; length-limited responses are rejected."""

    max_context_tokens: int = 4096
    """Maximum prompt plus reserved output tokens, bounded by 32768."""

    system_prompt: str = SYSTEM_PROMPT
    """Native system instruction, prepended to each request history."""

    repetition_penalty: float = 1.05
    """LFM-recommended repetition penalty; use 1.0 for controlled comparisons."""

    device: Literal["CPU", "GPU"] = "GPU"
    """OpenVINO device, GPU by default; loading errors never trigger fallback."""

    def __post_init__(self) -> None:
        """Reject invalid serving settings before allocating a model."""
        if self.device not in ("CPU", "GPU"):
            raise ValueError("Device must be CPU or GPU")
        if (
            self.threads not in (1, 2)
            or not 0 < self.max_new_tokens < self.max_context_tokens <= 32768
        ):
            raise ValueError("Invalid thread count or context/output budget")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("Repetition penalty must be finite and positive")
        if not self.system_prompt.strip():
            raise ValueError("System prompt must not be empty")

    @classmethod
    def from_cache(
        cls,
        cache_dir: Path = DEFAULT_CACHE,
        threads: int | None = None,
        device: Literal["CPU", "GPU"] = "GPU",
    ) -> Self:
        """Resolve the prepared export without network access.

        Args:
            cache_dir:
                Cache root used by `tools/prepare_llm.py`.

            threads:
                One or two CPU workers; None selects one for GPU, two for CPU.

            device:
                Explicit OpenVINO execution device, GPU by default, without fallback.

        """
        manifest = json.loads((cache_dir / "lfm/model.json").read_text())
        if manifest["model_id"] != "LiquidAI/LFM2.5-350M":
            raise ValueError("Unexpected model in LFM manifest")
        return cls(
            Path(manifest["path"]),
            cache_dir=cache_dir,
            threads=threads if threads is not None else (1 if device == "GPU" else 2),
            device=device,
        )


@dataclass(slots=True)
class LFM2:
    """Own a local LFM pipeline, serialize requests, and intercept typed tools.

    Histories are caller-owned; there is no implicit conversation state or tool
    execution. Use `tools.dispatch(result.calls)` for explicit validated dispatch.
    """

    config: LFMConfig
    """Immutable serving settings."""

    tools: ToolRegistry
    """Shared typed tool registry and explicit dispatcher."""

    _model: Any = field(default=None, init=False, repr=False)
    """Owned OpenVINO pipeline or None while unloaded."""

    _tokenizer: Any = field(default=None, init=False, repr=False)
    """Official local tokenizer and chat template."""

    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    """Serializes model loading, inference, and resource release."""

    def load(self) -> Self:
        """Load upstream or fine-tuned fast-tokenizer artifacts; failures leave unloaded."""
        # Native inference libraries remain lazy for standalone parser/tool use.
        import openvino_genai
        from transformers import PreTrainedTokenizerFast

        with self._lock:
            if self._model is not None:
                return self
            path = self.config.model_path.resolve()
            if not (path / "openvino_model.xml").is_file():
                raise FileNotFoundError(f"Prepare LFM first: {path}")
            metadata = json.loads((path / "tokenizer_config.json").read_text())
            tokenizer_class = metadata.pop("tokenizer_class")
            if tokenizer_class == "TokenizersBackend":
                if metadata.pop("backend") != "tokenizers":
                    raise ValueError("Unexpected LFM tokenizer backend metadata")
                # Translate upstream Transformers 5 names for the 4.x runtime.
                metadata["additional_special_tokens"] = metadata.pop(
                    "extra_special_tokens"
                )
                metadata["extra_special_tokens"] = metadata.pop(
                    "model_specific_special_tokens"
                )
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(path / "tokenizer.json"),
                    chat_template=(path / "chat_template.jinja").read_text(),
                    **metadata,
                )
            elif tokenizer_class == "PreTrainedTokenizerFast":
                if "backend" in metadata and metadata["backend"] != "tokenizers":
                    raise ValueError("Unexpected LFM tokenizer backend metadata")
                tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    path, local_files_only=True
                )
            else:
                raise ValueError("Unexpected LFM tokenizer backend metadata")
            cache = self.config.cache_dir.resolve() / "lfm/compiled"
            cache.mkdir(parents=True, exist_ok=True)
            model = openvino_genai.LLMPipeline(
                str(path),
                self.config.device,
                CACHE_DIR=str(cache),
                CACHE_MODE="OPTIMIZE_SPEED",
                PERFORMANCE_HINT="LATENCY",
                NUM_STREAMS="1",
                **(
                    {
                        "INFERENCE_NUM_THREADS": self.config.threads,
                        "INFERENCE_PRECISION_HINT": "f32",
                    }
                    if self.config.device == "CPU"
                    else {
                        "INFERENCE_PRECISION_HINT": "f16",
                        "GPU_QUEUE_THROTTLE": "LOW",
                        "COMPILATION_NUM_THREADS": 1,
                    }
                ),
            )
            self._model, self._tokenizer = model, tokenizer
            logger.info("Loaded LFM2.5 %s model from %s", self.config.device, path)
            return self

    def close(self) -> None:
        """Wait for inference, release owned references, and permit reloading."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            gc.collect()

    def __enter__(self) -> Self:
        """Load the model on context entry."""
        return self.load()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release owned model resources without suppressing exceptions.

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
        """Generate one independent user turn without executing tools.

        Args:
            user_text:
                Nonempty user request.

        """
        if not user_text.strip():
            raise ValueError("User request must not be empty")
        return self.generate_messages([{"role": "user", "content": user_text}])

    def generate_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> Generation:
        """Canonicalize user language and generate with the native structured template.

        Args:
            messages:
                Nonempty Hugging Face-style history. System/developer roles are
                rejected because the configured system prompt is inserted here.
                Tool calls use function name/argument mappings; tool results may
                contain JSON-compatible mappings or strings. External user text is
                canonicalized with basic punctuation retained; structured history is preserved. No history
                is retained.

            on_text:
                Optional incremental raw-text callback preserving native protocol.
                Callback exceptions abort generation and propagate.

        """
        # Tensor wrapping is required only when actually generating a response.
        import openvino as ov

        with self._lock:
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Load LFM2 before generating")
            if not messages or any(
                message["role"] not in ("user", "assistant", "tool")
                for message in messages
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
            prompt = self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.config.system_prompt},
                    *canonical_user_messages(messages),
                ],
                tools=tool_declarations(self.tools),
                tokenize=False,
                add_generation_prompt=True,
            )
            # CPU int64 input IDs have shape (batch=1, prompt_tokens).
            inputs = self._tokenizer(
                prompt, add_special_tokens=False, return_tensors="np"
            )
            input_count = inputs.input_ids.shape[1]
            if (
                input_count + self.config.max_new_tokens
                > self.config.max_context_tokens
            ):
                raise ValueError("Request exceeds configured context budget")
            output = self._model.generate(
                ov.Tensor(inputs.input_ids),
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
                repetition_penalty=self.config.repetition_penalty,
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
            count = len(output.tokens[0])
            if count >= self.config.max_new_tokens:
                raise ValueError("Generation reached output limit")
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
                count,
                time.perf_counter() - started,
                output.perf_metrics.get_ttft().mean / 1000,
            )
