"""Benchmark isolated LFM weather routing without dispatching tools.

Example (coordinate exclusive GPU use first)::

    python -m tools.benchmark_lfm_gpu --checkpoint models/openvino-int8 \
        --device 'UHD Graphics 630' --output weather-results.jsonl

The three requests are fixed synthetic replay text. Both backends use the same
checkpoint, tokenizer, tool declarations, synthetic music-off context, greedy
settings, and output limits. The GPU GenAI oracle and custom OpenCL runner are
loaded sequentially; the live service is not controlled. The first cached
request includes prefix construction; subsequent requests reuse the snapshot.
Results are full-generation wall times, including prompt rendering and parsing,
not kernel timings. GenAI and the custom runner use different inference engines
and precisions. Competing workloads, warmup and cache conditions can affect
results. The output is exclusive-created JSON Lines; on failure it contains the
traceback and any completed records. No provider tools or credentials are used.
"""

import argparse
import hashlib
import json
import logging
import statistics
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from hoast.input_text import canonical_user_messages
from hoast.lfm import LFM2, LFMConfig, tool_declarations
from hoast.lfm_gpu.generation import GenerationConfig, GPUGeneration
from hoast.lights import declare_light_tool
from hoast.llm import Generation, ToolRegistry
from hoast.music import declare_music_tools
from hoast.prompts import build_system_prompt
from hoast.weather import declare_weather_tool

LOGGER = logging.getLogger(__name__)
REQUESTS = (
    "what's the weather like today",
    "what's the weather i like today",
    "what's the temperature now",
)


def sha256(path: Path) -> str:
    """Stream a file and return its SHA-256 digest.

    Args:
        path:
            Artifact file to identify.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def common_prefix(sequences: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Return the longest exact shared token prefix, rejecting empty inputs.

    Args:
        sequences:
            Nonempty batch of nonempty token sequences.
    """
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Expected nonempty token sequences")
    prefix: list[int] = []
    for values in zip(*sequences):
        if any(value != values[0] for value in values[1:]):
            break
        prefix.append(values[0])
    if not prefix:
        raise ValueError("Prompts have no common token prefix")
    return tuple(prefix)


def choose_device(devices: Sequence[Any], match: str) -> Any:
    """Select exactly one device by case-insensitive name substring.

    Args:
        devices:
            Enumerated device objects exposing a name attribute.

        match:
            Nonempty case-insensitive device-name substring.
    """
    if not match.strip():
        raise ValueError("GPU device selector must be nonempty")
    selected = [
        device for device in devices if match.casefold() in device.name.casefold()
    ]
    if len(selected) != 1:
        raise ValueError(f"GPU selector matched {len(selected)} devices")
    return selected[0]


def result_record(request: str, result: Generation, wall: float) -> dict[str, Any]:
    """Serialize one completed generation, without executing its calls.

    Args:
        request:
            Synthetic user request.

        result:
            Validated generation from either backend.

        wall:
            External request elapsed time in seconds.
    """
    return {
        "request": request,
        "raw": result.raw,
        "text": result.text,
        "calls": [{"name": c.name, "arguments": c.arguments} for c in result.calls],
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_seconds": wall,
        "first_token_seconds": result.first_token_seconds,
    }


def run_requests(
    generate: Callable[[str], Generation], requests: Sequence[str]
) -> list[dict[str, Any]]:
    """Measure completed requests in order, propagating inference failures.

    Args:
        generate:
            Non-dispatching backend generation callable.

        requests:
            Requests to render and generate independently.
    """
    records = []
    for request in requests:
        started = time.perf_counter()
        result = generate(request)
        records.append(result_record(request, result, time.perf_counter() - started))
    return records


def parity(
    oracle: Sequence[dict[str, Any]], actual: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare raw responses and calls for aligned requests without truncation.

    Args:
        oracle:
            Reference results in request order.

        actual:
            Candidate results in the same order.
    """
    if len(oracle) != len(actual):
        raise ValueError("Result counts differ")
    rows = []
    for reference, candidate in zip(oracle, actual, strict=True):
        if reference["request"] != candidate["request"]:
            raise ValueError("Result requests differ")
        rows.append(
            {
                "request": reference["request"],
                "raw_equal": reference["raw"] == candidate["raw"],
                "calls_equal": reference["calls"] == candidate["calls"],
            }
        )
    return rows


class RunnerTrace:
    """Delegate runner methods and measure only the prefix snapshot operation."""

    runner: Any
    """Exclusively owned underlying runner."""

    workspace: Any
    """Underlying workspace and fixed dimensions."""

    cache_build_seconds: list[float]
    """Measured cache-prefix calls in seconds."""

    def __init__(self, runner: Any) -> None:
        """Wrap a runner while preserving its public generation interface.

        Args:
            runner:
                Exclusively owned real runner.
        """
        self.runner = runner
        self.workspace = runner.workspace
        self.cache_build_seconds = []

    def reset(self) -> None:
        """Reset live runner state without discarding its saved prefix."""
        self.runner.reset()

    def prefill(self, token_ids: list[int]) -> Any:
        """Prefill a full prompt from empty state.

        Args:
            token_ids:
                Full prompt token IDs.
        """
        return self.runner.prefill(token_ids)

    def cache_prefix(self, token_ids: list[int]) -> Any:
        """Capture one prefix and record elapsed time after queue completion.

        Args:
            token_ids:
                Exact stable prefix token IDs.
        """
        started = time.perf_counter()
        value = self.runner.cache_prefix(token_ids)
        self.cache_build_seconds.append(time.perf_counter() - started)
        return value

    def prefill_cached(self, token_ids: list[int]) -> Any:
        """Restore a prefix and evaluate the matching prompt suffix.

        Args:
            token_ids:
                Full prompt beginning with the saved prefix.
        """
        return self.runner.prefill_cached(token_ids)

    def decode(self, token_id: int) -> Any:
        """Append one generated token.

        Args:
            token_id:
                Vocabulary token ID.
        """
        return self.runner.decode(token_id)


def execute(args: argparse.Namespace, emit: Callable[[dict[str, Any]], None]) -> None:
    """Run serialized GPU GenAI and custom GPU full-generation measurements.

    Args:
        args:
            CLI options naming checkpoint, GPU substring, and repetition count.

        emit:
            Durable record sink flushed after each stage.
    """
    # Native runtimes are optional and expensive until this explicit benchmark runs.
    import openvino as ov
    import pyopencl as cl

    from hoast.lfm_gpu.runner import Runner, WorkspaceSpec

    checkpoint = args.checkpoint.resolve()
    artifacts = {
        name: sha256(checkpoint / name)
        for name in (
            "openvino_model.xml",
            "openvino_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "config.json",
        )
    }
    devices = [
        device
        for platform in cl.get_platforms()
        for device in platform.get_devices(device_type=cl.device_type.GPU)
    ]
    device = choose_device(devices, args.device)
    core = ov.Core()
    gpu_ids = [
        name
        for name in core.available_devices
        if name == "GPU" or name.startswith("GPU.")
    ]
    if len(gpu_ids) != 1:
        raise ValueError(
            "OpenVINO must expose exactly one GPU for matched device selection"
        )
    ov_name = str(core.get_property(gpu_ids[0], "FULL_DEVICE_NAME"))
    if args.device.casefold() not in ov_name.casefold():
        raise ValueError("OpenVINO GPU does not match the requested OpenCL device")
    tools = ToolRegistry(
        [declare_weather_tool(), *declare_music_tools(), declare_light_tool()]
    )
    system = build_system_prompt({"status": "observed", "state": "off"})
    emit(
        {
            "stage": "context",
            "checkpoint": str(checkpoint),
            "sha256": artifacts,
            "opencl_device": device.name,
            "opencl_driver": device.driver_version,
            "openvino_device": gpu_ids[0],
            "openvino_name": ov_name,
            "openvino_version": ov.__version__,
            "requests": REQUESTS,
            "music_state": {"status": "observed", "state": "off"},
            "max_new_tokens": 128,
            "max_context_tokens": 4096,
            "repetition_penalty": 1.05,
            "chunk_size": 128,
            "repeats": args.repeats,
            "timing": "full generation; sequential backends; first cached includes build",
        }
    )
    # GenAI's compilation cache is private and temporary, not the live service cache.
    with tempfile.TemporaryDirectory(prefix="lfm-bench-") as cache:
        oracle = LFM2(
            LFMConfig(
                checkpoint,
                cache_dir=Path(cache),
                device="GPU",
                system_prompt=system,
                max_new_tokens=128,
                max_context_tokens=4096,
            ),
            tools,
        )
        started = time.perf_counter()
        oracle.load()
        try:
            emit({"stage": "oracle_load", "seconds": time.perf_counter() - started})
            tokenizer = oracle._tokenizer
            if tokenizer is None:
                raise RuntimeError("Oracle tokenizer was not initialized")
            reference = run_requests(oracle.generate, REQUESTS)
            emit({"stage": "oracle", "results": reference})
        finally:
            oracle.close()
    sequences = []
    for request in REQUESTS:
        messages = canonical_user_messages([{"role": "user", "content": request}])
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, *messages],
            tools=tool_declarations(tools),
            tokenize=False,
            add_generation_prompt=True,
        )
        sequences.append(
            [
                int(token)
                for token in tokenizer(
                    rendered, add_special_tokens=False, return_tensors="np"
                ).input_ids[0]
            ]
        )
    prefix = common_prefix(sequences)
    emit(
        {
            "stage": "prefix",
            "tokens": len(prefix),
            "prompt_lengths": [len(x) for x in sequences],
        }
    )
    started = time.perf_counter()
    runner = Runner(
        cl.CommandQueue(cl.Context([device])),
        checkpoint,
        WorkspaceSpec(4096, 128, prepared_prefill=True, prefix_cache=True),
    )
    traced = RunnerTrace(runner)
    emit(
        {
            "stage": "runner_load",
            "seconds": time.perf_counter() - started,
            "workspace_bytes": sum(runner.workspace.spec.buffer_sizes.values()),
        }
    )
    config = GenerationConfig(
        max_new_tokens=128, max_context_tokens=4096, system_prompt=system
    )
    uncached = GPUGeneration(traced, tokenizer, tools, config)
    plain_results = run_requests(uncached.generate, REQUESTS)
    emit(
        {
            "stage": "uncached",
            "results": plain_results,
            "parity": parity(reference, plain_results),
        }
    )
    cached = GPUGeneration(traced, tokenizer, tools, config, prefix)
    first = run_requests(cached.generate, REQUESTS[:1])
    if len(traced.cache_build_seconds) != 1:
        raise RuntimeError("Expected exactly one successful prefix build")
    emit(
        {
            "stage": "cache_build",
            "prefix_build_seconds": traced.cache_build_seconds[0],
            "first_generation": first[0],
            "parity": parity(reference[:1], first),
        }
    )
    for iteration in range(args.repeats):
        results = run_requests(cached.generate, REQUESTS)
        emit(
            {
                "stage": "warm_cached",
                "iteration": iteration,
                "results": results,
                "parity": parity(reference, results),
                "uncached_parity": parity(plain_results, results),
                "median_wall_seconds": statistics.median(
                    row["wall_seconds"] for row in results
                ),
            }
        )
    if len(traced.cache_build_seconds) != 1:
        raise RuntimeError("Cached warm requests unexpectedly rebuilt the prefix")
    emit({"stage": "complete"})


def main() -> None:
    """Parse explicit options and write durable JSONL progress or failure records."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True, help="Unique GPU name substring")
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSONL output path"
    )
    parser.add_argument("--repeats", type=int, default=1, help="Warm cached rounds")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    with args.output.open("x", encoding="utf-8") as stream:

        def emit(record: dict[str, Any]) -> None:
            """Write and flush one JSONL benchmark record.

            Args:
                record:
                    JSON-serializable evidence or diagnostic data.
            """
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

        try:
            execute(args, emit)
        except Exception:
            emit({"stage": "error", "traceback": traceback.format_exc()})
            LOGGER.exception("LFM benchmark failed; partial results preserved")
            raise
    LOGGER.info("Benchmark results saved to %s", args.output)


if __name__ == "__main__":
    main()
