"""Profile the isolated custom LFM OpenCL runner, not OpenVINO GenAI.

Coordinate exclusive GPU access before running. Example::

    python -m tools.profile_lfm_gpu --checkpoint models/openvino-int8 \
        --device 'UHD Graphics 630' --output profile.jsonl

The fixed weather prompt uses the deployed chat template and declarations. It
must tokenize to exactly 930 tokens; a mismatch fails rather than substituting
a synthetic workload. No provider tool is executed. Each repetition resets the
runner, prefills the complete prompt, then performs a fixed number of greedy
argmax decode steps (not the production generation policy). Kernel timings are
OpenCL event execution durations; wall times include runner host work and its
existing blocking transfers. Transfers and reset are not attributed to kernels.
Instrumentation and profiling can perturb execution. JSONL is exclusively
created and flushed after each record, including full failure traceback.
Use --prepared-kn with --prepared-prefill for K-major prepared projections;
--tiled-attention and --vec4-matvec independently select optimized kernels.
Projection and normalization events retain their IR names. Directly dispatched
kernels use their kernel name and runner source line as an operation label, so
separate call sites are distinguishable; line labels may change with the source.
"""

import argparse
import ast
import inspect
import json
import logging
import textwrap
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np

from hoast.input_text import canonical_user_messages
from hoast.lfm import tool_declarations
from hoast.lfm_gpu.runner import WorkspaceSpec
from hoast.lights import declare_light_tool
from hoast.llm import ToolRegistry
from hoast.music import declare_music_tools
from hoast.prompts import build_system_prompt
from hoast.weather import declare_weather_tool
from tools.benchmark_lfm_gpu import REQUESTS, choose_device, sha256

LOGGER = logging.getLogger(__name__)


def layer_loop_lines(method: Callable[..., Any]) -> tuple[int, int]:
    """Locate the layer loop in a runner chunk method, failing on ambiguity.

    Args:
        method:
            Original Python chunk method whose source is inspected.
    """
    source, first = inspect.getsourcelines(method)
    tree = ast.parse(textwrap.dedent("".join(source)))
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "layer"
    ]
    if len(loops) != 1 or loops[0].end_lineno is None:
        raise ValueError("Expected exactly one layer loop in runner chunk")
    return first + loops[0].lineno - 1, first + loops[0].end_lineno - 1


class KernelTrace:
    """Temporarily replace a runner's dispatch with event-retaining dispatch."""

    runner: Any
    """Exclusively owned runner."""
    original: Any
    """Original bound dispatch method."""
    bounds: tuple[int, int]
    """Inclusive source line bounds of the layer loop."""
    events: list[tuple[dict[str, Any], Any]]
    """Labels and live OpenCL events, in enqueue order."""

    def __init__(self, runner: Any) -> None:
        """Prepare tracing for an exclusively owned runner.

        Args:
            runner:
                Initialized runner on an in-order profiling queue.
        """
        self.runner = runner
        self.original = runner._run
        self.bounds = layer_loop_lines(type(runner)._chunk)
        self.events = []

    def dispatch(
        self,
        name: str,
        global_size: tuple[int, ...],
        local_size: tuple[int, ...] | None,
        *args: Any,
    ) -> None:
        """Enqueue exactly the runner's kernel and retain its event and labels.

        Args:
            name:
                Original kernel name.

            global_size:
                Global work dimensions.

            local_size:
                Local work dimensions, or None.

            *args:
                Unmodified kernel arguments.
        """
        frame = inspect.currentframe()
        operation = None
        layer = None
        line = None
        try:
            frame = frame.f_back if frame else None
            while frame is not None:
                if frame.f_code.co_name in ("_project", "_norm"):
                    operation = frame.f_locals["name"]
                if frame.f_code.co_name == "_chunk":
                    line = frame.f_lineno
                    if self.bounds[0] <= line <= self.bounds[1]:
                        layer = frame.f_locals["layer"]
                    break
                frame = frame.f_back
        finally:
            del frame
        if line is None:
            raise RuntimeError("Kernel dispatch outside runner chunk")
        if operation is None:
            operation = f"{name}@line:{line}"
        event = self.runner.kernels[name](
            self.runner.workspace.queue, global_size, local_size, *args
        )
        self.events.append(
            (
                {
                    "kernel": name,
                    "operation": operation,
                    "layer": layer,
                    "source_line": line,
                    "global_size": global_size,
                    "local_size": local_size,
                },
                event,
            )
        )

    def __enter__(self) -> Self:
        """Install the temporary per-instance dispatch hook."""
        self.runner._run = self.dispatch
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Restore the original dispatch even when inference fails.

        Args:
            exc_type:
                Exception class, if any.

            exc:
                Exception instance, if any.

            tb:
                Exception traceback, if any.
        """
        self.runner._run = self.original

    def drain(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Read completed events and aggregate durations by layer and operation.

        Caller must first complete the phase; this method adds no queue finish.
        """
        rows = []
        totals: dict[tuple[Any, Any, str], list[int]] = defaultdict(list)
        for label, event in self.events:
            start, end = int(event.profile.start), int(event.profile.end)
            if end < start:
                raise ValueError("Invalid OpenCL event timestamps")
            duration = end - start
            rows.append(
                {**label, "start_ns": start, "end_ns": end, "duration_ns": duration}
            )
            totals[(label["layer"], label["operation"], label["kernel"])].append(
                duration
            )
        self.events.clear()
        summary = [
            {
                "layer": layer,
                "operation": operation,
                "kernel": kernel,
                "count": len(values),
                "device_ns": sum(values),
            }
            for (layer, operation, kernel), values in totals.items()
        ]
        return rows, summary


def prompt_ids(tokenizer: Any) -> list[int]:
    """Render the fixed weather request and require the intended token count.

    Args:
        tokenizer:
            Local checkpoint tokenizer with its native chat template.
    """
    tools = ToolRegistry(
        [declare_weather_tool(), *declare_music_tools(), declare_light_tool()]
    )
    system = build_system_prompt({"status": "observed", "state": "off"})
    messages = canonical_user_messages([{"role": "user", "content": REQUESTS[2]}])
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, *messages],
        tools=tool_declarations(tools),
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = [
        int(token)
        for token in tokenizer(
            rendered, add_special_tokens=False, return_tensors="np"
        ).input_ids[0]
    ]
    if len(ids) != 930:
        raise ValueError(f"Expected 930 prompt tokens; got {len(ids)}")
    return ids


def workspace_spec(args: argparse.Namespace) -> WorkspaceSpec:
    """Build the exact workspace configuration used and reported by the profiler.

    Args:
        args:
            Parsed profiling options, including runner feature switches.
    """
    return WorkspaceSpec(
        930 + args.decode_steps,
        args.chunk_size,
        prepared_prefill=args.prepared_prefill,
        prepared_kn=args.prepared_kn,
        tiled_attention=args.tiled_attention,
        vec4_matvec=args.vec4_matvec,
    )


def execute(args: argparse.Namespace, emit: Callable[[dict[str, Any]], None]) -> None:
    """Run isolated prefill and fixed-step decode event profiling.

    Args:
        args:
            Validated command-line arguments.

        emit:
            Flushed JSONL record sink.
    """
    # Native GPU and tokenizer dependencies are loaded only for an explicit run.
    import pyopencl as cl
    from transformers import AutoTokenizer

    from hoast.lfm_gpu.runner import Runner

    checkpoint = args.checkpoint.resolve()
    spec = workspace_spec(args)
    names = (
        "openvino_model.xml",
        "openvino_model.bin",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    hashes = {name: sha256(checkpoint / name) for name in names}
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    ids = prompt_ids(tokenizer)
    devices = [
        device
        for platform in cl.get_platforms()
        for device in platform.get_devices(device_type=cl.device_type.GPU)
    ]
    device = choose_device(devices, args.device)
    runner_source = Path(inspect.getfile(Runner)).resolve()
    sources = [
        runner_source,
        Path(__file__).resolve(),
        *sorted(runner_source.parent.glob("*.cl")),
    ]
    source_hashes = {str(path): sha256(path) for path in sources}
    emit(
        {
            "stage": "context",
            "checkpoint": str(checkpoint),
            "sha256": hashes,
            "source_sha256": source_hashes,
            "request": REQUESTS[2],
            "prompt_token_ids": ids,
            "prompt_tokens": len(ids),
            "device": device.name,
            "driver": device.driver_version,
            "platform": device.platform.name,
            "chunk_size": args.chunk_size,
            "prepared_prefill": spec.prepared_prefill,
            "prepared_kn": spec.prepared_kn,
            "tiled_attention": spec.tiled_attention,
            "vec4_matvec": spec.vec4_matvec,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "decode_steps": args.decode_steps,
            "wall_only": args.wall_only,
            "timing": "kernel device execution only; wall includes host and blocking transfers; reset excluded; not GenAI",
        }
    )
    queue = cl.CommandQueue(
        cl.Context([device]), properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    runner = Runner(queue, checkpoint, spec)
    trace = KernelTrace(runner)
    with nullcontext() if args.wall_only else trace:
        for iteration in range(args.warmups + args.repeats):
            runner.reset()
            warmup = iteration < args.warmups
            started = time.perf_counter_ns()
            logits = runner.prefill(ids)
            prefill_wall = time.perf_counter_ns() - started
            rows, summary = trace.drain()
            emit(
                {
                    "stage": "prefill",
                    "iteration": iteration,
                    "warmup": warmup,
                    "wall_ns": prefill_wall,
                    "events": rows,
                    "summary": summary,
                }
            )
            for step in range(args.decode_steps):
                token = int(np.argmax(logits))
                started = time.perf_counter_ns()
                logits = runner.decode(token)
                wall = time.perf_counter_ns() - started
                rows, summary = trace.drain()
                emit(
                    {
                        "stage": "decode",
                        "iteration": iteration,
                        "warmup": warmup,
                        "step": step,
                        "input_token": token,
                        "wall_ns": wall,
                        "events": rows,
                        "summary": summary,
                    }
                )
    emit({"stage": "complete"})


def main() -> None:
    """Parse options and preserve all completed records and full errors."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--prepared-prefill", action="store_true")
    parser.add_argument(
        "--prepared-kn",
        action="store_true",
        help="Use K-major prepared weights; requires --prepared-prefill",
    )
    parser.add_argument(
        "--tiled-attention",
        action="store_true",
        help="Use the tiled attention-score kernel",
    )
    parser.add_argument(
        "--vec4-matvec",
        action="store_true",
        help="Use the vectorized single-row projection kernel",
    )
    parser.add_argument(
        "--wall-only",
        action="store_true",
        help="Measure same workload without kernel dispatch instrumentation",
    )
    args = parser.parse_args()
    if (
        args.warmups < 0
        or args.repeats < 1
        or args.decode_steps < 1
        or not 1 <= args.chunk_size <= 930
    ):
        parser.error("Invalid warmups, repeats, decode steps or chunk size")
    if args.prepared_kn and not args.prepared_prefill:
        parser.error("--prepared-kn requires --prepared-prefill")
    with args.output.open("x", encoding="utf-8") as stream:

        def emit(record: dict[str, Any]) -> None:
            """Write and flush a JSONL record.

            Args:
                record:
                    Serializable profile or diagnostic record.
            """
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

        try:
            execute(args, emit)
        except BaseException:
            emit({"stage": "error", "traceback": traceback.format_exc()})
            LOGGER.exception("Profile failed; partial results preserved")
            raise
    LOGGER.info("Profile saved to %s", args.output)


if __name__ == "__main__":
    main()
