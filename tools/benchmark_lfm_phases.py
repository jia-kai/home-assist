"""Matched stateful LFM phase benchmark; coordinate exclusive GPU access first.

Each invocation runs exactly one backend in a fresh process. Supply the same
JSON token fixture, checkpoint, affinity and options to ``cpu``, ``ov-gpu`` and
``custom``. The fixture is {"prompt": [IDs], "continuation": [IDs]}; decode
inputs are teacher forced, including EOS if supplied, and never sampled. The
model IR is unmodified. Prefill is one full prompt call; custom may internally
chunk it. Every decode has the same growing context. Reset and compilation are
outside timed phases. Output logits are copied to host on every backend.

Example: python -m tools.benchmark_lfm_phases --backend cpu --checkpoint PATH
    --tokens tokens.json --cpus 2 --output cpu.jsonl --logits cpu.npz
Run GPU backends only after coordination; compare using --reference cpu.npz.
Use separate wall-only and --profile invocations: profiling changes execution.
OpenVINO profiling node sums and custom OpenCL kernel sums exclude transfers and
host overhead and are not necessarily device busy time. CPU and GPU execution
precisions and internal implementations may differ. Reported RSS is process peak,
not GPU memory; workspace sizes are estimates and exclude driver overhead.
"""

import argparse
import inspect
import json
import logging
import os
import platform
import resource
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from tools.benchmark_lfm_gpu import choose_device, sha256
from tools.lfm_gpu_reference import compare, logit_summary

LOGGER = logging.getLogger(__name__)


def load_tokens(path: Path) -> tuple[list[int], list[int]]:
    """Read bounded vocabulary IDs from a strict two-field token fixture.

    Args:
        path:
            JSON fixture containing nonempty prompt and continuation arrays.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"prompt", "continuation"}:
        raise ValueError("Token fixture requires exactly prompt and continuation")
    for key in ("prompt", "continuation"):
        tokens = value[key]
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or not 0 <= token < 65536 for token in tokens)
        ):
            raise ValueError(f"Invalid nonempty token array: {key}")
    return value["prompt"], value["continuation"]


def affinity(value: str) -> list[int]:
    """Set the current process affinity to an explicit nonempty CPU list.

    Args:
        value:
            Comma-separated logical CPU numbers, available in current affinity.
    """
    parts = value.split(",")
    if any(not part.isascii() or not part.isdigit() for part in parts):
        raise ValueError("CPU affinity must be comma-separated decimal IDs")
    cpus = [int(part) for part in parts]
    if len(set(cpus)) != len(cpus) or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("CPU IDs must be unique and within the current affinity")
    os.sched_setaffinity(0, cpus)
    return cpus


class OpenVINOBackend:
    """Unmodified stateful IR request with synchronous host logit copies."""

    request: Any
    """Compiled OpenVINO inference request."""
    length: int
    """Number of tokens in the current request state."""
    profiling: bool
    """Whether node profiling is enabled."""
    output_shape: list[int]
    """Full raw IR logits shape from the latest inference."""

    def __init__(self, request: Any, profiling: bool) -> None:
        """Bind the compiled request.

        Args:
            request:
                Fresh stateful OpenVINO request.
            profiling:
                Whether the compiled model has profiling enabled.
        """
        self.request = request
        self.profiling = profiling
        self.length = 0
        self.output_shape = []

    def reset(self) -> None:
        """Clear model state and token count outside the measured region."""
        self.request.reset_state()
        self.length = 0

    def infer(self, tokens: list[int]) -> np.ndarray:
        """Append tokens and return independent final-position logits.

        Args:
            tokens:
                Nonempty token IDs appended at the current context length.
        """
        self.length += len(tokens)
        self.request.infer(
            {
                "input_ids": np.asarray([tokens], dtype=np.int64),
                "attention_mask": np.ones((1, self.length), dtype=np.int64),
                "beam_idx": np.array([0], dtype=np.int32),
            }
        )
        output = np.asarray(self.request.get_tensor("logits").data)
        self.output_shape = list(output.shape)
        return output.reshape(-1, 65536)[-1].copy()

    def device_ns(self) -> int | None:
        """Sum reported node real times, not necessarily elapsed device time."""
        if not self.profiling:
            return None
        return sum(
            int(item.real_time.total_seconds() * 1e9)
            for item in self.request.get_profiling_info()
        )


class CustomBackend:
    """Adapter for the custom GPU runner and optional kernel event tracing."""

    runner: Any
    """Exclusive custom runner."""
    trace: Any
    """Optional installed kernel tracer."""
    first: bool
    """Whether the next call is prefill."""

    def __init__(self, runner: Any, trace: Any) -> None:
        """Bind a runner and optional active trace.

        Args:
            runner:
                Initialized custom runner.
            trace:
                Entered KernelTrace or None.
        """
        self.runner = runner
        self.trace = trace
        self.first = True

    def reset(self) -> None:
        """Reset runner state outside the measured region."""
        self.runner.reset()
        self.first = True

    def infer(self, tokens: list[int]) -> np.ndarray:
        """Append the full prompt or one teacher-forced token.

        Args:
            tokens:
                Complete prompt on first call, exactly one ID subsequently.
        """
        if self.first:
            self.first = False
            return self.runner.prefill(tokens)
        if len(tokens) != 1:
            raise ValueError("Decode requires one token")
        return self.runner.decode(tokens[0])

    def device_ns(self) -> int | None:
        """Drain retained kernel events and sum execution durations."""
        if self.trace is None:
            return None
        rows, _ = self.trace.drain()
        return sum(row["duration_ns"] for row in rows)


def measure(
    backend: Any,
    prompt: list[int],
    continuation: list[int],
    warmups: int,
    repeats: int,
    emit: Callable[[dict[str, Any]], None],
) -> np.ndarray:
    """Run independent sequences and return logits from the last repetition.

    Args:
        backend:
            Stateful adapter with reset, infer, and device_ns methods.
        prompt:
            Complete input prompt token IDs.
        continuation:
            Teacher-forced token IDs appended one at a time.
        warmups:
            Number of excluded warmup sequences.
        repeats:
            Number of measured sequences, at least one.
        emit:
            Flushed JSON record callback.
    """
    if warmups < 0 or repeats < 1:
        raise ValueError("Invalid warmup or repetition count")
    last: list[np.ndarray] = []
    for iteration in range(warmups + repeats):
        backend.reset()
        last = []
        for step, tokens in enumerate([prompt, *([token] for token in continuation)]):
            start_epoch_ns = time.time_ns()
            start = time.perf_counter_ns()
            logits = backend.infer(tokens)
            wall_ns = time.perf_counter_ns() - start
            device_ns = backend.device_ns()
            if logits.shape != (65536,) or not np.isfinite(logits).all():
                raise ValueError("Invalid logits")
            last.append(logits.copy())
            emit(
                {
                    "stage": "phase",
                    "iteration": iteration,
                    "warmup": iteration < warmups,
                    "phase": "prefill" if step == 0 else "decode",
                    "decode_index": None if step == 0 else step - 1,
                    "input_token": None if step == 0 else tokens[0],
                    "context_tokens": len(prompt) + step,
                    "wall_ns": wall_ns,
                    "start_epoch_ns": start_epoch_ns,
                    "raw_output_shape": getattr(backend, "output_shape", [65536]),
                    "profile_sum_ns": device_ns,
                    "logits": logit_summary(logits),
                }
            )
    return np.stack(last)


def execute(args: argparse.Namespace, emit: Callable[[dict[str, Any]], None]) -> None:
    """Run a single backend and preserve phase timings and parity evidence.

    Args:
        args:
            Parsed benchmark options.
        emit:
            Flushed JSON record callback.
    """
    cpus = affinity(args.cpus)
    prompt, continuation = load_tokens(args.tokens)
    checkpoint = args.checkpoint.resolve()
    capacity = (
        args.capacity if args.capacity is not None else len(prompt) + len(continuation)
    )
    if capacity < len(prompt) + len(continuation):
        raise ValueError("Capacity must accommodate prompt and continuation")
    if args.chunk_size < 1 or args.chunk_size > capacity:
        raise ValueError("Chunk size must be within context capacity")
    hashes = {
        name: sha256(checkpoint / name)
        for name in ("openvino_model.xml", "openvino_model.bin", "config.json")
    }
    context: dict[str, Any] = {
        "stage": "context",
        "backend": args.backend,
        "checkpoint": str(checkpoint),
        "artifact_sha256": hashes,
        "fixture_sha256": sha256(args.tokens),
        "prompt": prompt,
        "continuation": continuation,
        "capacity": capacity,
        "chunk_size": args.chunk_size,
        "prepared_prefill": args.prepared_prefill,
        "prepared_kn": args.prepared_kn,
        "tiled_attention": args.tiled_attention,
        "vec4_matvec": args.vec4_matvec,
        "persistent_kn": args.persistent_kn,
        "persistent_kn_budget_bytes": args.persistent_kn_budget_bytes,
        "cpus": cpus,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "profiling": args.profile,
        "platform": platform.platform(),
        "correctness_atol": args.atol,
        "correctness_rtol": args.rtol,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": sha256(Path(__file__)),
        "timing": "synchronous infer plus host logits copy; excludes reset, profiling extraction and validation",
    }
    trace = None
    if args.backend == "custom":
        # Optional GPU dependencies are needed only for explicit device runs.
        import pyopencl as cl

        from hoast.lfm_gpu.runner import Runner, WorkspaceSpec
        from tools.profile_lfm_gpu import KernelTrace

        devices = [
            d
            for p in cl.get_platforms()
            for d in p.get_devices(device_type=cl.device_type.GPU)
        ]
        device = choose_device(devices, args.device)
        gpu_context = cl.Context([device])
        queue = (
            cl.CommandQueue(
                gpu_context, properties=cl.command_queue_properties.PROFILING_ENABLE
            )
            if args.profile
            else cl.CommandQueue(gpu_context)
        )
        spec = WorkspaceSpec(
            capacity,
            args.chunk_size,
            prepared_prefill=args.prepared_prefill,
            prepared_kn=args.prepared_kn,
            tiled_attention=args.tiled_attention,
            vec4_matvec=args.vec4_matvec,
            persistent_kn=args.persistent_kn,
            persistent_kn_budget_bytes=args.persistent_kn_budget_bytes,
        )
        load_start = time.perf_counter_ns()
        runner = Runner(queue, checkpoint, spec)
        load_ns = time.perf_counter_ns() - load_start
        if args.profile:
            trace = KernelTrace(runner)
            trace.__enter__()
        backend: Any = CustomBackend(runner, trace)
        context.update(
            {
                "device": device.name,
                "driver": device.driver_version,
                "global_mem_bytes": int(device.global_mem_size),
                "max_allocation_bytes": int(device.max_mem_alloc_size),
                "runner_construction_ns": load_ns,
                "workspace_buffer_bytes": spec.buffer_sizes,
                "runner_source_sha256": sha256(Path(inspect.getfile(Runner))),
                "kernel_sha256": {
                    p.name: sha256(p)
                    for p in Path(inspect.getfile(Runner)).parent.glob("*.cl")
                },
            }
        )
    else:
        # OpenVINO is optional for fixture-only imports and tests.
        import openvino as ov

        core = ov.Core()
        device_name = "CPU"
        if args.backend == "ov-gpu":
            ids = [
                name
                for name in core.available_devices
                if name == "GPU" or name.startswith("GPU.")
            ]
            if (
                len(ids) != 1
                or args.device.casefold()
                not in str(core.get_property(ids[0], "FULL_DEVICE_NAME")).casefold()
            ):
                raise ValueError(
                    "Expected one OpenVINO GPU matching the device selector"
                )
            device_name = ids[0]
        config: dict[str, Any] = {
            "PERFORMANCE_HINT": "LATENCY",
            "PERF_COUNT": args.profile,
        }
        if args.backend == "cpu":
            config.update(
                {
                    "INFERENCE_NUM_THREADS": len(cpus),
                    "NUM_STREAMS": 1,
                    "INFERENCE_PRECISION_HINT": "f32",
                }
            )
        model = core.read_model(str(checkpoint / "openvino_model.xml"))
        compiled = core.compile_model(model, device_name, config)
        backend = OpenVINOBackend(compiled.create_infer_request(), args.profile)
        context.update(
            {
                "device": device_name,
                "device_name": str(core.get_property(device_name, "FULL_DEVICE_NAME")),
                "openvino_version": ov.__version__,
                "compile_config": config,
            }
        )
    emit(context)
    try:
        logits = measure(
            backend, prompt, continuation, args.warmups, args.repeats, emit
        )
    finally:
        if trace is not None:
            trace.__exit__(None, None, None)
    with args.logits.open("xb") as output:
        np.savez_compressed(
            output,
            logits=logits,
            fixture_sha256=sha256(args.tokens),
            xml_sha256=hashes["openvino_model.xml"],
            bin_sha256=hashes["openvino_model.bin"],
        )
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as ref:
            for key, expected in (
                ("fixture_sha256", sha256(args.tokens)),
                ("xml_sha256", hashes["openvino_model.xml"]),
                ("bin_sha256", hashes["openvino_model.bin"]),
            ):
                if str(ref[key]) != expected:
                    raise ValueError(f"Reference {key} mismatch")
            reference = ref["logits"]
            if reference.shape != logits.shape:
                raise ValueError("Reference logits shape mismatch")
            for step, (a, b) in enumerate(zip(reference, logits, strict=True)):
                result = compare(a, b, atol=args.atol, rtol=args.rtol)
                emit(
                    {
                        "stage": "correctness",
                        "step": step,
                        **result,
                        "reference_top": logit_summary(a),
                        "candidate_top": logit_summary(b),
                    }
                )
    emit(
        {
            "stage": "complete",
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "logits_file": str(args.logits),
            "logits_sha256": sha256(args.logits),
        }
    )


def main() -> None:
    """Parse explicit reproducibility options and write exclusive durable JSONL."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cpu", "ov-gpu", "custom"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--logits", required=True, type=Path)
    parser.add_argument(
        "--cpus",
        required=True,
        help="Explicit allowed logical CPU IDs, comma separated",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Unique GPU name substring, required for GPU backends",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--capacity", type=int, help="Custom workspace capacity, including continuation"
    )
    parser.add_argument("--prepared-prefill", action="store_true")
    parser.add_argument(
        "--prepared-kn", action="store_true", help="K-major prepared custom prefill"
    )
    parser.add_argument(
        "--tiled-attention", action="store_true", help="Tiled custom attention scores"
    )
    parser.add_argument(
        "--vec4-matvec",
        action="store_true",
        help="Vectorized custom decode projections",
    )
    parser.add_argument(
        "--persistent-kn",
        action="store_true",
        help="Retain K-major projections across resets",
    )
    parser.add_argument(
        "--persistent-kn-budget-bytes",
        type=int,
        default=0,
        help="Explicit byte budget for retained K-major projections",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()
    if args.backend != "cpu" and not args.device.strip():
        parser.error("GPU backends require --device")
    if args.prepared_kn and not args.prepared_prefill:
        parser.error("--prepared-kn requires --prepared-prefill")
    if args.persistent_kn and not args.prepared_kn:
        parser.error("--persistent-kn requires --prepared-kn")
    if args.persistent_kn_budget_bytes < 0 or (
        not args.persistent_kn and args.persistent_kn_budget_bytes
    ):
        parser.error("A nonnegative persistent budget requires --persistent-kn")
    with args.output.open("x", encoding="utf-8") as output:

        def emit(record: dict[str, Any]) -> None:
            """Write and flush a JSON record.

            Args:
                record:
                    Serializable benchmark evidence.
            """
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()

        try:
            execute(args, emit)
        except Exception:
            emit({"stage": "error", "traceback": traceback.format_exc()})
            LOGGER.exception("Benchmark failed; partial results preserved")
            raise
    LOGGER.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
