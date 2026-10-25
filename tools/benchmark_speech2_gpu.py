"""Opt-in synthetic packed-INT8 speech microbenchmarks; never loads real models.

Run with native .venv/bin/python (3.14), after coordinating GPU use. Results and
complete failure traces go to .cache/speech-bench/results/speech2-gpu/<tag>.json.
--offline validates quantization/reference arithmetic without creating an OpenCL
context or building kernels. Device results are experimental, not qualification
of SenseVoice/Melo accuracy, latency or production integration.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import psutil

from hoast.logging import configure_logging, get_logger
from hoast.runtime import configure_cpu_budget

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse explicit tensor-shape sweeps, bounded iteration counts and result tag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("matmul", "conv1d"), default="matmul")
    parser.add_argument("--rows", type=int, nargs="+", default=[32])
    parser.add_argument("--reduction", type=int, default=512)
    parser.add_argument("--outputs", type=int, nargs="+", default=[512])
    parser.add_argument("--channels", type=int, nargs="+", default=[64])
    parser.add_argument("--output-channels", type=int, default=None)
    parser.add_argument("--taps", type=int, nargs="+", default=[3])
    parser.add_argument("--dilation", type=int, nargs="+", default=[1])
    parser.add_argument("--frames", type=int, nargs="+", default=[32])
    parser.add_argument("--upsampling", type=int, nargs="+", default=[8])
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--resident-iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--device", default="630")
    parser.add_argument(
        "--variant",
        choices=("prescale", "postscale"),
        default="prescale",
        help="Reference per-weight FP32 scaling, or experimental final output scaling",
    )
    parser.add_argument("--transfer", choices=("copy", "map"), default="copy")
    parser.add_argument("--allocate-per-call", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--root", type=Path, default=Path(".cache/speech-bench"))
    args = parser.parse_args()
    positive = [
        args.reduction,
        args.iterations,
        args.resident_iterations,
        *args.rows,
        *args.outputs,
        *args.channels,
        *args.taps,
        *args.dilation,
        *args.frames,
        *args.upsampling,
    ]
    if args.output_channels is not None:
        positive.append(args.output_channels)
    if min(positive) <= 0 or args.warmup < 0 or any(k % 2 != 1 for k in args.taps):
        parser.error(
            "Dimensions/iterations must be positive, taps odd, warmup nonnegative"
        )
    if not args.tag or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in args.tag
    ):
        parser.error("Tag must contain only letters, digits, hyphen or underscore")
    return args


def cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Construct shape cases; convolution time equals frames times cumulative upsampling.

    Args:
        args:
            Validated CLI options; --channels chooses equal CI/CO unless CO is explicit.

    """
    if args.op == "matmul":
        return [
            {"input_shape": (m, args.reduction), "weight_shape": (n, args.reduction)}
            for m, n in itertools.product(args.rows, args.outputs)
        ]
    return [
        {
            "input_shape": (channels, frames * upsampling),
            "weight_shape": (args.output_channels or channels, channels, taps),
            "dilation": dilation,
            "frames": frames,
            "upsampling": upsampling,
        }
        for channels, taps, dilation, frames, upsampling in itertools.product(
            args.channels, args.taps, args.dilation, args.frames, args.upsampling
        )
    ]


def benchmark_case(
    case: dict[str, Any], args: argparse.Namespace, gpu: Any
) -> dict[str, Any]:
    """Validate references, time CPU arithmetic and profile one packed device case.

    Args:
        case:
            Shape and optional dilation/decoder temporal-length metadata.

        args:
            Validated CLI timing and output-transfer policy.

        gpu:
            ExperimentalGPU, or None for offline-only CPU validation.

    """
    # Heavy numerical imports happen only after the entry point enforces CPU affinity.
    import numpy as np

    from hoast.speech2_gpu import (
        FP32_ATOL,
        FP32_RTOL,
        dequantize_reference,
        error_metrics,
        event_ms,
        quantize_weights,
        reference_conv1d,
        reference_matmul,
        reference_postscale,
    )

    rng = np.random.default_rng(args.seed)
    data = rng.normal(0, 0.5, case["input_shape"]).astype(np.float32)
    weights = rng.normal(0, 0.02, case["weight_shape"]).astype(np.float32)
    started = time.perf_counter()
    packed = quantize_weights(weights)
    quantization_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    dequantized = dequantize_reference(packed)
    dequantization_reference_ms = (time.perf_counter() - started) * 1000

    def reference(source: Any) -> Any:
        """Compute this case's FP32 CPU operator with the selected weight source.

        Args:
            source:
                Original FP32 or CPU-expanded quantized weights.

        """
        if args.op == "matmul":
            return reference_matmul(data, source)
        return reference_conv1d(data, source, case["dilation"])

    original = reference(weights)
    expected = reference(dequantized)
    started = time.perf_counter()
    candidate_expected = (
        reference_postscale(data, packed, case.get("dilation", 1))
        if args.variant == "postscale"
        else expected
    )
    candidate_reference_ms = (time.perf_counter() - started) * 1000
    cpu_times: dict[str, list[float]] = {}
    for label, source in (
        ("original_fp32", weights),
        ("dequantized_int8_fp32", dequantized),
    ):
        for _ in range(args.warmup):
            reference(source)
        durations = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            reference(source)
            durations.append((time.perf_counter() - started) * 1000)
        cpu_times[label] = durations
    record: dict[str, Any] = {
        **case,
        "scaling_variant": args.variant,
        "correctness_tolerance": {"atol": FP32_ATOL, "rtol": FP32_RTOL},
        "fixture": {"seed": args.seed, "input_std": 0.5, "weight_std": 0.02},
        "quantization_ms": quantization_ms,
        "cpu_reference_dequantization_ms": dequantization_reference_ms,
        "cpu_arithmetic_ms": cpu_times,
        "cpu_arithmetic_median_ms": {
            key: float(np.median(v)) for key, v in cpu_times.items()
        },
        "weight_quantization_error": error_metrics(dequantized, weights),
        "output_quantization_error": error_metrics(expected, original),
        "cpu_candidate_reference_ms": candidate_reference_ms,
        "postscale_reassociation_error_vs_dequantized_int8": (
            error_metrics(candidate_expected, expected)
            if args.variant == "postscale"
            else None
        ),
        "storage": {
            "original_fp32_weight_bytes": weights.nbytes,
            "packed_int8_weight_bytes": packed.values.nbytes,
            "fp32_scale_bytes": packed.scales.nbytes,
            "resident_weight_bytes": packed.nbytes,
            "expanded_device_weight_bytes": 0,
            "fixture_host_array_bytes": sum(
                v.nbytes
                for v in (
                    data,
                    weights,
                    packed.values,
                    packed.scales,
                    dequantized,
                    original,
                    expected,
                )
            )
            + (candidate_expected.nbytes if args.variant == "postscale" else 0),
            "cpu_postscale_reference_temporary_fp32_weight_bytes": (
                weights.nbytes if args.variant == "postscale" else 0
            ),
            "device_local_bytes_per_workgroup": 2048,
            "register_float_weight_values_per_workitem": 4,
        },
    }
    if gpu is None:
        record["status"] = "offline_only_no_device_test"
        return record
    if gpu.variant != args.variant:
        raise ValueError("Benchmark label does not match compiled scaling variant")
    operation = gpu.prepare(packed, case.get("dilation", 1))
    workspace = None
    try:
        workspace = gpu.workspace(operation, data.shape, args.transfer)
        actual, first = gpu.execute(operation, workspace, data)
        record["first_execute"] = first
        record["kernel_arithmetic_error_vs_dequantized_int8"] = error_metrics(
            actual, expected
        )
        record["total_error_vs_original_fp32"] = error_metrics(actual, original)
        record["kernel_arithmetic_error_vs_candidate_cpu_reference"] = error_metrics(
            actual, candidate_expected
        )
        np.testing.assert_allclose(actual, expected, rtol=FP32_RTOL, atol=FP32_ATOL)
        np.testing.assert_allclose(
            actual, candidate_expected, rtol=FP32_RTOL, atol=FP32_ATOL
        )
        record["weight_setup"] = operation.setup
        record["workspace_setup"] = workspace.setup
        for _ in range(args.warmup):
            gpu.execute(operation, workspace, data)
        trials = []
        for _ in range(args.iterations):
            if args.allocate_per_call:
                started = time.perf_counter()
                transient = gpu.workspace(operation, data.shape, args.transfer)
                try:
                    _, timing = gpu.execute(operation, transient, data)
                    timing["allocation_ms"] = float(transient.setup["allocation_ms"])
                finally:
                    transient.close()
                timing["allocation_inclusive_wall_ms"] = (
                    time.perf_counter() - started
                ) * 1000
            else:
                _, timing = gpu.execute(operation, workspace, data)
            trials.append(timing)
        # Input remains on the device: this separate loop has no transfers or allocation.
        started = time.perf_counter()
        events = [
            gpu.launch(operation, workspace) for _ in range(args.resident_iterations)
        ]
        events[-1].wait()
        resident_wall_ms = (time.perf_counter() - started) * 1000
        resident = [event_ms(event) for event in events]
        record.update(
            status="validated_synthetic_experimental",
            trials=trials,
            median_ms={
                key: float(np.median([trial[key] for trial in trials]))
                for key in trials[0]
            },
            resident_kernel_ms=resident,
            resident_kernel_median_ms=float(np.median(resident)),
            resident_loop_wall_ms=resident_wall_ms,
            resident_loop_calls=args.resident_iterations,
            per_call_allocations={
                "device_count": 2 if args.allocate_per_call else 0,
                "device_bytes": int(workspace.setup["device_bytes"])
                if args.allocate_per_call
                else 0,
                "owned_host_array_count": 1 if args.allocate_per_call else 0,
                "owned_host_array_bytes": workspace.host_output.nbytes
                if args.allocate_per_call
                else 0,
                "mapping_view_count": 1 if args.transfer == "map" else 0,
            },
        )
        logger.info("gpu.microbench shape=%s medians_ms=%s", case, record["median_ms"])
        return record
    finally:
        gpu.close()
        if workspace is not None:
            workspace.close()
        operation.close()


def worker_affinity() -> list[dict[str, Any]]:
    """Verify native helper threads remain within the process's one/two-CPU budget."""
    allowed = set(os.sched_getaffinity(0))
    result = []
    for thread in psutil.Process().threads():
        try:
            cpus = set(os.sched_getaffinity(thread.id))
        except ProcessLookupError:
            continue
        if not cpus <= allowed:
            raise RuntimeError(f"Thread {thread.id} escaped affinity budget: {cpus}")
        result.append({"thread_id": thread.id, "cpus": sorted(cpus)})
    return result


def main() -> None:
    """Enforce native Python/CPU policy and persist incremental results and failures."""
    args = parse_args()
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("Use native project .venv/bin/python (Python 3.14)")
    root = args.root / "results/speech2-gpu"
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"{args.tag}.json"
    log_path = root / f"{args.tag}.log"
    if output.exists() or log_path.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {output}")
    output.touch(exist_ok=False)
    configure_logging(log_file=log_path, color=False)
    cpus = configure_cpu_budget(args.threads)
    # Optional numerical/device imports occur after affinity/environment configuration.
    from hoast.speech2_gpu import ExperimentalGPU

    configure_cpu_budget(args.threads)
    record: dict[str, Any] = {
        "status": "incomplete",
        "python": sys.version,
        "experimental": True,
        "model_integrated": False,
        "real_model_qualified": False,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cpu_affinity": cpus,
        "policy": {
            "weights": "resident signed INT8, per-output FP32 scales, register float4 conversion",
            "arithmetic": "FP32 activations, weights in registers, fused FP32 accumulation, FP32 outputs",
            "scaling_variant": args.variant,
            "scale_placement": (
                "inside reduction, per converted weight (reference)"
                if args.variant == "prescale"
                else "once after raw-byte FP32 dot (candidate; reassociated rounding and overflow range)"
            ),
            "transfers": "explicit input upload; explicit output copy or map+CPU copy+unmap",
            "zero_copy_claim": False,
            "allocations": "requested application buffer bytes; driver/compiler hidden allocations not measured",
            "cpu_baselines": "original FP32 and CPU-dequantized INT8 FP32 CPU arithmetic; output allocation included, weight expansion excluded",
        },
        "initial_rss_bytes": psutil.Process().memory_info().rss,
        "cases": [],
    }
    gpu = None
    try:
        output.write_text(json.dumps(record, indent=2) + "\n")
        if not args.offline:
            gpu = ExperimentalGPU(args.device, variant=args.variant)
            record["device"] = {
                "name": gpu.device.name,
                "vendor": gpu.device.vendor,
                "driver": gpu.device.driver_version,
                "opencl": gpu.device.version,
                "setup_build_ms": gpu.setup_ms,
            }
        for case in cases(args):
            record["current_case"] = case
            output.write_text(json.dumps(record, indent=2) + "\n")
            record["cases"].append(benchmark_case(case, args, gpu))
            output.write_text(json.dumps(record, indent=2) + "\n")
        record["worker_affinity"] = worker_affinity()
        record["final_rss_bytes"] = psutil.Process().memory_info().rss
        record["status"] = "offline_complete" if args.offline else "synthetic_complete"
        record.pop("current_case", None)
    except BaseException:
        record["status"] = "failed"
        record["exception"] = traceback.format_exc()
        logger.exception(
            "speech2.gpu benchmark failed; metadata=%s", record.get("current_case")
        )
        raise
    finally:
        try:
            if gpu is not None:
                gpu.close()
        finally:
            output.write_text(json.dumps(record, indent=2) + "\n")
    logger.info("speech2.gpu results=%s status=%s", output, record["status"])


if __name__ == "__main__":
    main()
