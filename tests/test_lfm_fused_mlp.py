"""Synthetic batch-one fused MLP tests; GPU access is explicitly opt-in."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1] / "hoast/lfm_gpu"


def make_case(n: int, k: int) -> tuple[np.ndarray, ...]:
    """Create reproducible independent U8 projections and a shared activation.

    Args:
        n: Number of output channels.

        k: Number of input channels.
    """
    rng = np.random.default_rng(710 + n + k)
    x = rng.normal(0, 0.3, k).astype(np.float32)
    tensors: list[np.ndarray] = [x]
    for _ in range(2):
        tensors.extend(
            (
                rng.integers(0, 256, (n, k), dtype=np.uint8),
                rng.integers(0, 256, n, dtype=np.uint8),
                rng.uniform(0.0001, 0.01, n).astype(np.float16).astype(np.float32),
            )
        )
    return tuple(tensors)


def cpu_reference(case: tuple[np.ndarray, ...]) -> np.ndarray:
    """Compute FP16-rounded U8 weights, FP32 projections and SwiGLU.

    Args:
        case: Shared activation and two (weights, zero points, scales) tuples.
    """
    x, q1, z1, s1, q3, z3, s3 = case
    projections = []
    for q, z, s in ((q1, z1, s1), (q3, z3, s3)):
        weight = ((q.astype(np.float32) - z[:, None]) * s[:, None]).astype(np.float16)
        projections.append(weight.astype(np.float32) @ x)
    gate, up = projections
    return (gate / (1 + np.exp(-gate))) * up


def test_cpu_fixture() -> None:
    """Exercise FP16 rounding and independent gate/up projection semantics."""
    x = np.array([1, 2], np.float32)
    q1 = np.array([[4, 7]], np.uint8)
    q3 = np.array([[3, 9]], np.uint8)
    z1, z3 = np.array([2], np.uint8), np.array([5], np.uint8)
    s1 = np.array([np.float16(0.1)], np.float32)
    s3 = np.array([np.float16(0.2)], np.float32)
    gate = np.float32(np.float16(2 * s1[0])) + 2 * np.float32(np.float16(5 * s1[0]))
    up = np.float32(np.float16(-2 * s3[0])) + 2 * np.float32(np.float16(4 * s3[0]))
    assert float(np.float16(5 * s1[0])) != float(5 * s1[0])
    np.testing.assert_allclose(
        cpu_reference((x, q1, z1, s1, q3, z3, s3)),
        [gate / (1 + np.exp(-gate)) * up],
        rtol=1e-6,
    )


@pytest.fixture(scope="module")
def gpu() -> tuple[Any, Any, Any, dict[int, Any]]:
    """Build baseline and local-size variants only after explicit GPU opt-in."""
    if os.environ.get("LFM_GPU_MLP_TEST") != "1":
        pytest.skip("Set LFM_GPU_MLP_TEST=1 after GPU coordination")
    import pyopencl as cl  # Optional GPU dependency is intentionally lazy.

    match = os.environ.get("LFM_GPU_MLP_DEVICE", "").lower()
    devices = [
        d
        for p in cl.get_platforms()
        for d in p.get_devices(device_type=cl.device_type.GPU)
        if match in d.name.lower()
    ]
    if len(devices) != 1:
        raise RuntimeError(f"Expected one matching GPU, found {len(devices)}")
    device = devices[0]
    if "cl_khr_fp16" not in device.extensions.split() or not (
        device.half_fp_config & cl.device_fp_config.DENORM
    ):
        pytest.skip("Requires FP16 and FP16 denormal support")
    ctx = cl.Context([device])
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    source = "\n".join(
        (ROOT / name).read_text()
        for name in ("lfm_gpu_kernels.cl", "lfm_gpu_runner.cl", "lfm_fused_mlp.cl")
    )
    programs = {
        size: cl.Program(ctx, source).build(
            options=["-cl-std=CL1.2", f"-DLFM_MLP_LOCAL={size}"]
        )
        for size in (64, 128, 256)
        if size <= device.max_work_group_size
    }
    return cl, ctx, queue, programs


def run_case(
    gpu: tuple[Any, Any, Any, dict[int, Any]], n: int, k: int, benchmark: bool = False
) -> None:
    """Check variants against scalar, vec4 and CPU; optionally profile.

    Args:
        gpu: OpenCL module, context, profiling queue and compiled variants.

        n: Output channel count.

        k: Input channel count.

        benchmark: Collect warm device-event timings at the checkpoint shape.
    """
    cl, ctx, queue, programs = gpu
    case = make_case(n, k)
    flags = cl.mem_flags
    inputs = [
        cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=a) for a in case
    ]
    x, q1, z1, s1, q3, z3, s3 = inputs
    # An extra sentinel detects output overruns; read back all immutable input state.
    initial = np.full(n + 1, -12345, np.float32)

    def output() -> Any:
        """Allocate an initialized output including its trailing canary."""
        return cl.Buffer(ctx, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=initial)

    gate, up, fused = output(), output(), output()
    I = np.int32
    p = programs[128] if 128 in programs else next(iter(programs.values()))
    swiglu = cl.Kernel(p, "lfm_swiglu")
    expected = cpu_reference(case)
    rounds = 12 if benchmark else 1
    baselines: dict[str, np.ndarray] = {}
    for name in ("scalar", "vec4"):
        suffix = "_vec4" if name == "vec4" else ""
        matvec = cl.Kernel(p, "lfm_matvec_u8_f16_dequant" + suffix)
        spans: list[float] = []
        sums: list[float] = []
        for _ in range(rounds):
            e1 = matvec(
                queue, (128, (n + 7) // 8), (128, 1), x, q1, z1, s1, gate, I(k), I(n)
            )
            e2 = matvec(
                queue, (128, (n + 7) // 8), (128, 1), x, q3, z3, s3, up, I(k), I(n)
            )
            e3 = swiglu(queue, (n,), None, gate, up, I(n))
            e3.wait()
            spans.append((e3.profile.end - e1.profile.start) / 1e6)
            sums.append(
                sum(e.profile.end - e.profile.start for e in (e1, e2, e3)) / 1e6
            )
        baseline = np.empty_like(initial)
        cl.enqueue_copy(queue, baseline, gate).wait()
        assert baseline[-1] == initial[-1]
        np.testing.assert_allclose(baseline[:-1], expected, rtol=5e-4, atol=5e-4)
        baselines[name] = baseline
        if benchmark:
            LOGGER.warning(
                "device=%s N=%d K=%d baseline=%s span_ms=%.5f kernel_sum_ms=%.5f",
                queue.device.name,
                n,
                k,
                name,
                np.median(spans[2:]),
                np.median(sums[2:]),
            )
    for size, program in programs.items():
        kernel = cl.Kernel(program, "lfm_fused_mlp_ar")
        if (
            kernel.get_work_group_info(
                cl.kernel_work_group_info.WORK_GROUP_SIZE, queue.device
            )
            < size
        ):
            continue
        times: list[float] = []
        for _ in range(rounds):
            event = kernel(
                queue,
                (size, (n + size // 16 - 1) // (size // 16)),
                (size, 1),
                *inputs,
                fused,
                I(k),
                I(n),
            )
            event.wait()
            times.append((event.profile.end - event.profile.start) / 1e6)
        result = np.empty_like(initial)
        cl.enqueue_copy(queue, result, fused).wait()
        assert result[-1] == initial[-1]
        np.testing.assert_allclose(
            result[:-1], baselines["scalar"][:-1], rtol=2e-5, atol=2e-5
        )
        np.testing.assert_allclose(
            result[:-1], baselines["vec4"][:-1], rtol=5e-4, atol=5e-4
        )
        np.testing.assert_allclose(result[:-1], expected, rtol=5e-4, atol=5e-4)
        if benchmark:
            LOGGER.warning(
                "device=%s N=%d K=%d local=%d fused_ms=%.5f max_abs_scalar=%.7g max_abs_vec4=%.7g",
                queue.device.name,
                n,
                k,
                size,
                np.median(times[2:]),
                np.max(np.abs(result[:-1] - baselines["scalar"][:-1])),
                np.max(np.abs(result[:-1] - baselines["vec4"][:-1])),
            )
    for host, device in zip(case, inputs, strict=True):
        saved = np.empty_like(host)
        cl.enqueue_copy(queue, saved, device).wait()
        np.testing.assert_array_equal(saved, host)


@pytest.mark.parametrize("n,k", [(1, 1), (5, 33), (17, 65), (19, 127)])
def test_gpu_tails(gpu: tuple[Any, Any, Any, dict[int, Any]], n: int, k: int) -> None:
    """Compare tail dimensions and unchanged input state on each local size.

    Args:
        gpu: OpenCL fixture.

        n: Output channel count.

        k: Input channel count.
    """
    run_case(gpu, n, k)


@pytest.mark.skipif(
    os.environ.get("LFM_GPU_MLP_BENCH") != "1", reason="opt-in benchmark"
)
def test_gpu_checkpoint_shape(gpu: tuple[Any, Any, Any, dict[int, Any]]) -> None:
    """Compare and profile the actual batch-one 4608 x 1024 projections.

    Args:
        gpu: OpenCL fixture.
    """
    run_case(gpu, 4608, 1024, benchmark=True)
