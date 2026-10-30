"""Opt-in, synthetic fused LFM correctness and benchmark checks."""

import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytest

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1] / "hoast/lfm_gpu"


@pytest.fixture(scope="module")
def gpu() -> tuple[Any, Any, Any, Any]:
    """Build candidate and reference kernels on an explicitly selected GPU."""
    if os.environ.get("LFM_GPU_OPS_TEST") != "1":
        pytest.skip("Set LFM_GPU_OPS_TEST=1 for serialized real GPU tests")
    import pyopencl as cl  # Optional native dependency, loaded only on opt-in.

    match = os.environ.get("LFM_GPU_OPS_DEVICE", "").lower()
    devices = [
        d
        for p in cl.get_platforms()
        for d in p.get_devices(device_type=cl.device_type.GPU)
        if match in d.name.lower()
    ]
    if len(devices) != 1:
        raise RuntimeError(f"Expected one matching GPU, found {len(devices)}")
    ctx = cl.Context(devices)
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    source = "\n".join(
        (ROOT / name).read_text()
        for name in ("lfm_gpu_ops.cl", "lfm_gpu_runner.cl", "lfm_fused_ops.cl")
    )
    return cl, ctx, queue, cl.Program(ctx, source).build(options=["-cl-std=CL1.2"])


def buf(cl: Any, ctx: Any, data: np.ndarray) -> Any:
    """Allocate and initialize a device buffer from contiguous FP32 data.

    Args:
        cl: PyOpenCL module.

        ctx: Device context.

        data: Contiguous FP32 host tensor.
    """
    return cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data
    )


def fetch(cl: Any, queue: Any, device: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Read an FP32 device buffer.

    Args:
        cl: PyOpenCL module.

        queue: In-order command queue.

        device: Source buffer.

        shape: Result dimensions.
    """
    result = np.empty(shape, np.float32)
    cl.enqueue_copy(queue, result, device).wait()
    return result


@pytest.mark.parametrize("count", [1, 2, 7, 64, 930])
def test_fused_conv(gpu: tuple[Any, Any, Any, Any], count: int) -> None:
    """Compare both convolution paths, including nonzero state and reset.

    Args:
        gpu: OpenCL module, context, queue and program.

        count: Chunk length.
    """
    cl, ctx, queue, p = gpu
    rng = np.random.default_rng(99 + count)
    width = 129
    x = rng.standard_normal((count, 3, width), dtype=np.float32)
    w = rng.standard_normal((width, 3), dtype=np.float32)
    initial = rng.standard_normal((width, 3), dtype=np.float32)
    xb, wb = buf(cl, ctx, x), buf(cl, ctx, w)
    bx, gate = (
        buf(cl, ctx, np.zeros((count, width), np.float32)),
        buf(cl, ctx, np.zeros((count, width), np.float32)),
    )
    ref, ar, pre = [
        buf(cl, ctx, np.zeros((count, width), np.float32)) for _ in range(3)
    ]
    state, state_ar, state_pre, state_in = [
        buf(cl, ctx, initial.copy()) for _ in range(4)
    ]
    I = np.int32
    for reset in (False, False, True):
        if reset:
            for b in (state, state_ar, state_in):
                cl.enqueue_fill_buffer(queue, b, np.float32(0), 0, initial.nbytes)
        cl.Kernel(p, "lfm_conv_split")(
            queue, (count * width,), None, xb, bx, gate, I(count), I(width)
        )
        cl.Kernel(p, "lfm_causal_conv3")(
            queue, (width,), None, bx, gate, wb, state, ref, I(count), I(width)
        )
        cl.Kernel(p, "lfm_fused_conv_ar")(
            queue, (width,), None, xb, wb, state_ar, ar, I(count), I(width)
        )
        cl.Kernel(p, "lfm_fused_conv_prefill")(
            queue,
            (width, count),
            None,
            xb,
            wb,
            state_in,
            state_pre,
            pre,
            I(count),
            I(width),
        )
        for candidate in (ar, pre):
            np.testing.assert_array_equal(
                fetch(cl, queue, candidate, (count, width)),
                fetch(cl, queue, ref, (count, width)),
            )
        for candidate in (state_ar, state_pre):
            np.testing.assert_array_equal(
                fetch(cl, queue, candidate, (width, 3)),
                fetch(cl, queue, state, (width, 3)),
            )
        state_in, state_pre = state_pre, state_in


@pytest.mark.parametrize("count", [1, 3, 64, 930])
def test_add_norm(gpu: tuple[Any, Any, Any, Any], count: int) -> None:
    """Compare residual and normalized output bitwise to the runner operations.

    Args:
        gpu: OpenCL module, context, queue and program.

        count: Row count.
    """
    cl, ctx, queue, p = gpu
    rng = np.random.default_rng(count)
    width = 1024
    x = rng.standard_normal((count, width), dtype=np.float32)
    upd = rng.standard_normal(x.shape, dtype=np.float32)
    w = rng.standard_normal(width, dtype=np.float32)
    a, b, u, wb = [buf(cl, ctx, z) for z in (x, x, upd, w)]
    y, yr = [buf(cl, ctx, np.zeros_like(x)) for _ in range(2)]
    I = np.int32
    cl.Kernel(p, "lfm_add")(queue, (count * width,), None, a, u, I(count * width))
    cl.Kernel(p, "lfm_rmsnorm_parallel")(
        queue, (64, count), (64, 1), a, wb, yr, I(count), I(width), np.float32(1e-5)
    )
    cl.Kernel(p, "lfm_fused_add_rms")(
        queue, (64, count), (64, 1), b, u, wb, y, I(count), I(width), np.float32(1e-5)
    )
    np.testing.assert_array_equal(
        fetch(cl, queue, a, x.shape), fetch(cl, queue, b, x.shape)
    )
    np.testing.assert_array_equal(
        fetch(cl, queue, y, x.shape), fetch(cl, queue, yr, x.shape)
    )


@pytest.mark.parametrize(
    "count,start,capacity",
    [(1, 0, 4), (3, 4, 7), (3, 5, 7), (1, -1, 7), (930, 0, 930), (1, 930, 931)],
)
def test_rope_append(
    gpu: tuple[Any, Any, Any, Any], count: int, start: int, capacity: int
) -> None:
    """Compare rotary output and complete caches, including invalid capacity.

    Args:
        gpu: OpenCL module, context, queue and program.

        count: Number of tokens.

        start: Absolute position.

        capacity: Cache token capacity.
    """
    cl, ctx, queue, p = gpu
    rng = np.random.default_rng(111)
    q = rng.standard_normal((count, 16, 64), dtype=np.float32)
    k = rng.standard_normal((count, 8, 64), dtype=np.float32)
    v = rng.standard_normal(k.shape, dtype=np.float32)
    cache = rng.standard_normal((capacity, 8, 64), dtype=np.float32)
    qb, kb, vb = [buf(cl, ctx, z) for z in (q, k, v)]
    qo, qr = [buf(cl, ctx, np.zeros_like(q)) for _ in range(2)]
    ko = buf(cl, ctx, np.zeros_like(k))
    kc, kr, vc, vr = [buf(cl, ctx, cache) for _ in range(4)]
    I = np.int32
    valid = start >= 0 and start + count <= capacity
    if valid:
        cl.Kernel(p, "lfm_rope_qk")(
            queue,
            (count, 16, 32),
            None,
            qb,
            kb,
            qr,
            ko,
            I(count),
            I(16),
            I(8),
            I(64),
            I(start),
            np.float32(1e6),
        )
        cl.Kernel(p, "lfm_kv_append")(
            queue,
            (count, 8, 64),
            None,
            ko,
            vb,
            kr,
            vr,
            I(count),
            I(8),
            I(64),
            I(start),
            I(capacity),
        )
    cl.Kernel(p, "lfm_fused_rope_append")(
        queue,
        (count, 16, 32),
        None,
        qb,
        kb,
        vb,
        qo,
        kc,
        vc,
        I(count),
        I(16),
        I(8),
        I(64),
        I(start),
        I(capacity),
        np.float32(1e6),
    )
    for a, b, shape in (
        (qo, qr, q.shape),
        (kc, kr, cache.shape),
        (vc, vr, cache.shape),
    ):
        np.testing.assert_array_equal(
            fetch(cl, queue, a, shape), fetch(cl, queue, b, shape)
        )


@pytest.mark.parametrize("count", [1, 64, 930])
@pytest.mark.parametrize("operation", ["conv", "norm", "rope"])
def test_benchmark_fused(
    gpu: tuple[Any, Any, Any, Any], count: int, operation: str
) -> None:
    """Measure synthetic runner-shaped operation latency, including launch overhead.

    Event time is the sum of device kernel durations; wall time covers enqueue and
    synchronization. Warmups precede five samples. No model weights are required.

    Args:
        gpu: OpenCL module, context, queue and compiled program.

        count: Number of tokens in the chunk.

        operation: Fused operation to measure.
    """
    if os.environ.get("LFM_FUSED_BENCH") != "1":
        pytest.skip("Set LFM_FUSED_BENCH=1 to run synthetic latency benchmarks")
    cl, ctx, queue, p = gpu
    rng = np.random.default_rng(1200 + count)
    I = np.int32

    def tensor(shape: tuple[int, ...]) -> Any:
        """Create a random device tensor.

        Args:
            shape: Tensor dimensions.
        """
        return buf(cl, ctx, rng.standard_normal(shape, dtype=np.float32))

    if operation == "conv":
        x, w = tensor((count, 3, 1024)), tensor((1024, 3))
        state, state2, state3 = tensor((1024, 3)), tensor((1024, 3)), tensor((1024, 3))
        bx, gate, out = [tensor((count, 1024)) for _ in range(3)]
        split, conv = cl.Kernel(p, "lfm_conv_split"), cl.Kernel(p, "lfm_causal_conv3")
        fused = cl.Kernel(
            p, "lfm_fused_conv_ar" if count == 1 else "lfm_fused_conv_prefill"
        )

        def reference() -> list[Any]:
            """Enqueue split then causal convolution."""
            return [
                split(queue, (count * 1024,), None, x, bx, gate, I(count), I(1024)),
                conv(queue, (1024,), None, bx, gate, w, state, out, I(count), I(1024)),
            ]

        def candidate() -> list[Any]:
            """Enqueue the AR or parallel-prefill convolution."""
            if count == 1:
                return [
                    fused(queue, (1024,), None, x, w, state2, out, I(count), I(1024))
                ]
            return [
                fused(
                    queue,
                    (1024, count),
                    None,
                    x,
                    w,
                    state2,
                    state3,
                    out,
                    I(count),
                    I(1024),
                )
            ]
    elif operation == "norm":
        x, upd, out = [tensor((count, 1024)) for _ in range(3)]
        w = tensor((1024,))
        add, norm = cl.Kernel(p, "lfm_add"), cl.Kernel(p, "lfm_rmsnorm_parallel")
        fused = cl.Kernel(p, "lfm_fused_add_rms")

        def reference() -> list[Any]:
            """Enqueue residual addition and normalization."""
            return [
                add(queue, (count * 1024,), None, x, upd, I(count * 1024)),
                norm(
                    queue,
                    (64, count),
                    (64, 1),
                    x,
                    w,
                    out,
                    I(count),
                    I(1024),
                    np.float32(1e-5),
                ),
            ]

        def candidate() -> list[Any]:
            """Enqueue fused addition and normalization."""
            return [
                fused(
                    queue,
                    (64, count),
                    (64, 1),
                    x,
                    upd,
                    w,
                    out,
                    I(count),
                    I(1024),
                    np.float32(1e-5),
                )
            ]
    else:
        q, qo = tensor((count, 16, 64)), tensor((count, 16, 64))
        k, v, ko = [tensor((count, 8, 64)) for _ in range(3)]
        kc, vc = tensor((count + 64, 8, 64)), tensor((count + 64, 8, 64))
        rotary, append = cl.Kernel(p, "lfm_rope_qk"), cl.Kernel(p, "lfm_kv_append")
        fused = cl.Kernel(p, "lfm_fused_rope_append")

        def reference() -> list[Any]:
            """Enqueue rotary then cache append."""
            return [
                rotary(
                    queue,
                    (count, 16, 32),
                    None,
                    q,
                    k,
                    qo,
                    ko,
                    I(count),
                    I(16),
                    I(8),
                    I(64),
                    I(64),
                    np.float32(1e6),
                ),
                append(
                    queue,
                    (count, 8, 64),
                    None,
                    ko,
                    v,
                    kc,
                    vc,
                    I(count),
                    I(8),
                    I(64),
                    I(64),
                    I(count + 64),
                ),
            ]

        def candidate() -> list[Any]:
            """Enqueue fused rotary and cache append."""
            return [
                fused(
                    queue,
                    (count, 16, 32),
                    None,
                    q,
                    k,
                    v,
                    qo,
                    kc,
                    vc,
                    I(count),
                    I(16),
                    I(8),
                    I(64),
                    I(64),
                    I(count + 64),
                    np.float32(1e6),
                )
            ]

    for label, run in (("reference", reference), ("fused", candidate)):
        for _ in range(3):
            run()[-1].wait()
        wall, device = [], []
        for _ in range(5):
            start = perf_counter()
            events = run()
            events[-1].wait()
            wall.append((perf_counter() - start) * 1e6)
            device.append(
                sum(
                    (event.profile.end - event.profile.start) / 1000 for event in events
                )
            )
        LOGGER.warning(
            "fused_bench op=%s count=%d path=%s event_us=%.1f wall_us=%.1f",
            operation,
            count,
            label,
            np.median(device),
            np.median(wall),
        )
