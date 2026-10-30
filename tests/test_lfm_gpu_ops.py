"""Opt-in device checks of batch-one LFM OpenCL primitives against NumPy."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "hoast/lfm_gpu/lfm_gpu_ops.cl"
ATOL = 3e-5
RTOL = 3e-5
LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def gpu() -> tuple[Any, Any, Any, Any]:
    """Build the primitives on one explicitly selected real GPU, when opted in."""
    if os.environ.get("LFM_GPU_OPS_TEST") != "1":
        pytest.skip("Set LFM_GPU_OPS_TEST=1 for serialized real OpenCL GPU tests")
    import pyopencl as cl  # Optional native dependency is only loaded on opt-in.

    devices = [
        d
        for p in cl.get_platforms()
        for d in p.get_devices(device_type=cl.device_type.GPU)
    ]
    match = os.environ.get("LFM_GPU_OPS_DEVICE", "").lower()
    devices = [d for d in devices if match in d.name.lower()]
    if len(devices) != 1:
        raise RuntimeError(f"Expected one matching OpenCL GPU, found {len(devices)}")
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)
    program = cl.Program(context, SOURCE.read_text()).build(options=["-cl-std=CL1.2"])
    return cl, context, queue, program


def buffer(cl: Any, context: Any, data: np.ndarray) -> Any:
    """Copy a contiguous float32 fixture to a read-write device buffer.

    Args:
        cl: PyOpenCL module.
        context: Owning OpenCL context.
        data: Contiguous FP32 host array.
    """
    assert data.dtype == np.float32 and data.flags.c_contiguous
    return cl.Buffer(
        context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data
    )


def read(cl: Any, queue: Any, device: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Copy a device buffer into an FP32 array of the requested shape.

    Args:
        cl: PyOpenCL module.
        queue: In-order OpenCL queue.
        device: Source OpenCL buffer.
        shape: Dimensions of the FP32 result.
    """
    result = np.empty(shape, dtype=np.float32)
    cl.enqueue_copy(queue, result, device).wait()
    return result


def rope(x: np.ndarray, start: int, theta: float) -> np.ndarray:
    """Compute rotate-half RoPE for token-major head tensors.

    Args:
        x: FP32 tensor [tokens, heads, head dimension].
        start: Absolute position of the first token.
        theta: Rotary frequency base.
    """
    dim = x.shape[-1]
    i = np.arange(dim // 2, dtype=np.float32)
    freq = np.power(np.float32(theta), -2 * i / np.float32(dim))
    angle = np.arange(start, start + len(x), dtype=np.float32)[:, None, None] * freq
    c, s = np.cos(angle), np.sin(angle)
    a, b = np.split(x, 2, axis=-1)
    return np.concatenate((a * c - b * s, b * c + a * s), axis=-1)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute causal GQA on complete token-major FP32 fixtures.

    Args:
        q: Query tensor [tokens, query heads, head dimension].
        k: Key tensor [tokens, KV heads, head dimension].
        v: Value tensor with the same shape as k.
    """
    out = np.empty_like(q)
    groups = q.shape[1] // k.shape[1]
    for t in range(len(q)):
        for h in range(q.shape[1]):
            kh = h // groups
            logits = k[: t + 1, kh] @ q[t, h] / np.sqrt(np.float32(q.shape[2]))
            p = np.exp(logits - logits.max())
            out[t, h] = (p / p.sum()) @ v[: t + 1, kh]
    return out


def conv(
    bx: np.ndarray, gate: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute width-three causal cross-correlation and final B*X history.

    Args:
        bx: FP32 B*X tensor [tokens, channels].
        gate: FP32 C gate tensor [tokens, channels].
        weight: FP32 cross-correlation taps [channels, 3].
    """
    state = np.zeros((bx.shape[1], 3), dtype=np.float32)
    result = np.empty_like(bx)
    for t in range(len(bx)):
        state[:, :-1] = state[:, 1:].copy()
        state[:, -1] = bx[t]
        result[t] = gate[t] * np.sum(state * weight, axis=1)
    return result, state


def test_rmsnorm_and_rope(gpu: tuple[Any, Any, Any, Any]) -> None:
    """Check distinct rows, non-power-of-two widths and absolute rotary positions.

    Args:
        gpu: OpenCL module, context, queue and compiled program.
    """
    cl, ctx, queue, program = gpu
    rng = np.random.default_rng(123)
    x = rng.normal(size=(5, 7)).astype(np.float32)
    x[0] = 0
    weight = rng.normal(size=7).astype(np.float32)
    xb, wb = buffer(cl, ctx, x), buffer(cl, ctx, weight)
    yb = buffer(cl, ctx, np.zeros_like(x))
    cl.Kernel(program, "lfm_rmsnorm")(
        queue, (5,), None, xb, wb, yb, np.int32(5), np.int32(7), np.float32(1e-5)
    )
    expected = (
        x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + np.float32(1e-5)) * weight
    )
    np.testing.assert_allclose(
        read(cl, queue, yb, x.shape), expected, rtol=RTOL, atol=ATOL
    )
    cl.Kernel(program, "lfm_rmsnorm_parallel")(
        queue, (64, 5), (64, 1), xb, wb, xb, np.int32(5), np.int32(7), np.float32(1e-5)
    )
    np.testing.assert_allclose(
        read(cl, queue, xb, x.shape), expected, rtol=RTOL, atol=ATOL
    )
    q = rng.normal(size=(3, 4, 6)).astype(np.float32)
    k = rng.normal(size=(3, 2, 6)).astype(np.float32)
    qb, kb = buffer(cl, ctx, q), buffer(cl, ctx, k)
    qo, ko = buffer(cl, ctx, np.zeros_like(q)), buffer(cl, ctx, np.zeros_like(k))
    cl.Kernel(program, "lfm_rope_qk")(
        queue,
        (3, 4, 3),
        None,
        qb,
        kb,
        qo,
        ko,
        np.int32(3),
        np.int32(4),
        np.int32(2),
        np.int32(6),
        np.int32(29),
        np.float32(1e6),
    )
    np.testing.assert_allclose(
        read(cl, queue, qo, q.shape), rope(q, 29, 1e6), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        read(cl, queue, ko, k.shape), rope(k, 29, 1e6), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("chunks", [(5,), (1, 1, 1, 1, 1), (2, 1, 2)])
def test_attention_conv_state_and_capacity(
    gpu: tuple[Any, Any, Any, Any], chunks: tuple[int, ...]
) -> None:
    """Verify prefill/chunk/decode, cache isolation, and capacity overflow backstop.

    Args:
        gpu: OpenCL module, context, queue and compiled program.
        chunks: Consecutive token counts making up one request.
    """
    cl, ctx, queue, program = gpu
    rng = np.random.default_rng(42)
    count, qh, kvh, dim, channels, capacity = 5, 4, 2, 6, 7, 7
    q = rng.normal(size=(count, qh, dim)).astype(np.float32)
    k = rng.normal(size=(count, kvh, dim)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    bx = rng.normal(size=(count, channels)).astype(np.float32)
    gate = rng.normal(size=bx.shape).astype(np.float32)
    w = rng.normal(size=(channels, 3)).astype(np.float32)
    kc0 = np.full((capacity, kvh, dim), 73, dtype=np.float32)
    vc0 = np.full_like(kc0, -91)
    kc, vc = buffer(cl, ctx, kc0), buffer(cl, ctx, vc0)
    other_kc, other_vc = buffer(cl, ctx, kc0), buffer(cl, ctx, vc0)
    state = buffer(cl, ctx, np.zeros((channels, 3), np.float32))
    other_state = buffer(cl, ctx, np.zeros((channels, 3), np.float32))
    weights = buffer(cl, ctx, w)
    workspace = buffer(cl, ctx, np.full((count, qh, capacity), 123, np.float32))
    scores = cl.Kernel(program, "lfm_gqa_scores")
    apply = cl.Kernel(program, "lfm_gqa_apply")
    outputs, fast_outputs, conv_outputs = [], [], []
    start = 0
    for n in chunks:
        kk, vv = (
            buffer(cl, ctx, k[start : start + n].copy()),
            buffer(cl, ctx, v[start : start + n].copy()),
        )
        qq = buffer(cl, ctx, q[start : start + n].copy())
        y = buffer(cl, ctx, np.zeros((n, qh, dim), np.float32))
        cl.Kernel(program, "lfm_kv_append")(
            queue,
            (n, kvh, dim),
            None,
            kk,
            vv,
            kc,
            vc,
            np.int32(n),
            np.int32(kvh),
            np.int32(dim),
            np.int32(start),
            np.int32(capacity),
        )
        cl.Kernel(program, "lfm_causal_gqa")(
            queue,
            (n, qh, dim),
            None,
            qq,
            kc,
            vc,
            y,
            np.int32(n),
            np.int32(qh),
            np.int32(kvh),
            np.int32(dim),
            np.int32(start),
            np.int32(capacity),
        )
        outputs.append(read(cl, queue, y, (n, qh, dim)))
        fast = buffer(cl, ctx, np.zeros((n, qh, dim), np.float32))
        args = (
            np.int32(n),
            np.int32(qh),
            np.int32(kvh),
            np.int32(dim),
            np.int32(start),
            np.int32(capacity),
        )
        scores(queue, (start + n, n * qh), None, qq, kc, workspace, *args)
        apply(queue, (64, n * qh), (64, 1), workspace, vc, fast, *args)
        fast_outputs.append(read(cl, queue, fast, (n, qh, dim)))
        bb, gg = (
            buffer(cl, ctx, bx[start : start + n].copy()),
            buffer(cl, ctx, gate[start : start + n].copy()),
        )
        cy = buffer(cl, ctx, np.zeros((n, channels), np.float32))
        cl.Kernel(program, "lfm_causal_conv3")(
            queue,
            (channels,),
            None,
            bb,
            gg,
            weights,
            state,
            cy,
            np.int32(n),
            np.int32(channels),
        )
        conv_outputs.append(read(cl, queue, cy, (n, channels)))
        start += n
    np.testing.assert_allclose(
        np.concatenate(outputs), attention(q, k, v), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        np.concatenate(fast_outputs), attention(q, k, v), rtol=RTOL, atol=ATOL
    )
    expected_conv, expected_state = conv(bx, gate, w)
    np.testing.assert_allclose(
        np.concatenate(conv_outputs), expected_conv, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        read(cl, queue, state, (channels, 3)), expected_state, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_array_equal(read(cl, queue, kc, kc0.shape)[:count], k)
    np.testing.assert_array_equal(read(cl, queue, vc, vc0.shape)[:count], v)
    np.testing.assert_array_equal(read(cl, queue, kc, kc0.shape)[count:], kc0[count:])
    np.testing.assert_array_equal(read(cl, queue, other_kc, kc0.shape), kc0)
    np.testing.assert_array_equal(read(cl, queue, other_vc, vc0.shape), vc0)
    np.testing.assert_array_equal(read(cl, queue, other_state, (channels, 3)), 0)
    before = read(cl, queue, kc, kc0.shape)
    cl.Kernel(program, "lfm_kv_append")(
        queue,
        (2, kvh, dim),
        None,
        buffer(cl, ctx, k[:2].copy()),
        buffer(cl, ctx, v[:2].copy()),
        kc,
        vc,
        np.int32(2),
        np.int32(kvh),
        np.int32(dim),
        np.int32(capacity - 1),
        np.int32(capacity),
    )
    np.testing.assert_array_equal(read(cl, queue, kc, kc0.shape), before)
    sentinel = np.full((2, qh, dim), 17, dtype=np.float32)
    rejected = buffer(cl, ctx, sentinel)
    cl.Kernel(program, "lfm_causal_gqa")(
        queue,
        (2, qh, dim),
        None,
        buffer(cl, ctx, q[:2].copy()),
        kc,
        vc,
        rejected,
        np.int32(2),
        np.int32(qh),
        np.int32(kvh),
        np.int32(dim),
        np.int32(capacity - 1),
        np.int32(capacity),
    )
    np.testing.assert_array_equal(read(cl, queue, rejected, sentinel.shape), sentinel)
    before_workspace = read(cl, queue, workspace, (count, qh, capacity))
    bad_args = (
        np.int32(2),
        np.int32(qh),
        np.int32(kvh),
        np.int32(dim),
        np.int32(capacity - 1),
        np.int32(capacity),
    )
    scores(
        queue,
        (capacity, 2 * qh),
        None,
        buffer(cl, ctx, q[:2].copy()),
        kc,
        workspace,
        *bad_args,
    )
    apply(queue, (64, 2 * qh), (64, 1), workspace, vc, rejected, *bad_args)
    np.testing.assert_array_equal(
        read(cl, queue, workspace, before_workspace.shape), before_workspace
    )
    np.testing.assert_array_equal(read(cl, queue, rejected, sentinel.shape), sentinel)
    cl.Kernel(program, "lfm_kv_append")(
        queue,
        (2, kvh, dim),
        None,
        buffer(cl, ctx, k[:2].copy()),
        buffer(cl, ctx, v[:2].copy()),
        kc,
        vc,
        np.int32(2),
        np.int32(kvh),
        np.int32(dim),
        np.int32(count),
        np.int32(capacity),
    )
    np.testing.assert_array_equal(read(cl, queue, kc, kc0.shape)[count:], k[:2])


@pytest.mark.parametrize(
    ("count", "start", "capacity"),
    [(1, 0, 1), (3, 0, 7), (4, 0, 8), (5, 7, 12), (9, 8, 20), (5, 9, 14)],
)
def test_tiled_gqa_scores_causal_tails(
    gpu: tuple[Any, Any, Any, Any], count: int, start: int, capacity: int
) -> None:
    """Check every causal score and attention output, and preserve masked slots.

    Args:
        gpu: OpenCL module, context, queue and compiled program.
        count: Number of query tokens, including partial four-token tiles.
        start: Absolute query start, including partial eight-key tiles.
        capacity: Cache and score-row capacity, including exact-capacity cases.
    """
    cl, ctx, queue, program = gpu
    rng = np.random.default_rng(12000 + count * 100 + start)
    q = rng.normal(size=(count, 16, 64)).astype(np.float32)
    k = rng.normal(size=(capacity, 8, 64)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    initial = np.full((count, 16, capacity), 12345, dtype=np.float32)
    qb, kb, vb = buffer(cl, ctx, q), buffer(cl, ctx, k), buffer(cl, ctx, v)
    tiled, baseline = buffer(cl, ctx, initial), buffer(cl, ctx, initial)
    result = buffer(cl, ctx, np.zeros_like(q))
    ints = tuple(np.int32(x) for x in (count, 16, 8, 64, start, capacity))
    cl.Kernel(program, "lfm_gqa_scores_tiled_16_8_64")(
        queue,
        (((start + count + 7) // 8) * 64, (count + 3) // 4, 8),
        (64, 1, 1),
        qb,
        kb,
        tiled,
        np.int32(count),
        np.int32(start),
        np.int32(capacity),
    )
    cl.Kernel(program, "lfm_gqa_scores")(
        queue, (start + count, count * 16), None, qb, kb, baseline, *ints
    )
    actual = read(cl, queue, tiled, initial.shape)
    reference_scores = read(cl, queue, baseline, initial.shape)
    expected_output = np.empty_like(q)
    for t in range(count):
        n = start + t + 1
        for h in range(16):
            logits = k[:n, h // 2] @ q[t, h] * np.float32(0.125)
            assert np.all(np.isfinite(actual[t, h, :n]))
            np.testing.assert_allclose(actual[t, h, :n], logits, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(
                actual[t, h, :n], reference_scores[t, h, :n], rtol=RTOL, atol=ATOL
            )
            np.testing.assert_array_equal(actual[t, h, n:], initial[t, h, n:])
            weights = np.exp(logits - logits.max())
            expected_output[t, h] = (weights / weights.sum()) @ v[:n, h // 2]
    cl.Kernel(program, "lfm_gqa_apply")(
        queue, (64, count * 16), (64, 1), tiled, vb, result, *ints
    )
    output = read(cl, queue, result, q.shape)
    assert np.all(np.isfinite(output))
    np.testing.assert_allclose(output, expected_output, rtol=RTOL, atol=ATOL)

    # Invalid capacity must leave the score workspace untouched.
    rejected = buffer(cl, ctx, initial)
    cl.Kernel(program, "lfm_gqa_scores_tiled_16_8_64")(
        queue,
        (64, 1, 8),
        (64, 1, 1),
        qb,
        kb,
        rejected,
        np.int32(2),
        np.int32(capacity - 1),
        np.int32(capacity),
    )
    np.testing.assert_array_equal(read(cl, queue, rejected, initial.shape), initial)


def test_parallel_rms_realistic_width(gpu: tuple[Any, Any, Any, Any]) -> None:
    """Check 1024-element RMS rows, including a 64-lane reduction tail.

    Args:
        gpu: OpenCL module, context, queue and compiled program.
    """
    cl, ctx, queue, program = gpu
    rng = np.random.default_rng(64)
    for width in (64, 1001, 1024):
        x = rng.normal(size=(3, width)).astype(np.float32)
        w = rng.normal(size=width).astype(np.float32)
        xb, wb = buffer(cl, ctx, x), buffer(cl, ctx, w)
        y = buffer(cl, ctx, np.zeros_like(x))
        cl.Kernel(program, "lfm_rmsnorm_parallel")(
            queue,
            (64, 3),
            (64, 1),
            xb,
            wb,
            y,
            np.int32(3),
            np.int32(width),
            np.float32(1e-5),
        )
        ref = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-5)) * w
        np.testing.assert_allclose(
            read(cl, queue, y, x.shape), ref, rtol=RTOL, atol=ATOL
        )


@pytest.mark.skipif(
    os.environ.get("LFM_GPU_OPS_BENCH") != "1", reason="Explicit GPU benchmark opt-in"
)
def test_gpu_attention_benchmark(gpu: tuple[Any, Any, Any, Any]) -> None:
    """Profile reused buffers on model-shaped prefill/decode; compare both paths.

    Args:
        gpu: OpenCL module, context, queue and compiled program.
    """
    cl, ctx, _, program = gpu
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    rng = np.random.default_rng(91)
    max_query, capacity, qh, kvh, dim = 128, 1024, 16, 8, 64
    q = rng.normal(size=(max_query, qh, dim)).astype(np.float32)
    k = rng.normal(size=(capacity, kvh, dim)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    qb, kb, vb = buffer(cl, ctx, q), buffer(cl, ctx, k), buffer(cl, ctx, v)
    oldout = buffer(cl, ctx, np.zeros_like(q))
    newout = buffer(cl, ctx, np.zeros_like(q))
    work = buffer(cl, ctx, np.zeros((max_query, qh, capacity), np.float32))
    scores = cl.Kernel(program, "lfm_gqa_scores")
    apply = cl.Kernel(program, "lfm_gqa_apply")
    baseline = cl.Kernel(program, "lfm_causal_gqa")
    for label, count, start in (("prefill", 128, 0), ("decode", 1, 1023)):
        args = (
            np.int32(count),
            np.int32(qh),
            np.int32(kvh),
            np.int32(dim),
            np.int32(start),
            np.int32(capacity),
        )
        baseline_times, fast_times = [], []
        for iteration in range(4):
            old = baseline(queue, (count, qh, dim), None, qb, kb, vb, oldout, *args)
            first = scores(
                queue, (start + count, count * qh), None, qb, kb, work, *args
            )
            last = apply(queue, (64, count * qh), (64, 1), work, vb, newout, *args)
            last.wait()
            if iteration:
                baseline_times.append((old.profile.end - old.profile.start) / 1e6)
                fast_times.append((last.profile.end - first.profile.start) / 1e6)
        a = read(cl, queue, oldout, q.shape)[:count]
        b = read(cl, queue, newout, q.shape)[:count]
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-4)
        for t in {0, count - 1}:
            for h in (0, 7, 15):
                kh = h // (qh // kvh)
                logits = k[: start + t + 1, kh] @ q[t, h] / np.sqrt(np.float32(dim))
                probs = np.exp(logits - logits.max())
                expected = (probs / probs.sum()) @ v[: start + t + 1, kh]
                np.testing.assert_allclose(b[t, h], expected, rtol=1e-4, atol=1e-4)
        LOGGER.info(
            "LFM GPU benchmark %s capacity=%d q=%d baseline_ms=%.3f optimized_ms=%.3f workspace_bytes=%d",
            label,
            capacity,
            count,
            np.median(baseline_times),
            np.median(fast_times),
            max_query * qh * capacity * 4,
        )
    x = rng.normal(size=(128, 1024)).astype(np.float32)
    w = rng.normal(size=1024).astype(np.float32)
    xb, wb = buffer(cl, ctx, x), buffer(cl, ctx, w)
    oldnorm = buffer(cl, ctx, np.zeros_like(x))
    newnorm = buffer(cl, ctx, np.zeros_like(x))
    serial = cl.Kernel(program, "lfm_rmsnorm")
    parallel = cl.Kernel(program, "lfm_rmsnorm_parallel")
    for rows in (1, 128):
        old_times, new_times = [], []
        args = (np.int32(rows), np.int32(1024), np.float32(1e-5))
        for iteration in range(11):
            old = serial(queue, (rows,), None, xb, wb, oldnorm, *args)
            new = parallel(queue, (64, rows), (64, 1), xb, wb, newnorm, *args)
            new.wait()
            if iteration:
                old_times.append((old.profile.end - old.profile.start) / 1e6)
                new_times.append((new.profile.end - new.profile.start) / 1e6)
        np.testing.assert_allclose(
            read(cl, queue, newnorm, x.shape)[:rows],
            read(cl, queue, oldnorm, x.shape)[:rows],
            rtol=RTOL,
            atol=ATOL,
        )
        LOGGER.info(
            "LFM GPU RMS rows=%d width=1024 baseline_ms=%.3f optimized_ms=%.3f",
            rows,
            np.median(old_times),
            np.median(new_times),
        )
