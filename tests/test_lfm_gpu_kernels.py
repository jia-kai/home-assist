"""Device tests for the standalone LFM asymmetric U8 projection primitives.

Set LFM_GPU_BENCH=1 to profile representative checkpoint projection shapes.
Tests use synthetic arrays and require neither checkpoint files nor a GPU; they
skip if the optional PyOpenCL dependency or an OpenCL device is unavailable.
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

cl = pytest.importorskip("pyopencl")

LOGGER = logging.getLogger(__name__)
SOURCE = Path(__file__).resolve().parents[1] / "hoast/lfm_gpu/lfm_gpu_kernels.cl"


@pytest.fixture(scope="module")
def device_program() -> tuple[Any, Any, Any]:
    """Build the kernel on the first available GPU, or skip without OpenCL."""
    try:
        devices = [
            d
            for platform in cl.get_platforms()
            for d in platform.get_devices(device_type=cl.device_type.GPU)
        ]
    except cl.LogicError as error:
        pytest.skip(f"OpenCL unavailable: {error}")
    if not devices:
        pytest.skip("No OpenCL GPU")
    context = cl.Context([devices[0]])
    queue = cl.CommandQueue(
        context, properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    return context, queue, cl.Program(context, SOURCE.read_text()).build()


def run_case(device_program: tuple[Any, Any, Any], n: int, k: int) -> float:
    """Check one seeded [N,K] projection and return device kernel time in ms.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        n: Number of output channels.

        k: Number of input channels.
    """
    context, queue, program = device_program
    rng = np.random.default_rng(42 + n + k)
    x = rng.normal(size=k).astype(np.float32)
    q = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    zero = rng.integers(0, 256, size=n, dtype=np.uint8)
    scale = rng.uniform(0.0001, 0.02, size=n).astype(np.float32)
    flags = cl.mem_flags
    xb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=x)
    qb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=q)
    zb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=zero)
    sb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=scale)
    out = np.empty(n, dtype=np.float32)
    yb = cl.Buffer(context, flags.WRITE_ONLY, out.nbytes)
    kernel = cl.Kernel(program, "lfm_matvec_u8")
    times = []
    for _ in range(11):
        event = kernel(
            queue,
            (128, (n + 7) // 8),
            (128, 1),
            xb,
            qb,
            zb,
            sb,
            yb,
            np.int32(k),
            np.int32(n),
        )
        event.wait()
        times.append((event.profile.end - event.profile.start) / 1e6)
    cl.enqueue_copy(queue, out, yb).wait()
    expected = ((q.astype(np.float32) - zero[:, None]) * scale[:, None]) @ x
    np.testing.assert_allclose(out, expected, rtol=3e-4, atol=3e-4)
    elapsed = float(np.median(times[1:]))
    LOGGER.warning(
        "device=%s N=%d K=%d median_ms=%.4f max_abs_error=%.6g",
        queue.device.name,
        n,
        k,
        elapsed,
        float(np.max(np.abs(out - expected))),
    )
    return elapsed


@pytest.mark.parametrize("n,k", [(1, 1), (5, 33), (17, 65), (32, 1024)])
def test_matvec_tails(device_program: tuple[Any, Any, Any], n: int, k: int) -> None:
    """Compare arbitrary output and reduction tails with NumPy.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        n: Number of output channels.

        k: Number of input channels.
    """
    run_case(device_program, n, k)


@pytest.mark.skipif(
    os.environ.get("LFM_GPU_BENCH") != "1", reason="opt-in GPU benchmark"
)
@pytest.mark.parametrize(
    "n,k", [(1024, 1024), (3072, 1024), (4608, 1024), (1024, 4608), (65536, 1024)]
)
def test_matvec_benchmark(device_program: tuple[Any, Any, Any], n: int, k: int) -> None:
    """Profile synthetic checkpoint-sized projections and check against NumPy.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        n: Number of output channels.

        k: Number of input channels.
    """
    run_case(device_program, n, k)


def run_prefill_case(
    device_program: tuple[Any, Any, Any], m: int, n: int, k: int
) -> float:
    """Check seeded FP32 [M,K] and U8 [N,K] projection and return median ms.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        m: Number of token rows.

        n: Number of output channels.

        k: Number of input channels.
    """
    context, queue, program = device_program
    rng = np.random.default_rng(1729 + m + n + k)
    x = rng.normal(size=(m, k)).astype(np.float32)
    q = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    zero = rng.integers(0, 256, size=n, dtype=np.uint8)
    # Checkpoint scales are stored as FP16, then promoted for this kernel.
    scale = rng.uniform(0.0001, 0.02, size=n).astype(np.float16).astype(np.float32)
    flags = cl.mem_flags
    xb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=x)
    qb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=q)
    zb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=zero)
    sb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=scale)
    out = np.empty((m, n), dtype=np.float32)
    yb = cl.Buffer(context, flags.WRITE_ONLY, out.nbytes)
    kernel = cl.Kernel(program, "lfm_matmul_u8")
    times = []
    for _ in range(6):
        event = kernel(
            queue,
            (128, (n + 7) // 8, (m + 7) // 8),
            (128, 1, 1),
            xb,
            qb,
            zb,
            sb,
            yb,
            np.int32(m),
            np.int32(k),
            np.int32(n),
        )
        event.wait()
        times.append((event.profile.end - event.profile.start) / 1e6)
    cl.enqueue_copy(queue, out, yb).wait()
    expected = x @ ((q.astype(np.float32) - zero[:, None]) * scale[:, None]).T
    np.testing.assert_allclose(out, expected, rtol=8e-4, atol=8e-4)
    elapsed = float(np.median(times[1:]))
    LOGGER.warning(
        "prefill device=%s M=%d N=%d K=%d median_ms=%.4f max_abs_error=%.6g",
        queue.device.name,
        m,
        n,
        k,
        elapsed,
        float(np.max(np.abs(out - expected))),
    )
    return elapsed


@pytest.mark.parametrize(
    "m,n,k",
    [(1, 1, 1), (3, 5, 7), (7, 33, 33), (8, 32, 32), (9, 35, 65), (17, 29, 127)],
)
def test_prefill_tails(
    device_program: tuple[Any, Any, Any], m: int, n: int, k: int
) -> None:
    """Compare token, channel and reduction tile tails with NumPy.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        m: Number of token rows.

        n: Number of output channels.

        k: Number of input channels.
    """
    run_prefill_case(device_program, m, n, k)


@pytest.mark.skipif(
    os.environ.get("LFM_GPU_BENCH") != "1", reason="opt-in GPU benchmark"
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        (8, 1024, 1024),
        (32, 1024, 1024),
        (128, 1024, 1024),
        (32, 3072, 1024),
        (32, 4608, 1024),
        (32, 1024, 4608),
        (8, 65536, 1024),
    ],
)
def test_prefill_benchmark(
    device_program: tuple[Any, Any, Any], m: int, n: int, k: int
) -> None:
    """Profile synthetic checkpoint-sized projections and compare to NumPy.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        m: Number of token rows.

        n: Number of output channels.

        k: Number of input channels.
    """
    run_prefill_case(device_program, m, n, k)


def run_ir_case(
    device_program: tuple[Any, Any, Any],
    m: int,
    n: int,
    k: int,
    matvec: bool,
    rounding_case: bool = False,
) -> None:
    """Compare both dequantization modes with NumPy and profile device time.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        m: Number of token rows; must be one for matvec.

        n: Number of output channels.

        k: Number of input channels.

        matvec: Select the vector rather than prefill entry point.

        rounding_case: Use values that guarantee different rounded weights.
    """
    context, queue, program = device_program
    if "cl_khr_fp16" not in queue.device.extensions.split():
        pytest.skip("IR dequantization requires cl_khr_fp16")
    assert not matvec or m == 1
    rng = np.random.default_rng(2001 + m + n + k)
    x = rng.normal(size=(m, k)).astype(np.float32)
    q = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    zero = rng.integers(0, 256, size=n, dtype=np.uint8)
    scale = rng.uniform(0.0001, 0.02, size=n).astype(np.float16).astype(np.float32)
    if rounding_case:
        x.fill(1)
        q.fill(3)
        zero.fill(0)
        scale.fill(np.float32(np.float16(0.1)))
    flags = cl.mem_flags
    xb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=x)
    qb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=q)
    zb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=zero)
    sb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=scale)
    yb = cl.Buffer(context, flags.WRITE_ONLY, m * n * np.dtype(np.float32).itemsize)
    weights = (q.astype(np.float32) - zero[:, None]) * scale[:, None]
    half_weights = weights.astype(np.float16).astype(np.float32)
    results = []
    times = []
    for suffix in ("", "_f16_dequant"):
        name = ("lfm_matvec_u8" if matvec else "lfm_matmul_u8") + suffix
        kernel = cl.Kernel(program, name)
        global_size = (
            (128, (n + 7) // 8) if matvec else (128, (n + 7) // 8, (m + 7) // 8)
        )
        local_size = (128, 1) if matvec else (128, 1, 1)
        dims = (
            (np.int32(k), np.int32(n))
            if matvec
            else (np.int32(m), np.int32(k), np.int32(n))
        )
        elapsed = []
        for _ in range(6):
            event = kernel(queue, global_size, local_size, xb, qb, zb, sb, yb, *dims)
            event.wait()
            elapsed.append((event.profile.end - event.profile.start) / 1e6)
        out = np.empty((m, n), dtype=np.float32)
        cl.enqueue_copy(queue, out, yb).wait()
        expected = x @ (half_weights if suffix else weights).T
        np.testing.assert_allclose(out, expected, rtol=4e-4, atol=4e-4)
        if rounding_case:
            np.testing.assert_array_equal(out, expected)
        results.append(out)
        times.append(float(np.median(elapsed[1:])))
    if rounding_case:
        assert np.all(results[0] != results[1])
    LOGGER.warning(
        "ir device=%s matvec=%s M=%d N=%d K=%d original_ms=%.4f ir_ms=%.4f "
        "max_mode_difference=%.8g",
        queue.device.name,
        matvec,
        m,
        n,
        k,
        times[0],
        times[1],
        float(np.max(np.abs(results[0] - results[1]))),
    )


@pytest.mark.parametrize(
    "m,n,k,matvec,rounding_case",
    [
        (1, 5, 1, True, True),
        (9, 5, 1, False, True),
        (1, 17, 65, True, False),
        (3, 5, 7, False, False),
        (9, 35, 65, False, False),
        (17, 29, 127, False, False),
    ],
)
def test_ir_dequant_tails(
    device_program: tuple[Any, Any, Any],
    m: int,
    n: int,
    k: int,
    matvec: bool,
    rounding_case: bool,
) -> None:
    """Validate half-rounded weights and deliberately observable rounding.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        m: Number of token rows.

        n: Number of output channels.

        k: Number of input channels.

        matvec: Select the vector rather than prefill entry point.

        rounding_case: Use values that guarantee different rounded weights.
    """
    run_ir_case(device_program, m, n, k, matvec, rounding_case)


def test_ir_dequant_weight_edges(device_program: tuple[Any, Any, Any]) -> None:
    """Check individual positive, negative and subnormal weights exactly.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.
    """
    context, queue, program = device_program
    device = queue.device
    if "cl_khr_fp16" not in device.extensions.split():
        pytest.skip("IR dequantization requires cl_khr_fp16")
    if not device.half_fp_config & cl.device_fp_config.DENORM:
        pytest.skip("Exact subnormal check requires FP16 denormal support")
    q = np.array(
        [
            [3, 7, 128, 255, 0],
            [1, 2, 3, 128, 255],
            [0, 1, 127, 128, 255],
            [3, 11, 100, 200, 255],
        ],
        dtype=np.uint8,
    )
    zero = np.array([0, 0, 128, 255], dtype=np.uint8)
    scale = np.array([0.1, 2**-24, 2**-14, 0.33325], dtype=np.float16).astype(
        np.float32
    )
    x = np.eye(5, dtype=np.float32)
    weights = ((q.astype(np.float32) - zero[:, None]) * scale[:, None]).astype(
        np.float16
    )
    expected = weights.astype(np.float32).T
    flags = cl.mem_flags
    xb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=x)
    qb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=q)
    zb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=zero)
    sb = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=scale)
    out = np.empty((5, 4), dtype=np.float32)
    yb = cl.Buffer(context, flags.WRITE_ONLY, out.nbytes)
    cl.Kernel(program, "lfm_matmul_u8_f16_dequant")(
        queue,
        (128, 1, 1),
        (128, 1, 1),
        xb,
        qb,
        zb,
        sb,
        yb,
        np.int32(5),
        np.int32(5),
        np.int32(4),
    )
    cl.enqueue_copy(queue, out, yb).wait()
    np.testing.assert_array_equal(out, expected)
    vector = np.empty(4, dtype=np.float32)
    vector_buffer = cl.Buffer(context, flags.WRITE_ONLY, vector.nbytes)
    cl.Kernel(program, "lfm_matvec_u8_f16_dequant")(
        queue,
        (128, 1),
        (128, 1),
        xb,
        qb,
        zb,
        sb,
        vector_buffer,
        np.int32(5),
        np.int32(4),
    )
    cl.enqueue_copy(queue, vector, vector_buffer).wait()
    np.testing.assert_array_equal(vector, expected[0])
    prepared = cl.Buffer(context, flags.READ_WRITE, weights.nbytes)
    cl.Kernel(program, "lfm_prepare_u8_f16")(
        queue,
        (64,),
        (64,),
        qb,
        zb,
        sb,
        prepared,
        np.int32(5),
        np.int32(4),
    )
    stored = np.empty_like(weights)
    cl.enqueue_copy(queue, stored, prepared).wait()
    np.testing.assert_array_equal(stored, weights)
    cl.Kernel(program, "lfm_matmul_prepared_f16")(
        queue,
        (64, 1, 1),
        (64, 1, 1),
        xb,
        prepared,
        yb,
        np.int32(5),
        np.int32(5),
        np.int32(4),
    )
    cl.enqueue_copy(queue, out, yb).wait()
    np.testing.assert_array_equal(out, expected)


@pytest.mark.skipif(
    os.environ.get("LFM_GPU_BENCH") != "1", reason="opt-in GPU benchmark"
)
@pytest.mark.parametrize(
    "m,n,k,matvec",
    [
        (1, 1024, 1024, True),
        (1, 3072, 1024, True),
        (1, 1024, 4608, True),
        (8, 1024, 1024, False),
        (32, 1024, 1024, False),
        (128, 1024, 1024, False),
        (32, 1024, 4608, False),
    ],
)
def test_ir_dequant_benchmark(
    device_program: tuple[Any, Any, Any],
    m: int,
    n: int,
    k: int,
    matvec: bool,
) -> None:
    """Profile both arithmetic modes on synthetic checkpoint-sized projections.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        m: Number of token rows.

        n: Number of output channels.

        k: Number of input channels.

        matvec: Select the vector rather than prefill entry point.
    """
    run_ir_case(device_program, m, n, k, matvec)


def run_prepared_case(
    device_program: tuple[Any, Any, Any], m: int, n: int, k: int
) -> None:
    """Validate and profile a reusable half-workspace projection against NumPy.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        m: Number of token rows.

        n: Number of output channels.

        k: Number of input channels.
    """
    context, queue, program = device_program
    if "cl_khr_fp16" not in queue.device.extensions.split():
        pytest.skip("Prepared projection requires cl_khr_fp16")
    rng = np.random.default_rng(8317 + m + n + k)
    x = rng.normal(size=(m, k)).astype(np.float32)
    q = rng.integers(0, 256, (n, k), dtype=np.uint8)
    zero = rng.integers(0, 256, n, dtype=np.uint8)
    scale = rng.uniform(0.0001, 0.02, n).astype(np.float16).astype(np.float32)
    weights = ((q.astype(np.float32) - zero[:, None]) * scale[:, None]).astype(
        np.float16
    )
    expected = x @ weights.astype(np.float32).T
    flags = cl.mem_flags
    xb, qb, zb, sb = [
        cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=a)
        for a in (x, q, zero, scale)
    ]
    prepared = cl.Buffer(context, flags.READ_WRITE, weights.nbytes)
    yb = cl.Buffer(context, flags.WRITE_ONLY, m * n * np.dtype(np.float32).itemsize)
    prepare = cl.Kernel(program, "lfm_prepare_u8_f16")
    matmul = cl.Kernel(program, "lfm_matmul_prepared_f16")
    original = cl.Kernel(program, "lfm_matmul_u8_f16_dequant")
    all_times: list[list[float]] = [[], [], []]
    for _ in range(6):
        e1 = prepare(
            queue,
            (64 * ((n * k + 63) // 64),),
            (64,),
            qb,
            zb,
            sb,
            prepared,
            np.int32(k),
            np.int32(n),
        )
        e2 = matmul(
            queue,
            (64, (n + 7) // 8, (m + 7) // 8),
            (64, 1, 1),
            xb,
            prepared,
            yb,
            np.int32(m),
            np.int32(k),
            np.int32(n),
            wait_for=[e1],
        )
        e3 = original(
            queue,
            (128, (n + 7) // 8, (m + 7) // 8),
            (128, 1, 1),
            xb,
            qb,
            zb,
            sb,
            yb,
            np.int32(m),
            np.int32(k),
            np.int32(n),
            wait_for=[e2],
        )
        e3.wait()
        for times, event in zip(all_times, (e1, e2, e3), strict=True):
            times.append((event.profile.end - event.profile.start) / 1e6)
    stored = np.empty_like(weights)
    cl.enqueue_copy(queue, stored, prepared).wait()
    np.testing.assert_array_equal(stored, weights)
    matmul(
        queue,
        (64, (n + 7) // 8, (m + 7) // 8),
        (64, 1, 1),
        xb,
        prepared,
        yb,
        np.int32(m),
        np.int32(k),
        np.int32(n),
    )
    out = np.empty((m, n), dtype=np.float32)
    cl.enqueue_copy(queue, out, yb).wait()
    np.testing.assert_allclose(out, expected, rtol=4e-4, atol=4e-4)
    prepare_ms, matmul_ms, original_ms = [float(np.median(t[1:])) for t in all_times]
    LOGGER.warning(
        "prepared device=%s M=%d N=%d K=%d scratch_bytes=%d "
        "prepare_ms=%.4f matmul_ms=%.4f total_ms=%.4f baseline_ms=%.4f max_error=%.6g",
        queue.device.name,
        m,
        n,
        k,
        weights.nbytes,
        prepare_ms,
        matmul_ms,
        prepare_ms + matmul_ms,
        original_ms,
        float(np.max(np.abs(out - expected))),
    )


@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (3, 5, 7), (9, 35, 65), (17, 29, 127)])
def test_prepared_tails(
    device_program: tuple[Any, Any, Any], m: int, n: int, k: int
) -> None:
    """Check prepared weights and outputs across all tile tails.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        m: Number of token rows.

        n: Number of output channels.

        k: Number of input channels.
    """
    run_prepared_case(device_program, m, n, k)


@pytest.mark.skipif(
    os.environ.get("LFM_GPU_BENCH") != "1", reason="opt-in GPU benchmark"
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        (32, 1024, 1024),
        (128, 1024, 1024),
        (930, 1024, 1024),
        (930, 3072, 1024),
        (930, 1024, 4608),
    ],
)
def test_prepared_benchmark(
    device_program: tuple[Any, Any, Any], m: int, n: int, k: int
) -> None:
    """Measure end-to-end preparation plus prefill and the direct IR path.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        m: Number of token rows.

        n: Number of output channels.

        k: Number of input channels.
    """
    run_prepared_case(device_program, m, n, k)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1, 1),
        (3, 5, 7),
        (9, 35, 65),
        (33, 37, 33),
        (3, 19, 4608),
        (32, 4608, 1024),
        (32, 1024, 4608),
    ],
)
def test_prepared_kn_tails(
    device_program: tuple[Any, Any, Any], m: int, n: int, k: int
) -> None:
    """Check K-major packing and blocked multiplication across all tile tails.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        m: Number of token rows.

        n: Number of output channels.

        k: Number of input channels.
    """
    context, queue, program = device_program
    if "cl_khr_fp16" not in queue.device.extensions.split():
        pytest.skip("K-major projection requires cl_khr_fp16")
    rng = np.random.default_rng(9835 + m + n + k)
    x = rng.normal(size=(m, k)).astype(np.float32)
    q = rng.integers(0, 256, (n, k), dtype=np.uint8)
    zero = rng.integers(0, 256, n, dtype=np.uint8)
    scale = rng.uniform(0.0001, 0.02, n).astype(np.float16).astype(np.float32)
    weights = ((q.astype(np.float32) - zero[:, None]) * scale[:, None]).astype(
        np.float16
    )
    flags = cl.mem_flags
    xb, qb, zb, sb = [
        cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=a)
        for a in (x, q, zero, scale)
    ]
    prepared = cl.Buffer(context, flags.READ_WRITE, weights.nbytes)
    yb = cl.Buffer(context, flags.WRITE_ONLY, m * n * np.dtype(np.float32).itemsize)
    packed_event = cl.Kernel(program, "lfm_prepare_u8_f16_kn")(
        queue,
        (16 * ((k + 15) // 16), 16 * ((n + 15) // 16)),
        (16, 16),
        qb,
        zb,
        sb,
        prepared,
        np.int32(k),
        np.int32(n),
    )
    result_event = cl.Kernel(program, "lfm_matmul_prepared_kn_f16")(
        queue,
        (8 * ((n + 31) // 32), 8 * ((m + 31) // 32)),
        (8, 8),
        xb,
        prepared,
        yb,
        np.int32(m),
        np.int32(k),
        np.int32(n),
        wait_for=[packed_event],
    )
    stored = np.empty((k, n), dtype=np.float16)
    cl.enqueue_copy(queue, stored, prepared, wait_for=[packed_event]).wait()
    np.testing.assert_array_equal(stored.view(np.uint16), weights.T.view(np.uint16))
    out = np.empty((m, n), dtype=np.float32)
    cl.enqueue_copy(queue, out, yb, wait_for=[result_event]).wait()
    expected = x @ weights.astype(np.float32).T
    np.testing.assert_allclose(out, expected, rtol=8e-4, atol=8e-4)


def test_prepared_kn_rounding_edges(device_program: tuple[Any, Any, Any]) -> None:
    """Check packed bits for signed, subnormal, halfway and overflow weights.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.
    """
    context, queue, program = device_program
    if "cl_khr_fp16" not in queue.device.extensions.split():
        pytest.skip("K-major projection requires cl_khr_fp16")
    if not queue.device.half_fp_config & cl.device_fp_config.DENORM:
        pytest.skip("Exact subnormal check requires FP16 denormal support")
    scales = (
        np.array(
            [1, 2, 3, 127, 1023, 1024, 1025, 0x1400, 0x3555, 0x3C00, 0x5000, 0x7BFF],
            dtype=np.uint16,
        )
        .view(np.float16)
        .astype(np.float32)
    )
    zero = np.repeat(np.array([0, 1, 127, 128, 254, 255], dtype=np.uint8), len(scales))
    scale = np.tile(scales, 6)
    n, k = len(zero), 256
    q = np.tile(np.arange(k, dtype=np.uint8), (n, 1))
    with np.errstate(over="ignore"):
        expected = (
            ((q.astype(np.float32) - zero[:, None]) * scale[:, None])
            .astype(np.float16)
            .T.copy()
        )
    flags = cl.mem_flags
    qb, zb, sb = [
        cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=a)
        for a in (q, zero, scale)
    ]
    prepared = cl.Buffer(context, flags.WRITE_ONLY, expected.nbytes)
    event = cl.Kernel(program, "lfm_prepare_u8_f16_kn")(
        queue,
        (16 * ((k + 15) // 16), 16 * ((n + 15) // 16)),
        (16, 16),
        qb,
        zb,
        sb,
        prepared,
        np.int32(k),
        np.int32(n),
    )
    stored = np.empty_like(expected)
    cl.enqueue_copy(queue, stored, prepared, wait_for=[event]).wait()
    np.testing.assert_array_equal(stored.view(np.uint16), expected.view(np.uint16))


def vec4_edge_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return [N,257] U8 weights and exact FP16-rounded [N,257] reference.

    Rows cover signed differences, zeros, subnormal and normal scales, and
    halfway products. K=257 tests the scalar tail and unaligned row starts.
    Finite scales here keep all products finite so one-hot vectors isolate
    weights without zero times infinity contaminating the result.
    """
    scale_bits = np.array(
        [1, 2, 3, 127, 1023, 1024, 1025, 0x1400, 0x3555, 0x3C01, 0x5000, 0, 0xBC01],
        dtype=np.uint16,
    )
    scales = scale_bits.view(np.float16).astype(np.float32)
    zero = np.repeat(np.array([0, 1, 127, 128, 254, 255], dtype=np.uint8), len(scales))
    scale = np.tile(scales, 6)
    values = np.concatenate(
        (np.arange(256, dtype=np.uint8), np.array([3], dtype=np.uint8))
    )
    q = np.tile(values, (len(zero), 1))
    expected = ((q.astype(np.float32) - zero[:, None]) * scale[:, None]).astype(
        np.float16
    )
    return q, zero, scale, expected


def test_vec4_edge_fixture_reference() -> None:
    """Check explicit FP16 bit patterns for subnormal, signs and halfway tie."""
    q, zero, scale, expected = vec4_edge_fixture()
    assert q.shape == (78, 257)
    assert expected[0, 1].view(np.uint16) == 0x0001
    assert expected[0, 255].view(np.uint16) == 0x00FF
    assert expected[13, 0].view(np.uint16) == 0x8001
    assert expected[9, 3].view(np.uint16) == 0x4202
    assert zero[0] == 0 and scale[0] == np.float32(2**-24)


def test_vec4_exact_weight_edges(device_program: tuple[Any, Any, Any]) -> None:
    """Check each finite dequantized weight exactly via one-hot projections.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.
    """
    context, queue, program = device_program
    if "cl_khr_fp16" not in queue.device.extensions.split():
        pytest.skip("IR dequantization requires cl_khr_fp16")
    if not queue.device.half_fp_config & cl.device_fp_config.DENORM:
        pytest.skip("Exact subnormal check requires FP16 denormal support")
    q, zero, scale, expected = vec4_edge_fixture()
    n, k = q.shape
    flags = cl.mem_flags
    qb, zb, sb = [
        cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=a)
        for a in (q, zero, scale)
    ]
    x = np.zeros(k, dtype=np.float32)
    xb = cl.Buffer(context, flags.READ_ONLY, x.nbytes)
    out = np.empty(n, dtype=np.float32)
    yb = cl.Buffer(context, flags.WRITE_ONLY, out.nbytes)
    kernel = cl.Kernel(program, "lfm_matvec_u8_f16_dequant_vec4")
    for col in range(k):
        x.fill(0)
        x[col] = 1
        cl.enqueue_copy(queue, xb, x)
        kernel(
            queue,
            (128, (n + 7) // 8),
            (128, 1),
            xb,
            qb,
            zb,
            sb,
            yb,
            np.int32(k),
            np.int32(n),
        )
        cl.enqueue_copy(queue, out, yb).wait()
        np.testing.assert_array_equal(out, expected[:, col].astype(np.float32))


@pytest.mark.parametrize(
    "n,k", [(1, 1), (7, 3), (8, 4), (9, 5), (9, 63), (7, 64), (9, 65), (17, 129)]
)
def test_vec4_projection_tails(
    device_program: tuple[Any, Any, Any], n: int, k: int
) -> None:
    """Compare arbitrary vector and row tails with FP16-rounded dense weights.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        n: Number of output channels.

        k: Number of input channels.
    """
    context, queue, program = device_program
    if "cl_khr_fp16" not in queue.device.extensions.split():
        pytest.skip("IR dequantization requires cl_khr_fp16")
    rng = np.random.default_rng(235 + n + k)
    x = rng.normal(size=k).astype(np.float32)
    q = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    zero = rng.integers(0, 256, size=n, dtype=np.uint8)
    scale = rng.uniform(0.0001, 0.02, size=n).astype(np.float16).astype(np.float32)
    expected = ((q.astype(np.float32) - zero[:, None]) * scale[:, None]).astype(
        np.float16
    ).astype(np.float32) @ x
    flags = cl.mem_flags
    xb, qb, zb, sb = [
        cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=a)
        for a in (x, q, zero, scale)
    ]
    out = np.empty(n, dtype=np.float32)
    yb = cl.Buffer(context, flags.WRITE_ONLY, out.nbytes)
    cl.Kernel(program, "lfm_matvec_u8_f16_dequant_vec4")(
        queue,
        (128, (n + 7) // 8),
        (128, 1),
        xb,
        qb,
        zb,
        sb,
        yb,
        np.int32(k),
        np.int32(n),
    )
    cl.enqueue_copy(queue, out, yb).wait()
    np.testing.assert_allclose(out, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("activation", [1.0, -1.0, 0.0])
def test_vec4_overflow_and_nan(
    device_program: tuple[Any, Any, Any], activation: float
) -> None:
    """Check FP16 overflow and zero-times-infinity for the scalar K tail.

    Args:
        device_program: OpenCL context, profiling queue and compiled program.

        activation: Input activation, including zero for IEEE NaN behavior.
    """
    context, queue, program = device_program
    if "cl_khr_fp16" not in queue.device.extensions.split():
        pytest.skip("IR dequantization requires cl_khr_fp16")
    x = np.array([activation], dtype=np.float32)
    q = np.array([[255], [0]], dtype=np.uint8)
    zero = np.array([0, 255], dtype=np.uint8)
    scale = np.full(2, 65504, dtype=np.float32)
    flags = cl.mem_flags
    xb, qb, zb, sb = [
        cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=a)
        for a in (x, q, zero, scale)
    ]
    out = np.empty(2, dtype=np.float32)
    yb = cl.Buffer(context, flags.WRITE_ONLY, out.nbytes)
    cl.Kernel(program, "lfm_matvec_u8_f16_dequant_vec4")(
        queue, (128, 1), (128, 1), xb, qb, zb, sb, yb, np.int32(1), np.int32(2)
    )
    cl.enqueue_copy(queue, out, yb).wait()
    with np.errstate(over="ignore", invalid="ignore"):
        weights = (
            ((q.astype(np.float32) - zero[:, None]) * scale[:, None])
            .astype(np.float16)
            .astype(np.float32)
        )
        expected = weights[:, 0] * x[0]
    np.testing.assert_array_equal(out, expected)
