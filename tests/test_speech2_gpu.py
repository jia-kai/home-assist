"""Offline packed-weight invariants and explicitly opt-in OpenCL correctness tests."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hoast.runtime import configure_cpu_budget
from hoast.speech2_gpu import (
    FP32_ATOL,
    FP32_RTOL,
    ExperimentalGPU,
    dequantize_reference,
    error_metrics,
    quantize_weights,
    reference_conv1d,
    reference_matmul,
    reference_postscale,
)
from tools import benchmark_speech2_gpu


def test_packing_scales_and_zero_channel() -> None:
    """Preserve per-output scales, channel order, zero padding and source ownership."""
    source = np.array([[0, 127, -127], [0, 0, 0], [254, -254, 2]], dtype=np.float32)
    packed = quantize_weights(source)
    assert packed.values.shape == (3, 32)
    assert packed.values.dtype == np.int8
    assert packed.scales.dtype == np.float32
    np.testing.assert_array_equal(packed.scales[:3], [1, 1, 2])
    np.testing.assert_array_equal(packed.values[:, 0], [0, 127, -127])
    np.testing.assert_array_equal(packed.values[:, 1], [0, 0, 0])
    np.testing.assert_array_equal(packed.values[:, 2], [127, -127, 1])
    assert not packed.values[:, 3:].any()
    assert not packed.scales[3:].any()
    assert not packed.values.flags.writeable and not packed.scales.flags.writeable
    assert not np.shares_memory(source, packed.values)
    np.testing.assert_array_equal(dequantize_reference(packed), source)
    assert packed.nbytes == 3 * 32 + 32 * 4


@pytest.mark.parametrize("shape", [(33, 7), (17, 3, 7)])
def test_fixture_quantization_error_bound(shape: tuple[int, ...]) -> None:
    """Bound fixture weight error separately from FP32 operator arithmetic error.

    Args:
        shape:
            Output-major matrix or convolution geometry, including output tile tails.

    """
    rng = np.random.default_rng(9)
    original = rng.normal(0, 0.2, shape).astype(np.float32)
    packed = quantize_weights(original)
    restored = dequantize_reference(packed)
    assert restored.shape == original.shape
    channel_error = np.max(np.abs(restored - original).reshape(shape[0], -1), axis=1)
    assert np.all(channel_error <= packed.scales[: shape[0]] * 0.50002)
    metrics = error_metrics(restored, original)
    assert 0 < metrics["relative_l2"] < 0.02


def test_original_and_dequantized_matmul_are_distinct_baselines() -> None:
    """Do not mislabel quantization error as kernel arithmetic error."""
    data = np.array([[1, -2, 3], [-2, 0.5, 1]], dtype=np.float32)
    weights = np.array([[0.1, 0.3, 0.7], [0.9, -0.4, 0.05]], dtype=np.float32)
    dequantized = dequantize_reference(quantize_weights(weights))
    original = reference_matmul(data, weights)
    quantized = reference_matmul(data, dequantized)
    assert error_metrics(quantized, original)["max_abs"] > 0
    np.testing.assert_array_equal(quantized, data @ dequantized.T)
    assert error_metrics(quantized, quantized) == {
        "max_abs": 0,
        "rmse": 0,
        "relative_l2": 0,
    }


@pytest.mark.parametrize("dilation", [1, 3, 5])
def test_dilated_convolution_matches_scalar_fixture(dilation: int) -> None:
    """Check symmetric zero padding, temporal boundaries and channel/tap ordering.

    Args:
        dilation:
            Tap spacing that can exceed the small fixture's temporal length.

    """
    data = np.arange(10, dtype=np.float32).reshape(2, 5) / np.float32(10)
    weights = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / np.float32(20)
    expected = np.zeros((3, 5), dtype=np.float32)
    for output in range(3):
        for t in range(5):
            for channel in range(2):
                for tap in range(3):
                    position = t + (tap - 1) * dilation
                    if 0 <= position < 5:
                        expected[output, t] += (
                            data[channel, position] * weights[output, channel, tap]
                        )
    np.testing.assert_allclose(
        reference_conv1d(data, weights, dilation), expected, atol=1e-6
    )


@pytest.mark.parametrize(
    "weights",
    [
        np.ones((2, 3), dtype=np.float64),
        np.ones((2, 3, 2), dtype=np.float32),
        np.ones((0, 3), dtype=np.float32),
        np.array([[np.nan]], dtype=np.float32),
        np.array([[np.inf]], dtype=np.float32),
    ],
)
def test_invalid_quantization_inputs_fail(weights: np.ndarray) -> None:
    """Reject invalid external weights rather than silently quantizing them.

    Args:
        weights:
            Invalid dtype, geometry or finite-value fixture.

    """
    with pytest.raises(ValueError):
        quantize_weights(weights)


def test_tiny_and_zero_scales_remain_finite() -> None:
    """Avoid zero scales for subnormal FP32 channels and retain exact zero channels."""
    source = np.array(
        [[np.nextafter(np.float32(0), np.float32(1))], [0]], dtype=np.float32
    )
    packed = quantize_weights(source)
    assert np.all(packed.scales[:2] > 0)
    assert np.isfinite(dequantize_reference(packed)).all()


def test_device_prepare_never_expands_weights() -> None:
    """Mock the upload boundary and require INT8 weights plus FP32 scales only."""
    gpu = ExperimentalGPU.__new__(ExperimentalGPU)
    gpu.cl = MagicMock()
    gpu.context = MagicMock()
    gpu.queue = MagicMock()
    events = [MagicMock(), MagicMock()]
    for event in events:
        event.profile.start = 100
        event.profile.end = 1100
    gpu.cl.enqueue_copy.side_effect = events
    packed = quantize_weights(np.ones((64, 16, 3), dtype=np.float32))
    with patch(
        "hoast.speech2_gpu.dequantize_reference",
        side_effect=AssertionError("expansion"),
    ):
        operation = gpu.prepare(packed, 3)
    uploads = gpu.cl.enqueue_copy.call_args_list
    assert uploads[0].args[2] is packed.values
    assert uploads[0].args[2].dtype == np.int8
    assert uploads[1].args[2] is packed.scales
    assert gpu.cl.Buffer.call_args_list[0].args[2] == packed.values.nbytes
    assert gpu.cl.Buffer.call_args_list[1].args[2] == packed.scales.nbytes
    assert operation.setup["expanded_device_weight_bytes"] == 0
    operation.close()


@pytest.mark.parametrize("variant", ["prescale", "postscale"])
def test_variant_compiler_selection_without_device(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """Select explicit kernel names/options using a mocked compiler, not OpenCL.

    Args:
        monkeypatch:
            Replaces the optional PyOpenCL import with an entirely in-memory API.

        variant:
            Reference or candidate arithmetic selection.

    """
    api = MagicMock()
    device = MagicMock()
    device.name = "mock UHD 630"
    device.max_work_group_size = 64
    api.get_platforms.return_value = [MagicMock()]
    api.get_platforms.return_value[0].get_devices.return_value = [device]
    monkeypatch.setitem(sys.modules, "pyopencl", api)
    gpu = ExperimentalGPU(variant=variant)  # pyright: ignore[reportArgumentType]
    assert gpu.variant == variant
    options = api.Program.return_value.build.call_args.kwargs["options"]
    assert options == [
        "-cl-std=CL1.2",
        f"-DSPEECH_POSTSCALE={int(variant == 'postscale')}",
    ]
    suffix = "_postscale" if variant == "postscale" else ""
    assert [call.args[1] for call in api.Kernel.call_args_list] == [
        f"speech_matmul{suffix}",
        f"speech_conv1d{suffix}",
    ]


def test_invalid_variant_fails_before_opencl_import() -> None:
    """Reject unknown arithmetic choices before optional device setup begins."""
    with pytest.raises(ValueError, match="Scaling variant"):
        ExperimentalGPU(variant="unknown")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "shape,input_shape,dilation",
    [
        ((35, 37), (9, 37), 1),
        ((17, 3, 3), (3, 9), 1),
        ((16, 16, 7), (16, 19), 3),
        ((17, 3, 11), (3, 7), 5),
    ],
)
def test_postscale_fixture_reassociation(
    shape: tuple[int, ...], input_shape: tuple[int, int], dilation: int
) -> None:
    """Keep candidate error separate from quantization with tails/zero/varied scales.

    Args:
        shape:
            Output-major packed matrix or SAME-convolution weight geometry.

        input_shape:
            Activation geometry, including temporal/reduction tile tails.

        dilation:
            Convolution tap spacing in samples; unused by matrix fixtures.

    """
    rng = np.random.default_rng(313)
    data = rng.normal(0, 0.5, input_shape).astype(np.float32)
    weights = rng.normal(0, 0.02, shape).astype(np.float32)
    factors = np.geomspace(0.01, 10, shape[0]).astype(np.float32)
    weights *= factors.reshape((shape[0],) + (1,) * (len(shape) - 1))
    weights[0] = 0
    packed = quantize_weights(weights)
    dequantized = dequantize_reference(packed)
    expected = (
        reference_matmul(data, dequantized)
        if len(shape) == 2
        else reference_conv1d(data, dequantized, dilation)
    )
    actual = reference_postscale(data, packed, dilation)
    np.testing.assert_allclose(actual, expected, atol=FP32_ATOL, rtol=FP32_RTOL)
    zero = actual[:, 0] if len(shape) == 2 else actual[0]
    assert not zero.any()
    assert packed.values.dtype == np.int8
    assert packed.nbytes == packed.values.nbytes + packed.scales.nbytes


def test_postscale_extreme_activation_overflow_is_not_hidden() -> None:
    """Document the candidate's narrower raw-dot range despite a finite scaled dot."""
    packed = quantize_weights(np.full((1, 2), 1e-12, dtype=np.float32))
    data = np.array([[1e37, -1e37]], dtype=np.float32)
    assert np.isfinite(reference_matmul(data, dequantize_reference(packed))).all()
    with pytest.raises(FloatingPointError):
        reference_postscale(data, packed)


def test_candidate_cli_preserves_existing_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse candidate result tags that already own a log, before any execution.

    Args:
        tmp_path:
            Empty fixture artifact root; contains only an existing sentinel log.

        monkeypatch:
            Supplies offline candidate CLI arguments without model/device access.

    """
    root = tmp_path / "results/speech2-gpu"
    root.mkdir(parents=True)
    log = root / "keep.log"
    log.write_text("existing evidence")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--root",
            str(tmp_path),
            "--tag",
            "keep",
            "--offline",
            "--variant",
            "postscale",
        ],
    )
    with pytest.raises(FileExistsError):
        benchmark_speech2_gpu.main()
    assert log.read_text() == "existing evidence"
    assert not (root / "keep.json").exists()


@pytest.mark.skipif(
    os.environ.get("SPEECH2_GPU_TEST") != "1",
    reason="Opt-in only: native OpenCL build/device execution must be serialized",
)
@pytest.mark.parametrize("transfer", ["copy", "map"])
@pytest.mark.parametrize("variant", ["prescale", "postscale"])
@pytest.mark.parametrize(
    "weight_shape,input_shape,dilation",
    [
        ((35, 37), (9, 37), 1),
        ((512, 512), (8, 512), 1),
        ((2048, 512), (8, 512), 1),
        ((17, 3, 3), (3, 9), 1),
        ((32, 16, 7), (16, 19), 3),
        ((16, 16, 11), (16, 7), 5),
    ],
)
def test_opencl_against_dequantized_reference(
    transfer: str,
    variant: str,
    weight_shape: tuple[int, ...],
    input_shape: tuple[int, int],
    dilation: int,
) -> None:
    """Validate tails, dilations, explicit transfers and resident-kernel repetition.

    Args:
        transfer:
            Output retrieval policy, copy or map plus host copy.

        variant:
            Reference scaling before reduction, or experimental final postscale.

        weight_shape:
            Matrix or odd-tap convolution output-major weight dimensions.

        input_shape:
            Matrix (M,K) or convolution (CI,T) activation geometry.

        dilation:
            Positive convolution tap spacing, ignored for matmul.

    """
    configure_cpu_budget(2)
    gpu = ExperimentalGPU(variant=variant)  # pyright: ignore[reportArgumentType]
    rng = np.random.default_rng(28)
    data = rng.normal(0, 0.5, input_shape).astype(np.float32)
    weights = rng.normal(0, 0.02, weight_shape).astype(np.float32)
    packed = quantize_weights(weights)
    dequantized = dequantize_reference(packed)
    expected = (
        reference_matmul(data, dequantized)
        if len(weight_shape) == 2
        else reference_conv1d(data, dequantized, dilation)
    )
    operation = gpu.prepare(packed, dilation)
    workspace = None
    try:
        workspace = gpu.workspace(operation, input_shape, transfer)  # pyright: ignore[reportArgumentType]
        actual, timing = gpu.execute(operation, workspace, data)
        np.testing.assert_allclose(actual, expected, atol=FP32_ATOL, rtol=FP32_RTOL)
        if variant == "postscale":
            np.testing.assert_allclose(
                actual,
                reference_postscale(data, packed, dilation),
                atol=FP32_ATOL,
                rtol=FP32_RTOL,
            )
        assert timing["kernel_ms"] > 0 and timing["wall_ms"] > 0
        gpu.launch(operation, workspace).wait()
        repeated, _ = gpu.execute(operation, workspace, data)
        np.testing.assert_allclose(repeated, expected, atol=FP32_ATOL, rtol=FP32_RTOL)
    finally:
        gpu.close()
        if workspace is not None:
            workspace.close()
        operation.close()
