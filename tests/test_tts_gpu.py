"""Offline invariants for maintained TTS GPU operation identity and host imports."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as ops

from hoast.tts_gpu import CPUConvOffload, Layer, SharedGPU, decoder_convolutions


def _fake_gpu() -> tuple[SharedGPU, list[MagicMock]]:
    """Provide diagnostic-confirmed mock imports without requiring an OpenCL device."""
    gpu = SharedGPU.__new__(SharedGPU)
    gpu.cl = MagicMock()
    gpu.context = MagicMock()
    gpu.queue = MagicMock()
    gpu.diagnostics = []
    gpu.closed = False
    created: list[MagicMock] = []

    def create(context: object, flags: object, hostbuf: np.ndarray) -> MagicMock:
        """Emulate the driver's successful physical-sharing diagnostic.

        Args:
            context:
                Mock OpenCL context.

            flags:
                Mock buffer access flags.

            hostbuf:
                Imported page-aligned byte view.

        """
        gpu.diagnostics.append(
            f"clCreateBuffer pointer {hex(hostbuf.ctypes.data)} will share the same physical memory with CPU"
        )
        buffer = MagicMock()
        created.append(buffer)
        return buffer

    gpu.cl.Buffer.side_effect = create
    return gpu, created


def test_distinct_offloads_survive_graph_optimization() -> None:
    """Keep different weight/geometry identities separate despite a shared input."""
    gpu = SharedGPU.__new__(SharedGPU)
    data = ops.parameter([1, 3, -1], ov.Type.f32)
    attrs = {
        "strides": [1],
        "pads_begin": [1],
        "pads_end": [1],
        "dilations": [1],
        "auto_pad": "explicit",
    }
    first = CPUConvOffload(
        data.output(0),
        gpu,
        Layer("same_name", (4, 3, 3), attrs, None, None, None, "first_weights"),
    )
    second = CPUConvOffload(
        data.output(0),
        gpu,
        Layer("same_name", (8, 3, 3), attrs, None, None, None, "second_weights"),
    )
    assert (
        first.get_attributes()["weight_identity"]
        != second.get_attributes()["weight_identity"]
    )
    model = ov.Model([first, second], [data])  # pyright: ignore[reportGeneralTypeIssues]
    compiled = ov.Core().compile_model(
        model, "CPU", {"INFERENCE_NUM_THREADS": 2, "NUM_STREAMS": "1"}
    )

    def execute(layer: Layer, inputs: np.ndarray, output: np.ndarray) -> None:
        """Write a distinguishable result for each immutable synthetic layer.

        Args:
            layer:
                Synthetic fixed-weight identity.

            inputs:
                CPU input tensor; only its validated shape matters here.

            output:
                CPU result buffer allocated by shape inference.

        """
        assert inputs.shape[1] == 3
        output.fill(1 if layer.fingerprint == "first_weights" else 2)

    with patch.object(SharedGPU, "execute", side_effect=execute) as dispatch:
        for width in (17, 39):
            result = compiled([np.zeros((1, 3, width), dtype=np.float32)])
            np.testing.assert_array_equal(
                result[compiled.output(0)], np.ones((1, 4, width))
            )
            np.testing.assert_array_equal(
                result[compiled.output(1)], np.full((1, 8, width), 2)
            )
        assert dispatch.call_count == 4


def test_host_import_aliases_tensor_without_copy() -> None:
    """Align imports to containing pages while preserving the logical tensor offset."""
    gpu, _ = _fake_gpu()
    owner = np.zeros(16384, dtype=np.float32)
    values = owner[17:2049]
    _, view, offset = gpu.import_host(values, True)
    assert view.ctypes.data % 4096 == 0
    assert view.size % 64 == 0
    assert view.ctypes.data + offset * 4 == values.ctypes.data
    assert np.shares_memory(view, values)
    view.view(np.float32)[offset] = 42
    assert values[0] == 42


def test_overlapping_pages_use_one_import() -> None:
    """Adjacent logical tensors on a page share one OpenCL object, not overlapping ones."""
    gpu, created = _fake_gpu()
    owner = np.zeros(4096, dtype=np.float32)
    start = ((-owner.ctypes.data) % 4096) // 4 + 16
    data, output = owner[start : start + 16], owner[start + 16 : start + 32]
    x, y, _, _, xo, yo = gpu.import_pair(data, output)
    assert x is y and len(created) == 1
    assert yo - xo == 16
    with pytest.raises(AssertionError):
        gpu.import_pair(data, data)


def test_failed_enqueue_releases_imports_and_kernel_bindings() -> None:
    """Failure cleanup drains the queue and prevents stale CPU-arena bindings."""
    gpu, created = _fake_gpu()
    gpu.cl.enqueue_nd_range_kernel.side_effect = RuntimeError("enqueue failed")
    kernel = MagicMock()
    layer = Layer("failure", (1, 1, 1), {}, kernel, None, None, "failure_weights")
    data = np.zeros((1, 1, 128), dtype=np.float32)
    output = np.zeros_like(data)
    with pytest.raises(RuntimeError, match="enqueue failed"):
        gpu.execute(layer, data, output)
    gpu.queue.finish.assert_called_once()
    assert kernel.set_arg.call_args_list == [call(0, None), call(3, None)]
    assert created
    for buffer in created:
        buffer.release.assert_called_once()


def test_spilling_tile_is_reduced_and_choice_cached() -> None:
    """Choose a nonspilling kernel once and reuse its geometry across future calls."""
    gpu, _ = _fake_gpu()
    gpu.source_code = "fixture"
    gpu.device = MagicMock()
    gpu.programs = {}
    gpu.tile_choices = {}
    attrs = {"strides": [1], "pads_begin": [5], "dilations": [1]}
    with patch(
        "hoast.tts_gpu.spill_bytes",
        side_effect=[448, 0, 0],
    ):
        _, tile = gpu._kernel((128, 128, 11), attrs)
        assert tile == 16
        assert gpu.cl.Program.call_count == 2
        _, tile = gpu._kernel((128, 128, 11), attrs)
        assert tile == 16
        assert gpu.cl.Program.call_count == 2


def test_unconfirmed_sharing_releases_buffer() -> None:
    """Do not equate an imported pointer with verified physical zero-copy sharing."""
    gpu, _ = _fake_gpu()
    buffer = MagicMock()
    gpu.cl.Buffer.side_effect = None
    gpu.cl.Buffer.return_value = buffer
    with pytest.raises(RuntimeError, match="confirm zero-copy"):
        gpu.import_host(np.zeros(32, dtype=np.float32), False)
    buffer.release.assert_called_once()


def test_shutdown_failure_still_releases_resources() -> None:
    """Preserve a queue failure while clearing owned handles and allowing repeated close."""
    gpu, _ = _fake_gpu()
    queue = gpu.queue
    queue.finish.side_effect = RuntimeError("queue failed")
    gpu.programs = {(1,): object()}
    gpu.tile_choices = {(1,): 32}
    with pytest.raises(RuntimeError, match="queue failed"):
        gpu.close()
    assert gpu.closed and gpu.queue is None and gpu.context is None
    assert not gpu.programs and not gpu.tile_choices
    gpu.close()
    queue.finish.assert_called_once()


def test_decoder_coverage_excludes_frontend() -> None:
    """Recognize both export naming styles and reject incomplete decoder coverage."""
    data = ops.parameter([1, 16, -1], ov.Type.f32)
    tiny_data = ops.parameter([1, 1, -1], ov.Type.f32)
    weight = ops.constant(np.ones((2, 16, 3), dtype=np.float32))
    selected = []
    for index in range(67):
        node = ops.convolution(data, weight, [1], [1], [1], [1])
        node.set_friendly_name(f"/decoder/decoder/layer{index}/Conv")
        selected.append(node)
    tiny = []
    for name in ("F0_conv", "N_conv"):
        node = ops.convolution(
            tiny_data,
            ops.constant(np.ones((1, 1, 3), dtype=np.float32)),
            [1],
            [1],
            [1],
            [1],
        )
        node.set_friendly_name(f"__module.model.decoder.{name}/Convolution")
        tiny.append(node)
    frontend = ops.convolution(data, weight, [1], [1], [1], [1])
    frontend.set_friendly_name("/encoder/conv")
    model = ov.Model([*selected, *tiny, frontend], [data, tiny_data])  # pyright: ignore[reportGeneralTypeIssues]
    assert set(decoder_convolutions(model)) == set(selected)
    incomplete = ov.Model([*selected[:-1], *tiny, frontend], [data, tiny_data])  # pyright: ignore[reportGeneralTypeIssues]
    with pytest.raises(ValueError, match="Expected 69"):
        decoder_convolutions(incomplete)
