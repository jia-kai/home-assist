"""Kokoro decoder convolution on UHD 630 with zero-copy CPU tensor sharing."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import openvino as ov
import pyopencl as cl
from numpy.typing import NDArray
from openvino import opset13 as ops
from openvino.passes import ConstantFolding, Manager

from .logging import get_logger

logger = get_logger(__name__)
KERNEL_SOURCE = Path(__file__).with_name("kernels") / "conv_window.cl"
_NOTIFY = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p
)


def spill_bytes(kernel: Any, device: Any) -> int:
    """Return Intel's compiler spill footprint in bytes per hardware thread.

    Args:
        kernel:
            Compiled OpenCL convolution kernel.

        device:
            Intel GPU used to build the kernel.

    """
    query = ctypes.CDLL("libOpenCL.so.1").clGetKernelWorkGroupInfo
    query.restype = ctypes.c_int
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    value = ctypes.c_uint64()
    status = query(
        kernel.int_ptr,
        device.int_ptr,
        0x4109,
        ctypes.sizeof(value),
        ctypes.byref(value),
        None,
    )
    if status != 0:
        raise RuntimeError(f"Intel kernel spill query failed: {status}")
    return value.value


def _host_range(values: NDArray[np.float32]) -> tuple[int, int]:
    """Return page-aligned base and cache-line-rounded extent of a live CPU tensor.

    Args:
        values:
            Nonempty contiguous FP32 tensor, whose containing pages remain mapped.

    """
    assert values.flags.c_contiguous and values.dtype == np.float32 and values.size
    pointer = values.ctypes.data
    offset = pointer % 4096
    return pointer - offset, ((offset + values.nbytes + 63) // 64) * 64


@dataclass(slots=True, frozen=True)
class Layer:
    """Resident quantized weights, compiled kernel and immutable convolution geometry."""

    name: str
    """Original model operation name."""

    shape: tuple[int, int, int]
    """Weight dimensions (output channels, input channels, temporal filter size)."""

    attrs: Mapping[str, Any]
    """Original convolution strides, padding and dilation."""

    kernel: Any
    """PyOpenCL kernel compiled without temporal length specialization."""

    weights: Any
    """Resident signed INT8 OpenCL weight buffer."""

    scales: Any
    """Resident per-output-channel FP16 scale buffer."""

    fingerprint: str
    """Immutable digest of weights, scales, geometry, precision and kernel semantics."""

    tile_time: int = 32
    """Output time positions computed by one workgroup."""

    def __post_init__(self) -> None:
        """Freeze geometry metadata independently of mutable source dictionaries."""
        object.__setattr__(
            self,
            "attrs",
            MappingProxyType(
                {
                    key: tuple(value) if isinstance(value, list) else value
                    for key, value in self.attrs.items()
                }
            ),
        )


@dataclass(slots=True)
class SharedGPU:
    """Own one OpenCL context/queue and keep all kernels length-independent."""

    source_code: str = field(init=False, repr=False)
    """Source snapshot shared by all compiled programs and layer fingerprints."""

    cl: Any = field(init=False)
    """OpenCL API used by the executor."""

    context: Any = field(init=False)
    """Intel context with diagnostic callbacks enabled."""

    device: Any = field(init=False)
    """Target device used for compile-time resource queries."""

    queue: Any = field(init=False)
    """In-order command queue shared by serialized convolution calls."""

    callback: Any = field(init=False, repr=False)
    """Strong reference keeping the native diagnostic callback alive."""

    programs: dict[tuple[object, ...], Any] = field(default_factory=dict, init=False)
    """Compiled programs keyed only by layer geometry and types, never time length."""

    tile_choices: dict[tuple[object, ...], int] = field(
        default_factory=dict, init=False
    )
    """Largest nonspilling temporal tile selected once for each geometry."""

    diagnostics: list[str] = field(default_factory=list, init=False)
    """Reports for the current operation; detailed history is written to the logger."""

    closed: bool = field(default=False, init=False)
    """Whether the executor has released its queue and compilation cache."""

    def __post_init__(self) -> None:
        """Create a diagnostic context; no kernels or activation buffers are allocated."""
        if os.sysconf("SC_PAGE_SIZE") != 4096:
            raise RuntimeError("Shared GPU tensors require 4096-byte host pages")
        self.source_code = KERNEL_SOURCE.read_text()
        self.cl = cl
        devices = [
            d
            for p in self.cl.get_platforms()
            for d in p.get_devices(device_type=self.cl.device_type.GPU)
        ]
        if len(devices) != 1 or "630" not in devices[0].name:
            raise RuntimeError("The tuned TTS runtime requires one Intel UHD 630 GPU")
        self.device = devices[0]
        messages = self.diagnostics

        def notify(message: bytes, private: int, size: int, user: int) -> None:
            """Retain native diagnostic messages without propagating callback exceptions.

            Args:
                message:
                    Driver diagnostic UTF-8 text.

                private:
                    Unused implementation-specific data pointer.

                size:
                    Unused private-data byte count.

                user:
                    Unused user-data pointer.

            """
            text = message.decode("utf-8", errors="replace")
            messages.append(text)
            logger.debug("opencl.diagnostic %s", text)

        self.callback = _NOTIFY(notify)
        library = ctypes.CDLL("libOpenCL.so.1")
        create = library.clCreateContext
        create.restype = ctypes.c_void_p
        create.argtypes = [
            ctypes.POINTER(ctypes.c_ssize_t),
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
            _NOTIFY,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        properties = (ctypes.c_ssize_t * 3)(0x4106, 7, 0)
        handles = (ctypes.c_void_p * 1)(devices[0].int_ptr)
        error = ctypes.c_int()
        handle = create(
            properties, 1, handles, self.callback, None, ctypes.byref(error)
        )
        if error.value or not handle:
            raise RuntimeError(f"OpenCL diagnostic context failed: {error.value}")
        self.context = self.cl.Context.from_int_ptr(handle, retain=False)
        self.queue = self.cl.CommandQueue(self.context)

    def _kernel(
        self, shape: tuple[int, int, int], attrs: Mapping[str, Any]
    ) -> tuple[Any, int]:
        """Compile a reusable kernel and avoid scratch spills by reducing window tiles.

        Args:
            shape:
                Fixed weight dimensions (output channels, input channels, taps).

            attrs:
                Fixed convolution geometry; temporal input length is deliberately absent.

        """
        oc, ic, kw = shape
        stride, pad, dilation = (
            attrs["strides"][0],
            attrs["pads_begin"][0],
            attrs["dilations"][0],
        )
        if self.closed:
            raise RuntimeError("TTS GPU executor is closed")
        geometry = (oc, ic, kw, stride, pad, dilation)
        choices = (
            [self.tile_choices[geometry]]
            if geometry in self.tile_choices
            else [32, 16, 8]
        )
        for tile in choices:
            key = (*geometry, tile)
            options = [
                f"-DCI={ic}",
                f"-DCO={oc}",
                f"-DKW={kw}",
                f"-DSTRIDE={stride}",
                f"-DPAD={pad}",
                f"-DDILATION={dilation}",
                f"-DTILE={tile}",
            ]
            if key not in self.programs:
                self.programs[key] = self.cl.Program(
                    self.context, self.source_code
                ).build(options=options)
            kernel = self.cl.Kernel(self.programs[key], "hoast_conv")
            spill = spill_bytes(kernel, self.device)
            if spill:
                logger.debug(
                    "gpu.kernel geometry=%s tile=%d spill_bytes=%d; trying a smaller tile",
                    geometry,
                    tile,
                    spill,
                )
                continue
            self.tile_choices[geometry] = tile
            return kernel, tile
        raise RuntimeError(f"No nonspilling kernel for geometry {geometry}")

    def prepare(self, node: ov.Node) -> Layer:
        """Quantize fixed weights and select a nonspilling runtime-length GPU kernel.

        Args:
            node:
                Ordinary rank-three FP32 convolution with constant weights.

        """
        self.diagnostics.clear()
        weight_node = node.input_value(1).get_node()
        if weight_node.get_type_name() != "Constant":
            raise ValueError("Kokoro convolution requires constant weights")
        weights = weight_node.get_data().astype(np.float16)
        if weights.ndim != 3 or node.get_input_element_type(0) != ov.Type.f32:
            raise ValueError("Kokoro convolution requires rank-three FP32 activations")
        oc, ic, kw = weights.shape
        attrs = node.get_attributes()
        scales = (
            np.max(np.abs(weights.astype(np.float32)), axis=(1, 2), keepdims=True) / 127
        )
        scales[scales == 0] = 1
        if attrs["auto_pad"].lower() != "explicit":
            raise ValueError("Kokoro convolution requires explicit padding")
        stored = (
            np.rint(weights.astype(np.float32) / scales).clip(-127, 127).astype(np.int8)
        )
        kernel, tile = self._kernel((oc, ic, kw), attrs)
        flags = self.cl.mem_flags.READ_ONLY | self.cl.mem_flags.COPY_HOST_PTR
        stored = (
            np.pad(stored, ((0, (-oc) % 16), (0, 0), (0, 0))).transpose(1, 2, 0).copy()
        )
        scales = np.pad(scales.reshape(-1), (0, (-oc) % 16))
        fingerprint = hashlib.sha256(
            self.source_code.encode()
            + stored.tobytes()
            + scales.astype(np.float16).tobytes()
            + json.dumps(
                {
                    "attrs": attrs,
                    "shape": [oc, ic, kw],
                    "window_tile": tile,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return Layer(
            node.get_friendly_name(),
            (oc, ic, kw),
            attrs,
            kernel,
            self.cl.Buffer(self.context, flags, hostbuf=stored),
            self.cl.Buffer(self.context, flags, hostbuf=scales.astype(np.float16)),
            fingerprint,
            tile_time=tile,
        )

    def import_host(
        self, values: NDArray[np.float32], writable: bool
    ) -> tuple[Any, Any, int]:
        """Import mapped host pages without an application-side activation copy.

        Args:
            values:
                Contiguous live CPU tensor. Its first/last containing pages stay mapped
                during synchronous evaluation; kernels access only the tensor region.

            writable:
                Whether this buffer is the output. The imported page range can include
                neighboring allocator bytes, which the GPU must never access.

        """
        base, size = _host_range(values)
        buffer, view = self._import_region(base, size, writable)
        return buffer, view, (values.ctypes.data - base) // 4

    def _import_region(self, base: int, size: int, writable: bool) -> tuple[Any, Any]:
        """Import one region and require driver confirmation of physical sharing.

        Args:
            base:
                Page-aligned virtual address of mapped live tensor pages.

            size:
                Cache-line-rounded byte extent.

            writable:
                Whether the GPU can write this region.

        """
        owner = (ctypes.c_ubyte * size).from_address(base)
        view = np.ctypeslib.as_array(owner)
        flags = self.cl.mem_flags.USE_HOST_PTR | (
            self.cl.mem_flags.READ_WRITE if writable else self.cl.mem_flags.READ_ONLY
        )
        self.diagnostics.clear()
        buffer = self.cl.Buffer(self.context, flags, hostbuf=view)
        confirmed = any(
            "will share the same physical memory with CPU" in message
            and hex(base) in message
            for message in self.diagnostics
        )
        if not confirmed:
            buffer.release()
            raise RuntimeError("Intel driver did not confirm zero-copy host import")
        return buffer, view

    def import_pair(
        self, data: NDArray[np.float32], output: NDArray[np.float32]
    ) -> tuple[Any, Any, Any, Any, int, int]:
        """Merge overlapping page imports while rejecting logical in-place convolution.

        Args:
            data:
                Live contiguous input tensor.

            output:
                Live contiguous output tensor with a distinct logical byte range.

        """
        xp, yp = data.ctypes.data, output.ctypes.data
        assert xp + data.nbytes <= yp or yp + output.nbytes <= xp
        xb, xs = _host_range(data)
        yb, ys = _host_range(output)
        if xb < yb + ys and yb < xb + xs:
            base = min(xb, yb)
            size = max(xb + xs, yb + ys) - base
            buffer, owner = self._import_region(base, size, True)
            return buffer, buffer, owner, owner, (xp - base) // 4, (yp - base) // 4
        x_buffer, x_owner, x_offset = self.import_host(data, False)
        try:
            y_buffer, y_owner, y_offset = self.import_host(output, True)
        except BaseException:
            x_buffer.release()
            raise
        return x_buffer, y_buffer, x_owner, y_owner, x_offset, y_offset

    def execute(
        self, layer: Layer, data: NDArray[np.float32], output: NDArray[np.float32]
    ) -> None:
        """Dispatch one convolution and make shared output visible before CPU resumes.

        Args:
            layer:
                Prepared geometry, weights and reusable kernel.

            data:
                Live CPU input, float32 (1, input_channels, time).

            output:
                Live writable CPU output, float32 (1, output_channels, output_time).

        """
        if self.closed:
            raise RuntimeError("TTS GPU executor is closed")
        x_buffer, y_buffer, x_owner, y_owner, x_offset, y_offset = self.import_pair(
            data, output
        )
        width, out_width = data.shape[-1], output.shape[-1]
        oc = layer.shape[0]
        mapped = None
        mapping = None
        event = None
        try:
            layer.kernel.set_args(
                x_buffer,
                layer.weights,
                layer.scales,
                y_buffer,
                np.int32(width),
                np.int32(out_width),
                np.int32(x_offset),
                np.int32(y_offset),
            )
            event = self.cl.enqueue_nd_range_kernel(
                self.queue,
                layer.kernel,
                (
                    ((out_width + layer.tile_time - 1) // layer.tile_time) * 16,
                    (oc + 15) // 16,
                ),
                (16, 1),
            )
            event.wait()
            self.diagnostics.clear()
            mapped, mapping = self.cl.enqueue_map_buffer(
                self.queue,
                y_buffer,
                self.cl.map_flags.READ,
                y_offset * 4,
                output.shape,
                np.float32,
                is_blocking=True,
            )
            mapping.wait()
            assert mapped.ctypes.data == output.ctypes.data, (
                "Output mapping is not the CPU tensor address"
            )
            if not any(
                "clEnqueueMapBuffer" in message
                and "will not require any data copy" in message
                for message in self.diagnostics
            ):
                raise RuntimeError(
                    "Intel driver did not confirm copy-free output mapping"
                )
        finally:
            try:
                self.queue.finish()
                if mapped is not None:
                    mapped.base.release(self.queue).wait()
            finally:
                try:
                    # Release cached bindings before CPU arena memory can be resized/reused.
                    layer.kernel.set_arg(0, None)
                    layer.kernel.set_arg(3, None)
                finally:
                    mapped = mapping = event = None
                    x_buffer.release()
                    if y_buffer is not x_buffer:
                        y_buffer.release()
        assert x_owner.size >= data.nbytes and y_owner.size >= output.nbytes
        self.diagnostics.clear()

    def close(self) -> None:
        """Drain GPU work and release caches after the owning compiled graph is released."""
        if not self.closed:
            self.closed = True
            try:
                self.queue.finish()
            finally:
                self.programs.clear()
                self.tile_choices.clear()
                self.queue = None
                self.context = None
                self.device = None
                self.diagnostics.clear()


class CPUConvOffload(ov.Op):
    """CPU graph reference operation with a synchronous shared-memory GPU evaluator."""

    gpu: SharedGPU
    """Shared GPU execution context."""

    layer: Layer
    """Prepared immutable convolution geometry and GPU weights."""

    def __init__(self, data: ov.Output, gpu: SharedGPU, layer: Layer) -> None:
        """Bind one CPU activation edge to a prepared GPU convolution.

        Args:
            data:
                FP32 rank-three batch-one CPU activation output.

            gpu:
                Shared executor, kept alive by cloned operations.

            layer:
                Geometry and fixed weights for this convolution.

        """
        self.gpu, self.layer = gpu, layer
        super().__init__(self, [data])

    def validate_and_infer_types(self) -> None:
        """Use native convolution shape inference without allocating weight data."""
        weights = ops.parameter(self.layer.shape, ov.Type.f32)
        attrs = self.layer.attrs
        reference = ops.convolution(
            self.input_value(0),
            weights,
            attrs["strides"],
            attrs["pads_begin"],
            attrs["pads_end"],
            attrs["dilations"],
        )
        self.set_output_type(0, ov.Type.f32, reference.get_output_partial_shape(0))

    def clone_with_new_inputs(self, inputs: list[ov.Output]) -> CPUConvOffload:
        """Reuse kernels and weights when OpenVINO clones the graph.

        Args:
            inputs:
                One replacement FP32 activation edge.

        """
        assert len(inputs) == 1
        return CPUConvOffload(inputs[0], self.gpu, self.layer)

    def has_evaluate(self) -> bool:
        """Advertise the CPU reference callback."""
        return True

    def visit_attributes(self, visitor: Any) -> bool:
        """Expose immutable layer identity so graph optimizers cannot merge weights.

        Args:
            visitor:
                OpenVINO Python attribute visitor for cloning and equivalence checks.

        """
        visitor.on_attributes(
            {
                "layer_identity": self.layer.name,
                "weight_identity": self.layer.fingerprint,
                "weight_shape": list(self.layer.shape),
                **{
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in self.layer.attrs.items()
                },
            }
        )
        return True

    def evaluate(self, outputs: ov.TensorVector, inputs: ov.TensorVector) -> bool:
        """Dispatch GPU work synchronously while the CPU graph owns live tensor pages.

        Args:
            outputs:
                One writable CPU output tensor.

            inputs:
                One contiguous batch-one CPU input tensor.

        """
        shape = inputs[0].shape
        attrs = self.layer.attrs
        oc, ic, kw = self.layer.shape
        assert list(shape[:2]) == [1, ic]
        width = (
            shape[-1]
            + attrs["pads_begin"][0]
            + attrs["pads_end"][0]
            - attrs["dilations"][0] * (kw - 1)
            - 1
        ) // attrs["strides"][0] + 1
        outputs[0].set_shape([1, oc, width])
        self.gpu.execute(self.layer, inputs[0].data, outputs[0].data)
        return True


def decoder_convolutions(model: ov.Model) -> list[ov.Node]:
    """Validate Kokoro decoder coverage and return its 67 substantial filters.

    Args:
        model:
            Complete English ONNX or native Kokoro IR graph. Decoder names use
            slash-separated ONNX paths or dot-separated PyTorch module paths.

    """
    nodes = [
        node
        for node in model.get_ordered_ops()
        if node.get_type_name() == "Convolution"
        and re.search(r"(?:^|[./])decoder(?:[./]|$)", node.get_friendly_name())
    ]
    if len(nodes) != 69:
        raise ValueError(f"Expected 69 Kokoro decoder convolutions, found {len(nodes)}")
    selected = []
    tiny = []
    for node in nodes:
        shape = node.get_input_partial_shape(1)
        if not shape.is_static or shape.rank.get_length() != 3:
            raise ValueError(
                "Kokoro decoder weights must have static rank-three shapes"
            )
        weights = list(shape.to_shape())
        if weights[0] * weights[1] < 32:
            tiny.append(weights)
        else:
            selected.append(node)
    if len(selected) != 67 or tiny != [[1, 1, 3], [1, 1, 3]]:
        raise ValueError("Unexpected Kokoro decoder convolution layout")
    return selected


def replace_convolutions(model: ov.Model, gpu: SharedGPU) -> int:
    """Offload decoder convolutions while retaining frontend and tiny filters on CPU.

    Args:
        model:
            Mutable FP32-compute decoder graph.

        gpu:
            Shared executor that precompiles all geometry-specific kernels.

    """
    manager = Manager()
    manager.register_pass(ConstantFolding())
    manager.run_passes(model)
    count = 0
    for node in decoder_convolutions(model):
        layer = gpu.prepare(node)
        replacement = CPUConvOffload(node.input_value(0), gpu, layer)
        replacement.set_friendly_name(node.get_friendly_name() + "/gpu_shared")
        node.output(0).replace(replacement.output(0))
        count += 1
    model.validate_nodes_and_infer_types()
    logger.info(
        "gpu_shared.rewrite convolutions=%d programs=%d", count, len(gpu.programs)
    )
    return count
