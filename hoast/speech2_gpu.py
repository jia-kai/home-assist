"""Experimental packed INT8 speech operators, separate from production runtimes.

Only packed signed INT8 weights and FP32 per-output scales reside on the device.
FP32 activation arithmetic uses on-demand register dequantization, not an expanded
weight tensor. These portable OpenCL 1.2 kernels need neither integer dot hardware
nor FP16 support. Explicit copies/maps are intentional: unlike SharedGPU, this
wrapper does not assert driver-confirmed physical sharing or import CPU pages.

CPU reference functions may expand weights for offline validation; they are never
called by the device preparation/execution path. Convolution covers batch-one,
stride-one, odd-tap symmetric SAME padding; transposed convolution, bias and
activation fusion are not implemented. Decoder lengths can be F * upsampling.

prescale is the reference: converted weight bytes are scaled before accumulation.
postscale is an explicit candidate: accumulate raw converted bytes, then apply one
FP32 scale per output. It changes rounding and raw accumulator dynamic range;
extreme activations can overflow before scaling. Device qualification is separate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
Transfer = Literal["copy", "map"]
ScalingVariant = Literal["prescale", "postscale"]
FP32_ATOL = 3e-4
FP32_RTOL = 2e-4
SOURCE = Path(__file__).with_suffix(".cl")


@dataclass(slots=True, frozen=True)
class PackedWeights:
    """Host-side immutable packed weights with output-major source geometry."""

    values: NDArray[np.int8]
    """Signed INT8 (reduction, padded_outputs), padded to 32 output channels."""

    scales: FloatArray
    """FP32 (padded_outputs,) scales; zero-padded channels have scale zero."""

    shape: tuple[int, ...]
    """Source shape (outputs, reduction) or (outputs, input_channels, taps)."""

    @property
    def nbytes(self) -> int:
        """Return packed weight plus scale storage in bytes, including padding."""
        return self.values.nbytes + self.scales.nbytes


def quantize_weights(weights: FloatArray) -> PackedWeights:
    """Symmetrically quantize each output channel to [-127,127], rounding to even.

    Full finite-value scanning is done once at this preparation boundary. All-zero
    channels use scale 1; tiny nonzero scales are clamped to FP32's minimum normal.
    Returned owned arrays are read-only; the source tensor is not retained.

    Args:
        weights:
            Finite FP32 (outputs, reduction) or (outputs, input_channels, odd_taps).

    """
    if weights.dtype != np.float32 or weights.ndim not in (2, 3):
        raise ValueError("Weights must be FP32 with rank 2 or 3")
    if any(d <= 0 for d in weights.shape) or not np.isfinite(weights).all():
        raise ValueError("Weights must have nonempty dimensions and finite values")
    if weights.ndim == 3 and weights.shape[2] % 2 != 1:
        raise ValueError("Convolution requires odd tap count")
    outputs = weights.shape[0]
    flat = weights.reshape(outputs, -1)
    maxima = np.max(np.abs(flat), axis=1)
    scale = np.where(
        maxima == 0,
        np.float32(1),
        np.maximum(maxima / np.float32(127), np.finfo(np.float32).tiny),
    ).astype(np.float32)
    quantized = np.clip(np.rint(flat / scale[:, None]), -127, 127).astype(np.int8)
    padded = ((outputs + 31) // 32) * 32
    values = np.zeros((flat.shape[1], padded), dtype=np.int8)
    values[:, :outputs] = quantized.T
    scales = np.zeros(padded, dtype=np.float32)
    scales[:outputs] = scale
    values.flags.writeable = False
    scales.flags.writeable = False
    return PackedWeights(values, scales, tuple(weights.shape))


def dequantize_reference(packed: PackedWeights) -> FloatArray:
    """Expand weights on the CPU only, for an offline dequantized-INT8 reference.

    Args:
        packed:
            Quantized weights whose output-major geometry is restored.

    """
    outputs = packed.shape[0]
    return (
        packed.values[:, :outputs].T.astype(np.float32) * packed.scales[:outputs, None]
    ).reshape(packed.shape)


def reference_matmul(data: FloatArray, weights: FloatArray) -> FloatArray:
    """Compute FP32 A[M,K] @ W[N,K].T without quantization.

    Args:
        data:
            FP32 activations with rows/tokens M and reduction width K.

        weights:
            FP32 output-major (N,K) original or CPU-dequantized weights.

    """
    if data.dtype != np.float32 or weights.dtype != np.float32:
        raise ValueError("Reference arithmetic requires FP32")
    if data.ndim != 2 or weights.ndim != 2 or data.shape[1] != weights.shape[1]:
        raise ValueError("Matmul shapes must be (M,K) and (N,K)")
    return data @ weights.T


def reference_conv1d(
    data: FloatArray, weights: FloatArray, dilation: int = 1
) -> FloatArray:
    """Compute FP32 batch-one, stride-one SAME convolution without im2col.

    CPU tap/channel summation order differs from the GPU's sequential fused fma;
    validation therefore uses absolute/relative tolerances, not bit equality.

    Args:
        data:
            FP32 channel-major activations (input_channels, time).

        weights:
            FP32 (output_channels, input_channels, odd_taps).

        dilation:
            Positive temporal spacing of taps, in input samples.

    """
    if data.dtype != np.float32 or weights.dtype != np.float32:
        raise ValueError("Reference arithmetic requires FP32")
    if (
        data.ndim != 2
        or weights.ndim != 3
        or data.shape[0] != weights.shape[1]
        or weights.shape[2] % 2 != 1
        or dilation <= 0
    ):
        raise ValueError("Invalid SAME convolution geometry")
    width = data.shape[1]
    padding = dilation * (weights.shape[2] - 1) // 2
    output = np.zeros((weights.shape[0], width), dtype=np.float32)
    for tap in range(weights.shape[2]):
        offset = tap * dilation - padding
        begin, end = max(0, -offset), min(width, width - offset)
        if begin < end:
            output[:, begin:end] += (
                weights[:, :, tap] @ data[:, begin + offset : end + offset]
            )
    return output


def error_metrics(actual: FloatArray, expected: FloatArray) -> dict[str, float]:
    """Return max absolute, RMSE and relative L2 error using FP64 host reductions.

    Args:
        actual:
            Actual FP32 tensor; must match expected shape and be finite.

        expected:
            Finite reference tensor; relative L2 denominator is floored at 1e-30.

    """
    if actual.shape != expected.shape:
        raise ValueError("Error metric shapes differ")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Error metrics require finite tensors")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "relative_l2": float(
            np.linalg.norm(difference) / max(float(np.linalg.norm(expected)), 1e-30)
        ),
    }


def reference_postscale(
    data: FloatArray, packed: PackedWeights, dilation: int = 1
) -> FloatArray:
    """Compute the postscale reassociation on CPU only, not a bit-exact GPU emulator.

    Raw FP32 weights are expanded only in this offline helper. BLAS/tap summation
    differs from sequential GPU fma; compare with tolerance, not exact equality.
    This reference raises on overflow, including raw-dot overflow before scaling.

    Args:
        data:
            FP32 (M,K) matrix or (input_channels,time) SAME-convolution activations.

        packed:
            Signed INT8 reduction-major weights and FP32 output-channel scales.

        dilation:
            Positive convolution tap spacing; unused by matmul.

    """
    outputs = packed.shape[0]
    raw = packed.values[:, :outputs].T.astype(np.float32).reshape(packed.shape)
    with np.errstate(over="raise", invalid="raise"):
        if len(packed.shape) == 2:
            return reference_matmul(data, raw) * packed.scales[None, :outputs]
        return reference_conv1d(data, raw, dilation) * packed.scales[:outputs, None]


def event_ms(event: Any) -> float:
    """Return completed OpenCL event execution duration in milliseconds.

    Args:
        event:
            Event from a profiling-enabled queue, already waited or ordered complete.

    """
    return (event.profile.end - event.profile.start) / 1e6


@dataclass(slots=True)
class ResidentOperation:
    """Device buffers holding only packed INT8 weights and FP32 scales."""

    packed: PackedWeights
    """Host packed metadata; no source float weights are retained."""

    weights: Any
    """Resident read-only INT8 OpenCL buffer."""

    scales: Any
    """Resident read-only FP32 scale OpenCL buffer."""

    dilation: int
    """Temporal tap spacing, ignored by matmul."""

    setup: dict[str, float | int]
    """Allocation/upload wall times, event times and requested device storage bytes."""

    def close(self) -> None:
        """Release resident device allocations after all queued commands complete."""
        self.weights.release()
        self.scales.release()


@dataclass(slots=True)
class Workspace:
    """Reusable activation allocations for one shape and output-transfer policy."""

    input: Any
    """FP32 OpenCL activation buffer."""

    output: Any
    """FP32 OpenCL result buffer, optionally ALLOC_HOST_PTR for mapping."""

    host_output: FloatArray
    """Owned FP32 CPU result overwritten synchronously by each execution."""

    input_shape: tuple[int, int]
    """Activation shape (rows,reduction) or (input_channels,time)."""

    transfer: Transfer
    """Explicit output copy, or map plus CPU copy and unmap (not guaranteed zero-copy)."""

    setup: dict[str, float | int]
    """Requested allocation counts/bytes and measured allocation wall milliseconds."""

    def close(self) -> None:
        """Release activation device allocations after all queued commands complete."""
        self.input.release()
        self.output.release()


class ExperimentalGPU:
    """Serialized profiling executor, intentionally not integrated into speech models."""

    cl: Any
    """Lazy-loaded optional PyOpenCL API."""

    context: Any
    """Context containing exactly the selected GPU."""

    device: Any
    """Selected OpenCL device and its resource metadata."""

    queue: Any
    """In-order profiling-enabled OpenCL queue."""

    program: Any
    """Portable OpenCL 1.2 program shared by every runtime tensor shape."""

    matmul_kernel: Any
    """Reusable tiled matrix kernel; serialized calls mutate its argument bindings."""

    conv_kernel: Any
    """Reusable tiled SAME convolution kernel."""

    setup_ms: float
    """Context, queue and program build wall milliseconds, excluded from steady state."""

    variant: ScalingVariant
    """Reference per-load scaling, or experimental once-per-output postscale."""

    def __init__(
        self, device_match: str = "630", variant: ScalingVariant = "prescale"
    ) -> None:
        """Select a unique matching GPU and compile kernels without relaxed arithmetic.

        Args:
            device_match:
                Case-insensitive device-name substring; defaults to Intel UHD 630.

            variant:
                prescale preserves reference arithmetic; postscale reassociates the
                FP32 dot and final scale without changing packed weight storage.

        """
        if variant not in ("prescale", "postscale"):
            raise ValueError("Scaling variant must be prescale or postscale")
        self.variant = variant
        # Optional GPU dependency stays lazy so offline quantization needs no driver.
        import pyopencl as cl

        started = time.perf_counter()
        self.cl = cl
        devices = [
            device
            for platform in cl.get_platforms()
            for device in platform.get_devices(device_type=cl.device_type.GPU)
            if device_match.lower() in device.name.lower()
        ]
        if len(devices) != 1:
            raise RuntimeError(
                f"Expected one GPU matching {device_match!r}; found {len(devices)}"
            )
        self.device = devices[0]
        if self.device.max_work_group_size < 64:
            raise RuntimeError("Kernel requires a 64-workitem workgroup")
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(
            self.context, properties=cl.command_queue_properties.PROFILING_ENABLE
        )
        options = ["-cl-std=CL1.2", f"-DSPEECH_POSTSCALE={int(variant == 'postscale')}"]
        self.program = cl.Program(self.context, SOURCE.read_text()).build(
            options=options
        )
        suffix = "_postscale" if variant == "postscale" else ""
        self.matmul_kernel = cl.Kernel(self.program, f"speech_matmul{suffix}")
        self.conv_kernel = cl.Kernel(self.program, f"speech_conv1d{suffix}")
        self.setup_ms = (time.perf_counter() - started) * 1000

    def prepare(self, packed: PackedWeights, dilation: int = 1) -> ResidentOperation:
        """Allocate/upload packed weights and scales once; never expand device weights.

        Args:
            packed:
                Output of quantize_weights, with a reduction-major INT8 layout.

            dilation:
                Positive convolution tap spacing, ignored for matrix weights.

        """
        if dilation <= 0:
            raise ValueError("Dilation must be positive")
        started = time.perf_counter()
        weights = self.cl.Buffer(
            self.context, self.cl.mem_flags.READ_ONLY, packed.values.nbytes
        )
        try:
            scales = self.cl.Buffer(
                self.context, self.cl.mem_flags.READ_ONLY, packed.scales.nbytes
            )
        except BaseException:
            weights.release()
            raise
        allocated = time.perf_counter()
        try:
            w_event = self.cl.enqueue_copy(self.queue, weights, packed.values)
            s_event = self.cl.enqueue_copy(self.queue, scales, packed.scales)
            s_event.wait()
            setup = {
                "allocation_ms": (allocated - started) * 1000,
                "wall_ms": (time.perf_counter() - started) * 1000,
                "weight_upload_ms": event_ms(w_event),
                "scale_upload_ms": event_ms(s_event),
                "device_allocations": 2,
                "device_bytes": packed.nbytes,
                "expanded_device_weight_bytes": 0,
            }
            return ResidentOperation(packed, weights, scales, dilation, setup)
        except BaseException:
            self.queue.finish()
            weights.release()
            scales.release()
            raise

    def workspace(
        self,
        operation: ResidentOperation,
        input_shape: tuple[int, int],
        transfer: Transfer = "copy",
    ) -> Workspace:
        """Allocate reusable FP32 activations and CPU output; report requested bytes.

        Args:
            operation:
                Resident matrix or SAME convolution geometry.

            input_shape:
                Positive dimensions (M,K) or (CI,T); no batch axis.

            transfer:
                Copy output, or map ALLOC_HOST_PTR output then copy to owned CPU memory.

        """
        shape = operation.packed.shape
        if len(input_shape) != 2 or any(d <= 0 for d in input_shape):
            raise ValueError("Activation shape must have two positive dimensions")
        if transfer not in ("copy", "map"):
            raise ValueError("Output transfer must be copy or map")
        if len(shape) == 2:
            if input_shape[1] != shape[1]:
                raise ValueError("Matmul reduction dimensions differ")
            output_shape = (input_shape[0], shape[0])
        else:
            if input_shape[0] != shape[1]:
                raise ValueError("Convolution input channel dimensions differ")
            output_shape = (shape[0], input_shape[1])
        if max(*input_shape, *shape) > np.iinfo(np.int32).max:
            raise ValueError("Kernel dimensions exceed int32")
        if len(shape) == 3 and (
            operation.dilation * (shape[2] - 1) + input_shape[1]
            > np.iinfo(np.int32).max - 32
        ):
            raise ValueError("Dilated temporal indexing exceeds int32")
        # Kernel address arithmetic uses signed int32 indices.
        if max(
            np.prod(input_shape), np.prod(output_shape), operation.packed.values.size
        ) > (np.iinfo(np.int32).max):
            raise ValueError("Kernel tensor indexing exceeds int32")
        started = time.perf_counter()
        host_output = np.empty(output_shape, dtype=np.float32)
        x = self.cl.Buffer(
            self.context, self.cl.mem_flags.READ_ONLY, int(np.prod(input_shape)) * 4
        )
        flags = self.cl.mem_flags.READ_WRITE
        if transfer == "map":
            flags |= self.cl.mem_flags.ALLOC_HOST_PTR
        try:
            y = self.cl.Buffer(self.context, flags, host_output.nbytes)
        except BaseException:
            x.release()
            raise
        setup = {
            "allocation_ms": (time.perf_counter() - started) * 1000,
            "device_allocations": 2,
            "device_bytes": int(np.prod(input_shape)) * 4 + host_output.nbytes,
            "host_allocations": 1,
            "host_bytes": host_output.nbytes,
        }
        return Workspace(x, y, host_output, input_shape, transfer, setup)

    def launch(self, operation: ResidentOperation, workspace: Workspace) -> Any:
        """Enqueue one resident kernel without uploads, downloads or allocations.

        Args:
            operation:
                Resident packed weights and fixed convolution dilation.

            workspace:
                Compatible device activation buffers containing an uploaded input.

        """
        shape = operation.packed.shape
        outputs, padded = shape[0], operation.packed.values.shape[1]
        expected_shape = (
            (workspace.input_shape[0], outputs)
            if len(shape) == 2
            else (outputs, workspace.input_shape[1])
        )
        input_axis = 1 if len(shape) == 2 else 0
        if (
            workspace.input_shape[input_axis] != shape[1]
            or workspace.host_output.shape != expected_shape
        ):
            raise ValueError("Workspace does not match resident operation")
        arguments = [
            workspace.input,
            operation.weights,
            operation.scales,
            workspace.output,
        ]
        if len(shape) == 2:
            rows = workspace.input_shape[0]
            kernel = self.matmul_kernel
            dimensions = (rows, shape[1], outputs, padded)
        else:
            rows = workspace.input_shape[1]
            kernel = self.conv_kernel
            dimensions = (
                rows,
                shape[1],
                outputs,
                shape[2],
                operation.dilation,
                operation.dilation * (shape[2] - 1) // 2,
                padded,
            )
        kernel.set_args(*arguments, *(np.int32(value) for value in dimensions))
        return self.cl.enqueue_nd_range_kernel(
            self.queue,
            kernel,
            (((outputs + 31) // 32) * 8, ((rows + 7) // 8) * 8),
            (8, 8),
        )

    def execute(
        self, operation: ResidentOperation, workspace: Workspace, data: FloatArray
    ) -> tuple[FloatArray, dict[str, float]]:
        """Synchronously upload, compute and retrieve output with event/wall timings.

        Workspace allocation, weight upload, compilation, quantization and references
        are excluded from wall_ms and recorded separately. The returned array aliases
        workspace.host_output and is overwritten by the next call. Map mode includes
        a measured CPU copy; it makes no physical zero-copy claim.

        Args:
            operation:
                Prepared packed weights; input finite scanning is intentionally omitted.

            workspace:
                Allocated shape-compatible input/output buffers and transfer mode.

            data:
                Contiguous FP32 tensor exactly matching workspace.input_shape.

        """
        if (
            data.dtype != np.float32
            or not data.flags.c_contiguous
            or data.shape != workspace.input_shape
        ):
            raise ValueError("Input must match contiguous FP32 workspace shape")
        started = time.perf_counter()
        upload = self.cl.enqueue_copy(
            self.queue, workspace.input, data, is_blocking=False
        )
        kernel = self.launch(operation, workspace)
        timings: dict[str, float] = {}
        if workspace.transfer == "copy":
            download = self.cl.enqueue_copy(
                self.queue, workspace.host_output, workspace.output, is_blocking=False
            )
            download.wait()
            timings["download_ms"] = event_ms(download)
        else:
            mapped, mapping = self.cl.enqueue_map_buffer(
                self.queue,
                workspace.output,
                self.cl.map_flags.READ,
                0,
                workspace.host_output.shape,
                np.float32,
                is_blocking=False,
            )
            mapping.wait()
            try:
                copy_started = time.perf_counter()
                np.copyto(workspace.host_output, mapped)
                timings["mapped_host_copy_ms"] = (
                    time.perf_counter() - copy_started
                ) * 1000
            finally:
                unmap = mapped.base.release(self.queue)
                unmap.wait()
            timings["map_ms"] = event_ms(mapping)
            timings["unmap_ms"] = event_ms(unmap)
        timings.update(
            wall_ms=(time.perf_counter() - started) * 1000,
            upload_ms=event_ms(upload),
            kernel_ms=event_ms(kernel),
        )
        return workspace.host_output, timings

    def close(self) -> None:
        """Drain queue and clear kernel bindings before caller releases resident buffers."""
        self.queue.finish()
        for kernel in (self.matmul_kernel, self.conv_kernel):
            for index in range(4):
                kernel.set_arg(index, None)
