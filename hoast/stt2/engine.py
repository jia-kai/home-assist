"""One-request, one-stream explicit OpenVINO CPU/GPU execution of STT graphs."""

from pathlib import Path

import numpy as np
import openvino as ov
from numpy.typing import NDArray

from .settings import Backend, NodeProfile, compile_properties


class OpenVINOEngine:
    """Own one FP32 compiled graph and a reusable synchronous inference request.

    Callers serialize infer/profile/close operations. GPU is an explicit
    qualification candidate, not an automatic fallback or a qualified default.
    Close releases the native request before releasing its compiled graph.
    """

    compiled_model: ov.CompiledModel | None
    """CPU or GPU graph with explicit FP32 precision; None after resource release."""

    request: ov.InferRequest | None
    """Single request, accessed under the recognizer's serialization lock."""

    backend: Backend
    """Exact requested plugin, cpu or GPU; compilation failures never change it."""

    profiling: bool
    """Whether execution counters were enabled when compiling this graph."""

    profile_path: Path | None = None
    """OpenVINO exposes counter snapshots rather than a durable native trace file."""

    def __init__(
        self,
        model_path: Path,
        threads: int,
        backend: Backend = "cpu",
        profiling: bool = False,
    ) -> None:
        """Compile on the requested plugin without automatic selection or fallback.

        The owning STT constructor applies the process CPU-affinity/native-pool
        policy first. Direct diagnostic callers must apply that policy before
        constructing this engine. No native model/request survives a failed load.

        Args:
            model_path:
                Validated OpenVINO XML path with adjacent model.bin.

            threads:
                Exact CPU inference or GPU-host CPU budget, either one or two.

            backend:
                cpu selects CPU; GPU selects the explicit GPU candidate. The same
                FP32 graph is used on either backend, without CPU fallback.

            profiling:
                Enable per-node plugin execution counters, collected separately
                by get_profile(). False avoids counter-request overhead.

        """
        properties = compile_properties(backend, threads, profiling)
        self.backend = backend
        self.profiling = profiling
        self.compiled_model = None
        self.request = None
        core = ov.Core()
        model = core.read_model(model_path)
        names = {port.get_any_name() for port in model.inputs}
        if names != {"speech", "language", "textnorm"}:
            raise ValueError(f"STT2 graph input contract mismatch: {names}")
        speech = model.input("speech")
        speech_shape = speech.get_partial_shape()
        if (
            speech.get_element_type() != ov.Type.f32
            or speech_shape.rank != 3
            or speech_shape[0] != 1
            or speech_shape[2] != 560
            or not speech_shape[1].is_dynamic
        ):
            raise ValueError(
                "STT2 speech input must be float32 shaped (1, dynamic frames, 560)"
            )
        for name in ("language", "textnorm"):
            port = model.input(name)
            if (
                port.get_element_type() != ov.Type.i64
                or port.get_partial_shape() != ov.PartialShape([1])
            ):
                raise ValueError(f"STT2 {name} input must be int64 shaped (1,)")
        output = model.output("frame_ids")
        output_shape = output.get_partial_shape()
        if (
            output.get_element_type() != ov.Type.i64
            or output_shape.rank != 2
            or output_shape[0] != 1
            or not output_shape[1].is_dynamic
        ):
            raise ValueError(
                "STT2 frame_ids output must be int64 shaped (1, dynamic frames)"
            )
        self.compiled_model = core.compile_model(
            model,
            "CPU" if backend == "cpu" else "GPU",
            properties,
        )
        try:
            self.request = self.compiled_model.create_infer_request()
        except Exception:
            # Ownership cleanup must also cover plugin/request-construction failures.
            self.close()
            raise

    def infer(
        self, features: NDArray[np.float32], language_id: int, textnorm_id: int
    ) -> NDArray[np.int64]:
        """Return copied per-frame argmax IDs including the four metadata frames.

        Args:
            features:
                Contiguous float32 normalized speech matrix shaped (frames, 560).

            language_id:
                Validated prefix embedding row for auto or requested language.

            textnorm_id:
                Prefix embedding row 14 (ITN) or 15 (without ITN).

        """
        request = self.request
        if request is None:
            raise RuntimeError("STT2 engine is closed")
        if (
            features.dtype != np.float32
            or features.ndim != 2
            or features.shape[0] < 1
            or features.shape[1] != 560
        ):
            raise ValueError(
                "STT2 engine features must be float32 shaped (nonempty frames, 560)"
            )
        if language_id not in (0, 3, 4, 7, 11, 12) or textnorm_id not in (14, 15):
            raise ValueError("STT2 engine prefix controls are not supported")
        request.infer(
            {
                "speech": np.ascontiguousarray(features[None, :, :]),
                "language": np.array([language_id], dtype=np.int64),
                "textnorm": np.array([textnorm_id], dtype=np.int64),
            },
            share_inputs=True,
        )
        return np.asarray(
            request.get_tensor("frame_ids").data[0], dtype=np.int64
        ).copy()

    def get_profile(self) -> list[NodeProfile]:
        """Copy JSON-safe counters from the latest inference; return [] when disabled.

        Per-node durations can overlap and should not be summed to obtain wall
        latency. Fused nodes can include several source operations. No native
        ProfilingInfo objects are retained by the returned records.
        """
        request = self.request
        if request is None:
            raise RuntimeError("STT2 engine is closed")
        if not self.profiling:
            return []
        return [
            NodeProfile(
                node_name=info.node_name,
                node_type=info.node_type,
                execution_type=info.exec_type,
                status=str(info.status),
                real_seconds=info.real_time.total_seconds(),
                cpu_seconds=info.cpu_time.total_seconds(),
            )
            for info in request.get_profiling_info()
        ]

    def release_memory(self) -> None:
        """Release idle intermediate storage while retaining weights and the request.

        Call only after synchronous inference and output/profile copying, under
        the recognizer's serialization lock. Allocators may retain freed pages.
        """
        if self.compiled_model is None or self.request is None:
            raise RuntimeError("STT2 engine is closed")
        self.compiled_model.release_memory()

    def close(self) -> None:
        """Release the inference request and compiled graph; repeated calls are safe."""
        self.request = None
        self.compiled_model = None
