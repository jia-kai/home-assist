"""Lightweight explicit backend policies and JSON-safe profiling records."""

from typing import Literal, TypedDict

Backend = Literal["cpu", "GPU", "ort"]


class NodeProfile(TypedDict):
    """One native execution counter snapshot, with durations expressed in seconds."""

    node_name: str
    """Compiled execution-node name; fusion may combine several source operators."""

    node_type: str
    """Compiled operation kind, such as FullyConnected, Convolution or Reorder."""

    execution_type: str
    """Plugin implementation/kernel label, which may describe precision and layout."""

    status: str
    """Plugin counter status; optimized-out/nonexecuted nodes can have zero times."""

    real_seconds: float
    """Plugin-reported elapsed duration in seconds, not whole-request wall latency."""

    cpu_seconds: float
    """Reported CPU seconds; ORT does not report this and uses zero, not an estimate."""


def validate_settings(backend: Backend, threads: int, profiling: bool) -> None:
    """Validate the explicit runtime policy without importing native backends.

    Args:
        backend:
            cpu/GPU selects OpenVINO; ort selects ONNX Runtime CPU only.

        threads:
            Exactly one or two intra-operation workers and allowed CPU cores.

        profiling:
            Boolean instrumentation switch; disabled for headline measurements.

    """
    if backend not in ("cpu", "GPU", "ort"):
        raise ValueError("STT2 backend must be cpu, GPU or ort")
    if type(threads) is not int or threads not in (1, 2):
        raise ValueError("STT2 threads must be one or two")
    if type(profiling) is not bool:
        raise TypeError("STT2 profiling must be a boolean")


def compile_properties(
    backend: Backend, threads: int, profiling: bool = False
) -> dict[str, str | int | bool]:
    """Build an explicit latency/one-stream/FP32 policy without device fallback.

    CPU has an exact inference-thread setting. GPU driver/host workers are
    bounded by the process CPU-affinity policy rather than unsupported CPU-only
    plugin properties. Profiling counters are disabled by default because their
    collection changes performance and should be measured separately.

    Args:
        backend:
            Exact requested backend: cpu maps to OpenVINO CPU and GPU maps to
            OpenVINO GPU. AUTO, hybrid and implicit fallbacks are not supported.

        threads:
            One or two CPU inference/host threads within the process CPU budget.

        profiling:
            Enable OpenVINO execution counters when true; false avoids requesting
            or collecting per-node counters during transcription.

    """
    validate_settings(backend, threads, profiling)
    if backend == "ort":
        raise ValueError("ORT does not use OpenVINO compile properties")
    properties: dict[str, str | int | bool] = {
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": 1,
        "INFERENCE_PRECISION_HINT": "f32",
        "PERF_COUNT": profiling,
    }
    if backend == "cpu":
        properties["INFERENCE_NUM_THREADS"] = threads
        properties["ENABLE_CPU_PINNING"] = False
    return properties
