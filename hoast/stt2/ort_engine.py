"""Serialized, CPU-only ONNX Runtime execution of the native STT frame-ID graph."""

import json
import logging
from pathlib import Path
from threading import RLock
from uuid import uuid4

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from .settings import NodeProfile, validate_settings

logger = logging.getLogger(__name__)


class ORTEngine:
    """Own one CPU session; copy results and finalize optional profiles on close.

    The owning recognizer budgets process affinity/native pools before importing
    this module. Direct callers must apply that policy before construction.
    Profiling is disabled by default; ORT flushes its trace only at end_profiling,
    so node snapshots become available after close, not after each inference.
    """

    session: ort.InferenceSession | None
    """One synchronous CPUExecutionProvider session; None after resource release."""

    profiling: bool
    """Whether optional native trace instrumentation was enabled at construction."""

    profile_path: Path | None
    """Durable ORT JSON trace path, populated when a profiled session is closed."""

    _profile: list[NodeProfile]
    """Copied JSON-safe node durations from the finalized trace; empty until close."""

    _lock: RLock
    """Serializes inference, profile access and resource release, including direct calls."""

    def __init__(self, model_path: Path, threads: int, profiling: bool = False) -> None:
        """Load the transformed quantized ONNX graph with an explicit CPU policy.

        Args:
            model_path:
                Validated model.onnx with speech/language/textnorm -> frame_ids.
                Profiles go to .cache/hoast/diagnostics, outside model artifacts.

            threads:
                Exactly one or two intra-operation threads within the CPU budget.

            profiling:
                Enable a durable native trace; false avoids instrumentation.

        """
        validate_settings("ort", threads, profiling)
        self.session = None
        self.profiling = profiling
        self.profile_path = None
        self._profile = []
        self._lock = RLock()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.enable_profiling = profiling
        if profiling:
            diagnostics = Path(".cache/hoast/diagnostics").resolve()
            diagnostics.mkdir(parents=True, exist_ok=True)
            options.profile_file_prefix = str(diagnostics / f"stt2-ort-{uuid4().hex}")
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        try:
            if self.session.get_providers() != ["CPUExecutionProvider"]:
                raise ValueError("STT2 ORT requires CPUExecutionProvider exclusively")
            self.session.disable_fallback()
            inputs = {port.name: port for port in self.session.get_inputs()}
            if set(inputs) != {"speech", "language", "textnorm"}:
                raise ValueError("STT2 ORT graph input contract mismatch")
            speech = inputs["speech"]
            if (
                speech.type != "tensor(float)"
                or len(speech.shape) != 3
                or speech.shape[0] != 1
                or speech.shape[2] != 560
                or isinstance(speech.shape[1], int)
            ):
                raise ValueError(
                    "STT2 speech input must be float32 shaped (1, dynamic frames, 560)"
                )
            for name in ("language", "textnorm"):
                port = inputs[name]
                if port.type != "tensor(int64)" or port.shape != [1]:
                    raise ValueError(f"STT2 {name} input must be int64 shaped (1,)")
            outputs = self.session.get_outputs()
            if len(outputs) != 1:
                raise ValueError("STT2 ORT graph must output only frame_ids")
            output = outputs[0]
            if (
                output.name != "frame_ids"
                or output.type != "tensor(int64)"
                or len(output.shape) != 2
                or output.shape[0] != 1
                or isinstance(output.shape[1], int)
            ):
                raise ValueError(
                    "STT2 frame_ids output must be int64 shaped (1, dynamic frames)"
                )
        except Exception:
            self.close()
            raise

    def infer(
        self, features: NDArray[np.float32], language_id: int, textnorm_id: int
    ) -> NDArray[np.int64]:
        """Return owned int64 IDs shaped (frames+4,), including metadata frames.

        Args:
            features:
                Float32 normalized matrix shaped (nonempty speech frames, 560).

            language_id:
                Supported language embedding row 0, 3, 4, 7, 11 or 12.

            textnorm_id:
                Text normalization embedding row 14 (ITN) or 15 (non-ITN).

        """
        with self._lock:
            session = self.session
            if session is None:
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
            output = session.run(
                ["frame_ids"],
                {
                    "speech": np.ascontiguousarray(features[None, :, :]),
                    "language": np.array([language_id], dtype=np.int64),
                    "textnorm": np.array([textnorm_id], dtype=np.int64),
                },
            )[0]
            if (
                not isinstance(output, np.ndarray)
                or output.dtype != np.int64
                or output.shape != (1, features.shape[0] + 4)
            ):
                raise ValueError("STT2 ORT returned invalid frame_ids")
            return output[0].copy()

    def get_profile(self) -> list[NodeProfile]:
        """Return copied finalized node records; [] before close or when disabled.

        ORT reports node wall duration in microseconds, not per-node CPU time.
        cpu_seconds is zero to denote unavailable data, never a latency estimate.
        The complete native trace remains on disk; node durations across repeated
        requests are separate records and must not be summed as request latency.
        """
        with self._lock:
            return [record.copy() for record in self._profile]

    def release_memory(self) -> None:
        """Raise explicitly because ORT cannot release an idle session's CPU arena."""
        raise NotImplementedError("ONNX Runtime does not expose idle arena release")

    def close(self) -> None:
        """Finalize any enabled durable trace and release the session idempotently.

        Trace I/O/format errors propagate; the session is released even on error.
        Trace paths and copied node records survive close for diagnostic auditing.
        """
        with self._lock:
            session = self.session
            if session is None:
                return
            try:
                if self.profiling:
                    self.profile_path = Path(session.end_profiling())
                    events = json.loads(self.profile_path.read_text(encoding="utf-8"))
                    if not isinstance(events, list):
                        raise ValueError("STT2 ORT profile must contain an event list")
                    for event in events:
                        if event.get("cat") != "Node" or "dur" not in event:
                            continue
                        args = event.get("args", {})
                        self._profile.append(
                            NodeProfile(
                                node_name=str(event["name"]),
                                node_type=str(args.get("op_name", "")),
                                execution_type=str(args.get("provider", "")),
                                status="EXECUTED",
                                real_seconds=float(event["dur"]) / 1_000_000,
                                cpu_seconds=0.0,
                            )
                        )
                    logger.info(
                        "stt2.ort profile=%s nodes=%d",
                        self.profile_path,
                        len(self._profile),
                    )
            finally:
                self.session = None
