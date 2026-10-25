"""Streaming Silero v6 speech endpoint using a prepared, local ONNX model."""

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2
SILENCE_SAMPLES = 8000
DEFAULT_VAD_MODEL = Path("models/vad/silero_vad_v6.onnx")


class FrameVAD(Protocol):
    """Stateful probability source for consecutive mono 16 kHz int16 frames."""

    def reset(self) -> None:
        """Discard recurrent and context state before a new capture."""
        ...

    def __call__(self, pcm: bytes) -> float:
        """Return the current frame's speech probability in [0, 1].

        Args:
            pcm:
                Exactly 512 signed little-endian int16 samples at 16 kHz.

        """
        ...


@lru_cache(maxsize=1)
def _session(model_path: Path = DEFAULT_VAD_MODEL) -> ort.InferenceSession:
    """Cache a single-thread CPU session without state sharing or network access.

    Args:
        model_path:
            Prepared Silero v6 ONNX file; relative paths use the working directory.

    """
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing Silero VAD model: {model_path}; run python -m tools.prepare_vad"
        )
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 4
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"], sess_options=options
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        {item.name for item in inputs} != {"input", "h", "c"}
        or [item.name for item in outputs] != ["speech_probs", "hn", "cn"]
        or any(item.type != "tensor(float)" for item in [*inputs, *outputs])
    ):
        raise RuntimeError("Unsupported Silero v6 streaming model interface")
    shapes = {item.name: item.shape for item in [*inputs, *outputs]}
    if (
        len(shapes["input"]) != 2
        or shapes["input"][1] != 576
        or len(shapes["speech_probs"]) != 1
        or any(shapes[name] != [1, 1, 128] for name in ("h", "c", "hn", "cn"))
    ):
        raise RuntimeError("Unsupported Silero v6 streaming model shapes")
    return session


@dataclass(slots=True)
class SileroVAD:
    """Carry Silero v6 context and recurrent states across fixed 32 ms frames."""

    session: ort.InferenceSession = field(default_factory=_session)
    """Thread-safe ONNX session; state arrays belong to this stream, not the session."""

    input: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((1, 576), dtype=np.float32)
    )
    """Float32 normalized waveform shaped (batch=1, context=64 plus frame=512)."""

    h: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((1, 1, 128), dtype=np.float32)
    )
    """Float32 hidden state shaped (layers=1, batch=1, features=128)."""

    c: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((1, 1, 128), dtype=np.float32)
    )
    """Float32 cell state shaped (layers=1, batch=1, features=128)."""

    def reset(self) -> None:
        """Zero all context and recurrent state without loading another model."""
        self.input.fill(0)
        self.h.fill(0)
        self.c.fill(0)

    def __call__(self, pcm: bytes) -> float:
        """Infer one fixed-size frame, retaining context and recurrent output states.

        Args:
            pcm:
                Exactly 512 mono signed little-endian int16 samples at 16 kHz.

        """
        if len(pcm) != FRAME_BYTES:
            raise ValueError("Silero requires exactly 512 int16 samples per frame")
        self.input[:, :64] = self.input[:, -64:]
        self.input[:, 64:] = (
            np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        )
        probability, h, c = self.session.run(
            None, {"input": self.input, "h": self.h, "c": self.c}
        )
        self.h = np.asarray(h, dtype=np.float32)
        self.c = np.asarray(c, dtype=np.float32)
        probability = np.asarray(probability)
        if self.h.shape != (1, 1, 128) or self.c.shape != (1, 1, 128):
            raise RuntimeError("Silero returned invalid recurrent state shapes")
        if probability.size != 1:
            raise RuntimeError("Silero returned multiple frame probabilities")
        return float(probability.item())


@dataclass(slots=True)
class SpeechEndpoint:
    """Detect 500 ms trailing silence after speech, independently of packet boundaries.

    Speech starts at probability >= 0.5; after onset, probability >= 0.35 resets
    trailing silence. Sixteen silent 32 ms frames establish the 500 ms threshold
    (512 ms observed); the returned sample boundary retains exactly 500 ms after
    the last non-silent frame. Partial final frames are never padded into silence.
    """

    vad: FrameVAD = field(default_factory=SileroVAD)
    """Per-stream frame probability source with independently resettable state."""

    pending: bytearray = field(default_factory=bytearray, init=False)
    """Unclassified int16 PCM, normally fewer than 1024 bytes between calls."""

    samples: int = field(default=0, init=False)
    """Number of classified input samples since the latest reset."""

    speech: bool = field(default=False, init=False)
    """Whether any frame in this capture has crossed the speech-onset threshold."""

    silence: int = field(default=0, init=False)
    """Consecutive classified silent samples after speech onset."""

    end_sample: int | None = field(default=None, init=False)
    """Exclusive retained-audio sample boundary after detection, or None."""

    def reset(self) -> None:
        """Discard frame fragments and speech/silence/recurrent state for a new wake."""
        self.pending.clear()
        self.samples = self.silence = 0
        self.speech = False
        self.end_sample = None
        self.vad.reset()

    def feed(self, pcm: bytes) -> int | None:
        """Consume complete frames and return an absolute endpoint sample index if found.

        Audio after the first endpoint is not classified, even if the same packet
        contains renewed speech. Initial silence cannot end capture by itself.

        Args:
            pcm:
                Consecutive mono 16 kHz little-endian int16 PCM; byte length must
                be even. Callers bound each batch to four frames for loop fairness.

        """
        if len(pcm) % 2:
            raise ValueError("PCM packet contains a partial int16 sample")
        if self.end_sample is not None:
            return self.end_sample
        self.pending.extend(pcm)
        while len(self.pending) >= FRAME_BYTES:
            probability = self.vad(bytes(self.pending[:FRAME_BYTES]))
            del self.pending[:FRAME_BYTES]
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("VAD returned an invalid probability")
            self.samples += FRAME_SAMPLES
            if probability >= 0.5:
                self.speech = True
            if not self.speech:
                continue
            self.silence = self.silence + FRAME_SAMPLES if probability < 0.35 else 0
            if self.silence >= SILENCE_SAMPLES:
                self.end_sample = self.samples - self.silence + SILENCE_SAMPLES
                self.pending.clear()
                return self.end_sample
        return None
