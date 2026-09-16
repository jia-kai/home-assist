"""Streaming Silero speech endpoint using faster-whisper's bundled ONNX model."""

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import onnxruntime as ort
from faster_whisper.vad import get_vad_model
from numpy.typing import NDArray

FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2
SILENCE_SAMPLES = 9600


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


def _session() -> ort.InferenceSession:
    """Reuse the bundled single-thread CPU Silero session without sharing recurrent state."""
    session = get_vad_model().session
    if {item.name for item in session.get_inputs()} != {"input", "h", "c"}:
        raise RuntimeError("Unsupported bundled Silero streaming model interface")
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
        assert self.h.shape == self.c.shape == (1, 1, 128)
        return float(np.asarray(probability).item())


@dataclass(slots=True)
class SpeechEndpoint:
    """Detect 600 ms trailing silence after speech, independently of packet boundaries.

    Speech starts at probability >= 0.5; after onset, probability >= 0.35 resets
    trailing silence. Nineteen silent 32 ms frames establish the 600 ms threshold
    (608 ms observed); the returned sample boundary retains exactly 600 ms after
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
