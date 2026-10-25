"""Deterministic 16 kHz Kaldi-compatible FBANK, LFR and signed CMVN.

The numerical recipe follows FunASR's Apache-2.0 WavFrontend and Kaldi
compliance frontend. See LICENSES.md. Dither is zero by default, whereas the
reference WavFrontend defaults to one (in int16-scale sample units). Deterministic
parity comparisons must explicitly disable reference dither too. Inputs shorter
than a complete 400-sample frame are rejected instead of changing window size.
"""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.fft import rfft

FloatArray = NDArray[np.float32]
SAMPLE_RATE = 16000
FRAME_LENGTH = 400
FRAME_SHIFT = 160
FFT_LENGTH = 512
MEL_BINS = 80
LFR_WIDTH = 7
LFR_STRIDE = 6
FEATURE_DIM = MEL_BINS * LFR_WIDTH


def mel_filterbank() -> FloatArray:
    """Return float32 weights shaped (80 mel bins, 257 FFT bins).

    Triangles are equally spaced in Kaldi's 1127*log(1+Hz/700) mel scale from
    20 Hz to 8000 Hz; the Nyquist column is explicitly zero, as in Kaldi.
    """
    low_scalar = 1127.0 * math.log(1.0 + 20.0 / 700.0)
    high_scalar = 1127.0 * math.log(1.0 + 8000.0 / 700.0)
    low = np.float32(low_scalar)
    delta = np.float32((high_scalar - low_scalar) / (MEL_BINS + 1))
    index = np.arange(MEL_BINS, dtype=np.float32)[:, None]
    left = low + index * delta
    center = low + (index + np.float32(1)) * delta
    right = low + (index + np.float32(2)) * delta
    frequencies = np.arange(FFT_LENGTH // 2, dtype=np.float32) * np.float32(
        SAMPLE_RATE / FFT_LENGTH
    )
    mel = np.float32(1127) * np.log(np.float32(1) + frequencies / np.float32(700))
    triangles = np.maximum(
        np.float32(0),
        np.minimum((mel - left) / (center - left), (right - mel) / (right - center)),
    )
    return np.pad(triangles, ((0, 0), (0, 1))).astype(np.float32)


@dataclass(slots=True, frozen=True)
class CMVN:
    """Signed additive shifts and multiplicative scales for 560-wide LFR.

    Args:
        shift:
            Finite float32 signed AddShift vector shaped (560,). It is added
            directly, not negated or interpreted as a positive mean.

        scale:
            Positive finite float32 Rescale vector shaped (560,). Caller-owned
            vectors must remain unchanged while this normalization is in use.

    """

    shift: FloatArray
    """Signed AddShift vector shaped (560,), not a positive mean to subtract."""

    scale: FloatArray
    """Positive Rescale vector shaped (560,)."""

    def __post_init__(self) -> None:
        """Reject incorrect widths, non-finite values and nonpositive scales."""
        for vector in (self.shift, self.scale):
            if vector.shape != (FEATURE_DIM,) or vector.dtype != np.float32:
                raise ValueError("CMVN vectors must be float32 with shape (560,)")
            if not np.isfinite(vector).all():
                raise ValueError("CMVN vectors must contain only finite values")
        if np.any(self.scale <= 0):
            raise ValueError("CMVN scales must be positive")

    @classmethod
    def load(cls, path: Path) -> CMVN:
        """Read the explicit AddShift and Rescale vectors from Kaldi am.mvn.

        Args:
            path:
                UTF-8 text artifact containing exactly one section of each kind.

        """
        text = path.read_text(encoding="utf-8")
        vectors: list[FloatArray] = []
        for section in ("AddShift", "Rescale"):
            matches = re.findall(
                rf"<{section}>\s+560\s+560\s+<LearnRateCoef>\s+\S+\s+\[([^\]]+)\]",
                text,
            )
            if len(matches) != 1:
                raise ValueError(f"CMVN requires exactly one 560-wide {section}")
            values = [float(value) for value in matches[0].split()]
            vectors.append(np.asarray(values, dtype=np.float32))
        return cls(vectors[0], vectors[1])

    def apply(self, features: FloatArray) -> FloatArray:
        """Return float32 normalized features without mutating caller data.

        Args:
            features:
                Float32 LFR matrix shaped (frames, 560).

        """
        if (
            features.dtype != np.float32
            or features.ndim != 2
            or features.shape[1] != FEATURE_DIM
        ):
            raise ValueError("CMVN input must be float32 with shape (frames, 560)")
        return np.asarray((features + self.shift) * self.scale, dtype=np.float32)


def stack_lfr(features: FloatArray) -> FloatArray:
    """Stack seven frames every six frames, clipping both boundary indices.

    Args:
        features:
            Nonempty float32 FBANK matrix shaped (frames, 80). Output has shape
            (ceil(frames/6), 560), with slot k of row j selecting clip(6*j+k-3).

    """
    if features.dtype != np.float32 or features.ndim != 2:
        raise ValueError("LFR input must be a float32 matrix")
    frames, width = features.shape
    if frames == 0 or width != MEL_BINS:
        raise ValueError("LFR input must have shape (nonempty frames, 80)")
    count = (frames + LFR_STRIDE - 1) // LFR_STRIDE
    indices = (
        LFR_STRIDE * np.arange(count, dtype=np.int64)[:, None]
        + np.arange(LFR_WIDTH, dtype=np.int64)[None, :]
        - 3
    )
    indices = np.clip(indices, 0, frames - 1)
    return features[indices].reshape(count, FEATURE_DIM)


@dataclass(slots=True)
class Frontend:
    """Reusable frontend with precomputed mel weights and a Hamming window.

    Args:
        cmvn:
            Signed normalization for stacked 560-wide float32 features.

        dither:
            Gaussian noise standard deviation in int16-scale sample units.
            Zero is deterministic; reference WavFrontend defaults to one. The
            prepared STT runtime contract always selects zero.

    """

    cmvn: CMVN
    """Signed normalization applied after LFR stacking."""

    dither: float = 0.0
    """Gaussian noise standard deviation in int16-scale sample units."""

    _mel: FloatArray = field(default_factory=mel_filterbank, init=False, repr=False)
    """Cached filter weights shaped (80, 257)."""

    _window: FloatArray = field(
        default_factory=lambda: np.hamming(FRAME_LENGTH).astype(np.float32),
        init=False,
        repr=False,
    )
    """Nonperiodic Hamming coefficients shaped (400,)."""

    def __post_init__(self) -> None:
        """Reject negative or non-finite noise levels."""
        if not math.isfinite(self.dither) or self.dither < 0:
            raise ValueError("dither must be finite and nonnegative")

    def fbank(
        self, samples: FloatArray, rng: np.random.Generator | None = None
    ) -> FloatArray:
        """Extract log-power features from complete 25 ms frames only.

        Args:
            samples:
                Finite float32 mono waveform shaped (samples,) at 16 kHz,
                conventionally in [-1,1]; multiplied by 32768 before processing.
                Fewer than 400 samples raises ValueError, including empty input.

            rng:
                Noise generator when dither is nonzero; None creates an unseeded
                generator. Ignored for zero dither. Noise precedes DC removal.

        """
        if samples.dtype != np.float32 or samples.ndim != 1:
            raise ValueError("frontend samples must be mono float32")
        if samples.size < FRAME_LENGTH:
            raise ValueError(
                "STT audio must contain at least 400 samples (25 ms at 16 kHz)"
            )
        if not np.isfinite(samples).all():
            raise ValueError("frontend samples must contain only finite values")
        scaled = samples * np.float32(32768)
        frames = np.lib.stride_tricks.sliding_window_view(scaled, FRAME_LENGTH)[
            ::FRAME_SHIFT
        ].copy()
        if self.dither:
            generator = np.random.default_rng() if rng is None else rng
            frames += generator.standard_normal(
                frames.shape, dtype=np.float32
            ) * np.float32(self.dither)
        frames -= frames.mean(axis=1, keepdims=True, dtype=np.float32)
        previous = np.concatenate((frames[:, :1], frames[:, :-1]), axis=1)
        frames -= np.float32(0.97) * previous
        frames *= self._window
        spectrum = np.asarray(rfft(frames, n=FFT_LENGTH, axis=1, workers=1))
        power = np.square(np.abs(spectrum))
        energies = power @ self._mel.T
        return np.log(np.maximum(energies, np.finfo(np.float32).eps)).astype(np.float32)

    def extract(self, samples: FloatArray) -> FloatArray:
        """Return normalized float32 speech features shaped (LFR frames, 560).

        Args:
            samples:
                Mono normalized float32 waveform at 16 kHz; minimum 400 samples.

        """
        return self.cmvn.apply(stack_lfr(self.fbank(samples)))
