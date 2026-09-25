"""Measure correlated speaker echo in raw and processed reSpeaker captures."""

import argparse
import logging
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate

LOGGER = logging.getLogger(__name__)
RATE = 48_000


def load_wav(path: Path) -> NDArray[np.float32]:
    """Read a mono 48 kHz int16 or int32 PCM WAV as normalized float32 samples.

    Args:
        path:
            Local uncompressed mono test/reference WAV.

    Returns:
        One-dimensional normalized mono PCM.

    Raises:
        ValueError: If rate, channels or format differ from this experiment.

    """
    with wave.open(str(path), "rb") as recording:
        if recording.getnchannels() != 1 or recording.getframerate() != RATE:
            raise ValueError("AEC analysis needs mono 48 kHz WAV recordings")
        width = recording.getsampwidth()
        if width not in (2, 4):
            raise ValueError("AEC analysis needs signed 16- or 32-bit PCM")
        pcm = recording.readframes(recording.getnframes())
    dtype = "<i2" if width == 2 else "<i4"
    return (np.frombuffer(pcm, dtype=dtype).astype(np.float32)
            / np.float32(2 ** (8 * width - 1)))


def load_delay_pair(path: Path) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Read simultaneous SHF far-end and pre-gain raw-mic PCM from a debug WAV.

    Args:
        path:
            48 kHz stereo signed 32-bit WAV from diagnostic ``delay`` mode.

    Returns:
        Two mono float32 arrays (far-end reference, raw microphone), both
        indexed by the same XMOS I²S frame clock.

    Raises:
        ValueError: If the recording has an incompatible audio format.

    """
    with wave.open(str(path), "rb") as recording:
        if (recording.getnchannels() != 2 or recording.getframerate() != RATE
                or recording.getsampwidth() != 4):
            raise ValueError("Delay capture must be 48 kHz stereo int32 PCM")
        data = recording.readframes(recording.getnframes())
    channels = np.frombuffer(data, dtype="<i4").reshape(-1, 2).astype(np.float32)
    return channels[:, 0] / np.float32(2147483648), channels[:, 1] / np.float32(2147483648)


def echo_gain(
    reference: NDArray[np.float32], capture: NDArray[np.float32], max_lag_ms: float = 2500
) -> tuple[float, float]:
    """Estimate echo gain and lag within a bounded acoustic-delay window.

    Args:
        reference:
            Nonempty 48 kHz mono playback PCM, shape (frames,).

        capture:
            Nonempty 48 kHz mono microphone PCM, shape (frames,).

        max_lag_ms:
            Largest absolute lag to examine in milliseconds; the local trial
            starts playback shortly after opening the device's debug socket.

    Returns:
        Echo projection gain and signed best-lag milliseconds; this linear
        estimate is a diagnostic, not a complete room impulse response model.

    """
    if reference.ndim != 1 or capture.ndim != 1 or not len(reference) or not len(capture):
        raise ValueError("Expected nonempty mono PCM recordings")
    if not 0 < max_lag_ms <= 30_000:
        raise ValueError("Expected a positive bounded echo-lag window")
    corr = correlate(capture.astype(np.float64), reference.astype(np.float64), mode="full", method="fft")
    origin = len(reference) - 1
    limit = round(max_lag_ms * RATE / 1000)
    start = max(0, origin - limit)
    end = min(len(corr), origin + limit + 1)
    lag = start + int(np.argmax(np.abs(corr[start:end]))) - origin
    ref_start = max(0, -lag)
    mic_start = max(0, lag)
    length = min(len(reference) - ref_start, len(capture) - mic_start)
    if length <= 0:
        raise ValueError("Reference and capture have no overlap")
    source = reference[ref_start : ref_start + length].astype(np.float64)
    observed = capture[mic_start : mic_start + length].astype(np.float64)
    energy = float(np.dot(source, source))
    if not energy:
        raise ValueError("Reference audio contains no energy")
    gain = float(np.dot(source, observed) / energy)
    return gain, lag * 1000 / RATE


def normalized_echo_correlation(
    reference: NDArray[np.float32], capture: NDArray[np.float32], lag_ms: float
) -> float:
    """Return gain-independent reference correlation at a chosen acoustic lag.

    Args:
        reference:
            Mono 48 kHz stimulus with shape (frames,).

        capture:
            Mono 48 kHz device PCM with shape (frames,).

        lag_ms:
            Capture-minus-reference delay in milliseconds.

    Returns:
        Signed normalized correlation; zero denotes no linear echo match.

    """
    lag = round(lag_ms * RATE / 1000)
    ref_start = max(0, -lag)
    mic_start = max(0, lag)
    length = min(len(reference) - ref_start, len(capture) - mic_start)
    if length <= 0:
        raise ValueError("Reference and capture have no overlap")
    source = reference[ref_start : ref_start + length].astype(np.float64)
    observed = capture[mic_start : mic_start + length].astype(np.float64)
    denominator = np.linalg.norm(source) * np.linalg.norm(observed)
    if denominator == 0:
        raise ValueError("Reference or capture contains no energy")
    return float(np.dot(source, observed) / denominator)


def main() -> None:
    """Compare gain-independent raw and ASR correlations from local trials."""
    parser = argparse.ArgumentParser(description="Compare speaker echo in AEC recordings")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--farend", type=Path, default=None)
    parser.add_argument("--delay", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    reference = load_wav(args.reference)
    raw = load_wav(args.raw)
    processed = load_wav(args.processed)
    raw_gain, raw_lag = echo_gain(reference, raw)
    processed_gain, processed_lag = echo_gain(reference, processed)
    LOGGER.info(
        "aec.measurement raw_corr=%.4f processed_corr=%.4f raw_gain=%.5g processed_gain=%.5g raw_lag_ms=%.1f processed_lag_ms=%.1f",
        normalized_echo_correlation(reference, raw, raw_lag),
        normalized_echo_correlation(reference, processed, processed_lag),
        raw_gain, processed_gain, raw_lag, processed_lag,
    )
    if args.farend is not None:
        farend = load_wav(args.farend)
        _, farend_lag = echo_gain(reference, farend)
        LOGGER.info(
            "aec.farend correlation=%.4f lag_ms=%.1f",
            normalized_echo_correlation(reference, farend, farend_lag), farend_lag,
        )
    if args.delay is not None:
        farend, microphone = load_delay_pair(args.delay)
        _, lag_ms = echo_gain(farend, microphone, max_lag_ms=1500)
        LOGGER.info(
            "aec.delay farend_to_raw_mic_ms=%.1f correlation=%.4f",
            lag_ms, normalized_echo_correlation(farend, microphone, lag_ms),
        )


if __name__ == "__main__":
    main()
