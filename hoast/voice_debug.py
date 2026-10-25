"""Capture lifecycle tones and distinct opt-in cue-bracketed command replay."""

import threading
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .logging import get_logger
from .tts2 import TTS

logger = get_logger(__name__)
RATE = 16000
type CaptureCue = Literal["first_pcm", "silence_end"]


def _cue(frequency: float, *, peak: float = 0.4) -> NDArray[np.float32]:
    """Generate a 120 ms Hann-faded mono tone with bounded full-scale amplitude.

    Args:
        frequency:
            Tone frequency in Hz; 880 marks start and 440 marks end.

        peak:
            Maximum normalized amplitude; capture cues use 0.4, debug brackets 0.12.

    """
    if not 0 < frequency < RATE / 2 or not 0 < peak <= 1:
        raise ValueError("Cue frequency and peak must be within audio bounds")
    time = np.arange(1920, dtype=np.float32) / RATE
    return np.asarray(
        peak * np.sin(2 * np.pi * frequency * time) * np.hanning(time.size),
        dtype=np.float32,
    )


def play_capture_cue(kind: CaptureCue, cancelled: threading.Event, tts: TTS) -> None:
    """Play and drain a faded 120 ms capture tone using caller-owned native output.

    Run on an owned worker, not the network loop. Cancellation skips new output;
    any accepted samples drain before returning. Runtime output failures are
    logged with tracebacks and are recoverable so command PCM can reach STT;
    invalid arguments and programming errors propagate rather than being hidden.
    These tones are separate from the optional debug replay's bracket tones.

    Args:
        kind:
            First valid PCM arrival (880 Hz) or streaming silence endpoint (440 Hz).

        cancelled:
            Thread-safe abort/disconnect signal shared with the capture.

        tts:
            Shared playback owner; no synthesis/model inference or owner close.

    """
    if kind not in ("first_pcm", "silence_end"):
        raise ValueError("Unknown capture cue kind")
    if cancelled.is_set():
        logger.info("voice.cue kind=%s status=skipped reason=cancelled", kind)
        return
    logger.info("voice.cue kind=%s status=starting", kind)
    try:
        tts.play_samples(
            _cue(880 if kind == "first_pcm" else 440),
            RATE,
            blocking=True,
            buffer_seconds=0.2,
            cancelled=cancelled,
        )
    except RuntimeError:
        logger.exception("voice.cue kind=%s status=failed recovery=continue_asr", kind)
        return
    logger.info(
        "voice.cue kind=%s status=drained cancelled=%s", kind, cancelled.is_set()
    )


def replay_capture(pcm: bytes, cancelled: threading.Event, tts: TTS) -> None:
    """Play start cue, gap, exact captured audio, gap, end cue, then drain output.

    Run off the native event loop. TTS resamples the complete cue/audio waveform
    once to its 24 kHz queue rate. Cancellation is checked between 100 ms output
    blocks; accepted audio drains before returning. The caller owns the TTS engine.

    Args:
        pcm:
            Nonempty mono signed little-endian int16 PCM at 16 kHz, identical to
            the buffer subsequently passed to STT. Amplitude is not normalized.

        cancelled:
            Thread-safe abort/disconnect signal; cancelled captures are not started.

        tts:
            Existing synthesis/playback owner, sharing one queue with spoken replies.

    """
    if not pcm or len(pcm) % 2:
        raise ValueError("Replay requires nonempty complete int16 PCM samples")
    if cancelled.is_set():
        return
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    gap = np.zeros(1280, dtype=np.float32)
    audio = np.concatenate(
        [_cue(880, peak=0.12), gap, samples, gap, _cue(440, peak=0.12)]
    )
    logger.info("voice.debug_replay status=starting seconds=%.3f", samples.size / RATE)
    tts.play_samples(
        audio, RATE, blocking=True, buffer_seconds=0.2, cancelled=cancelled
    )
    logger.info(
        "voice.debug_replay status=%s", "cancelled" if cancelled.is_set() else "drained"
    )
