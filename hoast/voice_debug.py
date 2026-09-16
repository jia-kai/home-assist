"""Opt-in captured-command replay bracketed by short system-audio cues."""

import threading

import numpy as np
from numpy.typing import NDArray

from .logging import get_logger
from .tts import TTS

logger = get_logger(__name__)
RATE = 16000


def _cue(frequency: float) -> NDArray[np.float32]:
    """Generate a 120 ms, smoothly faded mono cue at a moderate amplitude.

    Args:
        frequency:
            Tone frequency in Hz; 880 marks start and 440 marks end.

    """
    time = np.arange(1920, dtype=np.float32) / RATE
    return np.asarray(
        0.12 * np.sin(2 * np.pi * frequency * time) * np.hanning(time.size),
        dtype=np.float32,
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
    audio = np.concatenate([_cue(880), gap, samples, gap, _cue(440)])
    logger.info("voice.debug_replay status=starting seconds=%.3f", samples.size / RATE)
    tts.play_samples(
        audio, RATE, blocking=True, buffer_seconds=0.2, cancelled=cancelled
    )
    logger.info(
        "voice.debug_replay status=%s", "cancelled" if cancelled.is_set() else "drained"
    )
