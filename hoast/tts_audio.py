"""Bounded PCM buffering with a playback-only worker and low device latency."""

import math
import sys
import threading
from collections import deque

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from .logging import get_logger

logger = get_logger(__name__)


def system_output_device() -> int | None:
    """Prefer Linux desktop mixing over direct ALSA default/dmix hardware access.

    Choose an output-capable ALSA pipewire adapter, then pulse, when advertised.
    Else use PortAudio's system default, including on non-Linux platforms. An
    advertised adapter that fails to open raises; no retry changes the route.
    """
    if sys.platform != "linux":
        return None
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    logger.debug("audio.output devices=%r hostapis=%r", devices, hostapis)
    for name in ("pipewire", "pulse"):
        for index, info in enumerate(devices):
            if (
                info["name"] == name
                and info["max_output_channels"] > 0
                and hostapis[info["hostapi"]]["name"] == "ALSA"
            ):
                logger.info("audio.output adapter=%s device=%d", name, index)
                return index
    logger.debug("audio.output adapter=portaudio_default")
    return None


class AudioPlayback:
    """Single-producer mono float32 queue; close drains and joins the worker."""

    _condition: threading.Condition
    """Protects queue, pending frame count, completion and failure state."""

    _queue: deque[NDArray[np.float32]]
    """Owned mono PCM blocks shaped (frames,)."""

    _pending: int
    """Queued frames plus frames currently being written to the device."""

    _capacity: int
    """Maximum pending frames, excluding the device buffer and caller's utterance."""

    _rate: int
    """PCM sample rate in Hz."""

    _closing: bool
    """Whether submission has ended and the worker should drain."""

    _error: BaseException | None
    """Worker failure propagated to the producer or closing caller."""

    _worker: threading.Thread
    """Playback-only thread; never performs synthesis or LLM inference."""

    def __init__(self, rate: int, buffer_seconds: float) -> None:
        """Start a worker that opens the device upon receiving its first block.

        Args:
            rate:
                Positive PCM sample rate in Hz.

            buffer_seconds:
                Positive finite queue capacity in seconds, rounded down to frames.
                Must accommodate at least one frame; device latency is independent.

        """
        if rate <= 0 or not math.isfinite(buffer_seconds) or buffer_seconds <= 0:
            raise ValueError(
                "Audio rate and buffer duration must be positive and finite"
            )
        self._capacity = int(rate * buffer_seconds)
        if self._capacity < 1:
            raise ValueError("Audio buffer must accommodate at least one frame")
        self._rate = rate
        self._condition = threading.Condition()
        self._queue = deque()
        self._pending = 0
        self._closing = False
        self._error = None
        self._worker = threading.Thread(
            target=self._run, name="tts-playback", daemon=True
        )
        self._worker.start()

    def submit(self, samples: NDArray[np.float32]) -> None:
        """Copy PCM into bounded blocks, waiting for capacity rather than playback end.

        Oversized utterances make progress in blocks of at most 20 ms. A device
        failure wakes blocked submission and raises with the original cause.

        Args:
            samples:
                Nonempty mono float32 PCM shaped (frames,) at the configured rate.
                The caller retains ownership; queued blocks are copied.

        """
        if samples.ndim != 1 or samples.dtype != np.float32 or not samples.size:
            raise ValueError("Expected nonempty mono float32 audio")
        block_size = min(self._capacity, max(1, self._rate // 50))
        for offset in range(0, samples.size, block_size):
            block = samples[offset : offset + block_size]
            with self._condition:
                while (
                    self._error is None
                    and not self._closing
                    and self._pending + block.size > self._capacity
                ):
                    self._condition.wait()
                self._raise_failure()
                if self._closing:
                    raise RuntimeError("Audio playback is closed")
                self._queue.append(block.copy())
                self._pending += block.size
                assert 0 < self._pending <= self._capacity
                self._condition.notify_all()

    def _raise_failure(self) -> None:
        """Raise a worker failure while the caller holds the condition lock."""
        if self._error is not None:
            raise RuntimeError("Audio playback failed") from self._error

    def close(self) -> None:
        """End submission, drain device playback and join; repeatable after success.

        Worker failures propagate after joining. A host device operation that hangs
        also blocks shutdown; no timeout or silent audio truncation is imposed.
        """
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._worker.join()
        with self._condition:
            self._raise_failure()

    def _run(self) -> None:
        """Write queued PCM on a low-latency device and publish any worker failure."""
        try:
            with self._condition:
                self._condition.wait_for(lambda: bool(self._queue) or self._closing)
                if not self._queue:
                    return
            with sd.OutputStream(
                samplerate=self._rate,
                channels=1,
                dtype="float32",
                latency="low",
                device=system_output_device(),
            ) as stream:
                logger.info("tts.audio status=opened device_latency=%s", stream.latency)
                while True:
                    with self._condition:
                        self._condition.wait_for(
                            lambda: bool(self._queue) or self._closing
                        )
                        if not self._queue:
                            break
                        block = self._queue.popleft()
                    if stream.write(block):
                        logger.warning(
                            "tts.audio status=underflow reason=output_buffer_underrun"
                        )
                    with self._condition:
                        self._pending -= block.size
                        self._condition.notify_all()
            logger.info("tts.audio status=drained")
        except BaseException as error:
            logger.exception("tts.audio status=failed")
            with self._condition:
                self._error = error
                self._queue.clear()
                self._pending = 0
                self._condition.notify_all()
