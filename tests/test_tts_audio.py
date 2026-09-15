"""Deterministic playback worker checks without an audio device."""

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from hoast.tts_audio import AudioPlayback


def test_start_backpressure_and_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start before filling capacity, bound in-flight PCM, and drain in order.

    Args:
        monkeypatch:
            Fixture replacing the device with event-controlled writes.

    """
    started = threading.Event()
    release = threading.Event()
    submitted = threading.Event()
    received: list[np.ndarray] = []
    caller = threading.get_ident()
    stream = MagicMock()

    def write(samples: np.ndarray) -> bool:
        """Hold playback to exercise producer backpressure.

        Args:
            samples:
                Mono float32 block supplied by the worker.

        """
        assert threading.get_ident() != caller
        started.set()
        assert release.wait(5)
        received.append(samples.copy())
        return False

    stream.write.side_effect = write
    factory = MagicMock()
    factory.return_value.__enter__.return_value = stream
    monkeypatch.setattr("hoast.tts_audio.sd.OutputStream", factory)
    playback = AudioPlayback(1000, 0.04)
    audio = np.arange(100, dtype=np.float32)
    playback.submit(audio[:20])
    assert started.wait(5)  # Only half the queue capacity was submitted.
    audio[:20] = -1  # Submission owns its copy.

    def produce() -> None:
        """Submit an utterance larger than capacity on a blocked producer."""
        playback.submit(audio[20:])
        submitted.set()

    producer = threading.Thread(target=produce)
    producer.start()
    try:
        with playback._condition:
            assert playback._pending <= 40
        assert not submitted.wait(0.05)
    finally:
        release.set()
        producer.join(5)
        playback.close()
    assert submitted.is_set()
    np.testing.assert_array_equal(np.concatenate(received), np.arange(100))
    factory.assert_called_once_with(
        samplerate=1000, channels=1, dtype="float32", latency="low"
    )
    factory.return_value.__exit__.assert_called_once()
    assert not playback._worker.is_alive()
    playback.close()
    with pytest.raises(RuntimeError, match="closed"):
        playback.submit(audio)


def test_worker_failure_wakes_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Propagate a device failure through blocked submission and draining.

    Args:
        monkeypatch:
            Fixture providing a device that fails to open.

    """
    factory = MagicMock(side_effect=RuntimeError("Device unavailable"))
    monkeypatch.setattr("hoast.tts_audio.sd.OutputStream", factory)
    playback = AudioPlayback(1000, 0.02)
    with pytest.raises(RuntimeError, match="Audio playback failed") as failure:
        playback.submit(np.zeros(100, dtype=np.float32))
    assert str(failure.value.__cause__) == "Device unavailable"
    with pytest.raises(RuntimeError, match="Audio playback failed"):
        playback.close()
    assert not playback._worker.is_alive()
