"""Deterministic playback worker checks without an audio device."""

import math
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
    stream.latency = 0.06

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
    monkeypatch.setattr("hoast.tts_audio.system_output_device", lambda: None)
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
    combined = np.concatenate(received)
    np.testing.assert_array_equal(combined[:100], np.arange(100))
    np.testing.assert_array_equal(combined[100:], np.zeros(80, dtype=np.float32))
    factory.assert_called_once_with(
        samplerate=1000,
        channels=1,
        dtype="float32",
        latency=0.06,
        blocksize=20,
        device=None,
    )
    stream.stop.assert_called_once_with(ignore_errors=False)
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
    monkeypatch.setattr("hoast.tts_audio.system_output_device", lambda: None)
    playback = AudioPlayback(1000, 0.02)
    with pytest.raises(RuntimeError, match="Audio playback failed") as failure:
        playback.submit(np.zeros(100, dtype=np.float32))
    assert str(failure.value.__cause__) == "Device unavailable"
    with pytest.raises(RuntimeError, match="Audio playback failed"):
        playback.close()
    assert not playback._worker.is_alive()


@pytest.mark.parametrize("reported_latency", [0.06, 0.0731])
def test_partial_tail_and_pcm_scale(
    monkeypatch: pytest.MonkeyPatch, reported_latency: float
) -> None:
    """Preserve PCM scale and a partial final block before flushing device silence.

    Args:
        monkeypatch:
            Replaces the device with an owned, copying stream fixture.

        reported_latency:
            Actual backend output latency in seconds, including a fractional frame.

    """
    stream = MagicMock(latency=reported_latency)
    received: list[np.ndarray] = []
    stream.write.side_effect = lambda block: received.append(block.copy()) or False
    factory = MagicMock()
    factory.return_value.__enter__.return_value = stream
    monkeypatch.setattr("hoast.tts_audio.sd.OutputStream", factory)
    monkeypatch.setattr("hoast.tts_audio.system_output_device", lambda: None)
    audio = np.array([-1, -0.5, -1 / 32768, 0, 1 / 32768, 0.5, 0.999], np.float32)
    playback = AudioPlayback(1000, 0.001)
    playback.submit(audio)
    playback.close()
    combined = np.concatenate(received)
    np.testing.assert_array_equal(combined[: audio.size], audio)
    guard_frames = math.ceil(reported_latency * 1000) + 20
    np.testing.assert_array_equal(combined[audio.size :], np.zeros(guard_frames))
    assert max(block.size for block in received) <= 20
    assert playback._pending == 0
    stream.stop.assert_called_once_with(ignore_errors=False)


@pytest.mark.parametrize("phase", ["write", "flush", "stop"])
def test_device_tail_failures_propagate(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Do not report a drained queue when speech, flush, or explicit stop fails.

    Args:
        monkeypatch:
            Supplies a stream failing at one controlled device operation.

        phase:
            Device operation that raises the original failure.

    """
    error = RuntimeError(f"device {phase} failed")
    stream = MagicMock(latency=0.06)
    if phase == "stop":
        stream.stop.side_effect = error
        stream.write.return_value = False
    else:
        stream.write.side_effect = [error] if phase == "write" else [False, error]
    factory = MagicMock()
    factory.return_value.__enter__.return_value = stream
    monkeypatch.setattr("hoast.tts_audio.sd.OutputStream", factory)
    monkeypatch.setattr("hoast.tts_audio.system_output_device", lambda: None)
    playback = AudioPlayback(1000, 0.02)
    playback.submit(np.ones(7, dtype=np.float32))
    with pytest.raises(RuntimeError, match="Audio playback failed") as failure:
        playback.close()
    assert failure.value.__cause__ is error
    assert not playback._worker.is_alive()
    assert playback._pending == 0


def test_empty_close_does_not_open_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled-before-submission queue closes without device access or flush.

    Args:
        monkeypatch:
            Records any unintended audio stream creation.

    """
    factory = MagicMock()
    monkeypatch.setattr("hoast.tts_audio.sd.OutputStream", factory)
    playback = AudioPlayback(24000, 1.0)
    playback.close()
    factory.assert_not_called()


@pytest.mark.parametrize("latency", [float("nan"), float("inf"), -0.1, (0.01, 0.06)])
def test_invalid_device_latency(
    monkeypatch: pytest.MonkeyPatch, latency: float | tuple[float, float]
) -> None:
    """Reject invalid backend latency before writing PCM or computing a flush size.

    Args:
        monkeypatch:
            Supplies an output stream with inconsistent latency metadata.

        latency:
            Invalid nonfinite, negative or duplex latency for an output-only stream.

    """
    stream = MagicMock(latency=latency)
    factory = MagicMock()
    factory.return_value.__enter__.return_value = stream
    monkeypatch.setattr("hoast.tts_audio.sd.OutputStream", factory)
    monkeypatch.setattr("hoast.tts_audio.system_output_device", lambda: None)
    playback = AudioPlayback(24000, 1.0)
    playback.submit(np.ones(7, dtype=np.float32))
    with pytest.raises(RuntimeError, match="Audio playback failed") as failure:
        playback.close()
    assert (
        str(failure.value.__cause__) == "Output device reported invalid scalar latency"
    )
    stream.write.assert_not_called()
