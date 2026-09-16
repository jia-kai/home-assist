"""Hardware-independent checks of ordered buffered playback and cleanup."""

import threading
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.signal import resample_poly

from hoast.tts import TTS, TTSConfig


def test_buffered_playback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep synthesis on the caller and reuse one stream until explicitly drained.

    Args:
        monkeypatch:
            Fixture replacing model inference and the physical audio device.

    """
    engine = object.__new__(TTS)
    engine._chinese_lock = threading.RLock()
    engine._output = None
    engine._output_buffer_seconds = None
    engine._closed = False
    samples = np.zeros(240, dtype=np.float32)
    caller = threading.get_ident()

    def synthesize(text: str) -> tuple[np.ndarray, int]:
        """Check caller-thread execution and return fixture mono audio.

        Args:
            text:
                Utterance submitted by the playback caller.

        """
        assert text
        assert threading.get_ident() == caller
        return samples, 24000

    monkeypatch.setattr(engine, "synthesize", synthesize)
    stream = Mock()
    factory = Mock(return_value=stream)
    monkeypatch.setattr("hoast.tts.AudioPlayback", factory)

    engine.play("First.", blocking=False)
    engine.play("Second.", blocking=False)
    factory.assert_called_once_with(24000, 1.0)
    assert stream.submit.call_count == 2
    stream.close.assert_not_called()
    with pytest.raises(ValueError, match="Drain playback"):
        engine.play("Third.", blocking=False, buffer_seconds=2.0)
    engine.wait_playback()
    engine.wait_playback()
    stream.close.assert_called_once()


def test_playback_failure_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Propagate output failures and discard the failed stream without retrying.

    Args:
        monkeypatch:
            Fixture replacing inference and a failing audio output device.

    """
    engine = object.__new__(TTS)
    engine._chinese_lock = threading.RLock()
    engine._output = None
    engine._output_buffer_seconds = None
    engine._closed = False
    monkeypatch.setattr(
        engine,
        "synthesize",
        Mock(return_value=(np.zeros(240, dtype=np.float32), 24000)),
    )
    stream = Mock()
    stream.submit.side_effect = RuntimeError("Device disconnected")
    monkeypatch.setattr("hoast.tts.AudioPlayback", Mock(return_value=stream))
    with pytest.raises(RuntimeError, match="Device disconnected"):
        engine.play("Hello.", blocking=False)
    stream.close.assert_called_once()
    assert engine._output is None
    assert engine._output_buffer_seconds is None


def test_raw_and_text_share_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw and synthesized submissions retain order in one queue until an explicit drain.

    Args:
        monkeypatch:
            Replaces the playback worker and model synthesis with in-memory fixtures.

    """
    engine = TTS(playback_only=True)
    stream = Mock()
    factory = Mock(return_value=stream)
    monkeypatch.setattr("hoast.tts.AudioPlayback", factory)
    raw = np.linspace(-0.2, 0.2, 1600, dtype=np.float32)
    speech = np.full(2400, 0.1, dtype=np.float32)
    monkeypatch.setattr(engine, "synthesize", Mock(return_value=(speech, 24000)))
    engine.play_samples(raw, 16000, blocking=False)
    engine.play("Reply.", blocking=False)
    engine.play_samples(speech, 24000, blocking=False)
    factory.assert_called_once_with(24000, 1.0)
    submitted = [call.args[0] for call in stream.submit.call_args_list]
    np.testing.assert_allclose(submitted[0], resample_poly(raw, 3, 2))
    assert submitted[0].size == 2400
    assert submitted[1] is speech and submitted[2] is speech
    with pytest.raises(ValueError, match="Drain playback"):
        engine.play_samples(raw, 16000, buffer_seconds=0.2)
    engine.wait_playback()
    stream.close.assert_called_once()
    engine.close()


def test_playback_only_needs_no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Playback-only initialization avoids all artifact and GPU access and closes safely.

    Args:
        monkeypatch:
            Makes any GPU construction fail if the playback path tries to load it.

    """
    gpu = Mock(side_effect=AssertionError("must not load GPU"))
    monkeypatch.setattr("hoast.tts.SharedGPU", gpu)
    engine = TTS(TTSConfig(), playback_only=True)
    gpu.assert_not_called()
    assert engine.model is None
    with pytest.raises(RuntimeError, match="playback-only"):
        engine.synthesize("Hello")
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.play_samples(np.zeros(24, dtype=np.float32), 24000)


@pytest.mark.parametrize(
    "samples,rate",
    [
        (np.zeros(0, dtype=np.float32), 24000),
        (np.zeros((2, 4), dtype=np.float32), 24000),
        (np.zeros(24, dtype=np.float64), 24000),
        (np.array([np.nan], dtype=np.float32), 24000),
        (np.array([np.inf], dtype=np.float32), 24000),
        (np.zeros(24, dtype=np.float32), 0),
        (np.zeros(24, dtype=np.float32), True),
    ],
)
def test_invalid_raw_audio(
    samples: np.ndarray, rate: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid external PCM cannot reach the output device.

    Args:
        samples:
            Invalid shape/dtype/value fixture or valid data paired with an invalid rate.

        rate:
            Declared sample rate in Hz.

        monkeypatch:
            Observes any accidental queue construction.

    """
    factory = Mock()
    monkeypatch.setattr("hoast.tts.AudioPlayback", factory)
    engine = TTS(playback_only=True)
    with pytest.raises(ValueError):
        engine.play_samples(samples, rate)
    factory.assert_not_called()
    engine.close()


def test_raw_cancellation_resamples_once_then_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resampling spans the complete waveform while cancellation bounds later submission.

    Args:
        monkeypatch:
            Replaces the playback worker with a cancelling sink.

    """
    engine = TTS(playback_only=True)
    cancelled = threading.Event()
    stream = Mock()
    factory = Mock(return_value=stream)
    monkeypatch.setattr("hoast.tts.AudioPlayback", factory)
    converter = Mock(wraps=resample_poly)
    monkeypatch.setattr("hoast.tts.resample_poly", converter)
    samples = np.linspace(-0.3, 0.3, 16000, dtype=np.float32)
    cancelled.set()
    engine.play_samples(samples, 16000, cancelled=cancelled)
    factory.assert_not_called()
    converter.assert_not_called()
    cancelled.clear()

    def cancel(block: np.ndarray) -> None:
        """Cancel once the first 100 ms output block has been accepted.

        Args:
            block:
                Initial portion of the once-resampled command waveform.

        """
        assert block.size == 2400
        cancelled.set()

    stream.submit.side_effect = cancel
    engine.play_samples(samples, 16000, buffer_seconds=0.2, cancelled=cancelled)
    converter.assert_called_once()
    assert converter.call_args.args[0] is samples
    stream.submit.assert_called_once()
    stream.close.assert_called_once()
    np.testing.assert_allclose(
        stream.submit.call_args.args[0], resample_poly(samples, 3, 2)[:2400]
    )
    engine.close()
