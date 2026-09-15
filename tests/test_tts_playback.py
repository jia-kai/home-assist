"""Hardware-independent checks of ordered buffered playback and cleanup."""

import threading
from unittest.mock import Mock

import numpy as np
import pytest

from hoast.tts import TTS


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
