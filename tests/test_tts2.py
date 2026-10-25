"""Hardware-independent duration expansion and native playback ownership checks."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from hoast.tts2 import TTS, TTSConfig
from hoast.tts2.engine import expand_prior


def test_sparse_duration_expansion() -> None:
    """Repeat priors exactly like a dense monotonic alignment including zero counts."""
    mean = np.arange(192 * 4, dtype=np.float32).reshape(1, 192, 4)
    scale = mean * np.float32(0.01)
    durations = np.array([2, 0, 1, 3], dtype=np.int64)
    expanded, logscale = expand_prior(mean, scale, durations, 6)
    alignment = np.zeros((4, 6), dtype=np.float32)
    alignment[np.repeat(np.arange(4), durations), np.arange(6)] = 1
    np.testing.assert_array_equal(expanded, mean @ alignment)
    np.testing.assert_array_equal(logscale, scale @ alignment)
    assert expanded.flags.c_contiguous and logscale.flags.c_contiguous
    assert not np.shares_memory(expanded, mean)


@pytest.mark.parametrize("counts", [[0, 0], [-1, 2], [3, 3]])
def test_duration_bounds(counts: list[int]) -> None:
    """Reject empty, negative and excessive durations instead of truncating speech.

    Args:
        counts:
            Invalid duration vector for a five-frame capacity.

    """
    prior = np.zeros((1, 192, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        expand_prior(prior, prior, np.array(counts, dtype=np.int64), 5)


def test_explicit_budget_and_precision() -> None:
    """Reject unsupported budgets and retain an explicit GPU floating-point hint."""
    assert TTSConfig(threads=1).gpu_precision == "f16"
    with pytest.raises(ValueError):
        TTSConfig(threads=3)
    with pytest.raises(ValueError):
        TTSConfig(threads=True)
    with pytest.raises(ValueError):
        TTSConfig(speed=float("nan"))
    with pytest.raises(ValueError):
        TTSConfig(max_phone_positions=0)


@pytest.mark.parametrize("tokens,phones", [(513, 20), (10, 2049)])
def test_input_attention_bounds(
    monkeypatch: pytest.MonkeyPatch, tokens: int, phones: int
) -> None:
    """Reject excessive quadratic-attention inputs before any neural request.

    Args:
        monkeypatch:
            Supplies an in-memory frontend without reading model artifacts.

        tokens:
            Prepared BERT sequence length including special tokens.

        phones:
            Prepared encoder sequence length including blanks.

    """
    engine = TTS(playback_only=True)
    engine._playback_only = False
    frontend = Mock()
    frontend.split.return_value = ["Hello"]
    frontend.prepare.return_value = SimpleNamespace(
        input_ids=np.zeros((1, tokens), dtype=np.int64),
        phones=np.zeros((1, phones), dtype=np.int64),
    )
    stage = Mock(device="CPU", precision_hint="f32")
    resources = SimpleNamespace(
        frontend=frontend,
        speakers={"EN-Default": 0},
        stages={"bert": stage},
        manifest={"precision": "FP16"},
    )
    monkeypatch.setattr(TTS, "_load", lambda self, language: resources)
    with pytest.raises(ValueError, match="limit"):
        engine.synthesize("Hello")
    stage.run.assert_not_called()
    engine.close()


def test_playback_cancel_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation stops new blocks while draining accepted audio exactly once.

    Args:
        monkeypatch:
            Replaces physical playback with a cancelling in-memory stream.

    """
    engine = TTS(playback_only=True)
    event = threading.Event()
    output = Mock()
    output.submit.side_effect = lambda samples: event.set()
    factory = Mock(return_value=output)
    monkeypatch.setattr("hoast.tts2.engine.AudioPlayback", factory)
    engine.play_samples(
        np.zeros(7200, dtype=np.float32), 24000, blocking=False, cancelled=event
    )
    output.submit.assert_called_once()
    output.close.assert_not_called()
    engine.close()
    engine.close()
    output.close.assert_called_once()
    assert engine._core is None and not engine._resources
    with pytest.raises(RuntimeError, match="closed"):
        engine.play_samples(np.zeros(24, dtype=np.float32), 24000)


def test_pre_cancelled_playback(monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-cancelled submission neither creates a stream nor resamples.

    Args:
        monkeypatch:
            Prevents access to audio hardware and records resampling requests.

    """
    engine = TTS(playback_only=True)
    event = threading.Event()
    event.set()
    factory = Mock()
    resample = Mock()
    monkeypatch.setattr("hoast.tts2.engine.AudioPlayback", factory)
    monkeypatch.setattr("hoast.tts2.engine.resample_poly", resample)
    engine.play_samples(np.zeros(441, dtype=np.float32), 44100, cancelled=event)
    factory.assert_not_called()
    resample.assert_not_called()
    engine.close()


def test_cancel_during_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip playback when cancellation arrives while synthesis is completing.

    Args:
        monkeypatch:
            Replaces synthesis and physical audio with in-memory mocks.

    """
    engine = TTS(playback_only=True)
    cancelled = threading.Event()

    def synthesize(self: TTS, text: str) -> tuple[np.ndarray, int]:
        """Set cancellation at the end of a simulated inference request.

        Args:
            self:
                Engine whose playback must remain idle.

            text:
                Requested speech discarded after transport loss.

        """
        cancelled.set()
        return np.zeros(441, dtype=np.float32), 44100

    monkeypatch.setattr(TTS, "synthesize", synthesize)
    output = Mock()
    monkeypatch.setattr("hoast.tts2.engine.AudioPlayback", output)
    engine.play("Hello", cancelled=cancelled)
    output.assert_not_called()
    engine.close()


def test_playback_failure_and_queue_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse one ordered queue, reject capacity changes and clean up device failures.

    Args:
        monkeypatch:
            Supplies a failing playback mock with no physical audio dependency.

    """
    engine = TTS(playback_only=True)
    output = Mock()
    monkeypatch.setattr("hoast.tts2.engine.AudioPlayback", Mock(return_value=output))
    audio = np.zeros(240, dtype=np.float32)
    engine.play_samples(audio, 24000, blocking=False)
    with pytest.raises(ValueError, match="Drain playback"):
        engine.play_samples(audio, 24000, blocking=False, buffer_seconds=2)
    output.submit.side_effect = RuntimeError("Device disconnected")
    with pytest.raises(RuntimeError, match="Device disconnected"):
        engine.play_samples(audio, 24000, blocking=False)
    output.close.assert_called_once()
    assert engine._output is None and engine._buffer_seconds is None
    engine.close()


@pytest.mark.parametrize("sample_rate", [24000, 44100, 48000])
def test_playback_resampling_preserves_amplitude(
    monkeypatch: pytest.MonkeyPatch, sample_rate: int
) -> None:
    """Resampling changes duration units, not normalized PCM amplitude or ownership.

    Args:
        monkeypatch:
            Supplies a copying queue without physical device access.

        sample_rate:
            Source sample rate in Hz; 24 kHz bypasses resampling.

    """
    received: list[np.ndarray] = []
    output = Mock()
    output.submit.side_effect = lambda block: received.append(block.copy())
    monkeypatch.setattr("hoast.tts2.engine.AudioPlayback", Mock(return_value=output))
    engine = TTS(playback_only=True)
    time = np.arange(sample_rate // 5, dtype=np.float32) / sample_rate
    audio = np.asarray(0.2 * np.sin(2 * np.pi * 600 * time), dtype=np.float32)
    original = audio.copy()
    engine.play_samples(audio, sample_rate)
    played = np.concatenate(received)
    assert played.size == 4800
    interior = played[240:-240]
    np.testing.assert_allclose(
        np.sqrt(np.mean(interior**2)), 0.2 / np.sqrt(2), rtol=0.003
    )
    np.testing.assert_array_equal(audio, original)
    if sample_rate == 24000:
        np.testing.assert_array_equal(played, audio)
    output.close.assert_called_once()
    engine.close()


def test_wav_output_preserves_normalized_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Write normalized decoder PCM once, with only normal PCM16 quantization.

    Args:
        monkeypatch:
            Replaces model synthesis with a normalized decoder waveform.

        tmp_path:
            Isolated output directory; no machine-local speech fixture is used.

    """
    audio = np.array([-1, -0.5, -1 / 32768, 0, 1 / 32768, 0.5, 1], np.float32)
    engine = TTS(playback_only=True)
    monkeypatch.setattr(engine, "synthesize", Mock(return_value=(audio, 44100)))
    path = tmp_path / "speech.wav"
    engine.write_wav("fixture", path)
    decoded, rate = sf.read(path, dtype="float32")
    assert rate == 44100 and sf.info(path).subtype == "PCM_16"
    expected = audio.copy()
    expected[-1] = 32767 / 32768
    np.testing.assert_array_equal(decoded, expected)
    engine.close()
