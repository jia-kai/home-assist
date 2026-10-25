"""Startup-only scalar loudness bounds and unchanged ordinary playback contracts."""

import numpy as np
import pytest
from scipy.signal import resample_poly

from hoast.application import _startup_audio
from hoast.voice_debug import _cue


@pytest.mark.parametrize("amplitude", [0.0, 0.001, 0.02, 0.2, 1.0])
def test_startup_gain_target_cap_and_preroll(amplitude: float) -> None:
    """Apply scalar gain without input mutation, honoring RMS/peak bounds.

    Args:
        amplitude:
            Constant fixture level spanning silence, gain-cap and attenuation cases.

    """
    samples = np.full(1600, amplitude, dtype=np.float32)
    original = samples.copy()
    audio = _startup_audio(samples, 16000)
    assert audio.dtype == np.float32 and audio.shape == (4800,)
    assert not np.any(audio[:3200])
    expected_gain = min(4.0, 0.07 / amplitude, 0.8 / amplitude) if amplitude else 1
    np.testing.assert_allclose(audio[3200:], samples * expected_gain)
    np.testing.assert_array_equal(samples, original)


def test_startup_peak_headroom_for_sparse_speech() -> None:
    """Peak safety overrides RMS targeting for a high-crest-factor waveform."""
    samples = np.zeros(4410, dtype=np.float32)
    samples[2205] = 0.5
    output = _startup_audio(samples, 44100)
    assert np.max(np.abs(output)) == pytest.approx(0.8)
    assert np.max(np.abs(resample_poly(output, 80, 147))) < 1
    assert np.count_nonzero(output) == 1


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_startup_nonfinite_rejected(value: float) -> None:
    """Fail explicitly rather than submitting invalid announcement samples.

    Args:
        value:
            Nonfinite sample from an invalid synthesis boundary.

    """
    with pytest.raises(ValueError, match="finite"):
        _startup_audio(np.array([value], dtype=np.float32), 16000)


@pytest.mark.parametrize("rate", [0, -1, True, 16000.0])
def test_startup_invalid_rate_rejected(rate: int) -> None:
    """Reject unsupported rate metadata before creating startup silence.

    Args:
        rate:
            Invalid runtime metadata, including integer-like non-integers.

    """
    with pytest.raises(ValueError, match="sample rate"):
        _startup_audio(np.ones(10, dtype=np.float32), rate)


@pytest.mark.parametrize("frequency", [440, 880])
def test_capture_cue_level_fade_and_pitch(frequency: float) -> None:
    """Capture tones are 10.46 dB above unchanged debug brackets, with no clipping.

    Args:
        frequency:
            End or first-PCM tone frequency in Hz.

    """
    cue = _cue(frequency)
    debug = _cue(frequency, peak=0.12)
    assert cue.dtype == np.float32 and cue.shape == (1920,)
    assert cue[0] == cue[-1] == 0
    assert 0.39 < np.max(np.abs(cue)) <= 0.4
    assert 0.17 < np.sqrt(np.mean(cue**2)) < 0.18
    np.testing.assert_allclose(cue, debug * (0.4 / 0.12), atol=5e-8)
    frequencies = np.fft.rfftfreq(cue.size, 1 / 16000)
    strongest = frequencies[np.argmax(np.abs(np.fft.rfft(cue)))]
    assert abs(strongest - frequency) < 16000 / cue.size
