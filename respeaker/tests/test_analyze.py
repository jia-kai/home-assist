"""Verify acoustic lag and suppression estimates on a known delayed echo."""

import wave
from pathlib import Path

import numpy as np
import pytest

from respeaker.tools.analyze import (
    echo_gain,
    load_delay_pair,
    normalized_echo_correlation,
)


def test_echo_gain_finds_sample_offset_and_attenuation() -> None:
    """Separate a known 20 ms echo from unrelated noise in both trial captures."""
    rng = np.random.default_rng(17)
    reference = rng.normal(0, 0.2, size=4_800).astype(np.float32)
    background = rng.normal(0, 0.002, size=6_000).astype(np.float32)
    raw = background.copy()
    processed = background.copy()
    raw[960:5760] += reference * 0.5
    processed[960:5760] += reference * 0.05
    gain_raw, lag_raw = echo_gain(reference, raw)
    gain_processed, lag_processed = echo_gain(reference, processed)
    assert lag_raw == lag_processed == 20.0
    assert gain_raw == pytest.approx(0.5, abs=0.001)
    assert gain_processed == pytest.approx(0.05, abs=0.001)
    assert normalized_echo_correlation(reference, raw, lag_raw) > 0.99


def test_delay_pair_preserves_simultaneous_frame_alignment(tmp_path: Path) -> None:
    """Interpret both I²S slots on one clock without accidental mono downmix."""
    path = tmp_path / "delay.wav"
    interleaved = np.array([[0, 2**30], [2**29, -2**29]], dtype="<i4")
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(2)
        recording.setsampwidth(4)
        recording.setframerate(48_000)
        recording.writeframes(interleaved.tobytes())
    farend, mic = load_delay_pair(path)
    np.testing.assert_allclose(farend, [0, 0.25])
    np.testing.assert_allclose(mic, [0.5, -0.25])
