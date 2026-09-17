"""Packet-independent 500 ms speech endpoints with mocked frame probabilities."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from hoast.endpoint import FRAME_BYTES, FrameVAD, SpeechEndpoint

SPEECH = b"\x01\0" * 512
SILENCE = b"\0\0" * 512


@dataclass(slots=True)
class MockVAD:
    """Classify fixture markers rather than real acoustic input."""

    calls: int = 0
    """Frames classified since the last reset."""

    def reset(self) -> None:
        """Reset the fixture's per-capture frame count."""
        self.calls = 0

    def __call__(self, pcm: bytes) -> float:
        """Map the frame marker to a controlled speech probability.

        Args:
            pcm:
                Full frame marked with zero for silence or one for speech.

        """
        assert len(pcm) == FRAME_BYTES
        self.calls += 1
        return 0.9 if pcm[0] else 0.1


@pytest.mark.parametrize("packet_bytes", [2, 98, 1024, 4096, 100000])
def test_packet_independent_endpoint(packet_bytes: int) -> None:
    """Ignore speech after an endpoint in the same packet; preserve exactly 500 ms.

    Args:
        packet_bytes:
            Even packet size, including individual samples and multi-frame bursts.

    """
    vad = MockVAD()
    endpoint = SpeechEndpoint(vad)
    pcm = SILENCE * 5 + SPEECH * 10 + SILENCE * 16 + SPEECH * 5
    result = None
    for offset in range(0, len(pcm), packet_bytes):
        result = endpoint.feed(pcm[offset : offset + packet_bytes])
        if result is not None:
            break
    assert result == 15 * 512 + 8000
    assert vad.calls == 31
    assert endpoint.feed(SPEECH) == result
    assert vad.calls == 31


def test_initial_silence_and_short_pause() -> None:
    """Initial silence never ends a command; a sub-500 ms pause resets on resumed speech."""
    endpoint = SpeechEndpoint(MockVAD())
    assert endpoint.feed(SILENCE * 60) is None
    assert endpoint.feed(SPEECH * 2) is None
    assert endpoint.feed(SILENCE * 15) is None
    assert endpoint.feed(SPEECH) is None
    assert endpoint.silence == 0
    last_speech_end = endpoint.samples
    assert endpoint.feed(SILENCE * 15 + SILENCE[:640]) is None
    assert endpoint.feed(SILENCE[640:]) == last_speech_end + 8000


def test_hysteresis_and_reset() -> None:
    """Ambiguous frames cannot establish onset, but prevent cutting existing speech."""
    vad = MagicMock(spec=FrameVAD)
    vad.return_value = 0.4
    endpoint = SpeechEndpoint(vad)
    endpoint.feed(SILENCE * 20)
    assert not endpoint.speech
    vad.return_value = 0.8
    endpoint.feed(SPEECH)
    vad.return_value = 0.1
    endpoint.feed(SILENCE * 15)
    vad.return_value = 0.4
    assert endpoint.feed(SILENCE) is None
    assert endpoint.silence == 0
    endpoint.feed(b"\0\0")
    endpoint.reset()
    vad.reset.assert_called_once()
    assert endpoint.samples == endpoint.silence == 0
    assert not endpoint.pending and not endpoint.speech
    assert endpoint.end_sample is None


def test_invalid_input_and_probability() -> None:
    """Reject partial samples and nonfinite model output instead of guessing silence."""
    vad = MagicMock(spec=FrameVAD, return_value=float("nan"))
    endpoint = SpeechEndpoint(vad)
    with pytest.raises(ValueError, match="partial"):
        endpoint.feed(b"\0")
    with pytest.raises(ValueError, match="probability"):
        endpoint.feed(SILENCE)
