"""Packet-independent 500 ms speech endpoints with mocked frame probabilities."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from hoast import endpoint as endpoint_module
from hoast.endpoint import FRAME_BYTES, FrameVAD, SileroVAD, SpeechEndpoint

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


def test_silero_normalized_pcm_and_context() -> None:
    """Feed normalized PCM16 once and retain normalized context between frames."""
    session = MagicMock()
    session.run.return_value = (
        np.array([[0.8]], dtype=np.float32),
        np.zeros((1, 1, 128), dtype=np.float32),
        np.zeros((1, 1, 128), dtype=np.float32),
    )
    vad = SileroVAD(session=session)
    pcm = np.resize(
        np.array([-32768, -16384, -1, 0, 1, 16384, 32767], dtype="<i2"), 512
    )
    normalized = pcm.astype(np.float32) / 32768
    vad(pcm.tobytes())
    first = session.run.call_args.args[1]["input"].copy()
    np.testing.assert_array_equal(first[0, :64], np.zeros(64))
    np.testing.assert_array_equal(first[0, 64:], normalized)
    vad(SILENCE)
    second = session.run.call_args.args[1]["input"]
    np.testing.assert_array_equal(second[0, :64], normalized[-64:])
    np.testing.assert_array_equal(second[0, 64:], np.zeros(512))
    vad.reset()
    assert not np.any(vad.input) and not np.any(vad.h) and not np.any(vad.c)


def test_silero_recurrent_stream_isolation() -> None:
    """Share a session, not hidden/cell/context arrays; retain exact recurrent outputs."""
    session = MagicMock()
    hidden = np.full((1, 1, 128), 0.25, dtype=np.float32)
    cell = np.full((1, 1, 128), 0.5, dtype=np.float32)
    session.run.return_value = (np.array([0.8], dtype=np.float32), hidden, cell)
    first, second = SileroVAD(session), SileroVAD(session)
    first(SPEECH)
    assert first.h is hidden and first.c is cell
    assert not np.any(second.h) and not np.any(second.c)
    first(SILENCE)
    assert session.run.call_args.args[1]["h"] is hidden
    assert session.run.call_args.args[1]["c"] is cell
    assert not np.any(second.input)
    first.reset()
    assert not np.any(first.h) and not np.any(first.c)


@pytest.mark.parametrize("invalid", ["frame", "state", "probability"])
def test_silero_rejects_invalid_frames_and_outputs(invalid: str) -> None:
    """Validate frame lengths and constant-time recurrent/probability output shapes.

    Args:
        invalid:
            The frame or model output contract to violate with a local mock.

    """
    session = MagicMock()
    session.run.return_value = (
        np.zeros(2 if invalid == "probability" else 1, dtype=np.float32),
        np.zeros((1, 128) if invalid == "state" else (1, 1, 128), dtype=np.float32),
        np.zeros((1, 1, 128), dtype=np.float32),
    )
    vad = SileroVAD(session)
    if invalid == "frame":
        with pytest.raises(ValueError, match="512"):
            vad(SILENCE[:-2])
        session.run.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="state shapes|probabilities"):
            vad(SILENCE)


@pytest.fixture
def prepared_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, MagicMock, MagicMock, MagicMock]:
    """Mock a complete ONNX interface without using a machine-local model file.

    Args:
        tmp_path:
            Isolated directory for an inert model-path fixture.

        monkeypatch:
            Restore ONNX constructors and clear session-cache methods after the test.

    """
    # Replace the cache wrapper per test so no mocked session survives restoration.
    monkeypatch.setattr(
        endpoint_module,
        "_session",
        lru_cache(maxsize=1)(endpoint_module._session.__wrapped__),
    )
    path = tmp_path / "vad.onnx"
    path.write_bytes(b"mock ONNX, never parsed")
    session = MagicMock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="input", type="tensor(float)", shape=["seq_len", 576]),
        SimpleNamespace(name="h", type="tensor(float)", shape=[1, 1, 128]),
        SimpleNamespace(name="c", type="tensor(float)", shape=[1, 1, 128]),
    ]
    session.get_outputs.return_value = [
        SimpleNamespace(name="speech_probs", type="tensor(float)", shape=["seq_len"]),
        SimpleNamespace(name="hn", type="tensor(float)", shape=[1, 1, 128]),
        SimpleNamespace(name="cn", type="tensor(float)", shape=[1, 1, 128]),
    ]
    constructor = MagicMock(return_value=session)
    options = MagicMock()
    monkeypatch.setattr(endpoint_module.ort, "InferenceSession", constructor)
    monkeypatch.setattr(
        endpoint_module.ort, "SessionOptions", MagicMock(return_value=options)
    )
    return path, session, constructor, options


def test_session_cached_single_thread_cpu(
    prepared_session: tuple[Path, MagicMock, MagicMock, MagicMock],
) -> None:
    """Retain the original ORT CPU threading, arena, logging and session cache settings.

    Args:
        prepared_session:
            Temporary model path and isolated mock session, constructor and options.

    """
    path, session, constructor, options = prepared_session
    assert endpoint_module._session(path) is session
    assert endpoint_module._session(path) is session
    constructor.assert_called_once_with(
        str(path), providers=["CPUExecutionProvider"], sess_options=options
    )
    assert options.inter_op_num_threads == options.intra_op_num_threads == 1
    assert options.enable_cpu_mem_arena is False
    assert options.log_severity_level == 4


@pytest.mark.parametrize("invalid", ["input", "output", "dtype", "waveform", "state"])
def test_session_rejects_incompatible_interface(
    prepared_session: tuple[Path, MagicMock, MagicMock, MagicMock], invalid: str
) -> None:
    """Reject incompatible ONNX metadata at loading time, before any frame inference.

    Args:
        prepared_session:
            Temporary model path and isolated mocked ONNX interface.

        invalid:
            Metadata contract to break in the otherwise compatible session.

    """
    path, session, _, _ = prepared_session
    if invalid == "input":
        session.get_inputs.return_value[0].name = "sr"
    elif invalid == "output":
        session.get_outputs.return_value.reverse()
    elif invalid == "dtype":
        session.get_inputs.return_value[0].type = "tensor(double)"
    elif invalid == "waveform":
        session.get_inputs.return_value[0].shape = ["seq_len", 512]
    else:
        session.get_outputs.return_value[1].shape = [2, 1, 128]
    with pytest.raises(RuntimeError, match="Unsupported Silero v6"):
        endpoint_module._session(path)


def test_session_missing_asset(
    prepared_session: tuple[Path, MagicMock, MagicMock, MagicMock],
) -> None:
    """Provide an actionable missing-file error without creating a session or downloading.

    Args:
        prepared_session:
            Temporary model path and isolated mocked ONNX constructor.

    """
    path, _, constructor, _ = prepared_session
    path.unlink()
    with pytest.raises(FileNotFoundError, match="tools.prepare_vad"):
        endpoint_module._session(path)
    constructor.assert_not_called()
