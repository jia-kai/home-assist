"""Offline native smoke-check fixtures with deterministic engine ownership."""

from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pytest
from numpy.typing import NDArray

from hoast.stt2 import STTConfig
from hoast.tts2 import TTSConfig
from tools import check_speech2


def test_canonical() -> None:
    """Ignore case/spacing/punctuation without losing spoken numeric content."""
    assert check_speech2.canonical("Hello, WORLD。") == "helloworld"
    assert check_speech2.canonical("十二分钟") != check_speech2.canonical("十分钟")
    assert check_speech2.canonical("１２") == "12"


@pytest.mark.parametrize("mismatch", [False, True])
def test_roundtrip_evidence_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: bool
) -> None:
    """Release both engines and retain each transcript even when its check fails.

    Args:
        tmp_path:
            Isolated directory retaining WAV and JSON fixture output.

        monkeypatch:
            Replace model loading with context-managed offline test engines.

        mismatch:
            Return a wrong first transcript to exercise durable rejection.

    """
    closed: list[str] = []
    texts = iter(["Hello world.", "The weather is sunny."])

    class FakeTTS:
        """Context-managed deterministic mono waveform source."""

        last_timings: ClassVar[dict[str, Any]] = {}
        """JSON-safe stage evidence."""

        def __init__(self, config: TTSConfig) -> None:
            """Accept supplied local configuration.

            Args:
                config:
                    Native synthesis configuration, without loading artifacts.

            """

        def __enter__(self) -> Self:
            """Return the owned source."""
            return self

        def __exit__(self, *args: object) -> None:
            """Record deterministic closure.

            Args:
                args:
                    Context-manager exception information.

            """
            closed.append("tts")

        def synthesize(self, text: str) -> tuple[NDArray[np.float32], int]:
            """Return a tiny finite native-rate fixture.

            Args:
                text:
                    Complete expected utterance.

            """
            return np.zeros(8, dtype=np.float32), 44100

    class FakeSTT:
        """Context-managed expected transcript source."""

        last_timings: ClassVar[dict[str, Any]] = {}
        """JSON-safe stage evidence."""

        def __init__(self, config: STTConfig) -> None:
            """Accept supplied local configuration.

            Args:
                config:
                    Native recognition configuration without graph loading.

            """

        def __enter__(self) -> Self:
            """Return the owned recognizer."""
            return self

        def __exit__(self, *args: object) -> None:
            """Record deterministic closure.

            Args:
                args:
                    Context-manager exception information.

            """
            closed.append("stt")

        def transcribe_samples(self, samples: NDArray[np.float32], rate: int) -> str:
            """Return expected or deliberately mismatched fixture text.

            Args:
                samples:
                    Mono float audio fixture.

                rate:
                    Fixture sample rate in Hz.

            """
            return "wrong" if mismatch else next(texts)

    monkeypatch.setattr(check_speech2, "TTS", FakeTTS)
    monkeypatch.setattr(check_speech2, "STT", FakeSTT)
    if mismatch:
        with pytest.raises(ValueError, match="mismatch"):
            check_speech2.check_round_trip(tmp_path, STTConfig(), TTSConfig(), False)
    else:
        check_speech2.check_round_trip(tmp_path, STTConfig(), TTSConfig(), False)
    assert closed == ["stt", "tts"]
    assert (tmp_path / "speech-0.wav").exists()
    assert (tmp_path / "roundtrip.json").exists()
