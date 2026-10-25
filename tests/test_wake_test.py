"""Model-free checks that microphone wake testing prepares replies with native Melo."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools import wake_test


@pytest.mark.parametrize("cached", [False, True])
def test_native_reply_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    """Publish a complete native TTS reply once without launching speech subprocesses.

    Args:
        tmp_path:
            Isolated reply and model paths, with no installed artifacts or audio device.

        monkeypatch:
            Replaces speech inference, port probing, logging and interactive listening.

        cached:
            Whether an existing complete reply should skip synthesis.

    """
    reply = tmp_path / "wake/im-here.wav"
    if cached:
        reply.parent.mkdir()
        reply.write_bytes(b"existing reply")
    engine = MagicMock()

    def write_wav(text: str, output: Path) -> None:
        """Simulate completed synthesis in the requested temporary WAV file.

        Args:
            text:
                Exact spoken wake acknowledgement.

            output:
                Temporary destination published only after synthesis succeeds.

        """
        assert text == "I'm here."
        output.write_bytes(b"native Melo reply")

    engine.write_wav.side_effect = write_wav
    factory = MagicMock()
    factory.return_value.__enter__.return_value = engine
    command = MagicMock(return_value=["fixture satellite"])
    run = MagicMock()
    monkeypatch.setattr(wake_test, "ROOT", tmp_path)
    monkeypatch.setattr(wake_test, "REPLY", reply)
    monkeypatch.setattr(wake_test, "TTS", factory)
    monkeypatch.setattr(wake_test, "configure_logging", MagicMock())
    monkeypatch.setattr(wake_test, "satellite_command", command)
    monkeypatch.setattr(wake_test.socket, "socket", MagicMock())
    monkeypatch.setattr(wake_test, "run_test", run)
    monkeypatch.setattr(wake_test.asyncio, "run", MagicMock())
    monkeypatch.setattr(sys, "argv", ["wake_test"])
    wake_test.main()
    if cached:
        factory.assert_not_called()
        assert reply.read_bytes() == b"existing reply"
    else:
        factory.assert_called_once()
        assert factory.call_args.args[0].model_path == tmp_path / "models/tts2"
        engine.write_wav.assert_called_once_with(
            "I'm here.", reply.with_suffix(".partial.wav")
        )
        assert reply.read_bytes() == b"native Melo reply"
        assert not reply.with_suffix(".partial.wav").exists()
        factory.return_value.__exit__.assert_called_once()
    command.assert_called_with(None, reply)
    run.assert_called_once_with(["fixture satellite"])
