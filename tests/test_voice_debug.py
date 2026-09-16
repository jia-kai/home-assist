"""Mock system playback for cue-bracketed command replay and cancellation."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from aioesphomeapi.client import APIClient
from aioesphomeapi.model import VoiceAssistantAudioSettings

from hoast import voice
from hoast.tts import TTS
from hoast.voice import Event, VoiceReceiver
from hoast.voice_debug import replay_capture


def test_replay_pcm_and_cue_order() -> None:
    """Submit one cue-bracketed waveform to the caller's TTS without changing command PCM."""
    tts = MagicMock(spec=TTS)
    original = np.arange(-2400, 2400, dtype="<i2")
    cancelled = threading.Event()
    replay_capture(original.tobytes(), cancelled, tts)
    tts.play_samples.assert_called_once()
    samples, rate = tts.play_samples.call_args.args
    assert rate == 16000
    assert tts.play_samples.call_args.kwargs == {
        "blocking": True,
        "buffer_seconds": 0.2,
        "cancelled": cancelled,
    }
    assert not np.array_equal(samples[:1920], samples[-1920:])
    assert np.max(np.abs(samples[:1920])) <= 0.12
    assert not np.any(samples[1920:3200]) and not np.any(samples[-3200:-1920])
    np.testing.assert_array_equal(
        samples[3200:-3200], original.astype(np.float32) / 32768.0
    )
    tts.close.assert_not_called()


def test_replay_cancellation_and_failure_cleanup() -> None:
    """Replay skips cancelled input and propagates failure without closing its owner's TTS."""
    tts = MagicMock(spec=TTS)
    cancelled = threading.Event()
    cancelled.set()
    replay_capture(b"\0\0", cancelled, tts)
    tts.play_samples.assert_not_called()
    cancelled.clear()

    tts.play_samples.side_effect = RuntimeError("output failed")
    with pytest.raises(RuntimeError, match="output failed"):
        replay_capture(b"\0\0", cancelled, tts)
    tts.close.assert_not_called()


@pytest.mark.parametrize("debug,cancel", [(False, False), (True, False), (True, True)])
def test_debug_replay_drains_before_stt(
    debug: bool, cancel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Receiver stops capture before replay and skips STT if replay is cancelled.

    Args:
        debug:
            Whether capture replay is enabled.

        cancel:
            Whether the mocked replay simulates disconnection before STT.

        monkeypatch:
            Supplies mock audible playback, without a device or model.

    """
    order: list[str] = []
    pcm = b"\x01\0" * 1600
    engine = MagicMock(spec=TTS)

    def replay(data: bytes, cancelled: threading.Event, tts: TTS) -> None:
        """Represent complete cue/audio/cue playback on the replay worker.

        Args:
            data:
                Exact captured bytes expected by STT.

            cancelled:
                Receiver-owned cancellation flag shared with the replay worker.

            tts:
                Existing application TTS instance, not a separately created queue.

        """
        assert data == pcm
        assert tts is engine
        order.extend(["start-cue", "command", "end-cue", "drained"])
        if cancel:
            cancelled.set()

    replay_mock = MagicMock(side_effect=replay)
    monkeypatch.setattr("hoast.voice.replay_capture", replay_mock)

    async def run() -> None:
        """Complete a bounded capture, then observe replay and STT ordering."""

        async def handler(data: bytes) -> str:
            """Require replay completion before accepting the captured command.

            Args:
                data:
                    Captured PCM after the optional debug replay.

            """
            assert data == pcm
            if debug:
                assert order[-1] == "drained"
            order.append("stt")
            return "test command"

        client = MagicMock(spec=APIClient)
        receiver = VoiceReceiver(
            client,
            handler,
            seconds=0.1,
            debug_audio=debug,
            replay_tts=engine if debug else None,
        )
        await receiver.start("", 0, VoiceAssistantAudioSettings(), None)
        await receiver.receive(pcm, None)
        assert not receiver.capturing
        assert receiver.task is not None
        await receiver.task
        assert (
            client.send_voice_assistant_event.call_args.args[0]
            == Event.VOICE_ASSISTANT_RUN_END
        )
        await receiver.close()

    asyncio.run(run())
    assert replay_mock.call_count == int(debug)
    assert ("stt" in order) is not cancel
    engine.close.assert_not_called()


@pytest.mark.parametrize("debug", [False, True])
def test_standalone_replay_owner(debug: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Standalone STT owns one model-free replay engine only when debugging is enabled.

    Args:
        debug:
            Whether the CLI enables captured-command replay.

        monkeypatch:
            Replaces speech, output and network boundaries without touching devices.

    """
    engine = MagicMock(spec=TTS)
    factory = MagicMock(return_value=engine)
    listener = AsyncMock()
    monkeypatch.setattr(voice, "TTS", factory)
    monkeypatch.setattr(voice, "STT", MagicMock())
    monkeypatch.setattr(voice, "listen", listener)
    monkeypatch.setattr(voice, "configure_cpu_budget", MagicMock())
    monkeypatch.setattr(voice, "configure_logging", MagicMock())
    monkeypatch.setattr("sys.argv", ["voice", *(["--debug-audio"] if debug else [])])
    voice.main()
    listener.assert_awaited_once()
    if debug:
        factory.assert_called_once_with(playback_only=True)
        assert listener.call_args.kwargs["replay_tts"] is engine
        engine.close.assert_called_once()
    else:
        factory.assert_not_called()
        assert listener.call_args.kwargs["replay_tts"] is None
