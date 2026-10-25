"""Mock-only capture tones, output ownership, cancellation and exact PCM tests."""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from aioesphomeapi.client import APIClient
from aioesphomeapi.model import VoiceAssistantAudioSettings, VoiceAssistantFeature
from numpy.typing import NDArray

from hoast import voice
from hoast.endpoint import FrameVAD, SpeechEndpoint
from hoast.tts2 import TTS
from hoast.voice import VoiceReceiver
from hoast.voice_debug import CaptureCue, _cue, play_capture_cue


def _receiver(
    handler: Callable[[bytes], Awaitable[str]],
    tts: TTS,
    *,
    endpoint: SpeechEndpoint | None = None,
    seconds: float = 6,
) -> VoiceReceiver:
    """Construct a cue-enabled receiver with mocked transport and shared output.

    Args:
        handler:
            Mock asynchronous recognizer, never a model invocation.

        tts:
            Mock caller-owned playback queue.

        endpoint:
            Optional detector backed by mocked frame probabilities.

        seconds:
            Maximum capture duration in seconds.

    """
    return VoiceReceiver(
        MagicMock(spec=APIClient),
        handler,
        seconds=seconds,
        endpoint=endpoint,
        replay_tts=tts,
        sound_cues=True,
    )


async def _start(receiver: VoiceReceiver) -> None:
    """Accept a fixture wake request without supplying any audio.

    Args:
        receiver:
            Isolated receiver whose public callbacks are under test.

    """
    assert await receiver.start("", 0, VoiceAssistantAudioSettings(), "wake") == 0


async def _finish(receiver: VoiceReceiver) -> str:
    """Join capture, return its end reason, and close without closing shared output.

    Args:
        receiver:
            Receiver with an accepted fixture capture.

    """
    assert receiver.task is not None
    await asyncio.wait_for(asyncio.shield(receiver.task), 3)
    reason = receiver.end_reason
    await receiver.close()
    assert receiver.cue_task is None
    return reason


def test_first_valid_nonempty_pcm_only_once_and_worker_thread(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wake/empty packets are silent; first complete PCM plays once off the loop.

    Args:
        caplog:
            Captures cue-start and accepted-output-drain diagnostics.

    """
    caplog.set_level("INFO")

    async def run() -> None:
        """Send empty and repeated chunks then verify exact PCM and native tone args."""
        loop_thread = threading.get_ident()
        playback_threads: list[int] = []
        tts = MagicMock(spec=TTS)
        tts.play_samples.side_effect = lambda *args, **kwargs: playback_threads.append(
            threading.get_ident()
        )
        handler = AsyncMock(return_value="command")
        receiver = _receiver(handler, tts)
        await _start(receiver)
        await asyncio.sleep(0)
        assert receiver.started_at is None
        tts.play_samples.assert_not_called()
        await receiver.receive(b"", None)
        await asyncio.sleep(0)
        tts.play_samples.assert_not_called()
        pcm = b"\x00\x80\xff\x7f\x01\x00"
        await receiver.receive(pcm[:2], None)
        started_at = receiver.started_at
        await receiver.receive(pcm[2:], b"ignored echo")
        assert receiver.started_at == started_at
        await receiver.stop(False)
        await _finish(receiver)
        handler.assert_awaited_once_with(pcm)
        tts.play_samples.assert_called_once()
        samples, rate = tts.play_samples.call_args.args
        np.testing.assert_array_equal(samples, _cue(880))
        assert samples.shape == (1920,) and samples.dtype == np.float32
        assert samples[0] == samples[-1] == 0
        assert 0.39 < np.max(np.abs(samples)) <= 0.4
        assert rate == 16000
        assert tts.play_samples.call_args.kwargs == {
            "blocking": True,
            "buffer_seconds": 0.2,
            "cancelled": receiver.cancelled,
        }
        assert playback_threads and all(t != loop_thread for t in playback_threads)
        tts.close.assert_not_called()

    asyncio.run(run())
    assert "voice.cue kind=first_pcm status=starting" in caplog.text
    assert "voice.cue kind=first_pcm status=drained" in caplog.text


@pytest.mark.parametrize(
    "scenario", ["malformed", "empty", "abort", "disconnect", "expired"]
)
def test_invalid_or_cancelled_input_skips_cues(scenario: str) -> None:
    """Unusable captures never play tones or reach the recognizer.

    Args:
        scenario:
            Invalid packet or cancellation/deadline boundary to exercise.

    """

    async def run() -> None:
        """Terminate an empty/invalid capture and inspect mock output and STT."""
        tts = MagicMock(spec=TTS)
        handler = AsyncMock(return_value="unexpected")
        receiver = _receiver(handler, tts)
        await _start(receiver)
        if scenario == "malformed":
            await receiver.receive(b"\x01", None)
        elif scenario == "abort":
            await receiver.stop(True)
            await receiver.receive(b"\x01\0", None)
        elif scenario == "disconnect":
            receiver.connected = False
            await receiver.receive(b"\x01\0", None)
            await receiver.stop(True)
        elif scenario == "expired":
            receiver.deadline = 0
            await receiver.receive(b"\x01\0", None)
        else:
            await receiver.receive(b"", None)
            await receiver.stop(False)
        await _finish(receiver)
        assert receiver.started_at is None
        handler.assert_not_awaited()
        tts.play_samples.assert_not_called()
        tts.close.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("termination", ["abort", "malformed", "disconnect"])
def test_cancelled_queued_first_cue_skips_playback(termination: str) -> None:
    """Cancellation before the scheduled worker runs skips new native output.

    Args:
        termination:
            Abort, malformed subsequent input, or disconnect before worker scheduling.

    """

    async def run() -> None:
        """Queue a first cue then cancel without yielding control to its worker."""
        tts = MagicMock(spec=TTS)
        handler = AsyncMock(return_value="unexpected")
        receiver = _receiver(handler, tts)
        await _start(receiver)
        await receiver.receive(b"\x01\0", None)
        assert receiver.cue_task is not None
        if termination == "malformed":
            await receiver.receive(b"\x01", None)
        elif termination == "disconnect":
            await receiver.close()
        else:
            await receiver.stop(True)
        await _finish(receiver)
        handler.assert_not_awaited()
        tts.play_samples.assert_not_called()
        tts.close.assert_not_called()

    asyncio.run(run())


def test_rearm_plays_one_first_pcm_cue_per_capture() -> None:
    """A reusable connected receiver resets first-PCM state after output is joined."""

    async def run() -> None:
        """Complete two independent remote-ended commands using one output owner."""
        tts = MagicMock(spec=TTS)
        handler = AsyncMock(return_value="command")
        receiver = _receiver(handler, tts)
        for _ in range(2):
            await _start(receiver)
            await receiver.receive(b"\x01\0", None)
            await receiver.stop(False)
            assert receiver.task is not None
            await receiver.task
            assert receiver.cue_task is None
        await receiver.close()
        assert tts.play_samples.call_count == 2
        assert handler.await_count == 2
        tts.close.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize(
    "reason", ["silence", "maximum_audio", "maximum_duration", "remote_end"]
)
def test_end_cue_only_on_streaming_silence(reason: str) -> None:
    """Only a streaming silence endpoint adds a second host output tone.

    Args:
        reason:
            Capture termination boundary to distinguish from silence.

    """

    async def run() -> None:
        """Keep mocked output separate from the exact endpoint-trimmed STT bytes."""
        tts = MagicMock(spec=TTS)
        handler = AsyncMock(return_value="command")
        vad = MagicMock(spec=FrameVAD)
        vad.side_effect = [0.9] * 10 + [0.1] * 16
        receiver = _receiver(
            handler,
            tts,
            endpoint=SpeechEndpoint(vad) if reason == "silence" else None,
            seconds=0.1 if reason == "maximum_audio" else 6,
        )
        await _start(receiver)
        pcm = b"\x01\0" * (512 * 26 if reason == "silence" else 1600)
        await receiver.receive(pcm, None)
        if reason == "maximum_duration":
            receiver.deadline = 0
            receiver.changed.set()
        elif reason == "remote_end":
            await receiver.stop(False)
        assert await _finish(receiver) == reason
        expected = pcm[: (10 * 512 + 8000) * 2] if reason == "silence" else pcm
        handler.assert_awaited_once_with(expected)
        assert tts.play_samples.call_count == (2 if reason == "silence" else 1)
        np.testing.assert_array_equal(
            tts.play_samples.call_args_list[0].args[0], _cue(880)
        )
        if reason == "silence":
            np.testing.assert_array_equal(
                tts.play_samples.call_args_list[1].args[0], _cue(440)
            )
        tts.close.assert_not_called()

    asyncio.run(run())


def test_receive_and_vad_continue_while_cue_blocks_then_drain_before_stt() -> None:
    """Blocked output cannot stall TCP buffering/VAD; cues drain before STT and reply."""

    async def run() -> None:
        """Hold first output in a worker while an entire command reaches its endpoint."""
        entered = threading.Event()
        release = threading.Event()
        order: list[str] = []
        tts = MagicMock(spec=TTS)

        def playback(
            samples: NDArray[np.float32],
            rate: int,
            *,
            blocking: bool,
            buffer_seconds: float,
            cancelled: threading.Event,
        ) -> None:
            """Block first tone to expose responsive input and serialized output.

            Args:
                samples:
                    Float32 mono cue waveform, never captured command PCM.

                rate:
                    Cue sample rate in Hz.

                blocking:
                    Must request accepted-output draining.

                buffer_seconds:
                    Bounded native queue duration in seconds.

                cancelled:
                    Capture-owned cancellation signal.

            """
            assert rate == 16000 and blocking and buffer_seconds == 0.2
            is_first = np.array_equal(samples, _cue(880))
            kind = "first" if is_first else "end"
            order.append(kind + "-start")
            if is_first:
                entered.set()
                assert release.wait(3)
            order.append(kind + "-drained")

        tts.play_samples.side_effect = playback
        handler = AsyncMock(side_effect=lambda pcm: order.append("stt") or "command")
        on_turn = AsyncMock(side_effect=lambda turn: order.append("reply"))
        vad = MagicMock(spec=FrameVAD, side_effect=[0.9] * 10 + [0.1] * 16)
        receiver = _receiver(handler, tts, endpoint=SpeechEndpoint(vad))
        receiver.on_turn = on_turn
        await _start(receiver)
        pcm = b"\x01\0" * 512 * 26
        await receiver.receive(pcm[:1024], None)
        assert await asyncio.to_thread(entered.wait, 3)
        try:
            await asyncio.wait_for(receiver.receive(pcm[1024:], None), 0.2)
            for _ in range(100):
                if receiver.end_reason == "silence":
                    break
                await asyncio.sleep(0)
            assert receiver.end_reason == "silence"
            assert not receiver.capturing
            assert bytes(receiver.audio) == pcm[: (10 * 512 + 8000) * 2]
            handler.assert_not_awaited()
            assert order == ["first-start"]
        finally:
            release.set()
        await _finish(receiver)
        assert order == [
            "first-start",
            "first-drained",
            "end-start",
            "end-drained",
            "stt",
            "reply",
        ]
        handler.assert_awaited_once_with(pcm[: (10 * 512 + 8000) * 2])
        tts.close.assert_not_called()

    asyncio.run(run())


def test_lifecycle_cues_drain_before_distinct_debug_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle tones precede the independent debug replay and exact-PCM STT.

    Args:
        monkeypatch:
            Replaces debug replay with a worker-only order recorder.

    """

    async def run() -> None:
        """End via silence and serialize all shared-owner output before recognition."""
        order: list[str] = []
        tts = MagicMock(spec=TTS)
        tts.play_samples.side_effect = lambda samples, *args, **kwargs: order.append(
            "first-drained" if np.array_equal(samples, _cue(880)) else "end-drained"
        )
        pcm = b"\x01\0" * 512 * 26
        trimmed = pcm[: (10 * 512 + 8000) * 2]
        loop_thread = threading.get_ident()

        def replay(data: bytes, cancelled: threading.Event, owner: TTS) -> None:
            """Require both capture cues to finish before mock debug playback begins.

            Args:
                data:
                    Exact endpoint-trimmed capture bytes.

                cancelled:
                    Capture-owned abort signal.

                owner:
                    Same caller-owned queue used by lifecycle cues.

            """
            assert owner is tts and data == trimmed
            assert not cancelled.is_set()
            assert threading.get_ident() != loop_thread
            assert order == ["first-drained", "end-drained"]
            order.append("replay-drained")

        monkeypatch.setattr(voice, "replay_capture", replay)
        handler = AsyncMock(side_effect=lambda data: order.append("stt") or "command")
        vad = MagicMock(spec=FrameVAD, side_effect=[0.9] * 10 + [0.1] * 16)
        receiver = _receiver(handler, tts, endpoint=SpeechEndpoint(vad))
        receiver.debug_audio = True
        await _start(receiver)
        await receiver.receive(pcm, None)
        await _finish(receiver)
        assert order == ["first-drained", "end-drained", "replay-drained", "stt"]
        handler.assert_awaited_once_with(trimmed)
        tts.close.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("cancel", ["none", "capture", "close"])
def test_close_joins_accepted_output_and_suppresses_further_cues(
    cancel: str,
) -> None:
    """Abort/disconnect and explicit task cancellation wait for accepted native output.

    Args:
        cancel:
            Optionally cancel capture or close itself while accepted output drains.

    """

    async def run() -> None:
        """Prevent receiver release while a fake native drain remains pending."""
        entered = threading.Event()
        release = threading.Event()
        tts = MagicMock(spec=TTS)

        def blocked(*args: object, **kwargs: object) -> None:
            """Represent accepted output that still needs joining.

            Args:
                args:
                    Native playback positional arguments.

                kwargs:
                    Native playback control arguments.

            """
            entered.set()
            assert release.wait(3)

        tts.play_samples.side_effect = blocked
        handler = AsyncMock(return_value="unexpected")
        receiver = _receiver(handler, tts)
        await _start(receiver)
        await receiver.receive(b"\x01\0", None)
        assert await asyncio.to_thread(entered.wait, 3)
        assert receiver.task is not None
        if cancel == "capture":
            receiver.task.cancel()
        closing = asyncio.create_task(receiver.close())
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not closing.done()
            assert receiver.cancelled.is_set()
            await receiver.receive(b"\x02\0", None)
            tts.play_samples.assert_called_once()
            if cancel == "close":
                closing.cancel()
                await asyncio.sleep(0)
                assert not closing.done()
        finally:
            release.set()
        results = await asyncio.gather(closing, return_exceptions=True)
        if cancel == "close":
            assert isinstance(results[0], asyncio.CancelledError)
        assert receiver.task.done()
        assert receiver.cue_task is None
        assert not receiver.audio
        handler.assert_not_awaited()
        tts.close.assert_not_called()

    asyncio.run(run())


def test_output_failure_is_logged_and_asr_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both tone failures are recoverable, visible with kind and traceback, and joined.

    Args:
        caplog:
            Captures diagnostic output without any real playback.

    """

    async def run() -> None:
        """Inject failed output at both capture boundaries but retain valid command PCM."""
        tts = MagicMock(spec=TTS)
        tts.play_samples.side_effect = RuntimeError("mock native output failed")
        handler = AsyncMock(return_value="command")
        vad = MagicMock(spec=FrameVAD, side_effect=[0.9] * 10 + [0.1] * 16)
        receiver = _receiver(handler, tts, endpoint=SpeechEndpoint(vad))
        await _start(receiver)
        await receiver.receive(b"\x01\0" * 512 * 26, None)
        await _finish(receiver)
        handler.assert_awaited_once()
        tts.close.assert_not_called()

    asyncio.run(run())
    failures = [
        r
        for r in caplog.records
        if "voice.cue" in r.message and "status=failed" in r.message
    ]
    assert len(failures) == 2
    assert "kind=first_pcm" in failures[0].message
    assert "kind=silence_end" in failures[1].message
    assert all(r.exc_info is not None for r in failures)


def test_sound_cue_owner_is_required_before_model_or_network() -> None:
    """Direct receivers opt out by default; enabled listener cues need a shared owner."""
    client = MagicMock(spec=APIClient)
    handler = AsyncMock()
    assert not VoiceReceiver(client, handler).sound_cues
    with pytest.raises(ValueError, match="Sound cues require"):
        VoiceReceiver(client, handler, sound_cues=True)
    with pytest.raises(ValueError, match="Sound cues require"):
        asyncio.run(voice.listen("unused", 6053, None, handler, 6))


@pytest.mark.parametrize("error_type", [ValueError, TypeError])
def test_cue_programming_errors_propagate(error_type: type[Exception]) -> None:
    """Only runtime output failures are recoverable; invalid calls fail loudly.

    Args:
        error_type:
            Invalid argument or programming error raised by the playback boundary.

    """
    tts = MagicMock(spec=TTS)
    tts.play_samples.side_effect = error_type("invalid output contract")
    with pytest.raises(error_type, match="invalid output contract"):
        play_capture_cue("first_pcm", threading.Event(), tts)


@pytest.mark.parametrize("kind", ["invalid"])
def test_unknown_cue_rejected(kind: CaptureCue) -> None:
    """Reject an invalid runtime cue label instead of silently choosing a tone.

    Args:
        kind:
            Deliberately invalid label injected by the untyped parametrization boundary.

    """
    tts = MagicMock(spec=TTS)
    with pytest.raises(ValueError, match="Unknown capture cue"):
        play_capture_cue(kind, threading.Event(), tts)
    tts.play_samples.assert_not_called()


def test_reconnect_waits_for_output_before_new_receiver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost transport joins first-cue output before disconnect/reconnect reuse.

    Args:
        monkeypatch:
            Replaces native transport and endpoint/model construction.

    """

    async def run() -> None:
        """Drive a mocked native disconnect while the shared output queue is blocked."""
        entered = threading.Event()
        release = threading.Event()
        first_connected = asyncio.Event()
        tts = MagicMock(spec=TTS)
        tts.play_samples.side_effect = lambda *args, **kwargs: (
            entered.set(),
            release.wait(3),
        )
        handler = AsyncMock(return_value="unexpected")
        client = MagicMock(spec=APIClient)
        client.device_info = AsyncMock(
            return_value=MagicMock(
                voice_assistant_feature_flags=VoiceAssistantFeature.API_AUDIO
            )
        )
        client.disconnect = AsyncMock()
        stop_callbacks: list[Callable[[bool], Awaitable[None]]] = []

        async def connect(
            *, on_stop: Callable[[bool], Awaitable[None]], login: bool
        ) -> None:
            """Capture the library's transport-loss callback without opening sockets.

            Args:
                on_stop:
                    Bound transport-loss callback for this connection.

                login:
                    Native authentication flag, unused by this mock.

            """
            stop_callbacks.append(on_stop)
            first_connected.set()

        client.connect = AsyncMock(side_effect=connect)
        factory = MagicMock(
            side_effect=[client, RuntimeError("mock reconnect reached")]
        )
        monkeypatch.setattr(voice, "APIClient", factory)
        monkeypatch.setattr(voice, "SpeechEndpoint", MagicMock(return_value=None))
        listening = asyncio.create_task(
            voice.listen("unused", 6053, None, handler, 6, replay_tts=tts)
        )
        await first_connected.wait()
        callbacks = client.subscribe_voice_assistant.call_args.kwargs
        await callbacks["handle_start"]("", 0, VoiceAssistantAudioSettings(), None)
        await callbacks["handle_audio"](b"\x01\0", None)
        assert await asyncio.to_thread(entered.wait, 3)
        await stop_callbacks[0](False)
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            client.disconnect.assert_not_awaited()
            assert factory.call_count == 1
        finally:
            release.set()
        for _ in range(100):
            if client.disconnect.await_count:
                break
            await asyncio.sleep(0.001)
        client.disconnect.assert_awaited_once()
        with pytest.raises(RuntimeError, match="mock reconnect reached"):
            await asyncio.wait_for(listening, 4)
        assert factory.call_count == 2
        handler.assert_not_awaited()
        tts.close.assert_not_called()

    asyncio.run(run())
