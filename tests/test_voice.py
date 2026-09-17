"""Hardware-independent native voice lifecycle and bounded PCM tests."""

import asyncio
from unittest.mock import MagicMock

from aioesphomeapi.client import APIClient
from aioesphomeapi.model import VoiceAssistantAudioSettings

from hoast.endpoint import FrameVAD, SpeechEndpoint
from hoast.voice import Event, VoiceReceiver


class Client(APIClient):
    """Record outgoing native voice events without opening a connection."""

    events: list[tuple[Event, dict[str, str] | None]]
    """Ordered protocol events emitted by the receiver."""

    def __init__(self) -> None:
        """Initialize isolated event storage for each test."""
        self.events = []

    def send_voice_assistant_event(
        self, event_type: Event, data: dict[str, str] | None
    ) -> None:
        """Record a native event for lifecycle assertions.

        Args:
            event_type:
                Native pipeline event type.

            data:
                Event arguments, including transcript or error code.

        """
        self.events.append((event_type, data))


def test_bounded_capture_and_rearm() -> None:
    """Two captures retain bounded PCM, reject overlap, and complete in order."""

    async def run() -> None:
        """Exercise the public receiver callbacks in one event loop."""
        client = Client()
        received: list[bytes] = []

        async def handler(pcm: bytes) -> str:
            """Record the exact bounded command.

            Args:
                pcm:
                    Primary-channel mono PCM bytes.

            """
            received.append(pcm)
            return "test command"

        receiver = VoiceReceiver(client, handler, seconds=0.1)
        for _ in range(2):
            assert (
                await receiver.start("", 0, VoiceAssistantAudioSettings(), "Okay Nabu")
                == 0
            )
            assert (
                await receiver.start("", 0, VoiceAssistantAudioSettings(), None) is None
            )
            await receiver.receive(b"\x01\x00" * 4000, b"ignored echo reference")
            assert len(receiver.audio) == 3200
            assert receiver.task is not None
            await receiver.task
        assert received == [b"\x01\x00" * 1600] * 2
        assert [kind for kind, _ in client.events] == [
            Event.VOICE_ASSISTANT_RUN_START,
            Event.VOICE_ASSISTANT_STT_START,
            Event.VOICE_ASSISTANT_STT_VAD_END,
            Event.VOICE_ASSISTANT_STT_END,
            Event.VOICE_ASSISTANT_RUN_END,
        ] * 2
        await receiver.close()

    asyncio.run(run())


def test_timeout_abort_malformed_and_handler_failure() -> None:
    """Empty windows, cancellation, bad PCM, and failed handlers all rearm."""

    async def run() -> None:
        """Exercise failure paths and assert no incomplete audio is transcribed."""
        client = Client()
        calls = 0

        async def handler(pcm: bytes) -> str:
            """Count valid captures and inject a transcription failure.

            Args:
                pcm:
                    Captured mono PCM bytes.

            """
            nonlocal calls
            calls += 1
            raise RuntimeError("test transcription failure")

        receiver = VoiceReceiver(client, handler, seconds=0.1)
        for scenario in ("timeout", "abort", "malformed", "handler"):
            assert await receiver.start("", 0, VoiceAssistantAudioSettings(), None) == 0
            if scenario == "abort":
                await receiver.receive(b"\0\0", None)
                await receiver.stop(True)
            elif scenario == "malformed":
                await receiver.receive(b"\0", None)
            elif scenario == "handler":
                await receiver.receive(b"\0\0", None)
                await receiver.stop(False)
            assert receiver.task is not None
            await receiver.task
            assert client.events[-1][0] == Event.VOICE_ASSISTANT_RUN_END
            assert client.events[-2][0] == Event.VOICE_ASSISTANT_ERROR
            assert not receiver.audio
        assert calls == 1
        await receiver.close()

    asyncio.run(run())


def test_disconnect_discards_recording() -> None:
    """Disconnect neither transcribes partial audio nor sends to a closed client."""

    async def run() -> None:
        """Close during capture before the deadline."""
        client = Client()

        async def handler(pcm: bytes) -> str:
            """Reject any accidental invocation for a disconnected session.

            Args:
                pcm:
                    Incomplete command audio that must be discarded.

            """
            raise AssertionError("must not transcribe")

        receiver = VoiceReceiver(client, handler)
        await receiver.start("", 0, VoiceAssistantAudioSettings(), None)
        await receiver.receive(b"\0\0", None)
        await receiver.close()
        assert not client.events
        assert not receiver.audio

    asyncio.run(run())


def test_disconnect_waits_for_stt_without_publishing() -> None:
    """Await an in-flight recognizer but discard its transcript after disconnect."""

    async def run() -> None:
        """Disconnect after transcription starts and before it completes."""
        client = Client()
        started = asyncio.Event()
        finish = asyncio.Event()
        published: list[str] = []

        async def handler(pcm: bytes) -> str:
            """Hold transcription until the connection has been closed.

            Args:
                pcm:
                    Complete command PCM received before disconnection.

            """
            started.set()
            await finish.wait()
            return "must not publish"

        receiver = VoiceReceiver(client, handler, 0.1, published.append)
        await receiver.start("", 0, VoiceAssistantAudioSettings(), None)
        await receiver.receive(b"\0\0" * 1600, None)
        await started.wait()
        closing = asyncio.create_task(receiver.close())
        await asyncio.sleep(0)
        assert not closing.done()
        finish.set()
        await closing
        assert not published
        assert not any(
            kind == Event.VOICE_ASSISTANT_STT_END for kind, _ in client.events
        )

    asyncio.run(run())


def test_streaming_silence_starts_stt_early() -> None:
    """Silence starts STT early and emits the satellite processing cue first."""

    async def run() -> None:
        """Verify exact retained PCM, native stop events, and detector reset on rearm."""
        client = Client()
        received: list[bytes] = []

        async def handler(pcm: bytes) -> str:
            """Observe the unmodified PCM prefix selected by the endpoint.

            Args:
                pcm:
                    Endpoint-trimmed command audio.

            """
            received.append(pcm)
            return "test command"

        vad = MagicMock(spec=FrameVAD)
        endpoint = SpeechEndpoint(vad)
        receiver = VoiceReceiver(client, handler, seconds=6, endpoint=endpoint)
        pcm = b"\x01\0" * 512 * 35
        for _ in range(2):
            vad.side_effect = [0.9] * 10 + [0.1] * 16
            await receiver.start("", 0, VoiceAssistantAudioSettings(), None)
            await receiver.receive(pcm, None)
            assert receiver.task is not None
            await asyncio.wait_for(receiver.task, timeout=1)
            assert receiver.end_reason == "silence"
        assert received == [pcm[: (10 * 512 + 8000) * 2]] * 2
        assert vad.reset.call_count == 2
        assert [kind for kind, _ in client.events].count(
            Event.VOICE_ASSISTANT_STT_VAD_END
        ) == 2
        assert [kind for kind, _ in client.events].count(
            Event.VOICE_ASSISTANT_INTENT_START
        ) == 2
        for start in range(0, len(client.events), 6):
            assert [kind for kind, _ in client.events][start : start + 5] == [
                Event.VOICE_ASSISTANT_RUN_START,
                Event.VOICE_ASSISTANT_STT_START,
                Event.VOICE_ASSISTANT_STT_VAD_END,
                Event.VOICE_ASSISTANT_INTENT_START,
                Event.VOICE_ASSISTANT_STT_END,
            ]
        await receiver.close()

    asyncio.run(run())


def test_vad_failure_aborts_without_transcription() -> None:
    """An endpoint failure ends the native run and does not strand busy state."""

    async def run() -> None:
        """Inject an inference failure during capture and verify the error lifecycle."""
        client = Client()

        async def handler(pcm: bytes) -> str:
            """Reject transcription of a capture whose VAD failed.

            Args:
                pcm:
                    Untrusted partial capture that must be discarded.

            """
            raise AssertionError("must not transcribe")

        vad = MagicMock(spec=FrameVAD, side_effect=RuntimeError("VAD failure"))
        receiver = VoiceReceiver(client, handler, endpoint=SpeechEndpoint(vad))
        await receiver.start("", 0, VoiceAssistantAudioSettings(), None)
        await receiver.receive(b"\0\0" * 512, None)
        assert receiver.task is not None
        await receiver.task
        assert client.events[-2][0] == Event.VOICE_ASSISTANT_ERROR
        assert client.events[-1][0] == Event.VOICE_ASSISTANT_RUN_END
        assert not receiver.capturing
        await receiver.close()

    asyncio.run(run())


def test_no_speech_maximum_keeps_partial_frame_unpadded() -> None:
    """A silent capture reaches its PCM cap without inventing extra VAD/audio samples."""

    async def run() -> None:
        """Deliver more than the cap and verify the retained tail is neither padded nor lost."""
        received: list[bytes] = []

        async def handler(pcm: bytes) -> str:
            """Retain the exact bounded input for assertions.

            Args:
                pcm:
                    Silent capture ending with a partial 32 ms VAD frame.

            """
            received.append(pcm)
            return ""

        vad = MagicMock(spec=FrameVAD, return_value=0.1)
        receiver = VoiceReceiver(
            Client(), handler, seconds=0.1, endpoint=SpeechEndpoint(vad)
        )
        await receiver.start("", 0, VoiceAssistantAudioSettings(), None)
        await receiver.receive(b"\0\0" * 4000, None)
        assert receiver.task is not None
        await asyncio.wait_for(receiver.task, 1)
        assert received == [b"\0\0" * 1600]
        assert vad.call_count == 3
        assert receiver.end_reason == "maximum_audio"
        await receiver.close()

    asyncio.run(run())


def test_deadline_during_vad_still_trims_buffered_endpoint() -> None:
    """A wall-clock cap freezes input without losing an earlier endpoint in queued PCM."""

    async def run() -> None:
        """Expire the deadline after four frames of an already buffered audio burst."""
        received: list[bytes] = []

        async def handler(pcm: bytes) -> str:
            """Retain the trimmed command selected from pre-deadline buffered audio.

            Args:
                pcm:
                    Expected prefix ending at the first 500 ms silence endpoint.

            """
            received.append(pcm)
            return "test command"

        vad = MagicMock(spec=FrameVAD)
        receiver = VoiceReceiver(
            Client(), handler, seconds=6, endpoint=SpeechEndpoint(vad)
        )
        calls = 0

        def probability(pcm: bytes) -> float:
            """Simulate deadline expiry while processing audio that arrived in time.

            Args:
                pcm:
                    One fixed-size frame from the pending audio burst.

            """
            nonlocal calls
            calls += 1
            if calls == 4:
                receiver.deadline = 0.0
            return 0.9 if calls <= 10 else 0.1

        vad.side_effect = probability
        pcm = b"\x01\0" * 512 * 35
        await receiver.start("", 0, VoiceAssistantAudioSettings(), None)
        await receiver.receive(pcm, None)
        assert receiver.task is not None
        await asyncio.wait_for(receiver.task, 1)
        assert receiver.end_reason == "silence"
        assert received == [pcm[: (10 * 512 + 8000) * 2]]
        await receiver.close()

    asyncio.run(run())
