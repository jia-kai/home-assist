"""Direct ESPHome voice capture with bounded recording and reusable STT handling."""

import argparse
import asyncio
import math
import os
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from aioesphomeapi.client import APIClient
from aioesphomeapi.core import APIConnectionError
from aioesphomeapi.model import (
    VoiceAssistantAudioSettings,
    VoiceAssistantCommandFlag,
    VoiceAssistantFeature,
)
from aioesphomeapi.model import VoiceAssistantEventType as Event

from .endpoint import FRAME_BYTES, SpeechEndpoint
from .logging import configure_logging, get_logger
from .runtime import configure_cpu_budget
from .stt import STT, STTConfig
from .tts import TTS
from .voice_debug import replay_capture

logger = get_logger(__name__)
type AudioHandler = Callable[[bytes], Awaitable[str]]


@dataclass(slots=True, frozen=True)
class VoiceTurn:
    """Recognized command and capture-time boundaries for an awaited agent turn."""

    text: str
    """Nonempty STT transcript; no tool execution has occurred."""

    started_at: float
    """Monotonic seconds when the first nonempty PCM chunk reached the controller."""

    ended_at: float
    """Monotonic seconds when capture ended, before STT or agent processing."""

    cancelled: threading.Event
    """Thread-safe signal set on abort or disconnect; dispatched effects persist."""


type TurnHandler = Callable[[VoiceTurn], Awaitable[None]]


@dataclass(slots=True)
class VoiceReceiver:
    """Serialize satellite captures and pass PCM to an asynchronous transcript handler.

    All methods run on one asyncio loop. The native listener supplies a streaming
    endpoint that finishes after 600 ms of silence following speech. Wall-clock
    and PCM-duration caps still bound silent, stalled, or continuously voiced input.
    Stop/cancel/disconnect discards incomplete audio. An already-running handler
    is awaited during shutdown; its external side effects cannot be cancelled.
    """

    client: APIClient
    """Connected native-API client; audio transport is TCP API_AUDIO."""

    handler: AudioHandler
    """Accept mono 16 kHz signed little-endian int16 PCM; return transcript text."""

    seconds: float = 6.0
    """Maximum capture duration in seconds, between 0.1 and 30 inclusive."""

    on_transcript: Callable[[str], None] | None = None
    """Optional nonblocking sink, called only for an uncancelled nonempty transcript."""

    on_turn: TurnHandler | None = None
    """Optional awaited agent/TTS hook; the satellite stays busy until it returns."""

    endpoint: SpeechEndpoint | None = None
    """Streaming detector supplied by listen; None permits detector-free test transports."""

    debug_audio: bool = False
    """Replay the captured command between cues before STT when explicitly enabled."""

    replay_tts: TTS | None = None
    """Caller-owned TTS engine shared by replay and spoken replies; never closed here."""

    changed: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    """Signals newly buffered PCM or capture termination to the endpoint consumer."""

    deadline: float = field(default=0.0, init=False)
    """Absolute monotonic hard deadline established at wake-request acceptance."""

    end_reason: str = field(default="maximum_duration", init=False)
    """Diagnostic reason for ending capture: silence, maximum, remote end or abort."""

    started_at: float | None = field(default=None, init=False)
    """First PCM arrival in monotonic seconds, or None until audio arrives."""

    cancelled: threading.Event = field(default_factory=threading.Event, init=False)
    """Cancellation signal for this capture and its optional agent/TTS hook."""

    task: asyncio.Task[None] | None = field(default=None, init=False)
    """Owned capture/handler task, awaited before releasing this receiver."""

    audio: bytearray = field(default_factory=bytearray, init=False)
    """Bounded command PCM, at most seconds times 32000 bytes."""

    done: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    """Capture ended by silence, a maximum bound, remote stop, or shutdown."""

    capturing: bool = field(default=False, init=False)
    """Whether incoming PCM belongs to the current command."""

    aborted: bool = field(default=False, init=False)
    """Whether the current command must be discarded."""

    connected: bool = field(default=True, init=False)
    """Whether event responses may still be sent to the satellite."""

    def __post_init__(self) -> None:
        """Reject unbounded or nonfinite capture windows."""
        if not math.isfinite(self.seconds) or not 0.1 <= self.seconds <= 30:
            raise ValueError("capture seconds must be between 0.1 and 30")
        if self.debug_audio and self.replay_tts is None:
            raise ValueError("Debug replay requires a shared TTS instance")

    def event(self, kind: Event, data: dict[str, str] | None = None) -> None:
        """Send a pipeline event only while connected.

        Args:
            kind:
                Native ESPHome voice-assistant event type.

            data:
                Optional protocol event arguments.

        """
        if self.connected:
            logger.debug("voice.event type=%s data=%r", kind.name, data)
            try:
                self.client.send_voice_assistant_event(kind, data)
            except APIConnectionError:
                logger.warning("voice.event status=disconnected", exc_info=True)
                self.connected = False
                self.aborted = True
                self.cancelled.set()

    async def start(
        self,
        conversation_id: str,
        flags: int,
        audio_settings: VoiceAssistantAudioSettings,
        wake_word: str | None,
    ) -> int | None:
        """Accept a local-wake request; return zero for TCP audio, None on rejection.

        Args:
            conversation_id:
                Satellite conversation reference, recorded in diagnostics.

            flags:
                Native command flags; server-side wake detection is unsupported.

            audio_settings:
                Requested host audio effects, logged but not applied.

            wake_word:
                Detected phrase, or None for a button-triggered capture.

        """
        logger.info(
            "voice.start wake=%r flags=%d conversation=%r",
            wake_word,
            flags,
            conversation_id,
        )
        logger.debug("voice.start audio_settings=%r", audio_settings)
        if (self.task is not None and not self.task.done()) or not self.connected:
            logger.warning("voice.start status=rejected reason=busy_or_disconnected")
            return None
        if flags & VoiceAssistantCommandFlag.USE_WAKE_WORD:
            logger.warning("voice.start status=rejected reason=server_wake_requested")
            return None
        self.audio.clear()
        self.done.clear()
        self.changed.clear()
        if self.endpoint is not None:
            self.endpoint.reset()
        self.deadline = time.monotonic() + self.seconds
        self.end_reason = "maximum_duration"
        self.aborted = False
        self.cancelled = threading.Event()
        self.started_at = None
        self.capturing = True
        self.task = asyncio.create_task(self._capture())
        return 0

    async def receive(self, data: bytes, data2: bytes | None) -> None:
        """Buffer primary-channel PCM for the endpoint consumer; reject partial samples.

        Args:
            data:
                Mono 16 kHz signed little-endian int16 PCM bytes.

            data2:
                Optional echo-reference channel; unused by this mono receiver.

        """
        if not self.capturing:
            return
        if time.monotonic() >= self.deadline:
            self.capturing = False
            self.end_reason = "maximum_duration"
            self.done.set()
            self.changed.set()
            return
        if data and self.started_at is None:
            self.started_at = time.monotonic()
        if len(data) % 2:
            logger.warning("voice.audio status=rejected reason=partial_sample")
            self.aborted = True
            self.cancelled.set()
            self.done.set()
            self.changed.set()
            self.capturing = False
            return
        remaining = int(self.seconds * 16000) * 2 - len(self.audio)
        self.audio.extend(data[:remaining])
        self.changed.set()
        if len(data) >= remaining:
            self.end_reason = "maximum_audio"
            self.capturing = False
            self.done.set()

    async def stop(self, abort: bool) -> None:
        """Finish or abort capture on the satellite's stop notification.

        Args:
            abort:
                True discards the command; False accepts already-received PCM.

        """
        logger.debug("voice.stop abort=%s", abort)
        if self.capturing or abort:
            self.end_reason = "aborted" if abort else "remote_end"
        self.aborted |= abort
        if abort:
            self.cancelled.set()
        self.capturing = False
        self.done.set()
        self.changed.set()

    async def close(self) -> None:
        """Suppress further events, cancel pending work, and await owned STT/agent work."""
        self.connected = False
        await self.stop(True)
        if self.task is not None:
            await self.task

    async def _wait_for_endpoint(self) -> None:
        """Process at most four VAD frames per loop slice until silence, cap or stop.

        Audio stays in a bounded buffer. Only this task mutates detector state;
        incoming callbacks append PCM without doing inference or awaiting work.
        The deadline freezes incoming audio, but already received PCM is fully
        classified so a burst containing an earlier endpoint is trimmed correctly.
        Partial VAD frames remain unpadded on remote stop or maximum duration.
        """
        processed = 0
        while not self.aborted:
            if self.endpoint is not None and processed < len(self.audio):
                stop = min(processed + 4 * FRAME_BYTES, len(self.audio))
                end_sample = self.endpoint.feed(bytes(self.audio[processed:stop]))
                processed = stop
                if end_sample is not None:
                    del self.audio[end_sample * 2 :]
                    self.end_reason = "silence"
                    break
                # Yield between bounded inference batches so network controls stay live.
                await asyncio.sleep(0)
                if time.monotonic() >= self.deadline and not self.done.is_set():
                    self.end_reason = "maximum_duration"
                    self.capturing = False
                    self.done.set()
                continue
            if self.done.is_set():
                break
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                break
            self.changed.clear()
            try:
                await asyncio.wait_for(self.changed.wait(), timeout=remaining)
            except TimeoutError:
                break
        self.capturing = False
        self.done.set()

    async def _capture(self) -> None:
        """Complete adaptive capture, optional replay, STT and agent/TTS before rearming."""
        try:
            self.event(Event.VOICE_ASSISTANT_RUN_START)
            self.event(Event.VOICE_ASSISTANT_STT_START)
            await self._wait_for_endpoint()
            ended_at = time.monotonic()
            self.event(Event.VOICE_ASSISTANT_STT_VAD_END)
            if self.aborted or not self.audio:
                self.event(
                    Event.VOICE_ASSISTANT_ERROR,
                    {
                        "code": "stt-no-text-recognized",
                        "message": "I didn't catch that.",
                    },
                )
                return
            pcm = bytes(self.audio)
            assert self.started_at is not None
            self.audio.clear()
            logger.info(
                "voice.capture status=ok bytes=%d seconds=%.3f reason=%s",
                len(pcm),
                len(pcm) / 32000,
                self.end_reason,
            )
            if self.debug_audio:
                assert self.replay_tts is not None
                await asyncio.to_thread(
                    replay_capture, pcm, self.cancelled, self.replay_tts
                )
            if self.cancelled.is_set():
                return
            text = await self.handler(pcm)
            if not self.aborted:
                if text.strip():
                    self.event(Event.VOICE_ASSISTANT_STT_END, {"text": text})
                    if self.connected and self.on_transcript is not None:
                        self.on_transcript(text)
                    if self.connected and not self.aborted and self.on_turn is not None:
                        self.event(Event.VOICE_ASSISTANT_INTENT_START)
                        await self.on_turn(
                            VoiceTurn(text, self.started_at, ended_at, self.cancelled)
                        )
                        if not self.aborted:
                            self.event(Event.VOICE_ASSISTANT_INTENT_END)
                else:
                    self.event(
                        Event.VOICE_ASSISTANT_ERROR,
                        {
                            "code": "stt-no-text-recognized",
                            "message": "I didn't catch that.",
                        },
                    )
        except Exception:
            logger.exception("voice.capture status=failed")
            self.event(
                Event.VOICE_ASSISTANT_ERROR,
                {"code": "unknown", "message": "I couldn't process that."},
            )
        finally:
            self.capturing = False
            self.audio.clear()
            self.event(Event.VOICE_ASSISTANT_RUN_END)


async def listen(
    host: str,
    port: int,
    key: str | None,
    receiver_handler: AudioHandler,
    seconds: float,
    on_transcript: Callable[[str], None] | None = None,
    on_turn: TurnHandler | None = None,
    debug_audio: bool = False,
    replay_tts: TTS | None = None,
) -> None:
    """Connect directly to a satellite and reconnect after native-API transport loss.

    Args:
        host:
            Satellite hostname or address; simulator defaults to loopback.

        port:
            Native API TCP port, normally 6053.

        key:
            Optional ESPHome Noise encryption key; never logged.

        receiver_handler:
            Asynchronous mono PCM-to-transcript handler, serialized across reconnects.

        seconds:
            Maximum capture duration; streaming VAD can finish earlier after silence.

        on_transcript:
            Optional nonblocking sink for uncancelled, nonempty transcripts.

        on_turn:
            Optional awaited agent/TTS hook invoked after STT; RUN_END waits for it.

        debug_audio:
            Replay captured PCM between cues before STT, retaining satellite busy state.

        replay_tts:
            Caller-owned shared TTS engine, required for debug replay. This listener
            retains it across reconnects but does not close it.

    """
    if debug_audio and replay_tts is None:
        raise ValueError("Debug replay requires a shared TTS instance")
    endpoint = SpeechEndpoint()
    logger.info("voice.endpoint model=silero_v6 silence_ms=600 frame_ms=32")
    while True:
        client = APIClient(host, port, password="", noise_psk=key)
        receiver = VoiceReceiver(
            client,
            receiver_handler,
            seconds,
            on_transcript,
            on_turn,
            endpoint,
            debug_audio,
            replay_tts,
        )
        disconnected = asyncio.Event()

        async def on_stop(
            expected: bool,
            receiver: VoiceReceiver = receiver,
            disconnected: asyncio.Event = disconnected,
        ) -> None:
            """Wake reconnect logic immediately on transport loss.

            Args:
                expected:
                    Whether the native client expected this disconnection.

                receiver:
                    Receiver bound to this particular connection attempt.

                disconnected:
                    Completion event bound to this connection attempt.

            """
            logger.warning("voice.connection status=closed expected=%s", expected)
            receiver.connected = False
            receiver.aborted = True
            receiver.cancelled.set()
            receiver.done.set()
            receiver.changed.set()
            disconnected.set()

        try:
            await client.connect(on_stop=on_stop, login=True)
            info = await client.device_info()
            features = info.voice_assistant_feature_flags
            if not features & VoiceAssistantFeature.API_AUDIO:
                raise ValueError("Satellite must support native API audio transport")
            client.subscribe_voice_assistant(
                handle_start=receiver.start,
                handle_stop=receiver.stop,
                handle_audio=receiver.receive,
            )
            logger.info("voice.connection status=ready host=%s port=%d", host, port)
            await disconnected.wait()
        except APIConnectionError:
            logger.warning("voice.connection status=retry host=%s", host, exc_info=True)
        finally:
            await receiver.close()
            await client.disconnect()
        await asyncio.sleep(2)


def main() -> None:
    """Emit one adaptive-capture transcript per wake for piping into hoast --text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6053)
    parser.add_argument(
        "--seconds",
        type=float,
        default=6.0,
        help="Maximum capture duration; silence ends it earlier",
    )
    parser.add_argument(
        "--debug-audio",
        action="store_true",
        help="Replay captured commands between tones before STT",
    )
    parser.add_argument("--key-env", default="ESPHOME_API_KEY")
    parser.add_argument("--language", default="en", help="Whisper language or auto")
    args = parser.parse_args()
    configure_logging(log_file=Path(".cache/hoast/diagnostics/voice.log"))
    get_logger("aioesphomeapi").setLevel("INFO")
    try:
        if not math.isfinite(args.seconds) or not 0.1 <= args.seconds <= 30:
            raise ValueError("capture seconds must be between 0.1 and 30")
        configure_cpu_budget(2)
        stt = STT(
            STTConfig(language=None if args.language == "auto" else args.language)
        )

        async def transcribe(pcm: bytes) -> str:
            """Run STT off the network loop and return the transcript.

            Args:
                pcm:
                    Mono 16 kHz signed little-endian int16 PCM command audio.

            """
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            return await asyncio.to_thread(stt.transcribe_samples, samples, 16000)

        def output(text: str) -> None:
            """Write one uncancelled transcript for a downstream agent process.

            Args:
                text:
                    Nonempty recognized command; whitespace is collapsed to one line.

            """
            sys.stdout.write(" ".join(text.split()) + "\n")
            sys.stdout.flush()

        replay_tts = TTS(playback_only=True) if args.debug_audio else None
        try:
            asyncio.run(
                listen(
                    args.host,
                    args.port,
                    os.environ.get(args.key_env),
                    transcribe,
                    args.seconds,
                    output,
                    debug_audio=args.debug_audio,
                    replay_tts=replay_tts,
                )
            )
        finally:
            if replay_tts is not None:
                replay_tts.close()
    except KeyboardInterrupt:
        logger.info("voice.run status=stopped")
    except Exception:
        logger.exception("voice.run status=failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
