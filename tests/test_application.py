"""Mocked complete voice pipeline: real orchestration, sessions, tools and events."""

import asyncio
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from aioesphomeapi.client import APIClient
from aioesphomeapi.model import VoiceAssistantAudioSettings
from numpy.typing import NDArray

from hoast import llm_cli
from hoast.agent import LocalAgent
from hoast.application import Assistant, VoiceApplication, bounded_speech
from hoast.config import SatelliteConfig, SwitchConfig
from hoast.lights import LightClient
from hoast.llm import FunctionGemma, Generation, LLMConfig, Tool, ToolCall, ToolRegistry
from hoast.llm_cli import run_voice
from hoast.music import MusicAssistantError, MusicClient
from hoast.session import Session
from hoast.stt import STT
from hoast.tts import TTS
from hoast.voice import AudioHandler, Event, TurnHandler, VoiceReceiver, VoiceTurn
from hoast.weather import WeatherArguments


@dataclass(slots=True)
class Pipeline:
    """Fixture that mocks only models, devices, and external service observations."""

    app: VoiceApplication
    """Real serialized voice application."""

    model: MagicMock
    """Weight-free model using a real configuration and tool registry."""

    stt: MagicMock
    """Speech recognizer with shape/rate assertions."""

    tts: MagicMock
    """Bilingual synthesis and playback mock."""

    music: MagicMock
    """Read-only music context and tool-result mock."""

    timeline: list[str]
    """Ordered inference, service, and playback operations."""

    output: list[str]
    """Grounded console lines."""

    histories: list[list[dict[str, Any]]]
    """Real session history submitted to each mocked model inference."""


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Pipeline:
    """Construct real agent/session/registry around mock external boundaries.

    Args:
        monkeypatch:
            Replaces Kasa relay access with a verified offline operation.

        tmp_path:
            Unused model root; no downloads or local model paths are required.

    """
    timeline: list[str] = []
    output: list[str] = []
    histories: list[list[dict[str, Any]]] = []
    music = MagicMock(spec=MusicClient)
    music.prompt_context.return_value = {
        "status": "observed",
        "state": "off",
        "title": "Current Song",
        "artist": "Current Artist",
    }
    music.resume_music.return_value = {
        "status": "resumed",
        "confirmation": "observed",
        "now_playing": {"title": "Current Song", "artist": "Current Artist"},
    }

    async def power(config: SwitchConfig, on: bool) -> None:
        """Stand in for the relay's verified state change.

        Args:
            config:
                Documentation-only fixture IP address.

            on:
                Requested power state.

        """
        timeline.append(f"light:{on}")

    monkeypatch.setattr("hoast.lights._set_power", AsyncMock(side_effect=power))
    tools = ToolRegistry(
        [
            Tool("get_weather", "Weather fixture", WeatherArguments, lambda args: None),
            LightClient(SwitchConfig("192.0.2.20")).tool(),
            *MusicClient.tools(music),
        ]
    )
    model = MagicMock(spec=FunctionGemma)
    model.config = LLMConfig(tmp_path)
    model.tools = tools

    def generate(
        messages: Sequence[dict[str, Any]],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Generation:
        """Route using mock inference while retaining real validation and dispatch.

        Args:
            messages:
                Real conversation history passed by Session.

            on_text:
                Optional model streaming callback, unused by this fixture.

        """
        timeline.append("llm")
        histories.append(list(messages))
        call = (
            ToolCall("resume_music", {})
            if "music" in messages[-1]["content"]
            else ToolCall("set_light", {"on": True})
        )
        return Generation("", (call,), "", 1, 1, 0.01, 0.001)

    def warm_model(text: str) -> Generation:
        """Propose a side-effecting tool during warm-up to verify it is not executed.

        Args:
            text:
                Synthetic warm-up prompt.

        """
        timeline.append("llm-warm")
        return Generation(
            "", (ToolCall("set_light", {"on": True}),), "", 1, 1, 0.01, 0.001
        )

    model.generate_messages.side_effect = generate
    model.generate.side_effect = warm_model
    stt = MagicMock(spec=STT)
    tts = MagicMock(spec=TTS)

    def synthesize(text: str) -> tuple[NDArray[np.float32], int]:
        """Generate small distinct in-memory English/Chinese waveforms.

        Args:
            text:
                Warm-up phrase selecting the fixture waveform.

        """
        chinese = "你好" in text
        timeline.append("tts-zh" if chinese else "tts-en")
        return np.full(240, 0.2 if chinese else 0.1, dtype=np.float32), 24000

    def transcribe(samples: NDArray[np.float32], sample_rate: int) -> str:
        """Verify dtype, mono shape, and warm-up versus captured-audio rates.

        Args:
            samples:
                Mono float32 generated or captured waveform.

            sample_rate:
                Native sample rate, 24 kHz for TTS and 16 kHz for input PCM.

        """
        assert samples.ndim == 1 and samples.dtype == np.float32
        assert sample_rate in (16000, 24000)
        timeline.append("stt-warm" if sample_rate == 24000 else "stt")
        return "Please switch the light on"

    def play(text: str, *, blocking: bool = True) -> None:
        """Record playback only after generation/tool completion and enforce bounds.

        Args:
            text:
                Bounded grounded speech passed to synthesis/playback.

            blocking:
                Must remain true to keep the satellite busy through device drain.

        """
        assert blocking and 0 < len(text) <= 400
        timeline.append("play:" + text)

    tts.synthesize.side_effect = synthesize
    tts.play.side_effect = play
    stt.transcribe_samples.side_effect = transcribe
    app = VoiceApplication(
        Assistant(LocalAgent(Session(model)), music), stt, tts, output.append
    )
    return Pipeline(app, model, stt, tts, music, timeline, output, histories)


@pytest.mark.parametrize("debug_audio", [False, True])
def test_mocked_complete_pipeline(
    pipeline: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    debug_audio: bool,
) -> None:
    """Run real startup, native callbacks, STT, LLM/light tool, and draining TTS twice.

    Args:
        pipeline:
            Weight-free external boundaries and real orchestration.

        monkeypatch:
            Substitutes constructors and the network transport with fixtures.

        tmp_path:
            Isolated artifact root accepted by mocked constructors.

        debug_audio:
            Whether debug replay must use the same mocked TTS instance as responses.

    """
    tts_factory = MagicMock(return_value=pipeline.tts)
    monkeypatch.setattr("hoast.llm_cli.TTS", tts_factory)
    monkeypatch.setattr("hoast.llm_cli.STT", MagicMock(return_value=pipeline.stt))
    events: list[Event] = []

    def event(kind: Event, data: dict[str, str] | None) -> None:
        """Observe protocol completion after system playback has drained.

        Args:
            kind:
                Native voice pipeline event.

            data:
                Event payload such as the STT transcript.

        """
        if kind == Event.VOICE_ASSISTANT_RUN_END:
            assert pipeline.timeline[-1] == "play:Light on."
        events.append(kind)

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
        """Replace network I/O with two full native voice sessions and real callbacks.

        Args:
            host:
                Configured fixture satellite host.

            port:
                Configured fixture native API port.

            key:
                Fixture encryption key passed separately from TOML.

            receiver_handler:
                Actual worker-backed STT callback supplied by run_voice.

            seconds:
                Configured capture duration in seconds.

            on_transcript:
                Optional text sink from the native listener.

            on_turn:
                Actual awaited worker-backed agent/TTS callback.

            debug_audio:
                Whether this invocation explicitly requests command replay.

            replay_tts:
                Existing application TTS engine passed through the native listener.

        """
        assert (host, port, key, seconds) == (
            "mock-satellite",
            6123,
            "fixture-key",
            0.1,
        )
        assert pipeline.timeline == [
            "tts-en",
            "tts-zh",
            "stt-warm",
            "stt-warm",
            "llm-warm",
        ]
        assert not pipeline.app.assistant.agent.session.history
        assert replay_tts is pipeline.tts
        client = MagicMock(spec=APIClient)
        client.send_voice_assistant_event.side_effect = event
        receiver = VoiceReceiver(
            client,
            receiver_handler,
            seconds,
            on_transcript,
            on_turn,
            debug_audio=debug_audio,
            replay_tts=replay_tts,
        )
        try:
            for _ in range(2):
                assert (
                    await receiver.start(
                        "", 0, VoiceAssistantAudioSettings(), "Okay Nabu"
                    )
                    == 0
                )
                await receiver.receive(b"\x10\x00" * 1600, None)
                assert receiver.task is not None
                await receiver.task
        finally:
            await receiver.close()

    monkeypatch.setattr("hoast.llm_cli.listen", listen)
    asyncio.run(
        run_voice(
            pipeline.app.assistant,
            SatelliteConfig("mock-satellite", port=6123, capture_seconds=0.1),
            "fixture-key",
            tmp_path,
            debug_audio,
        )
    )
    assert pipeline.timeline[5:] == ["stt", "llm", "light:True", "play:Light on."] * 2
    assert events.count(Event.VOICE_ASSISTANT_RUN_END) == 2
    assert len(pipeline.histories[0]) == 1 and len(pipeline.histories[1]) > 1
    assert '"state": "off"' in pipeline.model.config.system_prompt
    assert "Current Song" not in pipeline.model.config.system_prompt
    assert tts_factory.call_args.args[0].activation_kernel is None
    pipeline.tts.close.assert_called_once()
    assert pipeline.tts.play_samples.call_count == (2 if debug_audio else 0)


def test_session_audio_gap_boundary(pipeline: Pipeline) -> None:
    """Use audio timestamps rather than inference wall time for the strict 30 s gap.

    Args:
        pipeline:
            Weight-free agent and recorded real session histories.

    """
    for start, end in ((100.0, 106.0), (136.0, 142.0), (172.01, 178.0)):
        pipeline.app.respond(
            VoiceTurn("Please switch the light on", start, end, threading.Event())
        )
    assert [len(history) for history in pipeline.histories] == [1, 5, 1]
    assert pipeline.app.previous_audio_end == 178.0


def test_warmup_failure_closes_speech_before_listening(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Startup failures release TTS and never subscribe or dispatch a warm-up tool.

    Args:
        pipeline:
            Mock model and bilingual speech engines.

        monkeypatch:
            Substitutes speech constructors and observes the network listener.

        tmp_path:
            Unused artifact root for mock initialization.

    """
    pipeline.model.generate.side_effect = RuntimeError("warmup failure")
    monkeypatch.setattr(llm_cli, "TTS", MagicMock(return_value=pipeline.tts))
    monkeypatch.setattr(llm_cli, "STT", MagicMock(return_value=pipeline.stt))
    listener = AsyncMock()
    monkeypatch.setattr(llm_cli, "listen", listener)
    with pytest.raises(RuntimeError, match="warmup failure"):
        asyncio.run(
            run_voice(pipeline.app.assistant, SatelliteConfig("mock"), None, tmp_path)
        )
    listener.assert_not_called()
    pipeline.tts.close.assert_called_once()
    assert "light:True" not in pipeline.timeline


def test_live_music_prompt_and_announcement(pipeline: Pipeline) -> None:
    """Refresh on/off state only while keeping verified track labels in the spoken response.

    Args:
        pipeline:
            Mock current music state and verified resume outcome.

    """
    pipeline.app.respond(VoiceTurn("Please resume the music", 1, 2, threading.Event()))
    assert (
        pipeline.output[-1] == "Assistant: Now playing Current Song by Current Artist."
    )
    pipeline.tts.play.assert_called_once_with(
        "Now playing Current Song by Current Artist.", blocking=True
    )
    pipeline.music.prompt_context.return_value = {
        "status": "observed",
        "state": "on",
        "title": "Changed Track",
    }
    pipeline.app.respond(
        VoiceTurn("Please switch the light on", 3, 4, threading.Event())
    )
    assert '"state": "on"' in pipeline.model.config.system_prompt
    assert "Changed Track" not in pipeline.model.config.system_prompt
    assert "Current Song" not in pipeline.model.config.system_prompt
    assert "Current Artist" not in pipeline.model.config.system_prompt


def test_music_unavailable_is_not_stale_context(pipeline: Pipeline) -> None:
    """A failed status refresh marks music unavailable while allowing light requests.

    Args:
        pipeline:
            Mock music provider and light tool.

    """
    pipeline.app.assistant.refresh_music()
    pipeline.music.prompt_context.side_effect = MusicAssistantError("offline")
    pipeline.app.respond(
        VoiceTurn("Please switch the light on", 1, 2, threading.Event())
    )
    assert '"status": "unavailable"' in pipeline.model.config.system_prompt
    assert "Current Song" not in pipeline.model.config.system_prompt
    assert pipeline.timeline[-1] == "play:Light on."


def test_disconnect_during_inference_suppresses_tools_and_tts(
    pipeline: Pipeline,
) -> None:
    """Cancellation during model routing prevents tool effects and later playback.

    Args:
        pipeline:
            Real agent/session with controlled model inference.

    """
    cancelled = threading.Event()
    generate = pipeline.model.generate_messages.side_effect

    def disconnect(
        messages: Sequence[dict[str, Any]],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Generation:
        """Finish generation after simulated transport loss.

        Args:
            messages:
                Current session history.

            on_text:
                Optional raw model streaming callback.

        """
        cancelled.set()
        return generate(messages, on_text=on_text)

    pipeline.model.generate_messages.side_effect = disconnect
    pipeline.app.respond(VoiceTurn("Please switch the light on", 1, 2, cancelled))
    assert "light:True" not in pipeline.timeline
    pipeline.tts.play.assert_not_called()
    assert not pipeline.app.assistant.agent.session.history


@pytest.mark.parametrize(
    "text",
    [
        "Long word " * 100,
        "长句没有空格" * 100,
        "a" * 800,
        "Resume requested, but not confirmed. " + "More detail. " * 100,
    ],
)
def test_bound_before_tts(
    text: str, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep full console text while capping synthesis input at safe character bounds.

    Args:
        text:
            Long text including unbroken, Chinese, and uncertainty-bearing examples.

        pipeline:
            Application with mock playback and collected console output.

        monkeypatch:
            Supplies controlled fully generated output without model inference.

    """
    reply = MagicMock(return_value=text)
    monkeypatch.setattr(Assistant, "reply", reply)
    pipeline.app.respond(VoiceTurn("test", 1, 2, threading.Event()))
    assert pipeline.output[-1] == "Assistant: " + text
    spoken = pipeline.tts.play.call_args.args[0]
    assert len(spoken) <= 400
    assert spoken == bounded_speech(text)
    if text.startswith("Resume requested"):
        assert spoken.startswith("Resume requested, but not confirmed.")


def test_agent_failure_is_spoken_without_success(pipeline: Pipeline) -> None:
    """A failed model turn resets history and gives one concise failure response.

    Args:
        pipeline:
            Controlled failing model and mock system playback.

    """
    pipeline.model.generate_messages.side_effect = RuntimeError(
        "mock inference failure"
    )
    pipeline.app.respond(
        VoiceTurn("Please switch the light on", 1, 2, threading.Event())
    )
    assert not pipeline.app.assistant.agent.session.history
    assert "light:True" not in pipeline.timeline
    pipeline.tts.play.assert_called_once_with(
        "I couldn't complete that request. Please try again.", blocking=True
    )


def test_empty_stt_does_not_extend_session(pipeline: Pipeline) -> None:
    """Discard empty recognition without invoking the agent or refreshing audio time.

    Args:
        pipeline:
            Mock recognizer returning no speech and a real receiver/application.

    """
    pipeline.app.previous_audio_end = 10.0
    pipeline.stt.transcribe_samples.return_value = ""
    pipeline.stt.transcribe_samples.side_effect = None

    async def run() -> None:
        """Deliver a complete audio capture with an empty recognized transcript."""

        async def transcribe(pcm: bytes) -> str:
            """Run the empty recognizer using the real PCM conversion path.

            Args:
                pcm:
                    Captured mono int16 audio.

            """
            return await asyncio.to_thread(pipeline.app.transcribe, pcm)

        async def respond(turn: VoiceTurn) -> None:
            """Forward accepted text to the real application if incorrectly invoked.

            Args:
                turn:
                    Recognized command; empty recognition must not create one.

            """
            await asyncio.to_thread(pipeline.app.respond, turn)

        receiver = VoiceReceiver(
            MagicMock(spec=APIClient), transcribe, 0.1, on_turn=respond
        )
        await receiver.start("", 0, VoiceAssistantAudioSettings(), None)
        await receiver.receive(b"\0\0" * 1600, None)
        assert receiver.task is not None
        await receiver.task
        await receiver.close()

    asyncio.run(run())
    assert pipeline.app.previous_audio_end == 10.0
    pipeline.model.generate_messages.assert_not_called()
    pipeline.tts.play.assert_not_called()


def test_native_run_waits_for_playback_and_disconnect_drains(
    pipeline: Pipeline,
) -> None:
    """Keep native run busy through playback and await its drain during disconnect.

    Args:
        pipeline:
            Real pipeline with externally gated mock system playback.

    """
    playing = threading.Event()
    drained = threading.Event()

    def play(text: str, *, blocking: bool = True) -> None:
        """Hold the system output until the test permits its drain.

        Args:
            text:
                Grounded answer already committed by the agent.

            blocking:
                Required blocking playback mode.

        """
        assert text == "Light on." and blocking
        playing.set()
        assert drained.wait(timeout=5)

    pipeline.tts.play.side_effect = play

    async def run() -> None:
        """Observe busy admission and shutdown while the inference worker is in TTS."""

        async def transcribe(pcm: bytes) -> str:
            """Convert and recognize one bounded PCM capture.

            Args:
                pcm:
                    Captured mono int16 audio.

            """
            return await asyncio.to_thread(pipeline.app.transcribe, pcm)

        async def respond(turn: VoiceTurn) -> None:
            """Await the application's agent and blocking playback.

            Args:
                turn:
                    Recognized command with a cancellation event.

            """
            await asyncio.to_thread(pipeline.app.respond, turn)

        client = MagicMock(spec=APIClient)
        receiver = VoiceReceiver(client, transcribe, 0.1, on_turn=respond)
        try:
            await receiver.start("", 0, VoiceAssistantAudioSettings(), None)
            await receiver.receive(b"\0\0" * 1600, None)
            assert await asyncio.to_thread(playing.wait, 3)
            assert not any(
                call.args[0] == Event.VOICE_ASSISTANT_RUN_END
                for call in client.send_voice_assistant_event.call_args_list
            )
            assert (
                await receiver.start("", 0, VoiceAssistantAudioSettings(), None) is None
            )
            closing = asyncio.create_task(receiver.close())
            await asyncio.sleep(0)
            assert not closing.done()
            drained.set()
            await closing
        finally:
            drained.set()
            await receiver.close()

    asyncio.run(run())
    assert not pipeline.app.assistant.agent.session.history


def test_entry_point_uses_satellite_config(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Parse real TOML in main and select the configured voice orchestration by default.

    Args:
        pipeline:
            Mock loaded model used without artifact access.

        monkeypatch:
            Replaces only model construction, CPU setup, logging and the runner.

        tmp_path:
            Isolated TOML and cache paths.

    """
    path = tmp_path / "config.toml"
    (tmp_path / "int8.json").write_text('{"path":"unused-export"}')
    path.write_text(
        '[weather]\nlatitude=0\nlongitude=0\n[switch]\nip="192.0.2.20"\n[satellite]\nhost="configured-satellite"\nport=7000\nlanguage="auto"\n'
    )
    pipeline.model.__enter__.return_value = pipeline.model
    model_factory = MagicMock(return_value=pipeline.model)
    runner = AsyncMock()
    monkeypatch.setattr(llm_cli, "FunctionGemma", model_factory)
    monkeypatch.setattr(llm_cli, "run_voice", runner)
    monkeypatch.setattr(llm_cli, "configure_logging", MagicMock())
    monkeypatch.setattr(llm_cli, "configure_cpu_budget", MagicMock())
    monkeypatch.setattr(
        "sys.argv",
        [
            "hoast",
            "--config",
            str(path),
            "--cache",
            str(tmp_path),
            "--model",
            "functiongemma",
            "--debug-audio",
        ],
    )
    llm_cli.main()
    runner.assert_awaited_once()
    assert runner.call_args.args[4] is True
    assert runner.call_args.args[1] == SatelliteConfig(
        "configured-satellite", port=7000, language="auto"
    )
    registry = model_factory.call_args.args[1]
    assert {schema["function"]["name"] for schema in registry.schemas()} == {
        "get_weather",
        "set_light",
    }
