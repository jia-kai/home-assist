"""Mocked complete voice pipeline: real orchestration, sessions, tools and events."""

import asyncio
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock

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
from hoast.stt2 import STT
from hoast.tts2 import TTS
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
        chinese = "你好" in text or text == "系统初始化完毕"
        timeline.append(
            "tts-init"
            if text == "系统初始化完毕"
            else "tts-zh"
            if chinese
            else "tts-en"
        )
        return np.full(220, 0.2 if chinese else 0.1, dtype=np.float32), 44100

    def transcribe(samples: NDArray[np.float32], sample_rate: int) -> str:
        """Verify dtype, mono shape, and warm-up versus captured-audio rates.

        Args:
            samples:
                Mono float32 generated or captured waveform.

            sample_rate:
                Native sample rate, 44.1 kHz for Melo and 16 kHz for input PCM.

        """
        assert samples.ndim == 1 and samples.dtype == np.float32
        assert sample_rate in (16000, 44100)
        timeline.append("stt-warm" if sample_rate == 44100 else "stt")
        return "Please switch the light on"

    def play(
        text: str, *, blocking: bool = True, cancelled: threading.Event | None = None
    ) -> None:
        """Record playback only after generation/tool completion and enforce bounds.

        Args:
            text:
                Bounded grounded speech passed to synthesis/playback.

            blocking:
                Must remain true to keep the satellite busy through device drain.

            cancelled:
                Turn event forwarded to synthesis/playback, absent for warm-up.

        """
        assert blocking and 0 < len(text) <= 400
        timeline.append("play:" + text)

    tts.synthesize.side_effect = synthesize
    tts.play.side_effect = play

    def play_samples(
        samples: NDArray[np.float32], sample_rate: int, **kwargs: Any
    ) -> None:
        """Record startup's blocking leveled output separately from debug replay.

        Args:
            samples:
                Startup speech or opt-in captured command replay.

            sample_rate:
                Native startup rate or 16 kHz debug replay rate.

            kwargs:
                Blocking playback and optional cancellation controls.

        """
        assert kwargs["blocking"]
        if sample_rate == 44100:
            assert samples.dtype == np.float32
            assert not np.any(samples[:8820])
            timeline.append("play-samples:startup")

    tts.play_samples.side_effect = play_samples
    stt.transcribe_samples.side_effect = transcribe
    app = VoiceApplication(
        Assistant(LocalAgent(Session(model)), music), stt, tts, output.append
    )
    return Pipeline(app, model, stt, tts, music, timeline, output, histories)


def test_captured_pcm_scale(pipeline: Pipeline) -> None:
    """Convert captured little-endian PCM16 to normalized floats exactly once.

    Args:
        pipeline:
            Real application orchestration with a mocked STT boundary.

    """
    pcm = np.array([-32768, -16384, -1, 0, 1, 16384, 32767], dtype="<i2")
    pipeline.app.transcribe(pcm.tobytes())
    samples, rate = pipeline.stt.transcribe_samples.call_args.args
    assert rate == 16000 and samples.dtype == np.float32
    np.testing.assert_array_equal(samples, pcm.astype(np.float32) / 32768)
    np.testing.assert_array_equal(samples * np.float32(32768), pcm.astype(np.float32))


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
    stt_factory = MagicMock(return_value=pipeline.stt)
    monkeypatch.setattr("hoast.llm_cli.STT", stt_factory)
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
            "tts-init",
            "play-samples:startup",
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
    assert pipeline.timeline[7:] == ["stt", "llm", "light:True", "play:Light on."] * 2
    assert events.count(Event.VOICE_ASSISTANT_RUN_END) == 2
    assert [len(history) for history in pipeline.histories] == [1, 1]
    assert not pipeline.app.assistant.agent.session.history
    assert '"state": "off"' in pipeline.model.config.system_prompt
    assert "Current Song" not in pipeline.model.config.system_prompt
    assert tts_factory.call_args.args[0].model_path == Path("models/tts2")
    assert tts_factory.call_args.args[0].threads == 2
    assert stt_factory.call_args.args[0].model_path == Path("models/stt2")
    assert stt_factory.call_args.args[0].threads == 2
    pipeline.stt.close.assert_called_once()
    pipeline.tts.close.assert_called_once()
    assert pipeline.tts.play_samples.call_count == (3 if debug_audio else 1)


@pytest.mark.parametrize("gap", [0.0, 1.0, 30.0, 30.01, 600.0, -100.0])
def test_voice_turns_are_independent(pipeline: Pipeline, gap: float) -> None:
    """Use fresh voice history for every capture regardless of its timestamp gap.

    Args:
        pipeline:
            Weight-free agent and recorded real session histories.

        gap:
            Seconds between captures, including overlapping/reordered metadata.

    """
    for text, start, end in (
        ("Please resume the music", 100.0, 106.0),
        ("Please switch the light on", 106.0 + gap, 112.0 + gap),
    ):
        pipeline.app.respond(VoiceTurn(text, start, end, threading.Event()))
        assert not pipeline.app.assistant.agent.session.history
        assert not pipeline.app.assistant.agent.session.pending_calls
    assert [len(history) for history in pipeline.histories] == [1, 1]
    assert pipeline.histories[1] == [
        {"role": "user", "content": "Please switch the light on"}
    ]


def test_text_replies_retain_history_until_reset(pipeline: Pipeline) -> None:
    """Keep text-mode conversation reuse and explicit reset outside voice handling.

    Args:
        pipeline:
            Shared assistant with real text session state and mocked inference.

    """
    assistant = pipeline.app.assistant
    assistant.reply("Please resume the music")
    assistant.reply("Please switch the light on")
    assert [len(history) for history in pipeline.histories] == [1, 5]
    assert assistant.agent.session.history
    assistant.new_session()
    assert not assistant.agent.session.history
    assistant.reply("Please switch the light on")
    assert len(pipeline.histories[-1]) == 1


def test_text_cli_multiturn_and_reset(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep stdin requests multi-turn and clear history only for the /reset command.

    Args:
        pipeline:
            Real agent sessions around weight-free model generation.

        monkeypatch:
            Replaces model loading, stdin, process setup and the voice runner.

        tmp_path:
            Isolated text configuration and model-cache manifest.

    """
    config = tmp_path / "config.toml"
    config.write_text("[weather]\nlatitude=0\nlongitude=0\n")
    monkeypatch.chdir(tmp_path)
    selection = tmp_path / "models/functiongemma/int8.json"
    selection.parent.mkdir(parents=True)
    selection.write_text('{"path":"unused-export"}')
    pipeline.model.__enter__.return_value = pipeline.model
    monkeypatch.setattr(
        llm_cli, "FunctionGemma", MagicMock(return_value=pipeline.model)
    )
    monkeypatch.setattr(llm_cli, "configure_logging", MagicMock())
    monkeypatch.setattr(llm_cli, "configure_cpu_budget", MagicMock())
    voice = AsyncMock()
    monkeypatch.setattr(llm_cli, "run_voice", voice)
    monkeypatch.setattr(
        "builtins.input",
        MagicMock(
            side_effect=[
                "Please resume the music",
                "Please switch the light on",
                "/reset",
                "Please switch the light on",
                EOFError(),
            ]
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "hoast",
            "--text",
            "--model",
            "functiongemma",
            "--config",
            str(config),
            "--cache",
            str(tmp_path),
        ],
    )
    llm_cli.main()
    assert [len(history) for history in pipeline.histories] == [1, 5, 1]
    voice.assert_not_called()


def test_voice_discards_preexisting_text_history(pipeline: Pipeline) -> None:
    """Start accepted voice commands independently of any existing text session.

    Args:
        pipeline:
            Shared assistant with a populated real text session.

    """
    pipeline.app.assistant.reply("Please resume the music")
    assert pipeline.app.assistant.agent.session.history
    pipeline.app.respond(
        VoiceTurn("Please switch the light on", 1, 2, threading.Event())
    )
    assert [len(history) for history in pipeline.histories] == [1, 1]
    assert not pipeline.app.assistant.agent.session.history


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("language", ["auto", "zh"])
def test_prepare_native_artifacts_and_budget(
    pipeline: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    threads: int,
    language: str,
) -> None:
    """Native engines share an explicit budget and selectable local artifact roots.

    Args:
        pipeline:
            Weight-free speech and assistant boundaries.

        monkeypatch:
            Replaces native constructors without opening artifact files.

        tmp_path:
            Isolated selected artifact paths, intentionally absent on disk.

        threads:
            Qualified one/two-thread budget to forward unchanged.

        language:
            Auto detection or an explicit Chinese control.

    """
    stt_factory = MagicMock(return_value=pipeline.stt)
    tts_factory = MagicMock(return_value=pipeline.tts)
    monkeypatch.setattr(llm_cli, "STT", stt_factory)
    monkeypatch.setattr(llm_cli, "TTS", tts_factory)
    application = llm_cli.prepare_voice(
        pipeline.app.assistant,
        SatelliteConfig("mock", language=language),
        tmp_path / "cache",
        threads=threads,
        stt_model_dir=tmp_path / "selected-stt",
        tts_model_dir=tmp_path / "selected-tts",
    )
    stt_config = stt_factory.call_args.args[0]
    tts_config = tts_factory.call_args.args[0]
    assert stt_config.model_path == tmp_path / "selected-stt"
    assert tts_config.model_path == tmp_path / "selected-tts"
    assert stt_config.threads == tts_config.threads == threads
    assert stt_config.language == (None if language == "auto" else language)
    assert stt_config.backend == tts_config.backend == "cpu"
    pipeline.stt.close.assert_not_called()
    pipeline.tts.close.assert_not_called()
    application.close()
    pipeline.stt.close.assert_called_once()
    pipeline.tts.close.assert_called_once()


def test_stt_construction_failure_closes_tts(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing recognition artifacts propagate while releasing constructed TTS.

    Args:
        pipeline:
            Weight-free assistant and playback owner.

        monkeypatch:
            Injects a missing-artifact failure at the recognizer boundary.

        tmp_path:
            Isolated cache path with no speech assets.

    """
    monkeypatch.setattr(llm_cli, "TTS", MagicMock(return_value=pipeline.tts))
    monkeypatch.setattr(
        llm_cli, "STT", MagicMock(side_effect=FileNotFoundError("model.xml"))
    )
    with pytest.raises(FileNotFoundError, match="model.xml"):
        llm_cli.prepare_voice(pipeline.app.assistant, SatelliteConfig("mock"), tmp_path)
    pipeline.tts.close.assert_called_once()


def test_application_cleanup_releases_stt_when_playback_close_fails(
    pipeline: Pipeline,
) -> None:
    """A failed playback drain does not retain recognition resources.

    Args:
        pipeline:
            Native speech interfaces with mocked resource release.

    """
    pipeline.tts.close.side_effect = RuntimeError("drain failed")
    with pytest.raises(RuntimeError, match="drain failed"):
        pipeline.app.close()
    pipeline.stt.close.assert_called_once()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_listener_failure_closes_both_speech_engines(
    pipeline: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: type[BaseException],
) -> None:
    """Listener failure or task cancellation drains/releases both speech owners.

    Args:
        pipeline:
            Weight-free assistant and speech engines.

        monkeypatch:
            Injects failure after successful speech warmup.

        tmp_path:
            Mocked artifact root.

        failure:
            Exception class raised by the listener.

    """
    monkeypatch.setattr(llm_cli, "STT", MagicMock(return_value=pipeline.stt))
    monkeypatch.setattr(llm_cli, "TTS", MagicMock(return_value=pipeline.tts))
    monkeypatch.setattr(
        llm_cli, "listen", AsyncMock(side_effect=failure("listener stopped"))
    )
    with pytest.raises(failure, match="listener stopped"):
        asyncio.run(
            run_voice(pipeline.app.assistant, SatelliteConfig("mock"), None, tmp_path)
        )
    pipeline.stt.close.assert_called_once()
    pipeline.tts.close.assert_called_once()


def test_startup_cancellation_waits_for_warmup_and_cleanup(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cancellation awaits the active worker before releasing both speech engines.

    Args:
        pipeline:
            Weight-free engines with a gated warmup call.

        monkeypatch:
            Replaces constructors and blocks LLM warmup without inference.

        tmp_path:
            Mock artifact root.

    """
    started = threading.Event()
    finish = threading.Event()
    generate = pipeline.model.generate.side_effect

    def warmup(text: str) -> Generation:
        """Wait for cancellation while mimicking an in-flight synchronous model call.

        Args:
            text:
                Synthetic warmup request, never dispatched to tools.

        """
        started.set()
        assert finish.wait(5)
        return generate(text)

    pipeline.model.generate.side_effect = warmup
    monkeypatch.setattr(llm_cli, "STT", MagicMock(return_value=pipeline.stt))
    monkeypatch.setattr(llm_cli, "TTS", MagicMock(return_value=pipeline.tts))
    listener = AsyncMock()
    monkeypatch.setattr(llm_cli, "listen", listener)

    async def run() -> None:
        """Cancel startup, then permit the worker to finish and observe cleanup."""
        task = asyncio.create_task(
            run_voice(pipeline.app.assistant, SatelliteConfig("mock"), None, tmp_path)
        )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            pipeline.stt.close.assert_not_called()
            pipeline.tts.close.assert_not_called()
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            finish.set()

    asyncio.run(run())
    listener.assert_not_called()
    pipeline.stt.close.assert_called_once()
    pipeline.tts.close.assert_called_once()


def test_warmup_failure_closes_speech_before_listening(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Startup failures release both speech engines without subscribing or tools.

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
    pipeline.stt.close.assert_called_once()
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
        "Now playing Current Song by Current Artist.", blocking=True, cancelled=ANY
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
    """A failed status refresh maps to off while allowing light requests.

    Args:
        pipeline:
            Mock music provider and light tool.

    """
    pipeline.music.prompt_context.return_value = {"status": "observed", "state": "on"}
    pipeline.app.assistant.refresh_music()
    pipeline.music.prompt_context.side_effect = MusicAssistantError("offline")
    pipeline.app.respond(
        VoiceTurn("Please switch the light on", 1, 2, threading.Event())
    )
    assert '"status": "observed", "state": "off"' in pipeline.model.config.system_prompt
    assert "Current Song" not in pipeline.model.config.system_prompt
    assert pipeline.timeline[-1] == "play:Light on."


@pytest.mark.parametrize("status", ["player_required", "unavailable", "not_configured"])
def test_music_nonobserved_prompt_is_off(pipeline: Pipeline, status: str) -> None:
    """Map nonobserved music contexts to off without exposing provider metadata.

    Args:
        pipeline:
            Mock music provider and model configuration.

        status:
            Nonobserved provider status to map to off.

    """
    pipeline.music.prompt_context.return_value = {"status": status, "title": "Private"}
    pipeline.app.assistant.refresh_music()
    assert '"status": "observed", "state": "off"' in pipeline.model.config.system_prompt
    assert "Private" not in pipeline.model.config.system_prompt


def test_music_not_configured_prompt_is_off(pipeline: Pipeline) -> None:
    """Map an absent music client to off in the system prompt.

    Args:
        pipeline:
            Mock model configuration with the music client removed.

    """
    pipeline.app.assistant.music = None
    pipeline.app.assistant.refresh_music()
    assert '"status": "observed", "state": "off"' in pipeline.model.config.system_prompt


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


def test_pre_cancelled_turn_does_not_run(pipeline: Pipeline) -> None:
    """Reject a cancelled turn without inference, output, playback or state changes.

    Args:
        pipeline:
            Real assistant with recorded model and output boundaries.

    """
    cancelled = threading.Event()
    cancelled.set()
    session = pipeline.app.assistant.agent.session
    pipeline.app.respond(VoiceTurn("Please switch the light on", 1, 2, cancelled))
    assert pipeline.app.assistant.agent.session is session
    pipeline.model.generate_messages.assert_not_called()
    pipeline.tts.play.assert_not_called()
    assert not pipeline.output


def test_empty_agent_reply_discards_voice_history(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clean up a completed empty reply without speaking or retaining its request.

    Args:
        pipeline:
            Real agent/session with committed user, tool and assistant messages.

        monkeypatch:
            Suppresses the final answer after the real grounded reply completes.

    """
    original_reply = Assistant.reply

    def empty_reply(
        assistant: Assistant, text: str, cancelled: Callable[[], bool] | None = None
    ) -> str:
        """Commit a real tool turn, then simulate an empty application answer.

        Args:
            assistant:
                Real session owner whose history must be discarded afterward.

            text:
                Recognized voice request.

            cancelled:
                Turn cancellation predicate forwarded to the real agent.

        """
        original_reply(assistant, text, cancelled)
        assert assistant.agent.session.history
        return " "

    monkeypatch.setattr(Assistant, "reply", empty_reply)
    pipeline.app.respond(
        VoiceTurn("Please switch the light on", 1, 2, threading.Event())
    )
    pipeline.tts.play.assert_not_called()
    assert pipeline.output == ["You: Please switch the light on"]
    assert not pipeline.app.assistant.agent.session.history


def test_playback_failure_discards_voice_history(pipeline: Pipeline) -> None:
    """Propagate playback errors while clearing completed voice history and calls.

    Args:
        pipeline:
            Successful grounded inference followed by a failing playback boundary.

    """
    pipeline.tts.play.side_effect = RuntimeError("playback failed")
    with pytest.raises(RuntimeError, match="playback failed"):
        pipeline.app.respond(
            VoiceTurn("Please switch the light on", 1, 2, threading.Event())
        )
    assert "light:True" in pipeline.timeline
    assert not pipeline.app.assistant.agent.session.history
    assert not pipeline.app.assistant.agent.session.pending_calls


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
        "I couldn't complete that request. Please try again.",
        blocking=True,
        cancelled=ANY,
    )


def test_empty_stt_does_not_invoke_agent(pipeline: Pipeline) -> None:
    """Discard empty recognition without invoking the agent or creating a turn.

    Args:
        pipeline:
            Mock recognizer returning no speech and a real receiver/application.

    """
    session = pipeline.app.assistant.agent.session
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
    assert pipeline.app.assistant.agent.session is session
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

    def play(
        text: str, *, blocking: bool = True, cancelled: threading.Event | None = None
    ) -> None:
        """Hold the system output until the test permits its drain.

        Args:
            text:
                Grounded answer already committed by the agent.

            blocking:
                Required blocking playback mode.

            cancelled:
                Disconnect signal; accepted playback still drains.

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


@pytest.mark.parametrize("model_name", ["functiongemma", "lfm"])
@pytest.mark.parametrize("thread_option", ["auto", "1", "2"])
@pytest.mark.parametrize("memory_diagnostics", [False, True])
def test_entry_point_uses_satellite_config(
    pipeline: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
    thread_option: str,
    memory_diagnostics: bool,
) -> None:
    """Use configured voice and model settings with console-only default diagnostics.

    Args:
        pipeline:
            Mock loaded model used without artifact access.

        monkeypatch:
            Replaces only model construction, CPU setup, logging and the runner.

        tmp_path:
            Isolated TOML and cache paths.

        model_name:
            Model family, with LFM using a custom export without a cache manifest.

        thread_option:
            Shared CPU budget option; voice auto uses the provisional two threads.

        memory_diagnostics:
            Whether the optional voice memory monitor is requested.

    """
    path = tmp_path / "config.toml"
    monkeypatch.chdir(tmp_path)
    selection = tmp_path / "models/functiongemma/int8.json"
    selection.parent.mkdir(parents=True)
    selection.write_text('{"path":"unused-export"}')
    path.write_text(
        '[lfm]\nmodel_dir="models/openvino-int8"\n'
        '[weather]\nlatitude=0\nlongitude=0\n[switch]\nip="192.0.2.20"\n[satellite]\nhost="configured-satellite"\nport=7000\nlanguage="auto"\n'
    )
    pipeline.model.__enter__.return_value = pipeline.model
    model_factory = MagicMock(return_value=pipeline.model)
    runner = AsyncMock()
    monkeypatch.setattr(
        llm_cli,
        "FunctionGemma" if model_name == "functiongemma" else "LiveLFM2",
        model_factory,
    )
    monkeypatch.setattr(llm_cli, "run_voice", runner)
    logging_setup = MagicMock()
    monkeypatch.setattr(llm_cli, "configure_logging", logging_setup)
    budget = MagicMock()
    monkeypatch.setattr(llm_cli, "configure_cpu_budget", budget)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hoast",
            "--config",
            str(path),
            "--cache",
            str(tmp_path),
            "--model",
            model_name,
            "--debug-audio",
            "--threads",
            thread_option,
            "--stt-model-dir",
            str(tmp_path / "selected-stt"),
            "--tts-model-dir",
            str(tmp_path / "selected-tts"),
        ]
        + (["--memory-diagnostics"] if memory_diagnostics else []),
    )
    llm_cli.main()
    logging_setup.assert_called_once_with()
    threads = 2 if thread_option == "auto" else int(thread_option)
    budget.assert_called_once_with(threads)
    assert model_factory.call_args.args[0].threads == threads
    if model_name == "lfm":
        assert model_factory.call_args.args[0].model_path == Path(
            "models/openvino-int8"
        )
    runner.assert_awaited_once()
    assert runner.call_args.args[4] is True
    assert runner.call_args.kwargs == {
        "threads": threads,
        "memory_diagnostics": memory_diagnostics,
        "stt_model_dir": tmp_path / "selected-stt",
        "tts_model_dir": tmp_path / "selected-tts",
        "stt_backend": "cpu",
    }
    assert runner.call_args.args[1] == SatelliteConfig(
        "configured-satellite", port=7000, language="auto"
    )
    registry = model_factory.call_args.args[1]
    assert {schema["function"]["name"] for schema in registry.schemas()} == {
        "get_weather",
        "set_light",
    }
