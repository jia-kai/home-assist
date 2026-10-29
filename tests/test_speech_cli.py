"""Model-free backend forwarding tests for both native speech entry points."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict
from unittest.mock import AsyncMock, MagicMock

import pytest

from hoast import llm_cli, voice
from hoast.application import Assistant, VoiceApplication
from hoast.config import SatelliteConfig

type Backend = Literal["cpu", "GPU", "ort"]


class BackendOptions(TypedDict, total=False):
    """Optional backend keyword, absent when testing an API's default."""

    stt_backend: Backend
    """Explicit SenseVoice backend when the caller supplies one."""


@pytest.mark.parametrize("backend", [None, "cpu", "GPU", "ort"])
def test_prepare_voice_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: Backend | None
) -> None:
    """Forward speech backends and slower voice speed without loading real models.

    Args:
        monkeypatch:
            Replaces speech constructors and the application warmup boundary.

        tmp_path:
            Absent prepared artifact paths; no model files are needed.

        backend:
            Explicit backend, or None to exercise the omitted CPU default.

    """
    stt = MagicMock()
    tts = MagicMock()
    stt_factory = MagicMock(return_value=stt)
    tts_factory = MagicMock(return_value=tts)
    application = MagicMock(spec=VoiceApplication)
    application_factory = MagicMock(return_value=application)
    monkeypatch.setattr(llm_cli, "STT", stt_factory)
    monkeypatch.setattr(llm_cli, "TTS", tts_factory)
    monkeypatch.setattr(llm_cli, "VoiceApplication", application_factory)
    options = (
        BackendOptions() if backend is None else BackendOptions(stt_backend=backend)
    )
    assistant = MagicMock(spec=Assistant)
    result = llm_cli.prepare_voice(
        assistant, SatelliteConfig("mock", language="auto"), tmp_path, **options
    )
    settings = stt_factory.call_args.args[0]
    assert settings.backend == ("cpu" if backend is None else backend)
    assert settings.model_path == Path("models/stt2")
    assert settings.threads == 2 and settings.language is None
    assert settings.max_audio_seconds == 15.0
    assert settings.max_output_tokens == 256 and settings.max_output_chars == 512
    tts_settings = tts_factory.call_args.args[0]
    assert tts_settings.model_path == Path("models/tts2")
    assert tts_settings.threads == 2 and tts_settings.backend == "cpu"
    assert tts_settings.speed == 0.85
    assert tts_settings.max_input_chars == 400
    assert tts_settings.max_audio_seconds == 30.0
    assert not settings.release_after_call
    assert not tts_settings.release_after_call
    assert result is application
    application_factory.assert_called_once_with(assistant, stt, tts)
    application.warmup.assert_called_once_with()
    stt.close.assert_not_called()
    tts.close.assert_not_called()


@pytest.mark.parametrize("backend", [None, "cpu", "GPU", "ort"])
def test_run_voice_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: Backend | None
) -> None:
    """Forward backend selection through the executor without inference or networking.

    Args:
        monkeypatch:
            Replaces startup and the native listener with inert mocks.

        tmp_path:
            Explicit cache and speech artifact roots to retain unchanged.

        backend:
            Explicit backend, or None to exercise the omitted CPU default.

    """
    application = MagicMock(spec=VoiceApplication)
    prepare = MagicMock(return_value=application)
    listener = AsyncMock()
    monkeypatch.setattr(llm_cli, "prepare_voice", prepare)
    monkeypatch.setattr(llm_cli, "listen", listener)
    assistant = MagicMock(spec=Assistant)
    satellite = SatelliteConfig("mock")
    options = (
        BackendOptions() if backend is None else BackendOptions(stt_backend=backend)
    )
    asyncio.run(
        llm_cli.run_voice(
            assistant,
            satellite,
            None,
            tmp_path,
            threads=1,
            stt_model_dir=tmp_path / "selected-stt",
            tts_model_dir=tmp_path / "selected-tts",
            **options,
        )
    )
    prepare.assert_called_once_with(
        assistant,
        satellite,
        tmp_path,
        threads=1,
        stt_backend="cpu" if backend is None else backend,
        stt_model_dir=tmp_path / "selected-stt",
        tts_model_dir=tmp_path / "selected-tts",
    )
    listener.assert_awaited_once()
    application.close.assert_called_once_with()
    application.transcribe.assert_not_called()
    application.respond.assert_not_called()


@pytest.mark.parametrize("reclaim", [False, True])
@pytest.mark.parametrize("backend", [None, "cpu", "GPU", "ort"])
def test_assistant_cli_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: Backend | None,
    reclaim: bool,
) -> None:
    """Parse STT selection independently of the unchanged LLM device/model defaults.

    Args:
        monkeypatch:
            Replaces LLM construction, service clients, logging and the voice runner.

        tmp_path:
            Isolated TOML with an explicit LFM root avoiding manifest access.

        backend:
            Explicit backend, or None to exercise the omitted CPU default.

        reclaim:
            Default disabled cleanup or explicit boundary-reclamation opt-in.

    """
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[lfm]\nmodel_dir="unused-export"\n'
        '[weather]\nlatitude=0\nlongitude=0\n[satellite]\nhost="mock"\n'
    )
    model = MagicMock()
    model.__enter__.return_value = model
    model_factory = MagicMock(return_value=model)
    function_factory = MagicMock()
    runner = AsyncMock()
    budget = MagicMock()
    monkeypatch.setattr(llm_cli, "LFM2", model_factory)
    monkeypatch.setattr(llm_cli, "FunctionGemma", function_factory)
    monkeypatch.setattr(llm_cli, "WeatherClient", MagicMock())
    monkeypatch.setattr(llm_cli, "LocalAgent", MagicMock())
    monkeypatch.setattr(llm_cli, "run_voice", runner)
    monkeypatch.setattr(llm_cli, "configure_cpu_budget", budget)
    monkeypatch.setattr(llm_cli, "configure_logging", MagicMock())
    argv = ["hoast", "--config", str(config_path), "--cache", str(tmp_path)]
    if backend is not None:
        argv.extend(["--stt-backend", backend])
    if reclaim:
        argv.append("--memory-reclaim")
    monkeypatch.setattr("sys.argv", argv)
    llm_cli.main()
    runner.assert_awaited_once()
    assert runner.call_args.kwargs == {
        "threads": 2,
        "memory_diagnostics": False,
        "stt_backend": "cpu" if backend is None else backend,
        "stt_model_dir": None,
        "tts_model_dir": None,
    }
    assert runner.call_args.args[3] == tmp_path
    settings = model_factory.call_args.args[0]
    assert settings.device == "GPU" and settings.threads == 2
    assert settings.reclaim_after_call is reclaim
    assert settings.max_context_tokens == 4096
    assert settings.max_new_tokens == 384
    assert settings.max_output_chars == 4096
    function_factory.assert_not_called()
    budget.assert_called_once_with(2)


@pytest.mark.parametrize("backend", [None, "cpu", "GPU", "ort"])
def test_transcription_cli_backend(
    monkeypatch: pytest.MonkeyPatch, backend: Backend | None
) -> None:
    """Direct transcription selects STT explicitly while retaining command defaults.

    Args:
        monkeypatch:
            Replaces speech construction, logging, CPU setup and the network listener.

        backend:
            Explicit backend, or None to exercise the omitted CPU default.

    """
    stt = MagicMock()
    factory = MagicMock(return_value=stt)
    listener = AsyncMock()
    budget = MagicMock()
    tts_factory = MagicMock()
    monkeypatch.setattr(voice, "STT", factory)
    monkeypatch.setattr(voice, "TTS", tts_factory)
    monkeypatch.setattr(voice, "listen", listener)
    monkeypatch.setattr(voice, "configure_cpu_budget", budget)
    monkeypatch.setattr(voice, "configure_logging", MagicMock())
    argv = ["voice"]
    if backend is not None:
        argv.extend(["--stt-backend", backend])
    monkeypatch.setattr("sys.argv", argv)
    voice.main()
    settings = factory.call_args.args[0]
    assert settings.backend == ("cpu" if backend is None else backend)
    assert settings.model_path == voice.DEFAULT_MODEL
    assert settings.threads == 2 and settings.language == "en"
    listener.assert_awaited_once()
    assert listener.call_args.args[:2] == ("127.0.0.1", 6053)
    assert listener.call_args.args[4] == 6.0
    stt.close.assert_called_once_with()
    stt.transcribe_samples.assert_not_called()
    tts_factory.assert_called_once_with(playback_only=True)
    assert listener.call_args.kwargs["replay_tts"] is tts_factory.return_value
    tts_factory.return_value.close.assert_called_once_with()
    budget.assert_called_once_with(2)


@pytest.mark.parametrize("entry_point", [llm_cli.main, voice.main])
def test_cli_rejects_invalid_backend(
    monkeypatch: pytest.MonkeyPatch, entry_point: Callable[[], None]
) -> None:
    """Reject unsupported backend spellings before model or service startup.

    Args:
        monkeypatch:
            Replaces command arguments and construction boundaries.

        entry_point:
            Assistant or direct-transcription CLI main function.

    """
    speech = MagicMock()
    model = MagicMock()
    monkeypatch.setattr(llm_cli, "STT", speech)
    monkeypatch.setattr(voice, "STT", speech)
    monkeypatch.setattr(llm_cli, "LFM2", model)
    monkeypatch.setattr("sys.argv", ["hoast", "--stt-backend", "CPU"])
    with pytest.raises(SystemExit) as error:
        entry_point()
    assert error.value.code == 2
    speech.assert_not_called()
    model.assert_not_called()
