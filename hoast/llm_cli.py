"""Configured voice-first Home Assistant entry point with an explicit text mode."""

import argparse
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from .agent import SYSTEM_PROMPT, LocalAgent
from .application import Assistant, VoiceApplication
from .config import SatelliteConfig, load_config, music_token, satellite_key
from .lfm import LFM2, LFMConfig
from .lights import LightClient
from .llm import DEFAULT_CACHE, FunctionGemma, LLMConfig, Tool, ToolRegistry
from .llm import SYSTEM_PROMPT as FUNCTION_ACTIVATION
from .logging import configure_logging, get_logger
from .memory import log_memory_usage
from .music import MusicClient
from .runtime import configure_cpu_budget
from .session import Session
from .stt2 import DEFAULT_MODEL as DEFAULT_STT_MODEL
from .stt2 import STT, STTConfig
from .tts2 import DEFAULT_MODEL as DEFAULT_TTS_MODEL
from .tts2 import TTS, TTSConfig
from .voice import VoiceTurn, listen
from .weather import WeatherClient

logger = get_logger(__name__)


def prepare_voice(
    assistant: Assistant,
    config: SatelliteConfig,
    cache: Path,
    *,
    threads: int = 2,
    stt_backend: Literal["cpu", "GPU", "ort"] = "cpu",
    stt_model_dir: Path | None = None,
    tts_model_dir: Path | None = None,
) -> VoiceApplication:
    """Load native SenseVoice/Melo, warm speech and LLM, then announce readiness.

    Both speech engines receive the same one/two-thread process budget as LLM.
    Voice synthesis uses 0.85x speed for startup speech and spoken replies;
    standalone synthesis/benchmark defaults are unchanged.
    Voice TTS accepts at most 400 input characters and produces at most 30 seconds
    of audio. STT permits 15 seconds or the configured capture maximum, whichever
    is larger, and rejects more than 256 CTC tokens or 512 decoded characters.
    Artifact overrides are explicit; missing artifacts fail without downloads or
    backend fallback. Startup failure releases every constructed speech engine.

    Args:
        assistant:
            Loaded model and configured tool registry, without user history.

        config:
            Satellite settings supplying the transcription language.

        cache:
            Disposable cache root retained for call compatibility; not model selection.

        threads:
            Shared STT/TTS/LLM CPU budget, one or two. The local representative
            CPU speech measurements select two under the inclusive 1.2 ratio rule.

        stt_backend:
            Explicit SenseVoice execution backend; errors never select a fallback.

        stt_model_dir:
            Explicit prepared SenseVoice root; None selects models/stt2.

        tts_model_dir:
            Explicit prepared bilingual Melo root; None selects models/tts2.

    """
    with ExitStack() as cleanup:
        tts = TTS(
            TTSConfig(
                model_path=DEFAULT_TTS_MODEL
                if tts_model_dir is None
                else tts_model_dir,
                threads=threads,
                speed=0.85,
                max_input_chars=400,
                max_audio_seconds=30.0,
            )
        )
        cleanup.callback(tts.close)
        stt = STT(
            STTConfig(
                model_path=DEFAULT_STT_MODEL
                if stt_model_dir is None
                else stt_model_dir,
                threads=threads,
                language=None if config.language == "auto" else config.language,
                backend=stt_backend,
                max_audio_seconds=max(15.0, config.capture_seconds),
                max_output_tokens=256,
                max_output_chars=512,
            )
        )
        cleanup.callback(stt.close)
        application = VoiceApplication(assistant, stt, tts)
        application.warmup()
        cleanup.pop_all()
        return application


async def _monitor_memory_usage() -> None:
    """Sample counters every 60 seconds, including silence, until owner cancellation.

    Run on the event loop: these small counter reads do not submit inference work,
    spawn threads, retain samples, or touch models.
    """
    while True:
        await asyncio.sleep(60)
        log_memory_usage("voice.idle")


async def run_voice(
    assistant: Assistant,
    config: SatelliteConfig,
    key: str | None,
    cache: Path,
    debug_audio: bool = False,
    *,
    threads: int = 2,
    stt_backend: Literal["cpu", "GPU", "ort"] = "cpu",
    stt_model_dir: Path | None = None,
    tts_model_dir: Path | None = None,
) -> None:
    """Warm and run the native voice pipeline with one serialized inference worker.

    Own one periodic counter-only monitor after startup; cancel and join it before
    releasing speech resources on listener completion, failure, or cancellation.

    Args:
        assistant:
            Loaded grounded agent, with optional light/music tools.

        config:
            Validated satellite endpoint and capture settings.

        key:
            Optional Noise encryption key; never logged.

        cache:
            Disposable cache root retained for call compatibility; not model selection.

        debug_audio:
            Replay each captured command between tones before STT when true.

        threads:
            Shared STT/TTS/LLM CPU budget, one or two, passed to both speech engines.

        stt_backend:
            Explicit SenseVoice execution backend, forwarded unchanged to startup.

        stt_model_dir:
            Explicit SenseVoice artifact root; None selects models/stt2.

        tts_model_dir:
            Explicit bilingual Melo artifact root; None selects models/tts2.

    """
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="voice-inference"
    ) as worker:
        future = loop.run_in_executor(
            worker,
            lambda: prepare_voice(
                assistant,
                config,
                cache,
                threads=threads,
                stt_backend=stt_backend,
                stt_model_dir=stt_model_dir,
                tts_model_dir=tts_model_dir,
            ),
        )
        application: VoiceApplication | None = None
        memory_monitor: asyncio.Task[None] | None = None
        try:
            try:
                application = await asyncio.shield(future)
            except asyncio.CancelledError:
                application = await future
                raise
            log_memory_usage("voice.startup")
            memory_monitor = asyncio.create_task(
                _monitor_memory_usage(), name="voice-memory-monitor"
            )

            async def transcribe(pcm: bytes) -> str:
                """Run STT without blocking native API keepalives.

                Args:
                    pcm:
                        Mono 16 kHz int16 PCM captured by the receiver.

                """
                assert application is not None
                return await loop.run_in_executor(worker, application.transcribe, pcm)

            async def respond(turn: VoiceTurn) -> None:
                """Keep the native run active through tool execution and system playback.

                Args:
                    turn:
                        Transcript, capture boundaries, and cancellation signal.

                """
                assert application is not None
                await loop.run_in_executor(worker, application.respond, turn)

            assert application is not None
            await listen(
                config.host,
                config.port,
                key,
                transcribe,
                config.capture_seconds,
                on_turn=respond,
                debug_audio=debug_audio,
                replay_tts=application.tts,
            )
        finally:
            try:
                if memory_monitor is not None:
                    memory_monitor.cancel()
                    with suppress(asyncio.CancelledError):
                        await memory_monitor
            finally:
                if application is not None:
                    await loop.run_in_executor(worker, application.close)


def main() -> None:
    """Run configured voice turns by default, or stdin/stdout turns with --text.

    Voice startup warms English/Chinese TTS, STT and LLM before subscribing. Text
    mode uses EOF and /reset. Diagnostics go to stderr without persistent logs.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="TOML settings for weather, music, light relay, and the voice satellite",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Use stdin/stdout instead of satellite speech",
    )
    parser.add_argument(
        "--debug-audio",
        action="store_true",
        help="Replay captured voice commands between tones before STT",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv file for service credentials",
    )
    parser.add_argument(
        "--model",
        choices=("functiongemma", "lfm"),
        default="lfm",
        help="Prepared local language model",
    )
    parser.add_argument(
        "--device",
        choices=("CPU", "GPU"),
        default="GPU",
        help="Explicit OpenVINO device; loading errors do not select another device",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="Disposable compilation cache root (model selections live under models)",
    )
    parser.add_argument(
        "--stt-backend",
        choices=("cpu", "GPU", "ort"),
        default="cpu",
        help="Explicit SenseVoice backend; independent of --device, without fallback",
    )
    parser.add_argument(
        "--stt-model-dir",
        type=Path,
        help="Prepared native SenseVoiceSmall artifact root (default: models/stt2)",
    )
    parser.add_argument(
        "--tts-model-dir",
        type=Path,
        help="Prepared native bilingual MeloTTS artifact root (default: models/tts2)",
    )
    parser.add_argument(
        "--threads",
        choices=("auto", "1", "2"),
        default="auto",
        help="Shared STT/TTS/LLM CPU budget; voice auto uses the measured two-thread CPU speech policy; text auto uses one for GPU and two for CPU",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Maximum generated tokens per model call",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=4096,
        help="Maximum prompt plus reserved output tokens; oversized requests fail",
    )
    parser.add_argument(
        "--memory-reclaim",
        action="store_true",
        help="Opt into full Python GC/heap trimming at LLM boundaries; adds latency",
    )
    args = parser.parse_args()
    if args.text and args.debug_audio:
        parser.error("--debug-audio requires voice mode, without --text")
    configure_logging()
    try:
        threads = (
            (1 if args.text and args.device == "GPU" else 2)
            if args.threads == "auto"
            else int(args.threads)
        )
        configure_cpu_budget(threads)
        config = load_config(args.config)
        if not args.text and config.satellite is None:
            raise ValueError("Configure [satellite].host for voice mode, or use --text")
        key = (
            satellite_key(config.satellite, args.env_file)
            if not args.text and config.satellite is not None
            else None
        )
        registered: list[Tool[Any]] = [WeatherClient(config.weather).tool()]
        music: MusicClient | None = None
        if config.music is not None:
            music = MusicClient(config.music, music_token(config.music, args.env_file))
            music.require_airplay_2_ptp()
            registered.extend(music.tools())
        if config.switch is not None:
            registered.append(LightClient(config.switch).tool())
        tools = ToolRegistry(registered)
        model: FunctionGemma | LFM2
        if args.model == "functiongemma":
            settings = replace(
                LLMConfig.from_cache(args.cache, threads=threads, device=args.device),
                system_prompt=FUNCTION_ACTIVATION + ". " + SYSTEM_PROMPT,
                max_new_tokens=args.max_new_tokens,
                max_context_tokens=args.max_context_tokens,
                max_output_chars=4096,
                reclaim_after_call=args.memory_reclaim,
            )
            model = FunctionGemma(settings, tools)
        else:
            settings_lfm = replace(
                (
                    LFMConfig(
                        model_path=config.lfm_model_dir,
                        cache_dir=args.cache,
                        threads=threads,
                        device=args.device,
                    )
                    if config.lfm_model_dir is not None
                    else LFMConfig.from_cache(
                        args.cache, threads=threads, device=args.device
                    )
                ),
                system_prompt=SYSTEM_PROMPT,
                max_new_tokens=args.max_new_tokens,
                max_context_tokens=args.max_context_tokens,
                max_output_chars=4096,
                reclaim_after_call=args.memory_reclaim,
            )
            model = LFM2(settings_lfm, tools)
        with model:
            assistant = Assistant(LocalAgent(Session(model)), music)
            if not args.text:
                assert config.satellite is not None
                get_logger("aioesphomeapi").setLevel("INFO")
                asyncio.run(
                    run_voice(
                        assistant,
                        config.satellite,
                        key,
                        args.cache,
                        args.debug_audio,
                        threads=threads,
                        stt_backend=args.stt_backend,
                        stt_model_dir=args.stt_model_dir,
                        tts_model_dir=args.tts_model_dir,
                    )
                )
                return
            while True:
                try:
                    request = input("You: " if sys.stdin.isatty() else "")
                except EOFError:
                    break
                if request == "/reset":
                    assistant.new_session()
                    continue
                if not request.strip():
                    continue
                sys.stdout.write(assistant.reply(request) + "\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        logger.info("Home Assistant stopped")
    except Exception:
        logger.exception("Home Assistant failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
