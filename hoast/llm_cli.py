"""Configured voice-first Home Assistant entry point with an explicit text mode."""

import argparse
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import SYSTEM_PROMPT, LocalAgent
from .application import Assistant, VoiceApplication
from .config import SatelliteConfig, load_config, music_token, satellite_key
from .lfm import LFM2, LFMConfig
from .lights import LightClient
from .llm import DEFAULT_CACHE, FunctionGemma, LLMConfig, Tool, ToolRegistry
from .llm import SYSTEM_PROMPT as FUNCTION_ACTIVATION
from .logging import configure_logging, get_logger
from .music import MusicClient
from .runtime import configure_cpu_budget
from .session import Session
from .stt import STT, STTConfig
from .tts import TTS, TTSConfig
from .voice import VoiceTurn, listen
from .weather import WeatherClient

logger = get_logger(__name__)


def prepare_voice(
    assistant: Assistant, config: SatelliteConfig, cache: Path
) -> VoiceApplication:
    """Load and warm bilingual speech and LLM, then announce initialization.

    Args:
        assistant:
            Loaded model and configured tool registry, without user history.

        config:
            Satellite settings supplying the transcription language.

        cache:
            Prepared LLM and bilingual speech artifact root.

    """
    tts = TTS(
        TTSConfig(
            model_path=cache / "tts/kokoro-v1.0.onnx",
            voices_path=cache / "tts/voices-v1.0.bin",
            # The optional Snake extension crashes CPU inference with GenAI loaded.
            activation_kernel=None,
            chinese_model_dir=cache / "tts/chinese",
            chinese_python=cache / "speech-env/.venv/bin/python",
        )
    )
    try:
        stt = STT(
            STTConfig(
                model_path=cache / "stt/small",
                language=None if config.language == "auto" else config.language,
            )
        )
        application = VoiceApplication(assistant, stt, tts)
        application.warmup()
        return application
    except BaseException:
        tts.close()
        raise


async def run_voice(
    assistant: Assistant,
    config: SatelliteConfig,
    key: str | None,
    cache: Path,
    debug_audio: bool = False,
) -> None:
    """Warm and run the native voice pipeline with one serialized inference worker.

    Args:
        assistant:
            Loaded grounded agent, with optional light/music tools.

        config:
            Validated satellite endpoint and capture settings.

        key:
            Optional Noise encryption key; never logged.

        cache:
            Prepared speech/model artifact root.

        debug_audio:
            Replay each captured command between tones before STT when true.

    """
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="voice-inference"
    ) as worker:
        future = loop.run_in_executor(worker, prepare_voice, assistant, config, cache)
        application: VoiceApplication | None = None
        try:
            try:
                application = await asyncio.shield(future)
            except asyncio.CancelledError:
                application = await future
                raise

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
            if application is not None:
                await loop.run_in_executor(worker, application.tts.close)


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
        help="Prepared-model and compiled-cache root",
    )
    parser.add_argument(
        "--threads",
        choices=("auto", "1", "2"),
        default="auto",
        help="LLM worker budget; auto uses one for GPU and two for CPU; voice mode reserves two CPU cores for speech",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Maximum generated tokens per model call",
    )
    args = parser.parse_args()
    if args.text and args.debug_audio:
        parser.error("--debug-audio requires voice mode, without --text")
    configure_logging()
    try:
        threads = (
            (1 if args.device == "GPU" else 2)
            if args.threads == "auto"
            else int(args.threads)
        )
        configure_cpu_budget(threads if args.text else 2)
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
            )
            model = LFM2(settings_lfm, tools)
        with model:
            assistant = Assistant(LocalAgent(Session(model)), music)
            if not args.text:
                assert config.satellite is not None
                get_logger("aioesphomeapi").setLevel("INFO")
                asyncio.run(
                    run_voice(
                        assistant, config.satellite, key, args.cache, args.debug_audio
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
