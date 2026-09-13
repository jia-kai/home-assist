"""Interactive Home Assistant entry point."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .agent import SYSTEM_PROMPT, LocalAgent
from .config import load_config, music_token
from .lfm import LFM2, LFMConfig
from .llm import DEFAULT_CACHE, FunctionGemma, LLMConfig, ToolRegistry
from .llm import SYSTEM_PROMPT as FUNCTION_ACTIVATION
from .logging import configure_logging, get_logger
from .music import MusicClient
from .session import Session
from .weather import WeatherClient

logger = get_logger(__name__)


def main() -> None:
    """Load configured tools and one prepared model; stream interactive turns.

    EOF exits; /reset clears conversation history. Diagnostics go to stderr and
    a durable log; stdout carries conversational text for piping to a speech sink.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", choices=("functiongemma", "lfm"), default="lfm")
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    configure_logging(log_file=args.cache / "diagnostics/agent.log")
    try:
        config = load_config(args.config)
        registered = [WeatherClient(config.weather).tool()]
        if config.music is not None:
            registered.extend(
                MusicClient(
                    config.music, music_token(config.music, args.env_file)
                ).tools()
            )
        tools = ToolRegistry(registered)
        model: FunctionGemma | LFM2
        if args.model == "functiongemma":
            settings = replace(
                LLMConfig.from_cache(
                    args.cache, threads=args.threads, device=args.device
                ),
                system_prompt=FUNCTION_ACTIVATION + ". " + SYSTEM_PROMPT,
                max_new_tokens=384,
            )
            model = FunctionGemma(settings, tools)
        else:
            settings_lfm = replace(
                LFMConfig.from_cache(
                    args.cache, threads=args.threads, device=args.device
                ),
                system_prompt=SYSTEM_PROMPT,
                max_new_tokens=384,
            )
            model = LFM2(settings_lfm, tools)
        with model:
            agent = LocalAgent(Session(model))
            while True:
                try:
                    request = input("You: " if sys.stdin.isatty() else "")
                except EOFError:
                    break
                if request == "/reset":
                    agent.session.reset()
                    continue
                if not request.strip():
                    continue
                for text in agent.stream(request):
                    sys.stdout.write(text)
                    sys.stdout.flush()
                sys.stdout.write("\n")
    except Exception:
        logger.exception("Home Assistant failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
