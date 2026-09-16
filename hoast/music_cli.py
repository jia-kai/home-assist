"""Direct Music Assistant tool runner without model loading or inference."""

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

from pydantic import JsonValue

from hoast.config import MusicConfig, load_config, music_token
from hoast.llm import ToolCall, ToolRegistry
from hoast.logging import configure_logging, get_logger
from hoast.music import MusicClient

logger = get_logger(__name__)
_FAILURE_STATUSES = {
    "player_required",
    "cannot_resume",
    "not_playing",
    "not_found",
    "ambiguous",
    "cannot_volume",
}
_LOG_FILE = Path(".cache/hoast/diagnostics/music-cli.log")


def _parser() -> argparse.ArgumentParser:
    """Build music selection, player transport/next, and volume subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="System TOML (requires [weather])")
    parser.add_argument("--player", help="Override the configured player ID")
    parser.add_argument("--server", help="Override the Music Assistant HTTP(S) origin")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("players", "status", "resume", "next", "now-playing"):
        commands.add_parser(command)
    commands.add_parser("pause", aliases=["stop"])
    play_description = (
        "Resume existing playback when title and artist are blank; "
        "otherwise replace the queue with an Endless Mix"
    )
    play = commands.add_parser(
        "play", help=play_description, description=play_description
    )
    play.add_argument("--title", default="", help="Song title")
    play.add_argument("--artist", "--author", default="", help="Recording artist")
    volume = commands.add_parser(
        "volume", help="Louder, quieter, or a level from 1–100"
    )
    volume.add_argument("value", type=_volume_value, help="louder, quieter, or 1–100")
    return parser


def _volume_value(value: str) -> dict[str, JsonValue]:
    """Parse relative volume words or an absolute level for registered dispatch.

    Args:
        value:
            CLI argument: louder, quieter, or an integer from 1 through 100.

    """
    if value.casefold() in ("louder", "quieter"):
        return {"action": value.casefold()}
    try:
        level = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Use louder, quieter, or an integer from 1–100"
        ) from None
    if not 1 <= level <= 100:
        raise argparse.ArgumentTypeError("Volume must be from 1–100")
    return {"action": "set", "level": level}


def _redact(detail: str, token: str) -> str:
    """Remove literal, JSON-escaped, and repr-escaped credentials from diagnostics.

    Longer spellings are replaced first to avoid partially redacting an escaped
    credential. An unresolved empty token leaves the diagnostic text intact.

    Args:
        detail:
            Formatted diagnostic text, including exception chains and notes.

        token:
            Resolved credential, or an empty string before resolution succeeds.

    """
    if token:
        spellings = {token, json.dumps(token)[1:-1], repr(token)[1:-1]}
        for spelling in sorted(spellings, key=len, reverse=True):
            detail = detail.replace(spelling, "[REDACTED]")
    return detail


def main(argv: list[str] | None = None) -> int:
    """Emit one compact JSON result; return 0, 1 for errors, or 2 for refusals.

    Failures retain tracebacks, chains, and notes on stderr and in the working
    directory's cache. Literal and escaped forms of the resolved credential are
    redacted from diagnostics.
    Commands are never retried. Argument parsing errors exit with code 2.

    Args:
        argv:
            Explicit arguments for embedding/tests, or None for process arguments.

    """
    args = _parser().parse_args(argv)
    configure_logging(log_file=_LOG_FILE)
    token = ""
    try:
        config = (
            load_config(args.config).music or MusicConfig()
            if args.config is not None
            else MusicConfig()
        )
        if args.player is not None:
            config = replace(config, player_id=args.player)
        if args.server is not None:
            config = replace(config, server_url=args.server)
        token = music_token(config, args.env_file)
        client = MusicClient(config, token)
        if args.command == "players":
            result = client.players()
        elif args.command == "status":
            result = client.status()
        else:
            arguments: dict[str, JsonValue] = {}
            if args.command == "play":
                arguments = {"title": args.title, "artist": args.artist}
            elif args.command == "volume":
                arguments = args.value
            command = "pause" if args.command == "stop" else args.command
            results = ToolRegistry(client.tools()).dispatch(
                [
                    ToolCall(
                        {"next": "music_next", "now-playing": "what_is_playing"}.get(
                            command, f"{command}_music"
                        ),
                        arguments,
                    )
                ]
            )
            assert len(results) == 1
            result = results[0]
        sys.stdout.write(
            json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
        )
        if isinstance(result, dict) and result.get("status") in _FAILURE_STATUSES:
            return 2
        return 0
    except Exception:  # noqa: BLE001 - CLI boundary retains full failure diagnostics.
        # The CLI boundary reports failures once, including unexpected bugs.
        detail = _redact(traceback.format_exc(), token)
        logger.error("Music CLI failed (%s):\n%s", args.command, detail)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
