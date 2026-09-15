"""Switch a configured local Kasa smart plug on or off without LLM inference."""

import argparse
import asyncio
import sys
from pathlib import Path

from kasa import Discover

from hoast.config import SwitchConfig, load_config
from hoast.logging import configure_logging, get_logger

logger = get_logger(__name__)
_LOG_FILE = Path(".cache/hoast/diagnostics/switch-cli.log")


async def _set_power(config: SwitchConfig, on: bool) -> None:
    """Set the relay, verify its observed state, and close the device connection.

    Args:
        config:
            Validated local smart plug IP address.

        on:
            True to turn power on, False to turn it off.

    """
    logger.info("Switch tool ip=%s on=%s status=requested", config.ip, on)
    device = await Discover.discover_single(config.ip, timeout=5)
    if device is None:
        raise RuntimeError(f"No Kasa device found at {config.ip}")
    try:
        if on:
            await device.turn_on()
        else:
            await device.turn_off()
        await device.update()
        if device.is_on != on:
            raise RuntimeError("Smart plug did not confirm the requested power state")
        logger.info("Switch tool ip=%s on=%s status=confirmed", config.ip, on)
    finally:
        await device.disconnect()


def main(argv: list[str] | None = None) -> int:
    """Require one power action and report confirmed success or a logged failure.

    Returns zero on success and one on failure. Invalid CLI arguments exit with
    code two. Full failure diagnostics are retained in the working directory cache.

    Args:
        argv:
            Explicit CLI arguments, or None to read the process arguments.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--on", dest="on", action="store_true", help="Turn power on")
    action.add_argument("--off", dest="on", action="store_false", help="Turn power off")
    args = parser.parse_args(argv)
    configure_logging(log_file=_LOG_FILE)
    try:
        config = load_config(args.config).switch
        if config is None:
            raise ValueError("Configure [switch] with ip in the system TOML file")
        asyncio.run(_set_power(config, args.on))
    except Exception:
        logger.exception("Switch tool on=%s status=failed", args.on)
        return 1
    sys.stdout.write("Power on.\n" if args.on else "Power off.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
