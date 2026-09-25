"""Forward MA's native AirPlay reference Unix socket to the ESP over RTP."""

import argparse
import logging
import re
import signal
from pathlib import Path

from hoast.config import load_config

from .transport import forward_haec

LOGGER = logging.getLogger(__name__)
SOCKET_DIRECTORY = Path("/data/aec-reference")


def bridge_endpoints(config: Path, directory: Path) -> tuple[str, Path]:
    """Read the satellite address and MA player socket from Hoast's config.

    Args:
        config:
            Root Hoast TOML with [satellite].host and explicit [music].player_id.

        directory:
            Directory shared with Music Assistant for per-player Unix sockets.

    Returns:
        Device hostname/address and per-player socket path.

    Raises:
        ValueError: If the satellite or player ID is missing or the latter
            cannot safely form a socket filename.

    """
    settings = load_config(config)
    if settings.satellite is None:
        raise ValueError("Configure [satellite].host for the reSpeaker bridge")
    if settings.music is None or re.fullmatch(r"[A-Za-z0-9_.:-]+", settings.music.player_id) is None:
        raise ValueError("Configure a valid explicit Music Assistant player ID")
    return settings.satellite.host, directory / f"aec-reference-{settings.music.player_id}.sock"


def main() -> None:
    """Receive one selected player's HAEC reference until the bridge stops."""
    parser = argparse.ArgumentParser(description="Forward AirPlay reference PCM to reSpeaker")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--socket-directory", type=Path, default=SOCKET_DIRECTORY)
    args = parser.parse_args()
    log_path = Path(".cache/respeaker/diagnostics/bridge.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    def stop_on_term(signum: int, frame: object) -> None:
        """Unwind Unix socket ownership during container termination.

        Args:
            signum:
                Received termination signal number.

            frame:
                Interrupted Python frame, unused.

        """
        del signum, frame
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_term)
    try:
        host, path = bridge_endpoints(args.config, args.socket_directory)
        forward_haec(path, (host, 5070))
    except KeyboardInterrupt:
        LOGGER.info("reference.bridge status=stopped")


if __name__ == "__main__":
    main()
