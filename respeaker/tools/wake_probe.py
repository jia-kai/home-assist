"""Arm the ESPHome wake detector using Hoast's native audio transport."""

import argparse
import asyncio
import logging
from pathlib import Path

from hoast.voice import listen

from .status import local_api_key

LOGGER = logging.getLogger(__name__)


async def probe(host: str) -> None:
    """Keep the ESP voice subscription active without loading STT models.

    Args:
        host:
            Device IP address on the IoT access point.

    """
    async def receive(pcm: bytes) -> str:
        """Confirm receipt of one wake-triggered capture without transcription.

        Args:
            pcm:
                Mono 16 kHz signed little-endian command PCM.

        Returns:
            A short diagnostic transcript so the satellite rearms.

        """
        LOGGER.info("wake_probe.capture status=received samples=%d", len(pcm) // 2)
        return "Wake detected"

    await listen(host, 6053, local_api_key(Path(".env")), receive, 4.0)


def main() -> None:
    """Connect to the encrypted native API and log wake-triggered PCM reception."""
    parser = argparse.ArgumentParser(description="Check device wake and native PCM")
    parser.add_argument("--device-host", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(probe(args.device_host))


if __name__ == "__main__":
    main()
