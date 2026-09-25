"""Explicitly trigger the version-identical official XMOS test5 DFU image."""

import argparse
import asyncio
import logging
from pathlib import Path

from aioesphomeapi.client import APIClient

from .status import local_api_key

LOGGER = logging.getLogger(__name__)


async def request_test5_flash(host: str, key: str) -> None:
    """Ask the device to flash the pinned test5 binary embedded in its ESP image.

    Args:
        host:
            ESP IPv4 address on the IoT network.

        key:
            Private ESPHome Noise API key; never logged.

    Raises:
        ValueError: If the expected firmware-specific action is unavailable.

    """
    client = APIClient(host, 6053, password="", noise_psk=key)
    try:
        await client.connect(login=True)
        _, services = await client.list_entities_services()
        matches = [service for service in services if service.name == "flash_xmos_test5"]
        if len(matches) != 1 or matches[0].args:
            raise ValueError("Expected the no-argument official test5 flash action")
        await client.execute_service(matches[0], {})
        LOGGER.info("xmos.flash status=requested image=official_test5")
    finally:
        await client.disconnect()


def main() -> None:
    """Request DFU, leaving progress and version verification to device logs."""
    parser = argparse.ArgumentParser(description="Flash the official 1.0.7 48 kHz test5 image")
    parser.add_argument("--device-host", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    key = local_api_key(Path(".env"))
    asyncio.run(request_test5_flash(args.device_host, key))


if __name__ == "__main__":
    main()
