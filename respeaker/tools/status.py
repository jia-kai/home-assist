"""Read the board-reported XMOS version over Wi-Fi via ESPHome's native API."""

import argparse
import asyncio
import logging
from pathlib import Path

from aioesphomeapi.client import APIClient
from aioesphomeapi.model import TextSensorInfo, TextSensorState
from dotenv import dotenv_values

LOGGER = logging.getLogger(__name__)


def local_api_key(path: Path) -> str:
    """Load the Noise encryption key from the ignored root dotenv file.

    Args:
        path:
            Repository-root .env file holding ESPHOME_API_KEY.

    Returns:
        Local Noise API key without writing it to logs or standard output.

    Raises:
        ValueError: If ESPHOME_API_KEY is missing or empty.

    """
    key = dotenv_values(path, interpolate=False).get("ESPHOME_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Set ESPHOME_API_KEY in the root .env")
    return key


async def read_version(host: str, key: str) -> tuple[str, str | None]:
    """Read XMOS version and the optional wake-route state over native Wi-Fi API.

    Args:
        host:
            Device IP address reachable over the IoT AP.

        key:
            Local ESPHome Noise encryption key; never logged.

    Returns:
        XMOS firmware version and route readiness; the latter is None on
        firmware without a Wake Route text sensor.

    Raises:
        TimeoutError: If the configured sensors do not publish in their window.

    """
    client = APIClient(host, 6053, password="", noise_psk=key)
    try:
        await client.connect(login=True)
        entities, _ = await client.list_entities_services()
        versions = [entity for entity in entities
                    if isinstance(entity, TextSensorInfo) and entity.name == "Firmware Version"]
        routes = [entity for entity in entities
                  if isinstance(entity, TextSensorInfo) and entity.name == "Wake Route"]
        if len(versions) != 1 or len(routes) > 1:
            raise ValueError("Expected exactly one Firmware Version text sensor")
        version: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        route: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        def on_state(state: object) -> None:
            """Resolve the sensor update belonging to the XMOS version entity.

            Args:
                state:
                    ESPHome native entity update from the subscription.

            """
            if not isinstance(state, TextSensorState) or state.missing_state:
                return
            if state.key == versions[0].key and not version.done():
                version.set_result(state.state)
            if routes and state.key == routes[0].key and not route.done():
                route.set_result(state.state)

        client.subscribe_states(on_state)
        reported_version = await asyncio.wait_for(version, timeout=130)
        if not routes:
            return reported_version, None
        return reported_version, await asyncio.wait_for(route, timeout=15)
    finally:
        await client.disconnect()


def main() -> None:
    """Show the XMOS text sensor's version without exposing API credentials."""
    parser = argparse.ArgumentParser(description="Read XMOS Firmware Version over Wi-Fi")
    parser.add_argument("--device-host", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    version, route = asyncio.run(
        read_version(args.device_host, local_api_key(Path(".env")))
    )
    LOGGER.info("xmos.version status=observed version=%s wake_route=%s", version, route)


if __name__ == "__main__":
    main()
