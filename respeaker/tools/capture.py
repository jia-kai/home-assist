"""Capture either XVF3800 output through the device's development TCP socket."""

import argparse
import asyncio
import logging
import secrets
import socket
import threading
import time
import wave
from pathlib import Path

import numpy as np
from aioesphomeapi.client import APIClient
from aioesphomeapi.model import UserServiceArgType

from .status import local_api_key

LOGGER = logging.getLogger(__name__)
HELLO = b"HDBG\x01\x02\x20\x00"


async def arm_debug(device: str, token: str) -> None:
    """Authorize one short-lived PCM session over ESPHome's encrypted API.

    Args:
        device:
            ESP endpoint reachable on the IoT network.

        token:
            Fresh 64-character session credential, never logged.

    Raises:
        ValueError: If the flashed device lacks the guarded debug action.

    """
    client = APIClient(device, 6053, password="", noise_psk=local_api_key(Path(".env")))
    try:
        await client.connect(login=True)
        _, services = await client.list_entities_services()
        matches = [service for service in services if service.name == "arm_debug_audio"]
        if (
            len(matches) != 1
            or len(matches[0].args) != 1
            or matches[0].args[0].name != "token"
            or matches[0].args[0].type != UserServiceArgType.STRING
        ):
            raise ValueError("Device must expose the guarded arm_debug_audio action")
        await client.execute_service(matches[0], {"token": token})
    finally:
        await client.disconnect()


def capture(
    device: str, mode: str, seconds: float, output: Path,
    ready: threading.Event | None = None,
) -> None:
    """Write channel zero ASR or channel one raw mic to a mono WAV file.

    Args:
        device:
            ESP device IP address reachable on the IoT network.

        mode:
            ``processed`` selects ASR channel 0; ``wake`` reads normal channel
            1 without rerouting; ``raw`` temporarily routes pre-gain mic 0 to
            channel 1; ``reference`` selects the far-end signal entering
            the XMOS processing cores. ``delay`` records that far-end input
            and raw mic together in stereo. Diagnostic routing pauses wake
            inference for the session.

        seconds:
            Maximum duration of audio to receive, between 0 and 60 seconds.

        output:
            Local WAV file; its parent must exist.

        ready:
            Optional signal set after device routing and TCP negotiation complete.

    """
    if mode not in ("processed", "wake", "raw", "reference", "delay") or not 0 < seconds <= 60:
        raise ValueError("Select processed/wake/raw/reference/delay and a duration between 0 and 60 seconds")
    output.parent.mkdir(parents=True, exist_ok=True)
    channel = 0 if mode == "processed" else 1
    token = secrets.token_hex(32)
    asyncio.run(arm_debug(device, token))
    deadline = time.monotonic() + 4
    while True:
        try:
            client = socket.create_connection((device, 5072), timeout=1)
            break
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise ConnectionError("Guarded debug listener did not open") from None
            time.sleep(0.05)
    with client:
        client.sendall(
            token.encode("ascii")
            + {"processed": b"P", "wake": b"P", "raw": b"R", "reference": b"F", "delay": b"D"}[mode]
        )
        client.settimeout(4)
        hello = client.recv(len(HELLO), socket.MSG_WAITALL)
        if hello != HELLO:
            raise RuntimeError("Device debug socket did not agree on stereo 48 kHz s32le PCM")
        if ready is not None:
            ready.set()
        deadline = time.monotonic() + seconds
        pending = bytearray()
        with wave.open(str(output), "wb") as recording:
            recording.setnchannels(2 if mode == "delay" else 1)
            recording.setsampwidth(4)
            recording.setframerate(48_000)
            while time.monotonic() < deadline:
                data = client.recv(8192)
                if not data:
                    raise ConnectionError("Debug stream ended before the requested duration")
                pending.extend(data)
                valid_bytes = len(pending) // 8 * 8
                if not valid_bytes:
                    continue
                stereo = np.frombuffer(bytes(pending[:valid_bytes]), dtype="<i4").reshape(-1, 2)
                recording.writeframesraw(
                    stereo.tobytes() if mode == "delay" else stereo[:, channel].tobytes()
                )
                del pending[:valid_bytes]
    LOGGER.info("debug.capture status=saved mode=%s output=%s", mode, output)


def main() -> None:
    """Capture a bounded on-demand debug recording to an ignored WAV path."""
    parser = argparse.ArgumentParser(description="Record XVF3800 ASR or raw debug PCM")
    parser.add_argument("--device-host", required=True)
    parser.add_argument("--mode", choices=("raw", "wake", "processed", "reference", "delay"), required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    capture(args.device_host, args.mode, args.seconds, args.output)


if __name__ == "__main__":
    main()
