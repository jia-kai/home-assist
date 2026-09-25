"""Point the ignored Hoast configuration and Compose environment at the ESP."""

import argparse
import ipaddress
import logging
import os
import re
from pathlib import Path

from .status import local_api_key

LOGGER = logging.getLogger(__name__)
HOST_LINE = re.compile(r'^host\s*=\s*"[^"]*"\s*$', flags=re.MULTILINE)
KEY_LINE = re.compile(r'^key_env\s*=\s*"[^"]*"\s*$', flags=re.MULTILINE)


def activate(config: Path, backup: Path, host: str) -> None:
    """Point Hoast at the ESP while saving its original ignored TOML.

    Args:
        config:
            Existing ignored Hoast TOML with a [satellite] section.

        backup:
            New ignored backup of the original Hoast TOML.

        host:
            ESP IPv4 address reachable from the host-networked Hoast container.

    Raises:
        ValueError: If satellite settings cannot be updated unambiguously.

    """
    address = str(ipaddress.IPv4Address(host))
    source = config.read_text()
    sections = list(re.finditer(r"^\[([^]]+)\]\s*$", source, flags=re.MULTILINE))
    satellites = [(match.end(), sections[index + 1].start() if index + 1 < len(sections) else len(source))
                  for index, match in enumerate(sections) if match.group(1) == "satellite"]
    if len(satellites) != 1:
        raise ValueError("Expected exactly one [satellite] config section")
    start, end = satellites[0]
    current = source[start:end]
    host_lines = list(HOST_LINE.finditer(current))
    key_lines = list(KEY_LINE.finditer(current))
    if len(host_lines) != 1 or len(key_lines) > 1:
        raise ValueError("Ambiguous satellite host or API key setting")
    if key_lines and key_lines[0].group() != 'key_env = "ESPHOME_API_KEY"':
        raise ValueError("Existing satellite API key setting needs manual review")
    current = HOST_LINE.sub(f'host = "{address}"', current, count=1)
    if not key_lines:
        current = current.rstrip("\n") + '\nkey_env = "ESPHOME_API_KEY"\n'
    updated = source[:start] + current + source[end:]
    if backup.exists():
        raise FileExistsError("Satellite activation backup already exists")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup_fd = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(backup_fd, "w") as saved:
        saved.write(source)
    config.write_text(updated)
    LOGGER.info("satellite.activate status=updated config=%s backup=%s", config, backup)


def main() -> None:
    """Activate a hardware satellite without displaying its API key."""
    parser = argparse.ArgumentParser(description="Connect Hoast to the reSpeaker satellite")
    parser.add_argument("--device-host", required=True)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    local_api_key(Path(".env"))
    activate(
        args.config,
        Path(".cache/respeaker/config-before-device.toml"),
        args.device_host,
    )


if __name__ == "__main__":
    main()
