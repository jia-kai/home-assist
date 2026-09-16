"""Prepare and launch the pinned OHF microWakeWord/ESPHome microphone satellite."""

import argparse
import hashlib
import subprocess
from pathlib import Path

from hoast.logging import configure_logging, get_logger

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[1]
SATELLITE = ROOT / ".cache/hoast/voice-satellite"
REVISION = "7c6fbaa4ee3c9a2cdd25803ed40b32e108a99a4a"
REPOSITORY = "https://github.com/OHF-Voice/linux-voice-assistant.git"
LOG = ROOT / ".cache/hoast/diagnostics/voice-satellite.log"
MODEL_SHA256 = "d89128c4d16a72de429119fb2254ce46649553c2a24f5dd840175c80d7b9d094"


def verify_model() -> None:
    """Reject a model differing from official Okay Nabu release 20241226.3."""
    model = SATELLITE / "wakewords/okay_nabu.tflite"
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Satellite Okay Nabu model differs from pinned ESP model")
    logger.info("satellite.model status=verified sha256=%s", digest)


def prepare() -> None:
    """Install the pinned satellite in an isolated Python 3.13 environment.

    Preserve an existing checkout and reject a different revision or tracked edits.
    Subprocess output, including installation failures, is retained in the log.
    """
    SATELLITE.parent.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    if not SATELLITE.exists():
        commands.extend(
            [
                ["git", "clone", REPOSITORY, str(SATELLITE)],
                ["git", "-C", str(SATELLITE), "checkout", "--detach", REVISION],
            ]
        )
    with LOG.open("a") as output:
        for command in commands:
            logger.info("satellite.prepare command=%s", command)
            subprocess.run(command, stdout=output, stderr=output, check=True)
        revision = subprocess.check_output(
            ["git", "-C", str(SATELLITE), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != REVISION:
            raise ValueError(f"Satellite checkout must be at {REVISION}")
        subprocess.run(
            ["git", "-C", str(SATELLITE), "diff", "--exit-code", "HEAD"],
            stdout=output,
            stderr=output,
            check=True,
        )
        if not (SATELLITE / ".venv").exists():
            subprocess.run(
                ["uv", "venv", "--python", "3.13", str(SATELLITE / ".venv")],
                stdout=output,
                stderr=output,
                check=True,
            )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(SATELLITE / ".venv/bin/python"),
                str(SATELLITE),
                "pymicro-wakeword==2.4.1",
                "pymicro-features==2.0.2",
            ],
            stdout=output,
            stderr=output,
            check=True,
        )
    logger.info("satellite.prepare status=ok revision=%s", REVISION)
    verify_model()


def satellite_command(
    input_device: str | None = None, wakeup_sound: Path | None = None
) -> list[str]:
    """Validate the prepared model and construct the loopback satellite command.

    Args:
        input_device:
            PulseAudio/PipeWire input device name; None uses the default microphone.

        wakeup_sound:
            Optional existing audio file to play when the wake word is detected.

    """
    python = SATELLITE / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError("Run python -m tools.voice_satellite --prepare first")
    verify_model()
    command = [
        str(python),
        "-m",
        "linux_voice_assistant",
        "--name",
        "hoast-dev-mic",
        "--host",
        "127.0.0.1",
        "--port",
        "6053",
        "--wake-model",
        "okay_nabu",
        "--wake-word-dir",
        str(SATELLITE / "wakewords"),
        "--audio-input-channels",
        "1",
        "--disable-peripheral-api",
        "--debug",
    ]
    if input_device is not None:
        command.extend(["--audio-input-device", input_device])
    if wakeup_sound is not None:
        if not wakeup_sound.is_file():
            raise FileNotFoundError(wakeup_sound)
        command.extend(["--wakeup-sound", str(wakeup_sound.resolve())])
    return command


def main() -> None:
    """Prepare or run the local satellite, retaining upstream diagnostics on disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--list-input-devices", action="store_true")
    parser.add_argument("--input-device", help="PulseAudio/PipeWire microphone name")
    args = parser.parse_args()
    configure_logging(log_file=LOG)
    try:
        if args.prepare:
            prepare()
            return
        command = satellite_command(args.input_device)
        if args.list_input_devices:
            command.append("--list-input-devices")
        logger.info("satellite.run status=starting log=%s", LOG)
        with LOG.open("a") as output:
            subprocess.run(
                command, cwd=SATELLITE, stdout=output, stderr=output, check=True
            )
    except KeyboardInterrupt:
        logger.info("satellite.run status=stopped")
    except Exception:
        logger.exception("satellite.run status=failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
