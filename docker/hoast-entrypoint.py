"""Start the microphone satellite before Hoast and relay container shutdown."""

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

PYTHON = "/opt/hoast/.venv/bin/python"
SATELLITE_PORT = 6053
SATELLITE_DEVICE = "Built-in Audio Analog Stereo"
CHINESE_G2P_PYTHON = Path("/workspace/src/.cache/hoast/speech-env/.venv/bin/python")


def wait_for_satellite(satellite: subprocess.Popen[str], timeout_seconds: float) -> None:
    """Wait until the satellite accepts its loopback native-API connection.

    Args:
        satellite:
            Started satellite child process.

        timeout_seconds:
            Maximum readiness wait in seconds.

    Raises:
        RuntimeError: If the satellite exits or does not bind before the deadline.

    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if satellite.poll() is not None:
            raise RuntimeError(f"Voice satellite exited with status {satellite.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", SATELLITE_PORT)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError("Voice satellite did not bind port 6053")


def stop_process(process: subprocess.Popen[str]) -> int:
    """Terminate a child process and return its exit status.

    Args:
        process:
            Child process to stop when it remains running.

    Returns:
        Child process exit status after graceful or forced termination.

    """
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return process.wait()


def prepare_chinese_g2p() -> None:
    """Create the cached Chinese G2P environment when it is unavailable.

    The mounted development cache may contain a virtual environment pointing to a
    host-only Python installation. The Chinese TTS preparation command recreates
    that generated environment and reuses its existing model artifacts.
    """
    if CHINESE_G2P_PYTHON.is_file():
        return
    subprocess.run([PYTHON, "-m", "tools.prepare_tts", "--chinese"], check=True)


def main(arguments: Sequence[str]) -> int:
    """Prepare/start the satellite, then run Hoast until either process exits.

    Args:
        arguments:
            Hoast command arguments supplied by the Docker CMD.

    Returns:
        Hoast exit status, or a nonzero startup failure status.

    """
    if not arguments:
        raise ValueError("Hoast command arguments are required")
    stop_event = threading.Event()

    def request_shutdown(signum: int, frame: object) -> None:
        """Record a container termination signal for child-process shutdown.

        Args:
            signum:
                Received POSIX signal number.

            frame:
                Interrupted interpreter frame, unused by this handler.

        """
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    subprocess.run([PYTHON, "-m", "tools.voice_satellite", "--prepare"], check=True)
    satellite = subprocess.Popen(
        [
            PYTHON,
            "-m",
            "tools.voice_satellite",
            "--input-device",
            os.environ.get("HOAST_SATELLITE_INPUT", SATELLITE_DEVICE),
        ],
        text=True,
    )
    try:
        wait_for_satellite(satellite, timeout_seconds=60)
        prepare_chinese_g2p()
        hoast = subprocess.Popen([PYTHON, *arguments], text=True)
        try:
            while hoast.poll() is None and satellite.poll() is None:
                if stop_event.wait(0.2):
                    break
            if hoast.poll() is not None:
                return hoast.wait()
            if satellite.poll() is not None:
                raise RuntimeError(
                    f"Voice satellite exited with status {satellite.returncode}"
                )
            return stop_process(hoast)
        finally:
            stop_process(hoast)
    finally:
        stop_process(satellite)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
