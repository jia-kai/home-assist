"""Say 'Okay Nabu' and hear 'I'm here' using the real satellite wake detector."""

import argparse
import asyncio
import socket
import subprocess
import sys

from aioesphomeapi.client import APIClient
from aioesphomeapi.core import APIConnectionError
from aioesphomeapi.model import VoiceAssistantAudioSettings
from aioesphomeapi.model import VoiceAssistantEventType as Event

from hoast.logging import configure_logging, get_logger
from tools.voice_satellite import ROOT, SATELLITE, satellite_command

logger = get_logger(__name__)
LOG = ROOT / ".cache/hoast/diagnostics/wake-test.log"
REPLY = ROOT / ".cache/hoast/wake-test/im-here.wav"


async def run_test(command: list[str]) -> None:
    """Own a satellite process and native client; rearm after each audible reply.

    The satellite plays the configured reply before sending its wake request.
    This client ends each run without recording or transcribing command audio.
    Unexpected connection/process failure ends the test with durable diagnostics.

    Args:
        command:
            Prepared satellite argv with the spoken wake-up sound configured.

    """
    client = APIClient("127.0.0.1", 6053, password="")
    disconnected = asyncio.Event()

    async def on_disconnect(expected: bool) -> None:
        """Signal loss of the test connection.

        Args:
            expected:
                Whether the API client expected this disconnect.

        """
        logger.debug("wake_test.disconnect expected=%s", expected)
        disconnected.set()

    async def on_start(
        conversation_id: str,
        flags: int,
        audio_settings: VoiceAssistantAudioSettings,
        wake_word: str | None,
    ) -> int:
        """End the test run after the satellite's reply playback callback fires.

        Args:
            conversation_id:
                Native conversation reference, unused by the wake-only test.

            flags:
                Native request flags retained in diagnostics.

            audio_settings:
                Host audio-effect requests, unused without transcription.

            wake_word:
                Detected phrase provided by the satellite.

        """
        logger.info("wake_test.detected phrase=%r flags=%d", wake_word, flags)
        logger.debug(
            "wake_test.start conversation=%r settings=%r",
            conversation_id,
            audio_settings,
        )
        client.send_voice_assistant_event(Event.VOICE_ASSISTANT_RUN_START, None)
        client.send_voice_assistant_event(Event.VOICE_ASSISTANT_RUN_END, None)
        return 0

    async def on_stop(abort: bool) -> None:
        """Record a satellite stop notification without opening an audio session.

        Args:
            abort:
                Whether the satellite requested an abort rather than audio end.

        """
        logger.debug("wake_test.stop abort=%s", abort)

    async def on_audio(data: bytes, data2: bytes | None) -> None:
        """Accept and discard any in-flight audio while the run-end event arrives.

        Args:
            data:
                Primary-channel PCM bytes; the test performs no STT.

            data2:
                Optional reference-channel PCM, also unused.

        """
        logger.debug("wake_test.audio discarded_bytes=%d", len(data))

    with LOG.open("a") as output:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=SATELLITE,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
        try:
            for attempt in range(30):
                if process.returncode is not None:
                    raise RuntimeError(
                        f"Satellite exited with status {process.returncode}"
                    )
                try:
                    await client.connect(
                        login=True, on_stop=on_disconnect, log_errors=False
                    )
                    disconnected.clear()
                    break
                except APIConnectionError:
                    if attempt == 29:
                        raise
                    await asyncio.sleep(0.5)
            client.subscribe_voice_assistant(
                handle_start=on_start,
                handle_stop=on_stop,
                handle_audio=on_audio,
            )
            logger.info("Listening. Say 'Okay Nabu'; press Ctrl-C to stop.")
            await disconnected.wait()
            raise ConnectionError("Satellite disconnected during the wake test")
        finally:
            try:
                await client.disconnect()
            finally:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except TimeoutError:
                        process.kill()
                        await process.wait()


def main() -> None:
    """Prepare the spoken reply once and run an interactive microphone wake test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-device", help="PulseAudio/PipeWire microphone name")
    args = parser.parse_args()
    configure_logging(log_file=LOG)
    get_logger("aioesphomeapi").setLevel("WARNING")
    try:
        # A second satellite would compete for the same native API port.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 6053))
        satellite_command(args.input_device)
        if not REPLY.is_file():
            REPLY.parent.mkdir(parents=True, exist_ok=True)
            temporary = REPLY.with_suffix(".partial.wav")
            logger.info("Preparing the spoken reply.")
            with LOG.open("a") as output:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "hoast.tts",
                        "I'm here.",
                        "--output",
                        str(temporary),
                    ],
                    cwd=ROOT,
                    stdout=output,
                    stderr=output,
                    check=True,
                )
            temporary.replace(REPLY)
        asyncio.run(run_test(satellite_command(args.input_device, REPLY)))
    except KeyboardInterrupt:
        logger.info("Wake-word test stopped.")
    except Exception:
        logger.exception("wake_test status=failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
