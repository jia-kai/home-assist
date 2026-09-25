"""Run one matched local-speaker test with synchronized debug capture startup."""

import argparse
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Protocol

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from ..core.transport import FRAMES_PER_PACKET, SAMPLE_RATE, RtpSender
from .capture import capture

LOGGER = logging.getLogger(__name__)


class OutputTiming(Protocol):
    """PortAudio callback timestamps in seconds on a shared stream clock."""

    currentTime: float
    """Stream-clock time when the callback runs, in seconds."""

    outputBufferDacTime: float
    """Stream-clock time for the first queued DAC output frame, in seconds."""


def play_local(
    sender: RtpSender, seconds: float, device: str | None, reference_file: Path | None,
    reference_delay_ms: float = 0.0,
) -> None:
    """Play deterministic test PCM against its timestamped I²S reference.

    Args:
        sender:
            RTP sender addressed to the device's AEC receiver.

        seconds:
            Positive length of the local speaker experiment in seconds.

        device:
            Optional host output device; None uses the system default.

        reference_file:
            Optional local WAV path for later acoustic-correlation analysis.

        reference_delay_ms:
            Test-only offset in milliseconds added to reference presentation.

    """
    if not 0 < seconds <= 30:
        raise ValueError("Local stimulus duration must be between 0 and 30 seconds")
    if not -2000 <= reference_delay_ms <= 2000:
        raise ValueError("Reference presentation offset must be within two seconds")
    frames = int(SAMPLE_RATE * seconds)
    rng = np.random.default_rng(2026)
    noise = rng.standard_normal(frames).astype(np.float32)
    noise = np.convolve(noise, np.ones(7, dtype=np.float32) / 7, mode="same")
    signal = (np.clip(noise * 0.12, -0.8, 0.8) * 32767).astype(np.int16)
    if reference_file is not None:
        reference_file.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(reference_file), "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(SAMPLE_RATE)
            recording.writeframes(signal.tobytes())
    start_us = [0]
    xrun_times_ms: list[float] = []

    def speaker_output(
        output: NDArray[np.int16], frame_count: int, timing: OutputTiming, status: object
    ) -> None:
        """Put the test signal at its target DAC time in an open output stream.

        Args:
            output:
                Interleaved int16 output with shape (frames, 1).

            frame_count:
                Number of frames requested by PortAudio.

            timing:
                PortAudio's currentTime and outputBufferDacTime in seconds.

            status:
                Playback over/underflow flags for this callback.

        """
        output.fill(0)
        if not start_us[0]:
            return
        first_us = time.time_ns() // 1000 + round(
            (timing.outputBufferDacTime - timing.currentTime) * 1_000_000
        )
        if status and start_us[0] - 200_000 <= first_us <= start_us[0] + frames * 1_000_000 // SAMPLE_RATE:
            xrun_times_ms.append((first_us - start_us[0]) / 1000)
        first_sample = round((first_us - start_us[0]) * SAMPLE_RATE / 1_000_000)
        source_start = max(0, first_sample)
        output_start = max(0, -first_sample)
        count = min(frame_count - output_start, len(signal) - source_start)
        if count > 0:
            output[output_start : output_start + count, 0] = signal[
                source_start : source_start + count
            ]

    with sd.OutputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device,
        blocksize=8192, latency="high", callback=speaker_output,
    ):
        start_us[0] = time.time_ns() // 1000 + 1_200_000
        for index in range(0, frames, FRAMES_PER_PACKET):
            packet_us = (start_us[0] + round(reference_delay_ms * 1000)
                         + index * 1_000_000 // SAMPLE_RATE)
            delay = (packet_us - time.time_ns() // 1000 - 700_000) / 1_000_000
            if delay > 0:
                time.sleep(delay)
            sender.send(signal[index : index + FRAMES_PER_PACKET], packet_us)
        LOGGER.info("local_test.reference status=sent frames=%d", frames)
        end_us = start_us[0] + frames * 1_000_000 // SAMPLE_RATE + 150_000
        time.sleep(max(0, (end_us - time.time_ns() // 1000) / 1_000_000))
    active_xruns = [offset for offset in xrun_times_ms if 0 <= offset < seconds * 1000]
    LOGGER.info(
        "local_test.playback status=complete xruns=%d active_xruns=%d first_active_ms=%s",
        len(xrun_times_ms), len(active_xruns), active_xruns[0] if active_xruns else None,
    )
    if active_xruns:
        raise RuntimeError("Local speaker output underruns invalidate the acoustic trial")


def trial(
    device: str, mode: str, directory: Path, speaker: str | None,
    reference_delay_ms: float = 0.0,
) -> None:
    """Open device capture before sending a five-second matched speaker signal.

    Args:
        device:
            reSpeaker IP address on the trusted IoT network.

        mode:
            Diagnostic stream name: raw, processed, reference, or delay.

        directory:
            Ignored directory for the local reference and device WAVs.

        speaker:
            Optional system-output device name.

        reference_delay_ms:
            Extra reference presentation delay, in milliseconds, for a
            calibrated local-speaker experiment.

    """
    ready = threading.Event()
    failures: list[Exception] = []

    def record() -> None:
        """Capture in a worker and retain any failure for the caller."""
        try:
            capture(
                device, mode, 8.0 + max(0.0, reference_delay_ms / 1000),
                directory / f"{mode}.wav", ready,
            )
        except Exception as error:  # noqa: BLE001 - propagate worker failure on caller thread
            failures.append(error)
            ready.set()

    thread = threading.Thread(target=record, name="xmos-debug-record")
    thread.start()
    try:
        if not ready.wait(timeout=12):
            raise TimeoutError("Debug socket did not finish its handshake")
        if failures:
            raise failures[0]
        sender = RtpSender((device, 5070))
        try:
            play_local(sender, 5.0, speaker, directory / "stimulus.wav", reference_delay_ms)
        finally:
            sender.socket.close()
    finally:
        thread.join(timeout=12)
    if thread.is_alive():
        raise TimeoutError("Device audio capture did not finish")
    if failures:
        raise failures[0]
    LOGGER.info("local_trial status=complete mode=%s", mode)


def main() -> None:
    """Execute one test with matched system playback and an XMOS debug output."""
    parser = argparse.ArgumentParser(description="Capture matched local AEC trial")
    parser.add_argument("--device-host", required=True)
    parser.add_argument("--mode", required=True, choices=("raw", "processed", "reference", "delay"))
    parser.add_argument("--directory", type=Path, default=Path("respeaker/captures"))
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--reference-delay-ms", type=float, default=0.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    trial(args.device_host, args.mode, args.directory, args.speaker, args.reference_delay_ms)


if __name__ == "__main__":
    main()
