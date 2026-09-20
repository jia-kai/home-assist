"""Measure acoustic correlation between AirPlay AEC reference and microphone PCM."""

import argparse
import logging
import socket
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundcard
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
HEADER = struct.Struct("!4sBBBBIIQIIHH")
MAGIC = b"HAEC"
VERSION = 2
PACKET_AUDIO = 2
ENCODING_S16LE = 1
ENCODING_S32LE = 3
ENCODING_F32LE = 4


@dataclass(slots=True)
class ReferenceCapture:
    """AEC PCM packets captured from one native AirPlay player socket."""

    chunks: list[NDArray[np.float32]] = field(default_factory=list)
    """Interleaved reference chunks, downmixed to mono float32."""

    packets: int = 0
    """Accepted AEC audio datagram count."""

    dropped_sequences: int = 0
    """Detected missing sequence numbers within the received stream generation."""

    sample_rate: int = 0
    """AEC sample rate in Hz, set from the first accepted packet."""

    _stream_id: int | None = field(default=None, init=False)
    """Current packet stream generation used to reject stale references."""

    _sequence: int | None = field(default=None, init=False)
    """Last accepted packet sequence number."""

    def add(self, packet: bytes) -> None:
        """Validate and retain one AEC v2 audio datagram.

        Args:
            packet:
                Complete Unix datagram containing its AEC header and PCM payload.

        Raises:
            ValueError: If a packet header or PCM payload is malformed.

        """
        if len(packet) < HEADER.size:
            raise ValueError("AEC datagram is shorter than its header")
        (
            magic,
            version,
            packet_type,
            encoding,
            _,
            stream_id,
            sequence,
            _,
            _,
            sample_rate,
            channels,
            frame_size,
        ) = HEADER.unpack_from(packet)
        if magic != MAGIC or version != VERSION or packet_type != PACKET_AUDIO:
            return
        if channels < 1 or frame_size < channels:
            raise ValueError("AEC packet has an invalid PCM frame format")
        if self._stream_id is not None and stream_id != self._stream_id:
            self.chunks.clear()
            self.packets = 0
            self.dropped_sequences = 0
            self._sequence = None
        if self.sample_rate and sample_rate != self.sample_rate:
            raise ValueError("AEC sample rate changed during capture")
        self._stream_id = stream_id
        self.sample_rate = sample_rate
        if self._sequence is not None and sequence > self._sequence + 1:
            self.dropped_sequences += sequence - self._sequence - 1
        self._sequence = sequence
        payload = packet[HEADER.size :]
        if len(payload) % frame_size:
            raise ValueError("AEC payload does not contain whole PCM frames")
        samples = decode_pcm(payload, encoding, channels)
        self.chunks.append(samples)
        self.packets += 1

    def audio(self) -> NDArray[np.float32]:
        """Return captured mono AEC PCM, or an empty float32 array."""
        if not self.chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self.chunks)


def decode_pcm(payload: bytes, encoding: int, channels: int) -> NDArray[np.float32]:
    """Decode interleaved AEC PCM and downmix it to mono.

    Args:
        payload:
            Native AirPlay PCM payload.

        encoding:
            AEC v2 numeric PCM encoding.

        channels:
            Number of interleaved PCM channels.

    Returns:
        Mono normalized PCM samples in float32.

    Raises:
        ValueError: If the encoding is unsupported.

    """
    if encoding == ENCODING_S16LE:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif encoding == ENCODING_S32LE:
        samples = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    elif encoding == ENCODING_F32LE:
        samples = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported AEC PCM encoding: {encoding}")
    return samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)


def resample(audio: NDArray[np.float32], source_rate: int, target_rate: int) -> NDArray[np.float32]:
    """Resample PCM using linear interpolation for correlation only.

    Args:
        audio:
            Mono PCM samples.

        source_rate:
            Input sample rate in Hz.

        target_rate:
            Output sample rate in Hz.

    Returns:
        PCM at target_rate, preserving the source duration.

    """
    if source_rate == target_rate:
        return audio
    positions = np.arange(round(len(audio) * target_rate / source_rate))
    positions = positions * source_rate / target_rate
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def normalized_correlation(
    reference: NDArray[np.float32], microphone: NDArray[np.float32], sample_rate: int, max_lag: float
) -> tuple[float, float]:
    """Find the strongest reference-to-microphone delay and correlation.

    Args:
        reference:
            Mono AirPlay reference PCM.

        microphone:
            Mono microphone PCM sampled at sample_rate.

        sample_rate:
            Correlation sample rate in Hz.

        max_lag:
            Largest absolute delay to examine in seconds.

    Returns:
        Pair of microphone lag in milliseconds and signed normalized correlation.

    """
    reference = (reference - np.mean(reference)).astype(np.float32)
    microphone = (microphone - np.mean(microphone)).astype(np.float32)
    size = 1 << (len(reference) + len(microphone) - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(microphone, size) * np.fft.rfft(reference[::-1], size), size
    )[: len(reference) + len(microphone) - 1]
    lags = np.arange(-(len(reference) - 1), len(microphone))
    limit = round(max_lag * sample_rate)
    selection = np.abs(lags) <= limit
    index = np.flatnonzero(selection)[np.argmax(np.abs(correlation[selection]))]
    score = correlation[index] / (np.linalg.norm(reference) * np.linalg.norm(microphone))
    return float(lags[index] * 1000.0 / sample_rate), float(score)


def capture(
    socket_path: Path, microphone_name: str, seconds: float, sample_rate: int
) -> tuple[ReferenceCapture, NDArray[np.float32]]:
    """Capture native reference datagrams and microphone PCM for one time window.

    Args:
        socket_path:
            Per-player Unix datagram path to bind for the capture.

        microphone_name:
            Exact PulseAudio microphone name used by the voice satellite.

        seconds:
            Capture duration in seconds.

        sample_rate:
            Microphone capture sample rate in Hz.

    Returns:
        Reference packet capture and matching-duration microphone PCM.

    """
    if socket_path.exists():
        socket_path.unlink()
    reference = ReferenceCapture()
    microphone = soundcard.get_microphone(microphone_name, include_loopback=False)
    datagram = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    datagram.bind(str(socket_path))
    datagram.settimeout(0.1)
    microphone_chunks: list[NDArray[np.float32]] = []
    stopped = threading.Event()
    receiver_errors: list[Exception] = []

    def receive_reference() -> None:
        """Drain datagrams independently of microphone callback scheduling."""
        while not stopped.is_set():
            try:
                reference.add(datagram.recv(65_536))
            except TimeoutError:
                continue
            except OSError as error:
                if not stopped.is_set():
                    receiver_errors.append(error)
                return
            except ValueError as error:
                receiver_errors.append(error)
                return

    receiver = threading.Thread(target=receive_reference, name="aec-reference", daemon=True)
    receiver.start()
    try:
        with microphone.recorder(samplerate=sample_rate, channels=1) as recorder:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                microphone_chunks.append(recorder.record(numframes=960)[:, 0].astype(np.float32))
    finally:
        stopped.set()
        receiver.join(timeout=1)
        datagram.close()
        socket_path.unlink(missing_ok=True)
    if receiver_errors:
        raise receiver_errors[0]
    return reference, np.concatenate(microphone_chunks)


def write_wav(path: Path, audio: NDArray[np.float32], sample_rate: int) -> None:
    """Write mono float PCM as a standard signed-16-bit WAV file.

    Args:
        path:
            Destination WAV file, with missing parent directories created.

        audio:
            Mono normalized PCM samples in the range -1 through 1.

        sample_rate:
            WAV sample rate in Hz.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def main() -> None:
    """Capture AEC and microphone PCM, then log their strongest acoustic match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-id", required=True)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--max-lag-seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--microphone", default="Built-in Audio Analog Stereo")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.seconds <= 0 or args.max_lag_seconds <= 0 or args.sample_rate <= 0:
        raise ValueError("seconds, max-lag-seconds, and sample-rate must be positive")
    socket_path = Path("/data/aec-reference") / f"aec-reference-{args.player_id}.sock"
    reference, microphone = capture(socket_path, args.microphone, args.seconds, args.sample_rate)
    audio = reference.audio()
    if not len(audio):
        raise RuntimeError("No AEC packets received; confirm native AirPlay playback is active")
    reference_audio = resample(audio, reference.sample_rate, args.sample_rate)
    if args.output_directory is not None:
        write_wav(args.output_directory / "aec-reference.wav", reference_audio, args.sample_rate)
        write_wav(args.output_directory / "microphone.wav", microphone, args.sample_rate)
    lag_ms, score = normalized_correlation(
        reference_audio, microphone, args.sample_rate, args.max_lag_seconds
    )
    LOGGER.info(
        "aec.correlation packets=%d dropped_sequences=%d reference_seconds=%.3f "
        "microphone_seconds=%.3f lag_ms=%.1f correlation=%.4f",
        reference.packets,
        reference.dropped_sequences,
        len(reference_audio) / args.sample_rate,
        len(microphone) / args.sample_rate,
        lag_ms,
        score,
    )


if __name__ == "__main__":
    main()
