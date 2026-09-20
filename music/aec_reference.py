"""Nonblocking Unix-datagram export of native AirPlay reference PCM."""

import os
import socket
import struct
from dataclasses import dataclass, field

_HEADER = struct.Struct("!4sBBBBIIQIIHH")
_MAGIC = b"HAEC"
_VERSION = 2
_MAX_AUDIO_PAYLOAD_BYTES = 60_000
_ENCODINGS = {
    "s16le": 1,
    "s24le": 2,
    "s32le": 3,
    "f32le": 4,
    "f64le": 5,
    "s16be": 6,
    "s24be": 7,
    "s32be": 8,
}


@dataclass(slots=True)
class AecReferenceSink:
    """Export native AirPlay PCM without delaying the player stream."""

    player_id: str
    """Music Assistant player identifier associated with this reference stream."""

    socket_directory: str = field(
        default_factory=lambda: os.environ.get("MA_AEC_REFERENCE_DIRECTORY", "")
    )
    """Directory containing per-player Unix sockets; an empty value disables export."""

    stream_id: int = field(default=0, init=False)
    """Monotonically increasing presentation generation included in every packet."""

    _sequence: int = field(default=0, init=False)
    """Monotonically increasing packet sequence number within the stream generation."""

    _sample_rate: int = field(default=0, init=False)
    """PCM sample rate in Hz included in every datagram header."""

    _channels: int = field(default=0, init=False)
    """Interleaved PCM channel count included in every datagram header."""

    _frame_size: int = field(default=0, init=False)
    """Bytes per interleaved PCM sample frame included in every datagram header."""

    _encoding: int = field(default=0, init=False)
    """Protocol PCM encoding code included in every datagram header."""

    _socket: socket.socket | None = field(default=None, init=False)
    """Nonblocking Unix datagram socket, created only while export is enabled."""

    @property
    def socket_path(self) -> str:
        """Return this player's socket path, or an empty path when export is disabled."""
        if not self.socket_directory:
            return ""
        return os.path.join(self.socket_directory, f"aec-reference-{self.player_id}.sock")

    def stream_start(
        self,
        presentation_us: int,
        sample_rate: int,
        channels: int,
        frame_size: int,
        content_type: str,
    ) -> None:
        """Publish a host-timeline anchor and set format fields for every datagram.

        Args:
            presentation_us:
                Host Unix presentation time of the first PCM sample in microseconds.

            sample_rate:
                PCM sample rate in Hz.

            channels:
                Interleaved PCM channel count.

            frame_size:
                Bytes per interleaved PCM sample frame.

            content_type:
                Music Assistant PCM content-type name mapped to the protocol encoding.

        """
        try:
            encoding = _ENCODINGS[content_type]
        except KeyError as error:
            raise ValueError(f"Unsupported AEC PCM content type: {content_type}") from error
        self.stream_id += 1
        self._sequence = 0
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_size = frame_size
        self._encoding = encoding
        self._send(1, presentation_us, 0, b"")

    def send_audio(self, presentation_us: int, duration_us: int, data: bytes) -> None:
        """Export PCM in datagrams small enough for Unix datagram transport.

        Args:
            presentation_us:
                Host Unix presentation time of the first PCM sample in microseconds.

            duration_us:
                PCM chunk duration in microseconds, apportioned across datagrams.

            data:
                Native interleaved PCM bytes passed into the AirPlay player pipeline.

        """
        for offset in range(0, len(data), _MAX_AUDIO_PAYLOAD_BYTES):
            payload = data[offset : offset + _MAX_AUDIO_PAYLOAD_BYTES]
            payload_start_us = presentation_us + duration_us * offset // len(data)
            payload_end_us = presentation_us + duration_us * (offset + len(payload)) // len(data)
            self._send(2, payload_start_us, payload_end_us - payload_start_us, payload)

    def close(self) -> None:
        """Release the datagram socket when its native AirPlay session ends."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _send(
        self, packet_type: int, presentation_us: int, duration_us: int, payload: bytes
    ) -> None:
        """Send one self-describing packet and drop it when transport is unavailable.

        Args:
            packet_type:
                Start or audio packet discriminator.

            presentation_us:
                Host Unix PCM presentation time in microseconds.

            duration_us:
                PCM duration in microseconds, or zero for a start packet.

            payload:
                Empty start payload or PCM bytes for audio.

        """
        if not self.socket_path:
            return
        packet = _HEADER.pack(
            _MAGIC,
            _VERSION,
            packet_type,
            self._encoding,
            0,
            self.stream_id,
            self._sequence,
            presentation_us,
            duration_us,
            self._sample_rate,
            self._channels,
            self._frame_size,
        ) + payload
        self._sequence += 1
        try:
            self._get_socket().sendto(packet, self.socket_path)
        except (BlockingIOError, ConnectionRefusedError, FileNotFoundError, OSError):
            pass

    def _get_socket(self) -> socket.socket:
        """Return the best-effort nonblocking Unix datagram socket.

        Returns:
            Socket used only for outgoing reference packets.

        """
        if self._socket is None:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._socket.setblocking(False)
        return self._socket
