"""Nonblocking Unix-datagram export of native AirPlay reference PCM."""

import json
import os
import socket
import struct
from dataclasses import dataclass, field

_HEADER = struct.Struct("!4sB3xIQQ")
_MAGIC = b"HAEC"


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
        """Publish a host-timeline anchor and the dynamically selected PCM format.

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
                Music Assistant PCM content-type name.

        """
        self.stream_id += 1
        metadata = json.dumps(
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "frame_size": frame_size,
                "content_type": content_type,
            },
            separators=(",", ":"),
        ).encode()
        self._send(1, presentation_us, 0, metadata)

    def send_audio(self, presentation_us: int, duration_us: int, data: bytes) -> None:
        """Export one PCM chunk without blocking native AirPlay playback.

        Args:
            presentation_us:
                Host Unix presentation time of the first PCM sample in microseconds.

            duration_us:
                PCM chunk duration in microseconds.

            data:
                Native interleaved PCM bytes passed into the AirPlay player pipeline.

        """
        self._send(2, presentation_us, duration_us, data)

    def close(self) -> None:
        """Release the datagram socket when its native AirPlay session ends."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _send(
        self, packet_type: int, presentation_us: int, duration_us: int, payload: bytes
    ) -> None:
        """Send one packet, dropping it if the receiver is unavailable or saturated.

        Args:
            packet_type:
                Start or audio packet discriminator.

            presentation_us:
                Host Unix PCM presentation time in microseconds.

            duration_us:
                PCM duration in microseconds, or zero for a start packet.

            payload:
                JSON format metadata for start or PCM bytes for audio.

        """
        if not self.socket_path:
            return
        packet = _HEADER.pack(
            _MAGIC, packet_type, self.stream_id, presentation_us, duration_us
        ) + payload
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
