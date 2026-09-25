"""Translate Music Assistant's HAEC datagrams into timestamped RTP/L16 audio."""

import errno
import logging
import os
import select
import socket
import stat
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
HAEC_HEADER = struct.Struct("!4sBBBBIIQIIHH")
RTP_HEADER = struct.Struct("!BBHII")
RTCP_SR = struct.Struct("!BBHIIIIII")
SAMPLE_RATE = 48_000
FRAMES_PER_PACKET = 480
NTP_UNIX_OFFSET = 2_208_988_800
TARGET_LEAD_US = 3_000_000
PACKET_DURATION_US = 10_000
MAX_PENDING_PACKETS = 800
ENCODING_BYTES = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 2, 7: 3, 8: 4}


@dataclass(slots=True, frozen=True)
class ReferenceChunk:
    """PCM chunk anchored at the host's Unix presentation time in microseconds."""

    stream_id: int
    """Music Assistant presentation generation; resets resampler and RTP state."""

    sequence: int
    """HAEC packet sequence number within the stream generation."""

    presentation_us: int
    """Host Unix presentation time of the first source sample in microseconds."""

    sample_rate: int
    """Source PCM frame rate in Hz."""

    samples: NDArray[np.float32]
    """Mono, normalized float32 PCM with shape (source frames,)."""


def parse_haec(packet: bytes) -> ReferenceChunk | None:
    """Decode a HAEC v2 audio datagram; return None for its stream-start marker.

    Args:
        packet:
            Complete HAEC v2 Unix datagram, including its 36-byte header.

    Raises:
        ValueError: If header, PCM encoding, or frame alignment is invalid.

    """
    if len(packet) < HAEC_HEADER.size:
        raise ValueError("HAEC header truncated")
    (
        magic,
        version,
        kind,
        encoding,
        reserved,
        stream_id,
        sequence,
        presentation_us,
        duration_us,
        sample_rate,
        channels,
        frame_size,
    ) = HAEC_HEADER.unpack_from(packet)
    if magic != b"HAEC" or version != 2 or reserved or kind not in (1, 2):
        raise ValueError("Invalid HAEC v2 header")
    if encoding not in ENCODING_BYTES or not sample_rate or not channels:
        raise ValueError("Unsupported HAEC PCM format")
    if frame_size != channels * ENCODING_BYTES[encoding]:
        raise ValueError("Invalid HAEC frame size")
    payload = packet[HAEC_HEADER.size :]
    if kind == 1:
        if payload or duration_us:
            raise ValueError("Invalid HAEC start marker")
        return None
    if not payload or len(payload) % frame_size:
        raise ValueError("Misaligned HAEC audio")
    if encoding in (1, 6):
        dtype = "<i2" if encoding == 1 else ">i2"
        values = np.frombuffer(payload, dtype=dtype).astype(np.float32) / 32768.0
    elif encoding in (3, 8):
        dtype = "<i4" if encoding == 3 else ">i4"
        values = np.frombuffer(payload, dtype=dtype).astype(np.float32) / 2147483648.0
    elif encoding in (4, 5):
        dtype = "<f4" if encoding == 4 else "<f8"
        values = np.frombuffer(payload, dtype=dtype).astype(np.float32)
    else:
        sample_width = ENCODING_BYTES[encoding]
        data = np.frombuffer(payload, dtype=np.uint8).reshape(-1, sample_width)
        if encoding == 2:
            values_i32 = (
                (data[:, 0].astype(np.int32) << 8)
                | (data[:, 1].astype(np.int32) << 16)
                | (data[:, 2].astype(np.int32) << 24)
            )
        else:
            values_i32 = (
                (data[:, 2].astype(np.int32) << 8)
                | (data[:, 1].astype(np.int32) << 16)
                | (data[:, 0].astype(np.int32) << 24)
            )
        values = values_i32.astype(np.float32) / 2147483648.0
    mono = values.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return ReferenceChunk(stream_id, sequence, presentation_us, sample_rate, mono)


@dataclass(slots=True)
class RtpSender:
    """Send mono RTP/L16 and RTCP sender reports on the host presentation timeline."""

    destination: tuple[str, int]
    """Device IPv4 address and UDP RTP port; RTCP uses the next port."""

    ssrc: int = field(default_factory=lambda: int.from_bytes(os.urandom(4), "big"))
    """Random RTP synchronization source, regenerated for each sender process."""

    sequence: int = 0
    """Next 16-bit RTP sequence counter."""

    socket: socket.socket = field(
        default_factory=lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    )
    """Datagram transport, closed by the owner."""

    packet_count: int = 0
    """RTCP cumulative RTP packet count."""

    octet_count: int = 0
    """RTCP cumulative PCM payload bytes."""

    last_report_us: int = 0
    """Presentation time of the last RTCP sender report in microseconds."""

    first_packet: bool = True
    """Whether the next audio packet opens an RTP stream generation."""

    def send(self, samples: NDArray[np.int16], presentation_us: int) -> None:
        """Send contiguous 48 kHz mono blocks with timestamps of their first samples.

        Args:
            samples:
                Signed int16 mono PCM with shape (frames,).

            presentation_us:
                Host Unix presentation time of the first PCM sample in microseconds.

        """
        if samples.dtype != np.int16 or samples.ndim != 1:
            raise ValueError("RTP audio must be mono int16 PCM")
        for offset in range(0, len(samples), FRAMES_PER_PACKET):
            part = samples[offset : offset + FRAMES_PER_PACKET]
            timestamp_us = presentation_us + offset * 1_000_000 // SAMPLE_RATE
            rtp_time = timestamp_us * SAMPLE_RATE // 1_000_000 & 0xFFFFFFFF
            payload = part.astype(">i2", copy=False).tobytes()
            packet = (
                RTP_HEADER.pack(
                    0x80,
                    0x80 if self.first_packet else 0,
                    self.sequence,
                    rtp_time,
                    self.ssrc,
                )
                + payload
            )
            self.socket.sendto(packet, self.destination)
            self.first_packet = False
            self.sequence = (self.sequence + 1) & 0xFFFF
            self.packet_count += 1
            self.octet_count += len(payload)
            if timestamp_us - self.last_report_us >= 1_000_000:
                self._report(timestamp_us, rtp_time)
                self.last_report_us = timestamp_us

    def _report(self, presentation_us: int, rtp_time: int) -> None:
        """Map RTP sample time to host Unix presentation time using RTCP SR.

        Args:
            presentation_us:
                Host Unix presentation time of the RTP sample in microseconds.

            rtp_time:
                RTP timestamp corresponding to that presentation time.

        """
        seconds, micros = divmod(presentation_us, 1_000_000)
        report = RTCP_SR.pack(
            0x80,
            200,
            6,
            self.ssrc,
            (seconds + NTP_UNIX_OFFSET) & 0xFFFFFFFF,
            micros * (1 << 32) // 1_000_000,
            rtp_time,
            self.packet_count,
            self.octet_count,
        )
        self.socket.sendto(report, (self.destination[0], self.destination[1] + 1))


@dataclass(slots=True)
class ScheduledPacket:
    """One RTP packet and its earliest host send time in microseconds."""

    send_us: int
    """Earliest Unix time to transmit the packet."""

    presentation_us: int
    """Unix time of this packet's first audio sample."""

    pcm: NDArray[np.int16]
    """Signed 48 kHz mono samples, shape (up to 480,)."""


@dataclass(slots=True)
class PacedReference:
    """Buffer MA audio and send ten-millisecond RTP at a three-second lead."""

    sender: RtpSender
    """RTP/RTCP sender retaining the original sample presentation timestamps."""

    pending: deque[ScheduledPacket] = field(default_factory=deque)
    """Scheduled 48 kHz packets in playback order."""

    last_sent_us: int | None = None
    """Host send time of the preceding packet, when one was sent."""

    expired: int = 0
    """Packets rejected after their presentation deadlines."""

    def clear(self) -> None:
        """Discard audio from a previous MA stream generation or connection."""
        self.pending.clear()
        self.last_sent_us = None
        self.expired = 0

    def enqueue(self, chunk: ReferenceChunk, now_us: int) -> None:
        """Resample source audio and schedule RTP no faster than real time.

        Args:
            chunk:
                MA audio and its original host presentation timestamp.

            now_us:
                Current host Unix time in microseconds.

        """
        target_frames = round(len(chunk.samples) * SAMPLE_RATE / chunk.sample_rate)
        if not target_frames:
            return
        positions = np.arange(target_frames) * chunk.sample_rate / SAMPLE_RATE
        resampled = np.interp(positions, np.arange(len(chunk.samples)), chunk.samples)
        pcm = (np.clip(resampled, -1, 32767 / 32768) * 32768).astype(np.int16)
        packet_count = (len(pcm) + FRAMES_PER_PACKET - 1) // FRAMES_PER_PACKET
        if len(self.pending) + packet_count > MAX_PENDING_PACKETS:
            LOGGER.warning(
                "reference.buffer status=overflow queued=%d incoming=%d",
                len(self.pending),
                packet_count,
            )
            return
        previous_us = self.pending[-1].send_us if self.pending else self.last_sent_us
        for offset in range(0, len(pcm), FRAMES_PER_PACKET):
            presentation_us = chunk.presentation_us + offset * 1_000_000 // SAMPLE_RATE
            due_us = max(presentation_us - TARGET_LEAD_US, now_us)
            if previous_us is not None:
                due_us = max(due_us, previous_us + PACKET_DURATION_US)
            self.pending.append(
                ScheduledPacket(
                    due_us, presentation_us, pcm[offset : offset + FRAMES_PER_PACKET]
                )
            )
            previous_us = due_us

    def send_ready(self, now_us: int) -> None:
        """Send at most one due RTP packet; skip audio past presentation.

        Args:
            now_us:
                Current host Unix time in microseconds.

        """
        if not self.pending or self.pending[0].send_us > now_us:
            return
        packet = self.pending.popleft()
        if packet.presentation_us < now_us:
            self.expired += 1
            if self.expired == 1 or self.expired % 100 == 0:
                LOGGER.warning("reference.packet status=expired count=%d", self.expired)
        else:
            self.sender.send(packet.pcm, packet.presentation_us)
            self.last_sent_us = now_us
        if self.pending:
            self.pending[0].send_us = max(
                self.pending[0].send_us, now_us + PACKET_DURATION_US
            )

    def wait_seconds(self, now_us: int) -> float | None:
        """Return the time until the next scheduled send, if any.

        Args:
            now_us:
                Current host Unix time in microseconds.

        """
        if not self.pending:
            return None
        return max(0, self.pending[0].send_us - now_us) / 1_000_000


def bind_haec(path: Path) -> socket.socket:
    """Listen on one MA player socket, reclaiming only a confirmed stale socket.

    Args:
        path:
            Per-player Unix datagram socket to receive HAEC packets.

    Returns:
        Listening Unix sequence-packet socket owned by the caller.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not stat.S_ISSOCK(path.stat().st_mode):
            raise FileExistsError(f"AEC socket path is occupied: {path}")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            try:
                probe.connect(str(path))
            except OSError as error:
                if error.errno == errno.EPROTOTYPE:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as old_probe:
                        try:
                            old_probe.connect(str(path))
                        except ConnectionRefusedError:
                            pass
                        else:
                            raise FileExistsError(
                                f"AEC player socket is already bound: {path}"
                            ) from error
                elif not isinstance(error, ConnectionRefusedError):
                    raise
                LOGGER.warning("reference.socket status=stale path=%s", path)
                path.unlink()
            else:
                raise FileExistsError(f"AEC player socket is already bound: {path}")
        finally:
            probe.close()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        sock.bind(str(path))
        sock.listen(1)
    except OSError:
        sock.close()
        raise
    return sock


def forward_haec(path: Path, destination: tuple[str, int]) -> None:
    """Forward one MA player reference to the device until interrupted.

    Args:
        path:
            Unix datagram socket path for a single AirPlay player.

        destination:
            Device IPv4 address and RTP port.

    """
    sender = RtpSender(destination)
    listener = bind_haec(path)
    connection: socket.socket | None = None
    paced = PacedReference(sender)
    last_stream: int | None = None
    last_sequence: int | None = None
    try:
        while True:
            now_us = time.time_ns() // 1000
            paced.send_ready(now_us)
            readable: list[socket.socket] = [listener]
            if connection is not None:
                readable.append(connection)
            ready, _, _ = select.select(readable, [], [], paced.wait_seconds(now_us))
            if listener in ready:
                incoming, _ = listener.accept()
                if connection is not None:
                    connection.close()
                connection = incoming
                paced.clear()
                last_stream = None
                last_sequence = None
                LOGGER.info("reference.socket status=connected")
            if connection is None or connection not in ready:
                continue
            packet = connection.recv(65_536)
            if not packet:
                connection.close()
                connection = None
                LOGGER.warning("reference.socket status=disconnected")
                continue
            try:
                chunk = parse_haec(packet)
            except ValueError:
                LOGGER.warning("reference.packet status=invalid", exc_info=True)
                continue
            if chunk is None:
                paced.clear()
                last_stream = None
                last_sequence = None
                continue
            if chunk.stream_id != last_stream:
                paced.clear()
                last_stream = chunk.stream_id
                last_sequence = None
                sender.ssrc = int.from_bytes(os.urandom(4), "big")
                sender.last_report_us = 0
                sender.first_packet = True
                LOGGER.info(
                    "reference.stream status=start generation=%d", chunk.stream_id
                )
            if last_sequence is not None and chunk.sequence != last_sequence + 1:
                LOGGER.warning(
                    "reference.packet status=gap sequence=%d", chunk.sequence
                )
            last_sequence = chunk.sequence
            paced.enqueue(chunk, time.time_ns() // 1000)
    finally:
        if connection is not None:
            connection.close()
        listener.close()
        sender.socket.close()
        path.unlink()
