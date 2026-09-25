"""Contract tests for the MA-to-RTP reference bridge."""

import socket
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from respeaker.core.transport import (
    HAEC_HEADER,
    RTCP_SR,
    RTP_HEADER,
    PacedReference,
    ReferenceChunk,
    RtpSender,
    bind_haec,
    parse_haec,
)


def test_parse_haec_mono_stereo_and_start() -> None:
    """Preserve MA timestamps and downmix interleaved source samples."""
    header = HAEC_HEADER.pack(b"HAEC", 2, 2, 1, 0, 7, 3, 1_234_000, 1000, 48_000, 2, 4)
    chunk = parse_haec(header + struct.pack("<4h", 32767, -32768, 1000, 3000))
    assert chunk is not None
    assert (chunk.stream_id, chunk.sequence, chunk.presentation_us) == (7, 3, 1_234_000)
    np.testing.assert_allclose(chunk.samples, [-1 / 65536, 2000 / 32768], atol=1e-6)
    start = HAEC_HEADER.pack(b"HAEC", 2, 1, 1, 0, 8, 0, 2_000_000, 0, 48_000, 1, 2)
    assert parse_haec(start) is None
    with pytest.raises(ValueError, match="Misaligned"):
        parse_haec(header + b"a")


def test_rtp_and_rtcp_map_host_presentation_time() -> None:
    """Transmit big-endian L16 audio and an RTCP sender report for its sample time."""
    rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtcp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp.bind(("127.0.0.1", 0))
    rtcp.bind(("127.0.0.1", rtp.getsockname()[1] + 1))
    rtp.settimeout(1)
    rtcp.settimeout(1)
    sender = RtpSender(("127.0.0.1", rtp.getsockname()[1]), ssrc=55)
    try:
        sender.send(np.array([1234, -1234], dtype=np.int16), 1_750_000_000_000_000)
        audio, _ = rtp.recvfrom(2048)
        version, marker, seq, rtp_time, ssrc = RTP_HEADER.unpack_from(audio)
        assert (version, marker, seq, ssrc) == (0x80, 0x80, 0, 55)
        assert audio[RTP_HEADER.size :] == struct.pack(">2h", 1234, -1234)
        report, _ = rtcp.recvfrom(2048)
        (
            _,
            packet_type,
            length,
            report_ssrc,
            ntp_sec,
            ntp_frac,
            mapped_time,
            count,
            octets,
        ) = RTCP_SR.unpack(report)
        assert (packet_type, length, report_ssrc, mapped_time, count, octets) == (
            200,
            6,
            55,
            rtp_time,
            1,
            4,
        )
        assert ntp_sec == 1_750_000_000 + 2_208_988_800
        assert ntp_frac == 0
    finally:
        sender.socket.close()
        rtp.close()
        rtcp.close()


def test_rtp_sequence_and_timestamps_continue_across_haec_chunks() -> None:
    """Keep sample-time spacing and the same SSRC across ordinary source chunks."""
    rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp.bind(("127.0.0.1", 0))
    rtp.settimeout(1)
    sender = RtpSender(("127.0.0.1", rtp.getsockname()[1]), ssrc=11)
    try:
        sender.send(np.zeros(480, dtype=np.int16), 1_750_000_000_000_000)
        sender.send(np.zeros(480, dtype=np.int16), 1_750_000_000_010_000)
        one = RTP_HEADER.unpack_from(rtp.recvfrom(1024)[0])
        two = RTP_HEADER.unpack_from(rtp.recvfrom(1024)[0])
        assert one[1] == 0x80
        assert two[1] == 0
        assert two[2] == one[2] + 1
        assert (two[3] - one[3]) & 0xFFFFFFFF == 480
        assert one[4] == two[4] == 11
    finally:
        sender.socket.close()
        rtp.close()


def test_parse_haec_rejects_wrong_frame_size_and_decodes_s24le() -> None:
    """Catch invalid source formats and sign-extend MA's 24-bit PCM correctly."""
    header = HAEC_HEADER.pack(b"HAEC", 2, 2, 2, 0, 2, 0, 300, 1000, 48_000, 1, 3)
    data = parse_haec(header + b"\x00\x00\x80\xff\xff\x7f")
    assert data is not None
    np.testing.assert_allclose(data.samples, [-1, 1 - 1 / 8388608])
    invalid = HAEC_HEADER.pack(b"HAEC", 2, 2, 2, 0, 2, 0, 300, 1000, 48_000, 1, 4)
    with pytest.raises(ValueError, match="frame size"):
        parse_haec(invalid + b"\x00" * 4)


def test_unix_socket_reclaims_stale_path_but_refuses_live_owner(tmp_path: Path) -> None:
    """Recover after unclean shutdown without stealing an active player socket."""
    path = tmp_path / "aec-reference-test.sock"
    owner = bind_haec(path)
    try:
        with pytest.raises(FileExistsError, match="already bound"):
            bind_haec(path)
    finally:
        owner.close()
    # The previous Unix socket inode remains, but no process owns it.
    replacement = bind_haec(path)
    replacement.close()
    path.unlink()


def test_unix_socket_detects_live_old_datagram_owner(tmp_path: Path) -> None:
    """Refuse to replace an active datagram bridge during deployment."""
    path = tmp_path / "player.sock"
    old = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    old.bind(str(path))
    try:
        with pytest.raises(FileExistsError, match="already bound"):
            bind_haec(path)
    finally:
        old.close()
    successor = bind_haec(path)
    successor.close()
    path.unlink()


def test_paced_reference_sends_ten_millisecond_packets_with_original_timestamps() -> (
    None
):
    """A burst of PCM reaches UDP over time, three seconds ahead of playback."""
    rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp.bind(("127.0.0.1", 0))
    rtp.settimeout(0.1)
    sender = RtpSender(("127.0.0.1", rtp.getsockname()[1]), ssrc=42)
    paced = PacedReference(sender)
    now_us = 1_750_000_000_000_000
    chunk = ReferenceChunk(1, 1, now_us + 3_500_000, 48_000, np.zeros(1440, np.float32))
    try:
        paced.enqueue(chunk, now_us)
        assert paced.wait_seconds(now_us) == 0.5
        paced.send_ready(now_us + 499_999)
        with pytest.raises(socket.timeout):
            rtp.recv(1024)
        for offset in (500_000, 510_000, 520_000):
            paced.send_ready(now_us + offset)
            header = RTP_HEADER.unpack_from(rtp.recv(1024))
            assert (
                header[3]
                == ((now_us + 3_500_000 + (offset - 500_000)) * 48_000 // 1_000_000)
                & 0xFFFFFFFF
            )
        assert paced.wait_seconds(now_us + 520_000) is None
    finally:
        sender.socket.close()
        rtp.close()


def test_paced_reference_handles_late_burst_without_wifi_flood() -> None:
    """When MA arrives with less than three seconds' lead, space its RTP."""
    rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp.bind(("127.0.0.1", 0))
    rtp.settimeout(0.1)
    sender = RtpSender(("127.0.0.1", rtp.getsockname()[1]), ssrc=42)
    paced = PacedReference(sender)
    now_us = int(time.time_ns() // 1000)
    chunk = ReferenceChunk(1, 1, now_us + 2_800_000, 48_000, np.zeros(2400, np.float32))
    try:
        paced.enqueue(chunk, now_us)
        paced.send_ready(now_us)
        rtp.recv(1024)
        paced.send_ready(now_us)
        with pytest.raises(socket.timeout):
            rtp.recv(1024)
        assert paced.wait_seconds(now_us) == 0.01
        paced.clear()
        assert paced.wait_seconds(now_us) is None
    finally:
        sender.socket.close()
        rtp.close()


def test_paced_reference_bounds_queued_audio(caplog: pytest.LogCaptureFixture) -> None:
    """A large MA burst reports overflow without taking down the bridge."""
    sender = RtpSender(("127.0.0.1", 5070))
    paced = PacedReference(sender)
    try:
        chunk = ReferenceChunk(1, 1, 9_000_000, 48_000, np.zeros(801 * 480, np.float32))
        paced.enqueue(chunk, 1_000_000)
        assert not paced.pending
        assert "reference.buffer status=overflow" in caplog.text
    finally:
        sender.socket.close()
