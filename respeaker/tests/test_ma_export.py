"""Exercise Music Assistant's nonblocking local AEC reference exporter."""

import struct
import time
from pathlib import Path

from music.aec_reference import AecReferenceSink
from respeaker.core.transport import HAEC_HEADER, bind_haec


def test_ma_export_sends_ordered_records_from_background_writer(tmp_path: Path) -> None:
    """Transfer short bursts without blocking MA's playback loop or losing records."""
    listener = bind_haec(tmp_path / "aec-reference-test.sock")
    listener.settimeout(2)
    sink = AecReferenceSink("test", socket_directory=str(tmp_path))
    try:
        presentation_us = time.time_ns() // 1000 + 5_000_000
        sink.stream_start(presentation_us, 48_000, 1, 2, "s16le")
        for index in range(20):
            sink.send_audio(
                presentation_us + index * 10_000,
                10_000,
                struct.pack("<480h", *range(480)),
            )
        connection, _ = listener.accept()
        connection.settimeout(2)
        with connection:
            sequences = [
                HAEC_HEADER.unpack_from(connection.recv(65_536))[6] for _ in range(21)
            ]
        assert sequences == list(range(21))
        assert sink._dropped == 0
    finally:
        sink.close()
        listener.close()


def test_ma_export_reports_bounded_queue_overflow(tmp_path: Path) -> None:
    """An absent bridge cannot block MA or grow its pending audio without bound."""
    sink = AecReferenceSink("absent", socket_directory=str(tmp_path))
    try:
        presentation_us = time.time_ns() // 1000 + 30_000_000
        sink.stream_start(presentation_us, 48_000, 1, 2, "s16le")
        for index in range(150):
            sink.send_audio(presentation_us + index * 10_000, 10_000, b"\x00\x00" * 480)
        assert sink._dropped > 0
        assert sink._pending.qsize() <= sink._pending.maxsize
    finally:
        sink.close()


def test_ma_export_connects_when_bridge_appears(tmp_path: Path) -> None:
    """A short bridge restart preserves queued audio without holding up playback."""
    sink = AecReferenceSink("delayed", socket_directory=str(tmp_path))
    presentation_us = time.time_ns() // 1000 + 4_000_000
    try:
        sink.stream_start(presentation_us, 48_000, 1, 2, "s16le")
        sink.send_audio(presentation_us, 10_000, b"\x00\x00" * 480)
        listener = bind_haec(tmp_path / "aec-reference-delayed.sock")
        listener.settimeout(2)
        try:
            connection, _ = listener.accept()
            connection.settimeout(2)
            with connection:
                assert [
                    HAEC_HEADER.unpack_from(connection.recv(65_536))[6]
                    for _ in range(2)
                ] == [0, 1]
        finally:
            listener.close()
    finally:
        sink.close()


def test_ma_export_reconnects_after_bridge_restart(tmp_path: Path) -> None:
    """A broken local connection retries without dropping the next audio record."""
    path = tmp_path / "aec-reference-restarted.sock"
    listener = bind_haec(path)
    listener.settimeout(2)
    sink = AecReferenceSink("restarted", socket_directory=str(tmp_path))
    try:
        presentation_us = time.time_ns() // 1000 + 4_000_000
        sink.stream_start(presentation_us, 48_000, 1, 2, "s16le")
        connection, _ = listener.accept()
        connection.settimeout(2)
        assert HAEC_HEADER.unpack_from(connection.recv(65_536))[6] == 0
        connection.close()
        listener.close()
        path.unlink()
        listener = bind_haec(path)
        listener.settimeout(2)
        sink.send_audio(presentation_us, 10_000, b"\x00\x00" * 480)
        replacement, _ = listener.accept()
        replacement.settimeout(2)
        with replacement:
            assert HAEC_HEADER.unpack_from(replacement.recv(65_536))[6] == 1
    finally:
        sink.close()
        listener.close()


def test_ma_export_close_with_absent_bridge_exits_writer(tmp_path: Path) -> None:
    """Ending a session does not leave a background retrying worker behind."""
    sink = AecReferenceSink("absent", socket_directory=str(tmp_path))
    sink.stream_start(time.time_ns() // 1000 + 20_000_000, 48_000, 1, 2, "s16le")
    sink.close()
    assert sink._writer is not None
    sink._writer.join(timeout=2)
    assert not sink._writer.is_alive()
