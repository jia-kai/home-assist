"""Verify the guarded AirPlay patch and its HAEC clock conversion."""

import hashlib
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from music import apply_aec_reference_patch as installer

PATCH = Path(__file__).resolve().parents[1] / "music/aec_reference.patch"


def _patch_old_blocks(patch: str) -> list[str]:
    """Return the original source blocks represented by the patch's hunks.

    Args:
        patch:
            Unified diff text containing the reviewed AirPlay changes.
    """
    blocks: list[str] = []
    for section in patch.split("@@ ")[1:]:
        lines = section.splitlines(keepends=True)[1:]
        blocks.append(
            "".join(line[1:] for line in lines if line.startswith((" ", "-")))
        )
    return blocks


def test_patch_requires_unique_upstream_context() -> None:
    """All hunks apply exactly once; missing and duplicated context are rejected."""
    patch = PATCH.read_text()
    blocks = _patch_old_blocks(patch)
    source = "\n# separate hunks\n".join(blocks)
    patched = installer.apply_patch(source, patch)
    assert "from .aec_reference import AecReferenceSink" in patched
    assert "reference.send_audio(start_us, end_us - start_us, chunk)" in patched
    with pytest.raises(RuntimeError, match="not a unique source match"):
        installer.apply_patch(source + blocks[0], patch)
    with pytest.raises(RuntimeError, match="not a unique source match"):
        installer.apply_patch(source.replace(blocks[0], "missing"), patch)


def test_installer_fails_before_writing_on_unknown_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsupported upstream hash cannot modify the session or install a helper.

    Args:
        tmp_path:
            Isolated directory used for simulated installer inputs.

        monkeypatch:
            Fixture providing temporary installer paths and source guard.
    """
    session = tmp_path / "stream_session.py"
    session.write_text("unreviewed upstream\n")
    monkeypatch.setattr(installer, "SESSION_PATH", session)
    monkeypatch.setattr(
        installer, "EXPECTED_SHA256", hashlib.sha256(b"reviewed").hexdigest()
    )
    with pytest.raises(RuntimeError, match="Unsupported Music Assistant AirPlay"):
        installer.main()
    assert session.read_text() == "unreviewed upstream\n"
    assert not (tmp_path / "aec_reference.py").exists()


def test_airplay_clock_conversion_tracks_wall_clock_steps() -> None:
    """Start and audio hooks translate the raw-clock timeline to host Unix time."""
    patch = PATCH.read_text()
    added = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    start = added.index(
        "        # HAEC timestamps are host Unix time, not the cliairplay raw-clock timeline."
    )
    end = added.index("            )", start) + 1
    start_hook = textwrap.dedent("\n".join(added[start:end]))
    audio = added.index(
        "                    if self._aec_references and self.start_unix_ms:"
    )
    audio_end = (
        added.index(
            "                            reference.send_audio(start_us, end_us - start_us, chunk)",
            audio,
        )
        + 1
    )
    audio_hook = textwrap.dedent("\n".join(added[audio:audio_end]))

    sink = SimpleNamespace(stream_start=Mock(), send_audio=Mock())
    session = SimpleNamespace(
        _aec_references={"speaker": sink},
        start_unix_ms=110_000,
        pcm_format=SimpleNamespace(
            sample_rate=48_000, channels=1, content_type=SimpleNamespace(value="s16le")
        ),
        _pcm_frame_size=2,
        _pcm_total_fed=9_600,
        _pcm_byte_rate=96_000,
    )
    # AirPlay's clock is 100 seconds; wall clock has stepped ahead by 5 seconds.
    clock = SimpleNamespace(time_ns=lambda: 105_000_000_000)
    context = {
        "self": session,
        "time": clock,
        "cli_clock_time": lambda: 100.0,
        "target_ms": 110_000,
        "chunk": bytes(960),
    }
    # The executed statements are extracted solely from the tracked patch.
    exec(start_hook, context)  # noqa: S102
    exec(audio_hook, context)  # noqa: S102
    sink.stream_start.assert_called_once_with(115_000_000, 48_000, 1, 2, "s16le")
    sink.send_audio.assert_called_once_with(115_100_000, 10_000, bytes(960))
    # Recompute the offset for subsequent chunks after a wall-clock adjustment.
    clock.time_ns = lambda: 103_000_000_000
    exec(audio_hook, context)  # noqa: S102
    assert sink.send_audio.call_args.args[:2] == (113_100_000, 10_000)
