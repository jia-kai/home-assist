"""Desktop audio routing avoids direct ALSA dmix contention without hardware access."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hoast.tts_audio import system_output_device


@pytest.mark.parametrize(
    "expected,devices",
    [
        (
            2,
            [
                {"name": "default", "max_output_channels": 2, "hostapi": 0},
                {"name": "pulse", "max_output_channels": 2, "hostapi": 0},
                {"name": "pipewire", "max_output_channels": 2, "hostapi": 0},
            ],
        ),
        (
            1,
            [
                {"name": "pipewire", "max_output_channels": 0, "hostapi": 0},
                {"name": "pulse", "max_output_channels": 2, "hostapi": 0},
            ],
        ),
        (None, [{"name": "pipewire", "max_output_channels": 2, "hostapi": 1}]),
        (None, [{"name": "default", "max_output_channels": 2, "hostapi": 0}]),
    ],
)
def test_linux_desktop_output(
    expected: int | None,
    devices: list[dict[str, str | int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer output-capable ALSA desktop adapters and otherwise retain the system default.

    Args:
        expected:
            Selected device index, or None for PortAudio's default.

        devices:
            Mock PortAudio inventory including irrelevant and input-only adapters.

        monkeypatch:
            Replaces platform and PortAudio discovery with deterministic fixtures.

    """
    monkeypatch.setattr("hoast.tts_audio.sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        "hoast.tts_audio.sd.query_devices", MagicMock(return_value=devices)
    )
    monkeypatch.setattr(
        "hoast.tts_audio.sd.query_hostapis",
        MagicMock(return_value=[{"name": "ALSA"}, {"name": "Other"}]),
    )
    assert system_output_device() == expected


def test_other_platform_preserves_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-Linux playback does not probe ALSA adapters.

    Args:
        monkeypatch:
            Supplies a non-Linux platform without changing the process platform.

    """
    query = MagicMock()
    monkeypatch.setattr("hoast.tts_audio.sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr("hoast.tts_audio.sd.query_devices", query)
    assert system_output_device() is None
    query.assert_not_called()
