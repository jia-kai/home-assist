"""Offline checks for explicit smart plug actions and state confirmation."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hoast.config import SwitchConfig, load_config
from hoast.switch import _set_power, main


@pytest.mark.parametrize("argv", [[], ["--on", "--off"]])
def test_requires_exactly_one_action(argv: list[str]) -> None:
    """Reject missing or contradictory actions before device access.

    Args:
        argv:
            Invalid CLI action combination.

    """
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


@pytest.mark.parametrize("on", [True, False])
def test_power_confirmation(on: bool) -> None:
    """Send the requested action, read state, and disconnect.

    Args:
        on:
            Requested and observed relay state.

    """
    device = MagicMock(
        turn_on=AsyncMock(),
        turn_off=AsyncMock(),
        update=AsyncMock(),
        disconnect=AsyncMock(),
        is_on=on,
    )
    with patch("hoast.switch.Discover.discover_single", AsyncMock(return_value=device)):
        asyncio.run(_set_power(SwitchConfig("192.0.2.20"), on))
    (device.turn_on if on else device.turn_off).assert_awaited_once()
    (device.turn_off if on else device.turn_on).assert_not_awaited()
    device.update.assert_awaited_once()
    device.disconnect.assert_awaited_once()


def test_unconfirmed_state_disconnects() -> None:
    """A mismatched observed state fails without skipping connection cleanup."""
    device = MagicMock(
        turn_on=AsyncMock(),
        update=AsyncMock(),
        disconnect=AsyncMock(),
        is_on=False,
    )
    with (
        patch("hoast.switch.Discover.discover_single", AsyncMock(return_value=device)),
        pytest.raises(RuntimeError, match="did not confirm"),
    ):
        asyncio.run(_set_power(SwitchConfig("192.0.2.20"), True))
    device.disconnect.assert_awaited_once()


def test_switch_configuration(tmp_path: Path) -> None:
    """Load the explicit address and reject malformed switch settings.

    Args:
        tmp_path:
            Isolated directory for a minimal system configuration.

    """
    path = tmp_path / "config.toml"
    prefix = "[weather]\nlatitude=0\nlongitude=0\n[switch]\n"
    path.write_text(prefix + 'ip="192.0.2.20"\n')
    assert load_config(path).switch == SwitchConfig("192.0.2.20")
    for settings in ('ip="invalid"', "ip=20", 'address="192.0.2.20"'):
        path.write_text(prefix + settings + "\n")
        with pytest.raises((ValueError, TypeError)):
            load_config(path)
