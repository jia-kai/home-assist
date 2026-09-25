"""Check that the bridge derives its endpoints solely from Hoast config."""

from pathlib import Path

import pytest

from respeaker.core.bridge import bridge_endpoints


def test_bridge_uses_selected_player_and_satellite(tmp_path: Path) -> None:
    """Use the configured player ID under the shared Compose socket directory."""
    config = tmp_path / "config.toml"
    config.write_text(
        '[weather]\nlatitude = 0.0\nlongitude = 0.0\n'
        '[music]\nplayer_id = "airplay_test"\n'
        '[satellite]\nhost = "192.0.2.5"\n'
    )
    host, socket_path = bridge_endpoints(config, Path("/data/aec-reference"))
    assert host == "192.0.2.5"
    assert socket_path == Path("/data/aec-reference/aec-reference-airplay_test.sock")


def test_bridge_rejects_unsafe_or_missing_player(tmp_path: Path) -> None:
    """Never bind a guessed socket or let a player ID escape its directory."""
    config = tmp_path / "config.toml"
    for player in ('player_id = "../other"\n', ''):
        config.write_text(
            '[weather]\nlatitude = 0.0\nlongitude = 0.0\n'
            f'[music]\n{player}'
            '[satellite]\nhost = "192.0.2.5"\n'
        )
        with pytest.raises(ValueError, match="player ID"):
            bridge_endpoints(config, tmp_path)
