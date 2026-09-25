"""Verify hardware activation preserves other private Hoast configuration."""

from pathlib import Path

import pytest

from respeaker.tools.activate import activate


def test_activate_preserves_music_and_creates_private_backup(tmp_path: Path) -> None:
    """Write only satellite settings and a mode-0600 backup."""
    config = tmp_path / "config.toml"
    original = '[music]\nplayer_id = "test"\n\n[satellite]\nhost = "127.0.0.1"\nport = 6053\n'
    config.write_text(original)
    backup = tmp_path / "backup.toml"
    activate(config, backup, "192.0.2.5")
    assert config.read_text().startswith('[music]\nplayer_id = "test"')
    assert 'host = "192.0.2.5"' in config.read_text()
    assert 'key_env = "ESPHOME_API_KEY"' in config.read_text()
    assert backup.read_text() == original
    assert backup.stat().st_mode & 0o777 == 0o600


def test_activate_rejects_existing_unexpected_key_setting(tmp_path: Path) -> None:
    """Preserve an existing user-selected encryption source for manual review."""
    config = tmp_path / "config.toml"
    config.write_text('[satellite]\nhost = "127.0.0.1"\nkey_env = "OTHER_KEY"\n')
    with pytest.raises(ValueError, match="manual review"):
        activate(config, tmp_path / "backup.toml", "192.0.2.5")
