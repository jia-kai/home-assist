"""Ensure ESPHome build secrets come only from root config and dotenv."""

from pathlib import Path

import pytest

from respeaker.tools.prepare import GENERATED_MARKER, write_secrets


def test_prepare_generates_reproducible_private_esphome_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refresh generated YAML from root inputs without making new API keys.

    Args:
        tmp_path:
            Temporary directory used for isolated configuration fixtures.

        monkeypatch:
            Replaces AP discovery without depending on a local Wi-Fi adapter.

    """
    config = tmp_path / "config.toml"
    config.write_text('[weather]\nlatitude=0.0\nlongitude=0.0\n[satellite]\nhost="192.0.2.5"\n')
    env = tmp_path / ".env"
    env.write_text('RESPEAKER_WIFI_PASSWORD="test password"\nESPHOME_API_KEY="test-key"\n'
                   'ESPHOME_OTA_PASSWORD="test-ota"\n')
    monkeypatch.setattr("respeaker.tools.prepare.active_ssid", lambda: "test-ap")
    monkeypatch.setattr("respeaker.tools.prepare.ap_host_address", lambda host: "192.0.2.1")
    output = tmp_path / "secrets.yaml"
    write_secrets(output, config, env)
    generated = output.read_text()
    assert generated.startswith(GENERATED_MARKER)
    assert 'wifi_ssid: "test-ap"\n' in generated
    assert 'ntp_host: "192.0.2.1"\n' in generated
    assert 'wifi_password: "test password"\n' in generated
    assert 'api_key: "test-key"\n' in generated
    assert output.stat().st_mode & 0o777 == 0o600
    write_secrets(output, config, env)
    assert output.read_text() == generated
    env.write_text(env.read_text().replace("test-ota", "new-ota"))
    write_secrets(output, config, env)
    assert 'ota_password: "new-ota"\n' in output.read_text()


def test_prepare_rejects_missing_root_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail before generating firmware YAML when a required secret is absent.

    Args:
        tmp_path:
            Temporary directory for the missing-credential fixture.

        monkeypatch:
            Replaces AP discovery with deterministic test results.

    """
    config = tmp_path / "config.toml"
    config.write_text('[weather]\nlatitude=0.0\nlongitude=0.0\n[satellite]\nhost="192.0.2.5"\n')
    env = tmp_path / ".env"
    env.write_text('RESPEAKER_WIFI_PASSWORD="test password"\n')
    monkeypatch.setattr("respeaker.tools.prepare.active_ssid", lambda: "test-ap")
    monkeypatch.setattr("respeaker.tools.prepare.ap_host_address", lambda host: "192.0.2.1")
    with pytest.raises(ValueError, match="ESPHOME_API_KEY"):
        write_secrets(tmp_path / "secrets.yaml", config, env)
