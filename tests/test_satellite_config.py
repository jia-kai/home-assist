"""Strict satellite configuration and explicit encryption credential semantics."""

from pathlib import Path

import pytest

from hoast.config import SatelliteConfig, load_config, satellite_key


def test_satellite_config(tmp_path: Path) -> None:
    """Load a complete endpoint and reject missing/unknown fields and bad values.

    Args:
        tmp_path:
            Temporary TOML location independent of machine configuration.

    """
    path = tmp_path / "config.toml"
    prefix = "[weather]\nlatitude=0\nlongitude=0\n[satellite]\n"
    path.write_text(
        prefix
        + 'host="mock-satellite"\nport=1234\ncapture_seconds=5\nlanguage="auto"\n'
    )
    assert load_config(path).satellite == SatelliteConfig(
        "mock-satellite", port=1234, capture_seconds=5, language="auto"
    )
    for settings in (
        "port=6053",
        'host=""',
        'host="tcp://localhost"',
        'host="localhost:6053"',
        'host="localhost"\nport=true',
        'host="localhost"\nport=0',
        'host="localhost"\ncapture_seconds=nan',
        'host="localhost"\ncapture_seconds=31',
        'host="localhost"\nextra=1',
        'host="localhost"\nkey_env=""',
    ):
        path.write_text(prefix + settings + "\n")
        with pytest.raises((ValueError, TypeError)):
            load_config(path)


def test_satellite_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit empty/missing credentials fail; omitted key_env selects plaintext.

    Args:
        tmp_path:
            Isolated dotenv file location.

        monkeypatch:
            Isolates the selected credential environment variable.

    """
    name = "TEST_SATELLITE_KEY"
    monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / ".env"
    config = SatelliteConfig("localhost", key_env=name)
    assert satellite_key(SatelliteConfig("localhost"), dotenv) is None
    with pytest.raises(ValueError):
        satellite_key(config, dotenv)
    dotenv.write_text(f"{name}=fixture-key\n")
    assert satellite_key(config, dotenv) == "fixture-key"
    monkeypatch.setenv(name, "")
    with pytest.raises(ValueError):
        satellite_key(config, dotenv)
