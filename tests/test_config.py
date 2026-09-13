"""Offline strict TOML configuration tests."""

import os
from pathlib import Path

import pytest

from hoast.config import MusicConfig, WeatherConfig, load_config, music_token


def test_config(tmp_path: Path) -> None:
    """Load integer and floating-point home coordinates.

    Args:
        tmp_path:
            Isolated temporary directory.

    """
    path = tmp_path / "system.toml"
    path.write_text("[weather]\nlatitude = 90\nlongitude = -180.0\n")
    assert load_config(path).weather == WeatherConfig(90, -180)
    assert load_config(path).music is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[other]\nlatitude=1",
        "[weather]\nlatitude=1",
        "[weather]\nlatitude=1\nlongitude=2\nextra=3",
        "extra=3\n[weather]\nlatitude=1\nlongitude=2",
        "weather=1",
        "[weather]\nlatitude=true\nlongitude=2",
        '[weather]\nlatitude="1"\nlongitude=2',
        "[weather]\nlatitude=nan\nlongitude=2",
        "[weather]\nlatitude=1\nlongitude=inf",
        "[weather]\nlatitude=91\nlongitude=2",
        "[weather]\nlatitude=1\nlongitude=-181",
    ],
)
def test_invalid_config(tmp_path: Path, text: str) -> None:
    """Reject unknown, missing, mistyped, and invalid configuration values.

    Args:
        tmp_path:
            Isolated temporary directory.

        text:
            Invalid TOML configuration.

    """
    path = tmp_path / "system.toml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_file(tmp_path: Path) -> None:
    """Keep file errors visible to callers.

    Args:
        tmp_path:
            Isolated temporary directory.

    """
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.toml")


@pytest.mark.parametrize(
    "settings",
    ["", 'server_url="https://ma.example/"\nplayer_id="speaker"\ntoken_env="MA_TOKEN"'],
)
def test_music_config(tmp_path: Path, settings: str) -> None:
    """Accept an optional music table with defaults or every explicit field.

    Args:
        tmp_path:
            Isolated configuration directory.

        settings:
            Music table contents.

    """
    path = tmp_path / "system.toml"
    path.write_text("[weather]\nlatitude=1\nlongitude=2\n[music]\n" + settings)
    expected = (
        MusicConfig("https://ma.example/", "speaker", "MA_TOKEN")
        if settings
        else MusicConfig()
    )
    assert load_config(path).music == expected


@pytest.mark.parametrize(
    "settings",
    [
        "music=1\n",
        "[music]\nunknown=1",
        "[music]\ntoken='secret'",
        *[
            f"[music]\n{field}={value}"
            for field in ("server_url", "player_id", "token_env")
            for value in ("1", "true", "[]", "{}")
        ],
        *[
            f'[music]\nserver_url="{url}"'
            for url in (
                "",
                "ftp://host",
                "http://",
                "http://user:secret@host",
                "http://host/api",
                "http://host?token=secret",
                "http://host#fragment",
            )
        ],
        '[music]\ntoken_env=""',
        '[music]\ntoken_env="invalid-name"',
    ],
)
def test_invalid_music_config(tmp_path: Path, settings: str) -> None:
    """Reject unknown music keys, invalid origins, and mistyped fields.

    Args:
        tmp_path:
            Isolated configuration directory.

        settings:
            Invalid music configuration preceding the weather table.

    """
    path = tmp_path / "system.toml"
    path.write_text(settings + "\n[weather]\nlatitude=1\nlongitude=2\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_music_token_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Use the selected environment variable before dotenv without logging secrets.

    Args:
        tmp_path:
            Isolated dotenv directory.

        monkeypatch:
            Isolate credential environment variables.

        capsys:
            Capture both output streams.

    """
    config = MusicConfig(token_env="TEST_MA_TOKEN")
    path = tmp_path / ".env"
    path.write_text("TEST_MA_TOKEN=file-secret\n")
    monkeypatch.delenv(config.token_env, raising=False)
    assert music_token(config, path) == "file-secret"
    monkeypatch.setenv(config.token_env, " environment-secret ")
    assert music_token(config, path) == "environment-secret"
    monkeypatch.setenv(config.token_env, "")
    with pytest.raises(ValueError):
        music_token(config, path)
    monkeypatch.setenv(config.token_env, "secret with-whitespace")
    with pytest.raises(ValueError) as error:
        music_token(config, path)
    assert "secret" not in str(error.value)
    output = capsys.readouterr()
    assert output.out == output.err == ""


def test_music_token_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep missing/empty credentials explicit and dotenv uninterpolated and local.

    Args:
        tmp_path:
            Isolated dotenv directory.

        monkeypatch:
            Remove the selected environment variable.

    """
    config = MusicConfig(token_env="TEST_MA_TOKEN")
    monkeypatch.delenv(config.token_env, raising=False)
    path = tmp_path / ".env"
    with pytest.raises(ValueError):
        music_token(config, path)
    for text in ("OTHER=value", "TEST_MA_TOKEN", "TEST_MA_TOKEN=", "TEST_MA_TOKEN=' '"):
        path.write_text(text)
        with pytest.raises(ValueError):
            music_token(config, path)
    path.write_text("TEST_MA_TOKEN=${OTHER}\n")
    assert music_token(config, path) == "${OTHER}"
    assert config.token_env not in os.environ
