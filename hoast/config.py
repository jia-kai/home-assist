"""Strict system-wide TOML configuration."""

import math
import os
import tomllib
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values


@dataclass(slots=True, frozen=True)
class WeatherConfig:
    """Home coordinates in WGS84 decimal degrees."""

    latitude: float
    """Latitude in degrees, from -90 through 90."""

    longitude: float
    """Longitude in degrees, from -180 through 180."""

    def __post_init__(self) -> None:
        """Reject nonnumeric, boolean, nonfinite, and out-of-range coordinates."""
        for name, value, bound in (
            ("latitude", self.latitude, 90),
            ("longitude", self.longitude, 180),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not -bound <= value <= bound
            ):
                raise ValueError(f"weather.{name} must be finite and within ±{bound}")


@dataclass(slots=True, frozen=True)
class MusicConfig:
    """Music Assistant endpoint and configured player; credentials remain external."""

    server_url: str = "http://localhost:8095"
    """HTTP(S) server origin without credentials, query, fragment, or path."""

    player_id: str = ""
    """Stable MA player ID; empty requires a unique available enabled player."""

    token_env: str = "MUSIC_ASSISTANT_TOKEN"
    """Environment or .env variable containing the long-lived access token."""

    def __post_init__(self) -> None:
        """Reject invalid endpoints, nonstring settings, and invalid token names."""
        if not all(
            isinstance(value, str)
            for value in (self.server_url, self.player_id, self.token_env)
        ):
            raise ValueError("Music settings must be strings")
        url = urlsplit(self.server_url)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or url.path not in ("", "/")
        ):
            raise ValueError("music.server_url must be an HTTP(S) server origin")
        if not self.token_env.isidentifier():
            raise ValueError("music.token_env must be an environment variable name")


@dataclass(slots=True, frozen=True)
class SwitchConfig:
    """Local smart plug endpoint."""

    ip: str
    """Literal IP address of the smart plug."""

    def __post_init__(self) -> None:
        """Reject missing, nonstring, or malformed IP addresses."""
        if not isinstance(self.ip, str):
            raise TypeError("switch.ip must be an IP address string")
        ip_address(self.ip)


@dataclass(slots=True, frozen=True)
class SatelliteConfig:
    """Native ESPHome microphone endpoint and bounded command capture settings."""

    host: str
    """Required satellite IP address or hostname, without a URL scheme or port."""

    port: int = 6053
    """Native API TCP port, between 1 and 65535."""

    capture_seconds: float = 6.0
    """Maximum command capture duration in seconds, between 0.1 and 30."""

    language: str = "en"
    """Whisper language code, or auto for language detection."""

    key_env: str | None = None
    """Optional environment/dotenv variable containing the Noise encryption key."""

    def __post_init__(self) -> None:
        """Reject malformed endpoints, capture bounds, language, and secret names."""
        if (
            not isinstance(self.host, str)
            or not self.host
            or any(character.isspace() or character in "/@" for character in self.host)
        ):
            raise ValueError("satellite.host must be a hostname or IP address")
        if ":" in self.host:
            ip_address(self.host)
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("satellite.port must be an integer between 1 and 65535")
        if (
            type(self.capture_seconds) not in (int, float)
            or not math.isfinite(self.capture_seconds)
            or not 0.1 <= self.capture_seconds <= 30
        ):
            raise ValueError("satellite.capture_seconds must be between 0.1 and 30")
        if not isinstance(self.language, str) or not self.language.isalpha():
            raise ValueError("satellite.language must be a language code or auto")
        if self.key_env is not None and (
            not isinstance(self.key_env, str) or not self.key_env.isidentifier()
        ):
            raise ValueError("satellite.key_env must be an environment variable name")


@dataclass(slots=True, frozen=True)
class SystemConfig:
    """Validated system settings."""

    weather: WeatherConfig
    """Required home weather coordinates."""

    music: MusicConfig | None = None
    """Optional music integration, enabled by an explicit [music] table."""

    switch: SwitchConfig | None = None
    """Optional smart plug endpoint, enabled by an explicit [switch] table."""

    satellite: SatelliteConfig | None = None
    """Voice endpoint, required for voice mode and optional for text/tool CLIs."""

    lfm_model_dir: Path | None = None
    """Custom OpenVINO LFM directory, relative to cwd; None uses the cache manifest."""


def satellite_key(config: SatelliteConfig, env_file: Path = Path(".env")) -> str | None:
    """Read an explicitly configured satellite key; absent key_env means plaintext.

    Args:
        config:
            Satellite settings selecting an optional secret variable name.

        env_file:
            Dotenv file used only when the variable is absent from the environment.

    """
    if config.key_env is None:
        return None
    key = (
        os.environ[config.key_env]
        if config.key_env in os.environ
        else dotenv_values(env_file, interpolate=False).get(config.key_env)
    )
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"Set {config.key_env} in the environment or {env_file}")
    return key.strip()


def music_token(config: MusicConfig, env_file: Path = Path(".env")) -> str:
    """Read a token from the environment, then an explicit dotenv file.

    Does not modify the process environment, interpolate variables, or execute
    shell syntax. Empty explicit credentials fail rather than fall back.

    Args:
        config:
            Music settings selecting the token variable name.

        env_file:
            Dotenv file, relative to the working directory unless absolute.

    """
    if config.token_env in os.environ:
        token = os.environ[config.token_env]
    else:
        token = dotenv_values(env_file, interpolate=False).get(config.token_env)
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"Set {config.token_env} in the environment or {env_file}")
    if any(character.isspace() for character in token.strip()):
        raise ValueError("Music Assistant token must not contain whitespace")
    return token.strip()


def load_config(path: Path) -> SystemConfig:
    """Read weather, optional integrations, and an optional custom LFM directory.

    Args:
        path:
            TOML file with weather coordinates, integration tables, and optional
            [lfm].model_dir, resolved relative to the working directory.

    """
    with path.open("rb") as source:
        data = tomllib.load(source)
    if "weather" not in data or set(data) - {
        "weather", "music", "switch", "satellite", "lfm"
    }:
        raise ValueError(
            "System config requires [weather] and optional [music], [switch], [satellite], [lfm]"
        )
    weather = data["weather"]
    if not isinstance(weather, dict) or set(weather) != {"latitude", "longitude"}:
        raise ValueError("[weather] requires exactly latitude and longitude")
    music = None
    if "music" in data:
        settings = data["music"]
        if not isinstance(settings, dict) or set(settings) - {
            "server_url",
            "player_id",
            "token_env",
        }:
            raise ValueError("[music] accepts only server_url, player_id, token_env")
        music = MusicConfig(**settings)
    switch = None
    if "switch" in data:
        settings = data["switch"]
        if not isinstance(settings, dict) or set(settings) != {"ip"}:
            raise ValueError("[switch] requires exactly ip")
        switch = SwitchConfig(**settings)
    satellite = None
    if "satellite" in data:
        settings = data["satellite"]
        if (
            not isinstance(settings, dict)
            or "host" not in settings
            or set(settings)
            - {"host", "port", "capture_seconds", "language", "key_env"}
        ):
            raise ValueError(
                "[satellite] requires host; accepts port, capture_seconds, language, key_env"
            )
        satellite = SatelliteConfig(**settings)
    lfm_model_dir = None
    if "lfm" in data:
        settings = data["lfm"]
        if not isinstance(settings, dict) or set(settings) != {"model_dir"}:
            raise ValueError("[lfm] requires exactly model_dir")
        model_dir = settings["model_dir"]
        if not isinstance(model_dir, str) or not model_dir.strip():
            raise ValueError("lfm.model_dir must be a nonempty path string")
        lfm_model_dir = Path(model_dir)
    return SystemConfig(WeatherConfig(**weather), music, switch, satellite, lfm_model_dir)
