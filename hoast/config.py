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
class SystemConfig:
    """Validated system settings."""

    weather: WeatherConfig
    """Required home weather coordinates."""

    music: MusicConfig | None = None
    """Optional music integration, enabled by an explicit [music] table."""

    switch: SwitchConfig | None = None
    """Optional smart plug endpoint, enabled by an explicit [switch] table."""


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
    """Read required weather and optional music/switch settings; reject unknowns.

    Args:
        path:
            TOML file with weather coordinates and optional music/switch settings.

    """
    with path.open("rb") as source:
        data = tomllib.load(source)
    if "weather" not in data or set(data) - {"weather", "music", "switch"}:
        raise ValueError(
            "System config requires [weather] and optional [music], [switch]"
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
    return SystemConfig(WeatherConfig(**weather), music, switch)
