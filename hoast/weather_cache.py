"""Prepared home weather results with observation and destination-date expiry."""

import threading
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Literal
from urllib.error import URLError
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from .agent import render_weather
from .application import bounded_speech
from .llm import Tool
from .logging import get_logger
from .tts2 import TTS
from .weather import WeatherArguments, WeatherClient, observation_instant

logger = get_logger(__name__)


@dataclass(slots=True)
class PreparedWeather:
    """One provider result and its exact final spoken waveform."""

    result: dict[str, JsonValue]
    """Isolated home weather result."""
    text: str
    """Final bounded spoken text."""
    samples: NDArray[np.float32]
    """Mono waveform at the native synthesizer rate."""
    rate: int
    """Waveform sample rate in Hz."""
    started: float
    """Monotonic fetch start in seconds."""
    local_date: str
    """Destination-local date covered by the result."""
    zone: ZoneInfo
    """Destination timezone for midnight invalidation."""
    observation: datetime
    """Provider observation instant in UTC."""


class WeatherCache:
    """Share atomic prepared entries between the network and voice workers."""

    client: WeatherClient
    """Underlying live provider client."""
    entries: dict[str, PreparedWeather]
    """Prepared entries keyed by weather period."""
    last_hit: PreparedWeather | None
    """Most recent hit, used for waveform provenance."""
    turn_hits: list[PreparedWeather]
    """All hits during the current serialized voice turn."""
    turn_weather: list[PreparedWeather | None]
    """Cache provenance for each weather call, including live misses."""
    lock: threading.Lock
    """Protects publication and lookup of prepared entries."""
    voice_mode: bool
    """Whether the voice service is active; misses still use the live client."""

    def __init__(self, client: WeatherClient) -> None:
        """Attach the live client without I/O.

        Args:
            client:
                Configured home weather provider.

        """
        self.client = client
        self.entries = {}
        self.last_hit = None
        self.turn_hits = []
        self.turn_weather = []
        self.lock = threading.Lock()
        self.voice_mode = False

    def is_fresh(self, entry: PreparedWeather) -> bool:
        """Check fetch age, observation age and the destination-local date.

        Args:
            entry:
                Entry to validate independently of replacement in the cache.

        """
        now = datetime.now(UTC)
        age = now - entry.observation
        return (
            0 <= time.monotonic() - entry.started < 3600
            and timedelta(minutes=-5) <= age < timedelta(hours=1)
            and now.astimezone(entry.zone).date().isoformat() == entry.local_date
        )

    def valid(self, period: str) -> PreparedWeather | None:
        """Return an entry only while its observation and local date are valid.

        Args:
            period:
                Weather period key.

        """
        with self.lock:
            entry = self.entries.get(period)
        if entry is None:
            return None
        try:
            usable = (
                self.is_fresh(entry)
                and entry.result.get("period") == period
                and bounded_speech(render_weather(entry.result)) == entry.text
                and isinstance(entry.samples, np.ndarray)
                and entry.samples.dtype == np.float32
                and entry.samples.ndim == 1
                and entry.samples.size > 0
                and type(entry.rate) is int
                and entry.rate > 0
            )
        except AttributeError, KeyError, TypeError, ValueError, OverflowError:
            logger.warning(
                "weather.cache status=invalid period=%s", period, exc_info=True
            )
            return None
        return entry if usable else None

    def tool(self) -> Tool[WeatherArguments]:
        """Bind the normal weather schema to the cache-aware handler."""
        return replace(self.client.tool(), handler=self.get_weather)

    def get_weather(self, args: WeatherArguments) -> JsonValue:
        """Serve fresh home hits or live results, reporting provider failures safely.

        Provenance is retained only during voice turns for playback validation.

        Args:
            args:
                Schema-validated destination and period.

        """
        self.last_hit = None
        entry = self.valid(args.period)
        if (
            args.period in ("now", "today")
            and args.city.strip().casefold() in ("", "home")
            and entry is not None
        ):
            if self.voice_mode:
                self.last_hit = entry
                self.turn_hits.append(entry)
                self.turn_weather.append(entry)
            logger.info("get_weather status=cache_hit period=%s city=home", args.period)
            return deepcopy(entry.result)
        if self.voice_mode:
            self.turn_weather.append(None)
        try:
            return self.client.get_weather(args)
        except URLError, OSError, ValueError, TypeError, KeyError, OverflowError:
            logger.warning(
                "get_weather status=unavailable period=%s city=%r",
                args.period,
                args.city,
                exc_info=True,
            )
            return {"status": "unavailable"}

    def fetch(
        self, period: Literal["now", "today"]
    ) -> tuple[dict[str, JsonValue], float, str, ZoneInfo, datetime]:
        """Fetch and validate provider observation metadata without using TTS.

        Args:
            period:
                Home period to prepare.

        """
        if period not in ("now", "today"):
            raise ValueError("Unsupported cached weather period")
        started = time.monotonic()
        # The current request includes the observation timestamp even when preparing today.
        result = self.client.get_weather(WeatherArguments(period="now"))
        if not isinstance(result, dict):
            raise TypeError("Home weather is not an object")
        timezone_name = result.get("timezone")
        if not isinstance(timezone_name, str):
            raise TypeError("Home weather lacks a timezone")
        zone = ZoneInfo(timezone_name)
        raw_time = result.get("time")
        if not isinstance(raw_time, str):
            raise ValueError("Home weather lacks an observation time")  # noqa: TRY004
        now = datetime.now(UTC)
        observation = observation_instant(raw_time, zone, now)
        local_date = now.astimezone(zone).date().isoformat()
        days = result.get("days")
        if (
            not isinstance(days, list)
            or not days
            or not isinstance(days[0], dict)
            or days[0].get("date") != local_date
        ):
            raise ValueError("Home weather does not cover the current local date")
        fetched = (result, started, local_date, zone, observation)
        return self.today_from_now(fetched) if period == "today" else fetched

    @staticmethod
    def today_from_now(
        fetched: tuple[dict[str, JsonValue], float, str, ZoneInfo, datetime],
    ) -> tuple[dict[str, JsonValue], float, str, ZoneInfo, datetime]:
        """Derive today's distinct forecast from one validated current fetch.

        Args:
            fetched:
                Validated current forecast and unchanged freshness metadata.

        """
        result, started, local_date, zone, observation = fetched
        today = {
            key: value
            for key, value in result.items()
            if key
            not in (
                "current_temperature_c",
                "time",
                "interval_seconds",
                "temperature_c",
                "condition",
                "precipitation_mm",
            )
        }
        today["period"] = "today"
        return today, started, local_date, zone, observation

    def prepare(
        self,
        tts: TTS,
        fetched: tuple[dict[str, JsonValue], float, str, ZoneInfo, datetime],
    ) -> None:
        """Synthesize on the voice worker and atomically publish a fresh candidate.

        Args:
            tts:
                Synthesizer owned by the serialized voice worker.

            fetched:
                Validated network result and freshness metadata.

        """
        result, started, local_date, zone, observation = fetched
        text = bounded_speech(render_weather(result))
        samples, rate = tts.synthesize(text)
        candidate = PreparedWeather(
            deepcopy(result),
            text,
            samples,
            rate,
            started,
            local_date,
            zone,
            observation,
        )
        if not self.is_fresh(candidate):
            raise ValueError("Home weather expired during preparation")
        period = result["period"]
        if not isinstance(period, str):
            raise ValueError("Home weather lacks a period")  # noqa: TRY004
        with self.lock:
            self.entries[period] = candidate
        logger.info("weather.cache status=prepared period=%s", period)

    def refresh(self, tts: TTS, period: Literal["now", "today"]) -> None:
        """Fetch and prepare synchronously; intended for noninteractive callers.

        Args:
            tts:
                Serialized synthesizer.

            period:
                Home period to prepare.

        """
        self.prepare(tts, self.fetch(period))
