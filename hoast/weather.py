"""Compact Open-Meteo weather tool; network and provider errors propagate."""

import json
import logging
import math
from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Literal
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

from pydantic import Field, JsonValue, field_validator

from hoast.config import WeatherConfig
from hoast.llm import Tool, ToolArguments, declaration_only

logger = logging.getLogger(__name__)

_FORECAST = "https://api.open-meteo.com/v1/forecast"
_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
_DAILY_UNITS = {
    "time": "iso8601",
    "temperature_2m_min": "°C",
    "temperature_2m_max": "°C",
    "precipitation_probability_max": "%",
    "rain_sum": "mm",
    "showers_sum": "mm",
    "weather_code": "wmo code",
}
_CURRENT_UNITS = {
    "time": "iso8601",
    "interval": "seconds",
    "temperature_2m": "°C",
    "weather_code": "wmo code",
    "precipitation": "mm",
}
_CONDITIONS = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "heavy hailstorm",
}


class WeatherArguments(ToolArguments):
    """Weather period and optional destination."""

    period: Literal["now", "today", "tomorrow", "next_week"] = Field(
        default="now",
        description=(
            "now: current + today; today; tomorrow; next_week: next Monday–Sunday"
        ),
    )
    """Destination-local period; now includes current temperature with today."""

    city: str = Field(
        default="",
        max_length=200,
        description=(
            "Empty for home; otherwise city with supplied region/country, e.g. Ottawa, Ohio, US"
        ),
    )
    """City query; blank or trimmed, case-insensitive Home uses home coordinates."""

    @field_validator("period", mode="before")
    @classmethod
    def normalize_period(cls, value: object) -> object:
        """Normalize period strings; unknown strings select now, other types fail.

        Args:
            value:
                Raw period argument; case, surrounding whitespace, spaces, and
                hyphens are normalized before matching supported forecast periods.

        """
        if not isinstance(value, str):
            return value
        normalized = "_".join(value.strip().casefold().replace("-", " ").split())
        return normalized if normalized in {"today", "tomorrow", "next_week"} else "now"


def declare_weather_tool() -> Tool[WeatherArguments]:
    """Return the production weather schema without requiring home coordinates or a client."""
    return Tool(
        "get_weather",
        "Get weather, outside temperature, and rain forecast in Celsius.",
        WeatherArguments,
        declaration_only,
    )


class _NoRedirect(HTTPRedirectHandler):
    """Keep requests on the fixed provider endpoints."""

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        """Reject redirects through urllib's normal HTTPError path.

        Args:
            req:
                Original request.

            fp:
                Response stream.

            code:
                HTTP status.

            msg:
                HTTP reason.

            headers:
                Response headers.

            newurl:
                Rejected redirect target.

        """
        return


def _object(value: JsonValue) -> dict[str, JsonValue]:
    """Require a JSON object at the provider boundary.

    Args:
        value:
            Decoded provider value.

    """
    if not isinstance(value, dict):
        raise TypeError("Expected provider object")
    return value


def _text(value: JsonValue) -> str:
    """Require a nonempty provider string.

    Args:
        value:
            Decoded provider value.

    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty provider string")
    return value


def _number(
    value: JsonValue, low: float = -math.inf, high: float = math.inf
) -> float | None:
    """Preserve null measurements; reject invalid or nonfinite numeric values.

    Args:
        value:
            Provider measurement; booleans are invalid.

        low:
            Inclusive lower bound in the measurement's units.

        high:
            Inclusive upper bound in the measurement's units.

    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected numeric provider measurement or null")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Invalid provider measurement")
    return float(value)


def _condition(value: JsonValue) -> str | None:
    """Normalize WMO codes, preserving nulls and labeling unknown integral codes.

    Args:
        value:
            Provider weather code; nonnegative integers or null are accepted.

    """
    code = _number(value, 0)
    if code is None:
        return None
    if not code.is_integer():
        raise ValueError("Weather code must be integral")
    return _CONDITIONS.get(int(code), f"unknown ({int(code)})")


def _summary(days: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    """Aggregate selected days; each metric is null if any required input is null.

    Args:
        days:
            Nonempty chronological daily results. Rainy dates have combined
            large-scale rain and convective showers (rain_mm) > 0;
            probability is the maximum daily precipitation probability, not the
            probability of precipitation occurring at least once during the week.

    """
    assert days
    result: dict[str, JsonValue] = {}
    for key in (
        "temperature_min_c",
        "temperature_max_c",
        "precipitation_probability_max_pct",
    ):
        values = [_number(day[key]) for day in days]
        known = [value for value in values if value is not None]
        result[key] = (
            None
            if len(known) != len(values)
            else min(known)
            if key == "temperature_min_c"
            else max(known)
        )
    rain = [_number(day["rain_mm"], 0) for day in days]
    result["rainy_dates"] = (
        None
        if any(value is None for value in rain)
        else [
            day["date"]
            for day, value in zip(days, rain, strict=True)
            if value is not None and value > 0
        ]
    )
    return result


def _units(data: dict[str, JsonValue], expected: dict[str, str]) -> None:
    """Require exact units for every requested field.

    Args:
        data:
            Provider units object.

        expected:
            Requested field-to-unit mapping.

    """
    for key, unit in expected.items():
        if data[key] != unit:
            raise ValueError(f"Unexpected provider unit for {key}")


class WeatherClient:
    """Synchronous client with at most one geocoding and one forecast request."""

    config: WeatherConfig
    """Validated home coordinates."""

    clock: Callable[[], datetime] | None
    """Optional aware clock; otherwise use provider current.time for local dates."""

    def __init__(
        self, config: WeatherConfig, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        """Store home coordinates and an optional deterministic clock.

        Args:
            config:
                Home latitude/longitude.

            clock:
                Optional callable returning an aware datetime, converted to the
                destination timezone when selecting daily dates.

        """
        self.config = config
        self.clock = clock

    def tool(self) -> Tool[WeatherArguments]:
        """Bind the shared weather declaration to this client's validated handler."""
        return replace(declare_weather_tool(), handler=self.get_weather)

    def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, JsonValue]:
        """Fetch bounded JSON without retries or redirects; propagate HTTP errors.

        Args:
            endpoint:
                One of the two fixed Open-Meteo endpoints.

            params:
                URL-encoded query parameters.

        """
        if endpoint not in {_FORECAST, _GEOCODING}:
            raise ValueError("Unsupported weather endpoint")
        request = Request(
            endpoint + "?" + urlencode(params),
            headers={"User-Agent": "hoast/0.1", "Accept": "application/json"},
        )
        with build_opener(_NoRedirect()).open(request, timeout=10) as response:
            body = response.read(1_048_577)
        if len(body) > 1_048_576:
            raise ValueError("Weather response exceeds 1 MiB")
        data = _object(json.loads(body))
        if data.get("error"):
            raise ValueError(f"Open-Meteo error: {data.get('reason')}")
        return data

    def _locations(self, city: str) -> list[dict[str, JsonValue]]:
        """Filter qualifiers and retain exact-name, capital, and population metadata.

        Optional provider feature codes and population remain null when absent.
        Population, when present, must be a nonnegative integer count of people.

        Args:
            city:
                City followed by optional comma-separated region/country tokens.

        """
        parts = [part.strip() for part in city.split(",")]
        if any(not part for part in parts):
            raise ValueError("City qualifiers must not be empty")
        data = self._request(
            _GEOCODING,
            {
                "name": parts[0],
                "count": "10",
                "language": "en",
                "format": "json",
            },
        )
        results = data.get("results", [])
        if not isinstance(results, list):
            raise TypeError("Invalid geocoding results")
        places: list[dict[str, JsonValue]] = []
        for result in results:
            item = _object(result)
            name = _text(item["name"])
            country = _text(item["country"])
            code = _text(item["country_code"])
            regions = [
                _text(item[key])
                for key in ("admin1", "admin2")
                if key in item and item[key] is not None
            ]
            qualifiers = {value.casefold() for value in [country, code, *regions]}
            if not all(part.casefold() in qualifiers for part in parts[1:]):
                continue
            lat = _number(item["latitude"], -90, 90)
            lon = _number(item["longitude"], -180, 180)
            if lat is None or lon is None:
                raise ValueError("Missing geocoding coordinates")
            label = ", ".join(dict.fromkeys([name, *regions, country]))
            population = _number(item.get("population"), 0)
            if population is not None and not population.is_integer():
                raise ValueError("Geocoding population must be integral")
            feature = item.get("feature_code")
            place: dict[str, JsonValue] = {
                "city": label,
                "latitude": lat,
                "longitude": lon,
                "exact_name": name.casefold() == parts[0].casefold(),
                "feature_code": None if feature is None else _text(feature),
                "population": population,
            }
            if place not in places:
                places.append(place)
        return places

    def get_weather(self, args: WeatherArguments) -> JsonValue:
        """Fetch weather and log arguments, outcomes, and full failure tracebacks.

        Successful outcomes use info, location clarification uses warning, and
        complete normalized results (including candidate lists) use debug.
        Exceptions are logged and propagated without retries or replacement data.

        Args:
            args:
                Validated period and city query; blank or Home uses configured
                home coordinates. Period now includes today's forecast.

        """
        logger.info("get_weather requested period=%s city=%r", args.period, args.city)
        try:
            result = self._get_weather(args)
        except Exception:
            logger.exception(
                "get_weather status=failed period=%s city=%r", args.period, args.city
            )
            raise
        status = result.get("status", "success")
        logger.debug(
            "get_weather period=%s city=%r result=%r", args.period, args.city, result
        )
        if status in ("ambiguous_city", "city_not_found"):
            logger.warning(
                "get_weather status=%s period=%s city=%r reason=%s",
                status,
                args.period,
                args.city,
                "multiple locations matched; exact location required"
                if status == "ambiguous_city"
                else "no locations matched; exact location required",
            )
        else:
            logger.info(
                "get_weather status=success period=%s city=%r resolved_city=%r",
                args.period,
                args.city,
                result["city"],
            )
        return result

    def _distance_from_home(self, place: dict[str, JsonValue]) -> float:
        """Return great-circle distance from configured home in kilometers.

        Uses a spherical Earth of radius 6371 km and handles longitude wraparound.

        Args:
            place:
                Validated geocoding candidate with latitude and longitude in degrees.

        """
        latitude = _number(place["latitude"], -90, 90)
        longitude = _number(place["longitude"], -180, 180)
        assert latitude is not None and longitude is not None
        home_lat = math.radians(self.config.latitude)
        lat = math.radians(latitude)
        delta_lon = math.radians(longitude - self.config.longitude)
        haversine = (
            math.sin((lat - home_lat) / 2) ** 2
            + math.cos(home_lat) * math.cos(lat) * math.sin(delta_lon / 2) ** 2
        )
        return 6371 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, haversine))))

    def _location_rank(
        self, place: dict[str, JsonValue]
    ) -> tuple[int, int, float, float]:
        """Rank filtered candidates by exact name, capital, population, then distance.

        Smaller tuples rank first. Only PPLC identifies a current national capital;
        missing population ranks below known counts, including zero. Ties in all
        criteria retain provider order through stable minimum selection.

        Args:
            place:
                Validated candidate from _locations, including nullable metadata.

        """
        population = place["population"]
        assert population is None or isinstance(population, float)
        return (
            0 if place["exact_name"] else 1,
            0 if place["feature_code"] == "PPLC" else 1,
            -population if population is not None else 1,
            self._distance_from_home(place),
        )

    def _get_weather(self, args: WeatherArguments) -> dict[str, JsonValue]:
        """Return compact weather or city choices; null measurements remain null.

        Current precipitation_mm totals the interval_seconds preceding time.
        Daily rain_mm sums Open-Meteo large-scale rain_sum and convective
        showers_sum, excluding snow. Both components must use mm and be finite,
        nonnegative numbers or null; rain_mm is null if either component is null.
        precipitation_probability_max_pct includes all precipitation types.
        Next week includes days and a summary whose
        metrics are null if any selected day's relevant input is null (including
        rainy_dates). Missing keys and malformed provider data raise.
        Period now selects today's forecast and includes current_temperature_c,
        even when that observation is null; explicit today omits that field.
        Blank city or case-insensitive Home, after trimming whitespace, selects
        configured home coordinates without geocoding.
        Qualified candidates prefer exact names, national capitals, population,
        then great-circle proximity to home. Exact ties retain provider order.
        No matches request clarification.

        Args:
            args:
                Validated period and destination query.

        """
        location: dict[str, JsonValue] = {
            "city": "Home",
            "latitude": self.config.latitude,
            "longitude": self.config.longitude,
        }
        city = args.city.strip()
        if city and city.casefold() != "home":
            places = self._locations(city)
            logger.debug("get_weather city=%r candidates=%r", city, places)
            if not places:
                return {
                    "status": "city_not_found",
                    "period": args.period,
                    "city_query": args.city,
                    "choices": list(places),
                }
            location = min(places, key=self._location_rank)
            if len(places) > 1:
                logger.info(
                    "get_weather city=%r resolution=ranked_city candidates=%d "
                    "selected=%r rank=%r distance_km=%.1f "
                    "criteria=exact_name,capital,population,home_distance",
                    city,
                    len(places),
                    location["city"],
                    self._location_rank(location),
                    self._distance_from_home(location),
                )
        data = self._request(
            _FORECAST,
            {
                "latitude": str(location["latitude"]),
                "longitude": str(location["longitude"]),
                "timezone": "auto",
                "forecast_days": "16",
                "temperature_unit": "celsius",
                "precipitation_unit": "mm",
                "current": "temperature_2m,weather_code,precipitation",
                "daily": ",".join(key for key in _DAILY_UNITS if key != "time"),
            },
        )
        timezone = _text(data["timezone"])
        zone = ZoneInfo(timezone)
        current = _object(data["current"])
        _units(_object(data["current_units"]), _CURRENT_UNITS)
        interval = _number(current["interval"], 0)
        if interval is None or interval <= 0:
            raise ValueError("Current interval must be positive seconds")
        moment = datetime.fromisoformat(_text(current["time"]))
        local_date = (
            moment.date() if moment.tzinfo is None else moment.astimezone(zone).date()
        )
        if self.clock is not None:
            now = self.clock()
            if now.tzinfo is None or now.utcoffset() is None:
                raise ValueError("Injected weather clock must be timezone-aware")
            local_date = now.astimezone(zone).date()
        temperature = _number(current["temperature_2m"])
        precipitation = _number(current["precipitation"], 0)
        condition = _condition(current["weather_code"])
        output: dict[str, JsonValue] = {
            "city": location["city"],
            "timezone": timezone,
            "period": args.period,
        }
        daily = _object(data["daily"])
        _units(_object(data["daily_units"]), _DAILY_UNITS)
        arrays: dict[str, list[JsonValue]] = {}
        for key in _DAILY_UNITS:
            values = daily[key]
            if not isinstance(values, list):
                raise TypeError(f"Expected daily array: {key}")
            arrays[key] = values
        times = arrays["time"]
        if not 1 <= len(times) <= 16 or any(
            len(a) != len(times) for a in arrays.values()
        ):
            raise ValueError("Unaligned or oversized daily arrays")
        days: dict[date, dict[str, JsonValue]] = {}
        previous: date | None = None
        for index, value in enumerate(times):
            day = date.fromisoformat(_text(value))
            if previous is not None and day != previous + timedelta(days=1):
                raise ValueError("Daily dates must be consecutive and unique")
            previous = day
            minimum = _number(arrays["temperature_2m_min"][index])
            maximum = _number(arrays["temperature_2m_max"][index])
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError("Daily minimum exceeds maximum")
            rain = _number(arrays["rain_sum"][index], 0)
            showers = _number(arrays["showers_sum"][index], 0)
            days[day] = {
                "date": day.isoformat(),
                "temperature_min_c": minimum,
                "temperature_max_c": maximum,
                "condition": _condition(arrays["weather_code"][index]),
                "precipitation_probability_max_pct": _number(
                    arrays["precipitation_probability_max"][index], 0, 100
                ),
                "rain_mm": None if rain is None or showers is None else rain + showers,
            }
        start = local_date
        count = 1
        if args.period == "tomorrow":
            start += timedelta(days=1)
        elif args.period == "next_week":
            start += timedelta(days=7 - start.weekday())
            count = 7
        selected = [start + timedelta(days=offset) for offset in range(count)]
        if any(day not in days for day in selected):
            raise ValueError("Provider forecast does not cover requested dates")
        output["days"] = [days[day] for day in selected]
        if args.period == "now":
            output["current_temperature_c"] = temperature
            output.update(
                time=current["time"],
                interval_seconds=interval,
                temperature_c=temperature,
                condition=condition,
                precipitation_mm=precipitation,
            )
        if args.period == "next_week":
            output["summary"] = _summary([days[day] for day in selected])
        return output
