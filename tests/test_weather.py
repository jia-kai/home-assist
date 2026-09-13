"""Deterministic provider fixtures exercising the weather boundary."""

import logging
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from urllib.error import URLError
from urllib.request import Request

import pytest
from pydantic import JsonValue, ValidationError

from hoast import weather
from hoast.agent import render_weather
from hoast.config import WeatherConfig
from hoast.llm import ToolCall, ToolRegistry
from hoast.weather import WeatherArguments, WeatherClient


def forecast(
    start: str = "2026-12-28", zone: str = "Pacific/Kiritimati"
) -> dict[str, JsonValue]:
    """Build 16 days of large-scale rain and zero convective showers across New Year.

    Args:
        start:
            Destination-local first date, also the current date.

        zone:
            Provider destination timezone.

    """
    dates: list[JsonValue] = [
        (date.fromisoformat(start) + timedelta(days=i)).isoformat() for i in range(16)
    ]
    return {
        "timezone": zone,
        "current_units": {
            "time": "iso8601",
            "interval": "seconds",
            "temperature_2m": "°C",
            "weather_code": "wmo code",
            "precipitation": "mm",
        },
        "current": {
            "time": start + "T00:15",
            "interval": 900,
            "temperature_2m": 12,
            "weather_code": 61,
            "precipitation": 0.2,
        },
        "daily_units": {
            "time": "iso8601",
            "temperature_2m_min": "°C",
            "temperature_2m_max": "°C",
            "precipitation_probability_max": "%",
            "rain_sum": "mm",
            "showers_sum": "mm",
            "weather_code": "wmo code",
        },
        "daily": {
            "time": dates,
            "temperature_2m_min": list[JsonValue]([1] * 16),
            "temperature_2m_max": list[JsonValue]([12] * 16),
            "precipitation_probability_max": list[JsonValue]([30] * 16),
            "rain_sum": list[JsonValue]([2] * 16),
            "showers_sum": list[JsonValue]([0] * 16),
            "weather_code": list[JsonValue]([61] * 16),
        },
    }


class FixtureClient(WeatherClient):
    """Transport-free client retaining requests for contract assertions."""

    data: dict[str, JsonValue]
    """Forecast response."""

    geocoding: dict[str, JsonValue]
    """Search response."""

    requests: list[tuple[str, dict[str, str]]]
    """Ordered endpoint and parameter pairs."""

    def __init__(self) -> None:
        """Initialize a valid home and offline responses."""
        super().__init__(WeatherConfig(51, -1))
        self.data = forecast()
        self.geocoding = {}
        self.requests = []

    def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, JsonValue]:
        """Record calls and return fixture JSON.

        Args:
            endpoint:
                Requested provider URL.

            params:
                Request query parameters.

        """
        self.requests.append((endpoint, params))
        return self.geocoding if "geocoding" in endpoint else self.data


def test_tool_and_current() -> None:
    """Expose strict arguments and compact Celsius home measurements."""
    client = FixtureClient()
    tool = client.tool()
    assert tool.name == "get_weather"
    assert tool.arguments is WeatherArguments
    tool.schema()
    output = tool.handler(WeatherArguments.model_validate({"period": "current"}))
    assert isinstance(output, dict)
    assert output.pop("current_temperature_c") == 12
    assert isinstance(output.pop("days"), list)
    assert output == {
        "city": "Home",
        "timezone": "Pacific/Kiritimati",
        "period": "now",
        "time": "2026-12-28T00:15",
        "interval_seconds": 900.0,
        "temperature_c": 12.0,
        "condition": "light rain",
        "precipitation_mm": 0.2,
    }
    assert client.requests[0][1]["latitude"] == "51"
    assert client.requests[0][1]["forecast_days"] == "16"
    assert "weather_code" in client.requests[0][1]["daily"].split(",")
    assert {"rain_sum", "showers_sum"} <= set(client.requests[0][1]["daily"].split(","))
    assert WeatherArguments.model_validate({"period": "week"}).period == "now"
    with pytest.raises(ValidationError):
        WeatherArguments.model_validate({"period": "today", "city": 123})
    with pytest.raises(ValidationError):
        WeatherArguments.model_validate({"period": "today", "extra": 1})


def test_weather_success_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Retain request arguments, successful outcome, and detailed forecast in logs.

    Args:
        caplog:
            Captures weather records including debug-level normalized results.

    """
    caplog.set_level(logging.DEBUG, logger="hoast.weather")
    FixtureClient().get_weather(WeatherArguments(city="Home"))
    assert "get_weather requested period=now city='Home'" in caplog.text
    assert "get_weather status=success" in caplog.text
    assert any(
        record.levelno == logging.DEBUG and "current_temperature_c" in record.message
        for record in caplog.records
    )


def test_weather_clarification_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Log missing cities at warning and ranked selection with candidate metadata.

    Args:
        caplog:
            Captures status, diagnostic reason, and candidate detail records.

    """
    caplog.set_level(logging.DEBUG, logger="hoast.weather")
    client = FixtureClient()
    client.get_weather(WeatherArguments(city="Unknown"))
    assert any(
        record.levelno == logging.WARNING
        and "status=city_not_found" in record.message
        and "no locations matched" in record.message
        for record in caplog.records
    )
    caplog.clear()
    client.geocoding = {
        "results": [
            {
                "name": "Springfield",
                "country": "United States",
                "country_code": "US",
                "admin1": region,
                "latitude": latitude,
                "longitude": -90,
            }
            for region, latitude in [("Illinois", 39), ("Missouri", 37)]
        ]
    }
    client.get_weather(WeatherArguments(period="tomorrow", city="Springfield"))
    assert any(
        record.levelno == logging.INFO
        and "resolution=ranked_city" in record.message
        and "distance_km=" in record.message
        for record in caplog.records
    )
    assert any(
        record.levelno == logging.DEBUG
        and "Illinois" in record.message
        and "Missouri" in record.message
        for record in caplog.records
    )


def test_weather_exception_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Keep traceback information while propagating provider validation failures.

    Args:
        caplog:
            Captures the error record and its original exception information.

    """
    client = FixtureClient()
    client.data = {}
    with pytest.raises(KeyError, match="timezone") as raised:
        client.get_weather(WeatherArguments(period="tomorrow"))
    record = next(
        record for record in caplog.records if record.levelno == logging.ERROR
    )
    assert "get_weather status=failed period=tomorrow" in record.message
    assert record.exc_info is not None
    assert record.exc_info[1] is raised.value
    assert record.exc_info[2] is not None


@pytest.mark.parametrize(
    ("start", "period", "first", "last"),
    [
        ("2026-12-28", "next_week", "2027-01-04", "2027-01-10"),
        ("2027-01-03", "next_week", "2027-01-04", "2027-01-10"),
        ("2026-12-31", "tomorrow", "2027-01-01", "2027-01-01"),
        ("2026-12-28", "today", "2026-12-28", "2026-12-28"),
    ],
)
def test_dates(start: str, period: str, first: str, last: str) -> None:
    """Use provider-local calendar dates, including Monday and Sunday edges.

    Args:
        start:
            Provider local date.

        period:
            Requested daily period.

        first:
            Expected first date.

        last:
            Expected final date.

    """
    client = FixtureClient()
    client.data = forecast(start)
    output = client.get_weather(WeatherArguments.model_validate({"period": period}))
    assert isinstance(output, dict)
    days = output["days"]
    assert isinstance(days, list)
    assert isinstance(days[0], dict) and days[0]["date"] == first
    assert isinstance(days[-1], dict) and days[-1]["date"] == last
    assert len(days) == (7 if period == "next_week" else 1)


def test_clock_destination_timezone() -> None:
    """Convert an injected UTC Sunday to destination Monday before week selection."""
    client = FixtureClient()
    client.clock = lambda: datetime(2026, 12, 27, 12, tzinfo=UTC)
    output = client.get_weather(WeatherArguments(period="next_week"))
    assert isinstance(output, dict)
    assert isinstance(output["days"], list)
    assert isinstance(output["days"][0], dict)
    assert output["days"][0]["date"] == "2027-01-04"
    client.clock = lambda: datetime(2026, 12, 28)  # noqa: DTZ001 -- invalid clock fixture
    with pytest.raises(ValueError, match="aware"):
        client.get_weather(WeatherArguments(period="today"))


def test_daily_surface() -> None:
    """Return explicit daily names and normalized conditions without a summary."""
    assert FixtureClient().get_weather(WeatherArguments(period="today")) == {
        "city": "Home",
        "timezone": "Pacific/Kiritimati",
        "period": "today",
        "days": [
            {
                "date": "2026-12-28",
                "temperature_min_c": 1.0,
                "temperature_max_c": 12.0,
                "condition": "light rain",
                "precipitation_probability_max_pct": 30.0,
                "rain_mm": 2.0,
            }
        ],
    }


def test_default_period_rendering() -> None:
    """Default now includes current temperature; explicit today is forecast only."""
    client = FixtureClient()
    assert "period" not in client.tool().schema()["function"]["parameters"]["required"]
    output = client.get_weather(WeatherArguments())
    assert isinstance(output, dict)
    assert output["period"] == "now"
    assert output["current_temperature_c"] == 12
    assert render_weather(output) == (
        "It's light rain, 12 degrees outside now, 1 to 12 degrees, "
        "30% chance of light rain."
    )
    explicit = client.get_weather(WeatherArguments(period="today"))
    assert isinstance(explicit, dict)
    assert "current_temperature_c" not in explicit
    assert render_weather(explicit) == (
        "For today, it's light rain, 1 to 12 degrees, 30% chance of light rain."
    )
    output["city"] = "Paris, France"
    output["current_temperature_c"] = None
    assert render_weather(output).startswith(
        "In Paris, France, it's light rain, current temperature unavailable, "
    )


@pytest.mark.parametrize("city", ["Home", "home", "HOME", "  HoMe  "])
def test_home_alias(city: str) -> None:
    """Resolve model-authored Home labels locally without a geocoding request.

    Args:
        city:
            Home alias with varying case and surrounding whitespace.

    """
    client = FixtureClient()
    result = ToolRegistry([client.tool()]).dispatch(
        [ToolCall("get_weather", {"period": "tomorrow", "city": city})]
    )[0]
    assert isinstance(result, dict)
    assert result["city"] == "Home"
    assert result["period"] == "tomorrow"
    assert len(client.requests) == 1
    endpoint, params = client.requests[0]
    assert endpoint == weather._FORECAST
    assert params["latitude"] == str(client.config.latitude)
    assert params["longitude"] == str(client.config.longitude)


@pytest.mark.parametrize("period", ["now", "current", "whenever", "", " NOW "])
def test_now_period_dispatch(period: str) -> None:
    """Route explicit now and unknown strings through real validation to defaults.

    Args:
        period:
            Model-provided string expected to select now.

    """
    client = FixtureClient()
    registry = ToolRegistry([client.tool()])
    result = registry.dispatch([ToolCall("get_weather", {"period": period})])[0]
    assert result == client.get_weather(WeatherArguments())
    assert "12 degrees outside now, 1 to 12 degrees" in render_weather(result)


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        (" TODAY ", "today"),
        ("Tomorrow", "tomorrow"),
        ("next week", "next_week"),
        ("Next-Week", "next_week"),
    ],
)
def test_forecast_period_normalization(period: str, expected: str) -> None:
    """Preserve recognized forecast periods after textual normalization.

    Args:
        period:
            Model-provided period with varying case or separators.

        expected:
            Canonical forecast period.

    """
    assert WeatherArguments.model_validate({"period": period}).period == expected


@pytest.mark.parametrize("period", [None, 1, True, [], {}])
def test_period_wrong_type(period: JsonValue) -> None:
    """Reject non-string periods rather than treating malformed data as words.

    Args:
        period:
            Invalid non-string model argument.

    """
    with pytest.raises(ValidationError):
        WeatherArguments.model_validate({"period": period})


@pytest.mark.parametrize(
    ("condition", "rain", "probability", "expected"),
    [
        ("heavy rain", 10, 85.6, "86% chance of heavy rain"),
        ("light rain", 2, 30.4, "30% chance of light rain"),
        ("overcast", 2, 40, "40% chance of rain"),
        ("clear", 0, 10, "no rain"),
        ("light rain", 2, None, "rain chance unavailable"),
    ],
)
def test_compact_rain(
    condition: str, rain: float, probability: float | None, expected: str
) -> None:
    """Render rounded rain chances and provider-derived intensity.

    Args:
        condition:
            Normalized daily weather label.

        rain:
            Daily rain amount in millimeters.

        probability:
            Daily maximum precipitation probability in percent, or None.

        expected:
            Expected rain clause.

    """
    result: dict[str, JsonValue] = {
        "city": "Home",
        "period": "tomorrow",
        "days": [
            {
                "condition": condition,
                "rain_mm": rain,
                "precipitation_probability_max_pct": probability,
                "temperature_min_c": 1.6,
                "temperature_max_c": 12.4,
            }
        ],
    }
    assert render_weather(result) == (
        f"For tomorrow, it's {condition}, 2 to 12 degrees, {expected}."
    )


@pytest.mark.parametrize(
    ("rain", "showers", "expected"),
    [
        (2, 0.5, 2.5),
        (0, 1.5, 1.5),
        (2, 0, 2),
        (0, 0, 0),
        (None, 1, None),
        (1, None, None),
        (None, 0, None),
        (0, None, None),
        (None, None, None),
    ],
)
def test_daily_rain_total(
    rain: float | None, showers: float | None, expected: float | None
) -> None:
    """Sum large-scale rain and convective showers only when both are known.

    Args:
        rain:
            Large-scale rain in mm or null.

        showers:
            Convective showers in mm or null.

        expected:
            Combined daily rain in mm or null.

    """
    client = FixtureClient()
    daily = client.data["daily"]
    assert isinstance(daily, dict)
    daily["rain_sum"] = list[JsonValue]([rain] * 16)
    daily["showers_sum"] = list[JsonValue]([showers] * 16)
    output = client.get_weather(WeatherArguments(period="today"))
    assert isinstance(output, dict) and isinstance(output["days"], list)
    assert isinstance(output["days"][0], dict)
    assert output["days"][0]["rain_mm"] == expected


@pytest.mark.parametrize("component", ["rain_sum", "showers_sum"])
@pytest.mark.parametrize("missing", [None, 0, 7, 10, 13])
def test_weekly_showers_only(component: str, missing: int | None) -> None:
    """Include showers-only rainy dates; either null component makes dates unknown.

    Args:
        component:
            Rain component receiving an optional null measurement.

        missing:
            Null's provider index; zero is outside the selected week.

    """
    client = FixtureClient()
    daily = client.data["daily"]
    assert isinstance(daily, dict)
    daily["rain_sum"] = list[JsonValue]([0] * 16)
    showers = daily["showers_sum"]
    assert isinstance(showers, list)
    showers[0], showers[8], showers[12] = 4, 0.5, 2
    if missing is not None:
        values = daily[component]
        assert isinstance(values, list)
        values[missing] = None
    output = client.get_weather(WeatherArguments(period="next_week"))
    assert isinstance(output, dict) and isinstance(output["summary"], dict)
    assert output["summary"]["rainy_dates"] == (
        None
        if missing is not None and 7 <= missing <= 13
        else ["2027-01-05", "2027-01-09"]
    )


@pytest.mark.parametrize("component", ["rain_sum", "showers_sum"])
@pytest.mark.parametrize("other", [None, 2])
@pytest.mark.parametrize("invalid", [-1, float("inf"), float("nan"), True, "1"])
def test_invalid_rain_component(
    component: str, other: float | None, invalid: JsonValue
) -> None:
    """Validate each component even when its counterpart is null or offsets a negative.

    Args:
        component:
            Component containing an invalid measurement.

        other:
            Valid counterpart in mm or null.

        invalid:
            Negative, nonfinite, or nonnumeric provider measurement.

    """
    client = FixtureClient()
    daily = client.data["daily"]
    assert isinstance(daily, dict)
    for key in ("rain_sum", "showers_sum"):
        daily[key] = [invalid if key == component else other] * 16
    with pytest.raises((ValueError, TypeError)):
        client.get_weather(WeatherArguments(period="today"))


@pytest.mark.parametrize("missing", [None, 0, 7, 10, 13])
def test_weekly_summary(missing: int | None) -> None:
    """Aggregate only selected dates and propagate incomplete metrics independently.

    Args:
        missing:
            Optional provider index with missing minimum, probability, and rain.
            Index zero is outside the requested week.

    """
    client = FixtureClient()
    daily = client.data["daily"]
    assert isinstance(daily, dict)
    daily["temperature_2m_min"] = list[JsonValue](range(-10, 6))
    daily["temperature_2m_max"] = list[JsonValue](range(12, 28))
    daily["precipitation_probability_max"] = list[JsonValue](range(16))
    daily["rain_sum"] = list[JsonValue]([0] * 16)
    rain = daily["rain_sum"]
    assert isinstance(rain, list)
    rain[8], rain[12] = 0.1, 2
    if missing is not None:
        for key in ("temperature_2m_min", "precipitation_probability_max", "rain_sum"):
            values = daily[key]
            assert isinstance(values, list)
            values[missing] = None
    output = client.get_weather(WeatherArguments(period="next_week"))
    assert isinstance(output, dict)
    incomplete = missing is not None and 7 <= missing <= 13
    assert output["summary"] == {
        "temperature_min_c": None if incomplete else -3.0,
        "temperature_max_c": 25.0,
        "precipitation_probability_max_pct": None if incomplete else 13.0,
        "rainy_dates": None if incomplete else ["2027-01-05", "2027-01-09"],
    }
    assert isinstance(output["days"], list) and len(output["days"]) == 7


@pytest.mark.parametrize("value", [None, 0])
def test_weekly_nulls_and_dry_days(value: JsonValue) -> None:
    """Distinguish missing weekly measurements from known zero and dry dates.

    Args:
        value:
            Measurement shared by every day: null or known zero.

    """
    client = FixtureClient()
    daily = client.data["daily"]
    assert isinstance(daily, dict)
    for key in (
        "temperature_2m_min",
        "temperature_2m_max",
        "precipitation_probability_max",
        "rain_sum",
    ):
        daily[key] = [value] * 16
    output = client.get_weather(WeatherArguments(period="next_week"))
    assert isinstance(output, dict)
    assert output["summary"] == {
        "temperature_min_c": value,
        "temperature_max_c": value,
        "precipitation_probability_max_pct": value,
        "rainy_dates": None if value is None else [],
    }


@pytest.mark.parametrize(
    "code, condition", [(None, None), (999, "unknown (999)"), (73, "snow")]
)
def test_daily_condition(code: JsonValue, condition: str | None) -> None:
    """Normalize daily codes using the same semantics as current conditions.

    Args:
        code:
            Provider daily WMO code.

        condition:
            Expected normalized label or null.

    """
    client = FixtureClient()
    daily = client.data["daily"]
    assert isinstance(daily, dict)
    daily["weather_code"] = [code] * 16
    output = client.get_weather(WeatherArguments(period="today"))
    assert isinstance(output, dict) and isinstance(output["days"], list)
    assert isinstance(output["days"][0], dict)
    assert output["days"][0]["condition"] == condition


def test_null_and_unknown() -> None:
    """Preserve explicit nulls and label unknown WMO codes explicitly."""
    client = FixtureClient()
    current = client.data["current"]
    daily = client.data["daily"]
    assert isinstance(current, dict) and isinstance(daily, dict)
    current.update(temperature_2m=None, precipitation=None, weather_code=999)
    daily["rain_sum"] = list[JsonValue]([None] * 16)
    output = client.get_weather(WeatherArguments(period="now"))
    assert isinstance(output, dict)
    assert output["temperature_c"] is None and output["precipitation_mm"] is None
    assert output["condition"] == "unknown (999)"
    output = client.get_weather(WeatherArguments(period="today"))
    assert isinstance(output, dict) and isinstance(output["days"], list)
    assert isinstance(output["days"][0], dict)
    assert output["days"][0]["rain_mm"] is None
    current["weather_code"] = None
    output = client.get_weather(WeatherArguments(period="now"))
    assert isinstance(output, dict) and output["condition"] is None


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("current_units", "temperature_2m", "°F"),
        ("current_units", "interval", "minutes"),
        ("current", "interval", None),
        ("current", "interval", 0),
        ("current", "interval", -1),
        ("current", "interval", True),
        ("current", "interval", float("inf")),
        ("daily_units", "weather_code", "code"),
        ("daily", "weather_code", [1.5] * 16),
        ("daily", "weather_code", [True] * 16),
        ("daily", "weather_code", [-1] * 16),
        ("daily_units", "rain_sum", "inch"),
        ("daily_units", "showers_sum", "inch"),
        ("current", "temperature_2m", float("nan")),
        ("current", "weather_code", 1.5),
        ("current", "precipitation", True),
        ("current", "time", "not-a-date"),
        ("daily", "rain_sum", [0]),
        ("daily", "showers_sum", [0]),
        ("daily", "rain_sum", [-1] * 16),
        ("daily", "temperature_2m_min", [20] * 16),
        ("daily", "precipitation_probability_max", [101] * 16),
        ("daily", "time", ["2026-12-28"] * 16),
    ],
)
def test_malformed(section: str, key: str, value: JsonValue) -> None:
    """Reject malformed provider units, measurements, and aligned-date contracts.

    Args:
        section:
            Response section to corrupt.

        key:
            Field to replace.

        value:
            Invalid provider value.

    """
    client = FixtureClient()
    target = client.data[section]
    assert isinstance(target, dict)
    target[key] = value
    with pytest.raises((ValueError, TypeError)):
        client.get_weather(WeatherArguments(period="today"))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("current", "interval"),
        ("current_units", "interval"),
        ("daily", "weather_code"),
        ("daily_units", "weather_code"),
        ("daily", "rain_sum"),
        ("daily_units", "rain_sum"),
        ("daily", "showers_sum"),
        ("daily_units", "showers_sum"),
    ],
)
def test_missing_required_fields(section: str, key: str) -> None:
    """Require interval, daily conditions, both rain components, and their units.

    Args:
        section:
            Provider response object containing the required field.

        key:
            Required field to remove.

    """
    client = FixtureClient()
    target = client.data[section]
    assert isinstance(target, dict)
    del target[key]
    with pytest.raises(KeyError):
        client.get_weather(WeatherArguments(period="today"))


def test_missing_and_uncovered() -> None:
    """Missing measurements and unavailable forecast dates fail explicitly."""
    client = FixtureClient()
    current = client.data["current"]
    assert isinstance(current, dict)
    del current["temperature_2m"]
    with pytest.raises(KeyError):
        client.get_weather(WeatherArguments(period="now"))
    client.data = forecast()
    client.clock = lambda: datetime(2030, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="cover"):
        client.get_weather(WeatherArguments(period="today"))


def test_cities() -> None:
    """Choose the nearest city to home while honoring explicit region and country."""
    client = FixtureClient()
    client.geocoding = {
        "results": [
            {
                "name": "Paris",
                "country": "France",
                "country_code": "FR",
                "admin1": "Île-de-France",
                "latitude": 48.85,
                "longitude": 2.35,
            },
            {
                "name": "Paris",
                "country": "United States",
                "country_code": "US",
                "admin1": "Texas",
                "latitude": 33.66,
                "longitude": -95.55,
            },
        ]
    }
    output = client.get_weather(WeatherArguments(period="today", city="Paris"))
    assert isinstance(output, dict) and output["city"] == "Paris, Île-de-France, France"
    assert output["period"] == "today"
    assert len(client.requests) == 2
    output = client.get_weather(WeatherArguments(period="today", city="Paris, FR"))
    assert isinstance(output, dict) and output["city"] == "Paris, Île-de-France, France"
    assert client.requests[-1][1]["latitude"] == "48.85"
    output = client.get_weather(
        WeatherArguments(period="today", city="Paris, Texas, US")
    )
    assert isinstance(output, dict) and output["city"] == "Paris, Texas, United States"
    output = client.get_weather(WeatherArguments(period="today", city="Paris, Japan"))
    assert output == {
        "status": "city_not_found",
        "choices": [],
        "period": "today",
        "city_query": "Paris, Japan",
    }
    client.geocoding = {"results": None}
    with pytest.raises(TypeError):
        client.get_weather(WeatherArguments(period="today", city="Paris"))


@pytest.mark.parametrize(
    ("home_lon", "first_lon", "second_lon", "expected"),
    [(179, 170, -179, "Second"), (-95, 2, -96, "Second"), (0, -1, 1, "First")],
)
def test_nearest_city_distance(
    home_lon: float, first_lon: float, second_lon: float, expected: str
) -> None:
    """Select geographically nearest candidates across the dateline with stable ties.

    Args:
        home_lon:
            Configured equatorial home longitude in degrees.

        first_lon:
            First provider candidate's equatorial longitude in degrees.

        second_lon:
            Second candidate's equatorial longitude in degrees.

        expected:
            Selected candidate's region label.

    """
    client = FixtureClient()
    client.config = WeatherConfig(0, home_lon)
    candidates: list[JsonValue] = [
        {
            "name": "Example",
            "country": "Country",
            "country_code": "XX",
            "admin1": label,
            "latitude": 0,
            "longitude": longitude,
        }
        for label, longitude in [("First", first_lon), ("Second", second_lon)]
    ]
    client.geocoding = {"results": candidates}
    result = client.get_weather(WeatherArguments(city="Example"))
    assert isinstance(result, dict)
    assert result["city"] == f"Example, {expected}, Country"
    assert client.requests[-1][1]["longitude"] == str(
        float(first_lon if expected == "First" else second_lon)
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Beijing", "Beijing"),
        ("BEIJING", "Beijing"),
        ("Beijing, Jilin", "Jilin"),
        ("Beijing, Jilin, CN", "Jilin"),
    ],
)
def test_capital_city_ranking(query: str, expected: str) -> None:
    """Prefer a distant capital to nearby namesakes, respecting explicit qualifiers.

    Args:
        query:
            City query, optionally qualified by province and country.

        expected:
            Province of the selected candidate.

    """
    client = FixtureClient()
    client.config = WeatherConfig(41, 126)
    client.geocoding = {
        "results": [
            {
                "name": "Beijing",
                "country": "China",
                "country_code": "CN",
                "admin1": "Jilin",
                "latitude": 41,
                "longitude": 126,
                "feature_code": "PPL",
                "population": 100,
            },
            {
                "name": "Beijing",
                "country": "China",
                "country_code": "CN",
                "admin1": "Beijing",
                "latitude": 39.9,
                "longitude": 116.4,
                "feature_code": "PPLC",
                "population": 20000000,
            },
        ]
    }
    result = client.get_weather(WeatherArguments(city=query))
    assert isinstance(result, dict)
    assert result["city"] == (
        "Beijing, China" if expected == "Beijing" else "Beijing, Jilin, China"
    )
    assert client.requests[-1][1]["longitude"] == (
        "116.4" if expected == "Beijing" else "126.0"
    )


@pytest.mark.parametrize(
    (
        "first_name",
        "first_feature",
        "first_population",
        "second_population",
        "expected",
    ),
    [
        ("Exampleville", "PPLC", 1000000, 10, "Second"),
        ("Example", "PPLC", 10, 1000000, "First"),
        ("Example", "PPL", 10, 1000000, "Second"),
        ("Example", "PPL", None, 0, "Second"),
        ("Example", "PPL", 10, 10, "First"),
    ],
)
def test_city_ranking_precedence(
    first_name: str,
    first_feature: str,
    first_population: int | None,
    second_population: int,
    expected: str,
) -> None:
    """Rank exact names before capitals, population before distance, and handle nulls.

    Args:
        first_name:
            Name of the nearby first candidate.

        first_feature:
            Provider feature code of the nearby candidate.

        first_population:
            Nearby candidate's optional population count.

        second_population:
            Distant exact-name candidate's population count.

        expected:
            Selected candidate's region label.

    """
    client = FixtureClient()
    client.config = WeatherConfig(0, 0)
    client.geocoding = {
        "results": [
            {
                "name": first_name,
                "country": "Country",
                "country_code": "XX",
                "admin1": "First",
                "latitude": 0,
                "longitude": 0,
                "feature_code": first_feature,
                "population": first_population,
            },
            {
                "name": "Example",
                "country": "Country",
                "country_code": "XX",
                "admin1": "Second",
                "latitude": 20,
                "longitude": 20,
                "feature_code": "PPL",
                "population": second_population,
            },
        ]
    }
    result = client.get_weather(WeatherArguments(city="Example"))
    assert isinstance(result, dict)
    assert result["city"] == f"Example, {expected}, Country"


def test_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network failures propagate without fabricated weather or retries.

    Args:
        monkeypatch:
            Scoped replacement of the request boundary.

    """

    def fail(
        self: WeatherClient, endpoint: str, params: dict[str, str]
    ) -> dict[str, JsonValue]:
        """Simulate a transport failure.

        Args:
            self:
                Client under test.

            endpoint:
                Requested URL.

            params:
                Query parameters.

        """
        raise URLError("offline")

    monkeypatch.setattr(WeatherClient, "_request", fail)
    with pytest.raises(URLError, match="offline"):
        WeatherClient(WeatherConfig(0, 0)).get_weather(WeatherArguments(period="now"))


@pytest.mark.parametrize(
    "body",
    [
        b'{"error":true,"reason":"invalid request"}',
        b"not JSON",
        b"[]",
        b" " * 1_048_577,
    ],
)
def test_http_boundary(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    """Exercise bounded reads, fixed URLs, timeout, and provider/JSON failures.

    Args:
        monkeypatch:
            Scoped replacement of the urllib opener.

        body:
            Malformed or oversized provider response bytes.

    """

    class Opener:
        """Offline urllib opener asserting request limits."""

        def open(self, request: Request, timeout: int) -> BytesIO:
            """Return fixture bytes after checking request parameters.

            Args:
                request:
                    Encoded forecast request.

                timeout:
                    Socket timeout in seconds.

            """
            assert timeout == 10
            assert request.full_url.startswith(
                "https://api.open-meteo.com/v1/forecast?"
            )
            assert "forecast_days=16" in request.full_url
            return BytesIO(body)

    monkeypatch.setattr(weather, "build_opener", lambda handler: Opener())
    with pytest.raises((ValueError, TypeError)):
        WeatherClient(WeatherConfig(0, 0)).get_weather(WeatherArguments(period="now"))
