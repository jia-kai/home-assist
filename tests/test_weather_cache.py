"""Offline tests for prepared weather and exact speech provenance."""

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from hoast.agent import LocalAgent, render_weather
from hoast.application import Assistant, VoiceApplication
from hoast.lights import LightArguments
from hoast.llm import FunctionGemma, Generation, LLMConfig, Tool, ToolCall, ToolRegistry
from hoast.llm_cli import _refresh_weather
from hoast.session import Session
from hoast.stt2 import STT
from hoast.tts2 import TTS
from hoast.voice import VoiceTurn
from hoast.weather import WeatherArguments, WeatherClient, declare_weather_tool
from hoast.weather_cache import WeatherCache


def result(period: str) -> dict:
    """Construct a minimal current-date home forecast for the renderer."""
    value = {
        "city": "Home",
        "period": period,
        "timezone": "UTC",
        "days": [
            {
                "date": datetime.now(UTC).date().isoformat(),
                "temperature_min_c": 5,
                "temperature_max_c": 15,
                "condition": "clear",
                "precipitation_probability_max_pct": 10,
                "rain_mm": 0,
            }
        ],
    }
    if period == "now":
        value["current_temperature_c"] = 12
        value["condition"] = "clear"
        value["time"] = datetime.now(UTC).isoformat()
    return value


@pytest.fixture
def prepared() -> tuple[WeatherCache, MagicMock, MagicMock]:
    """Prepare distinct now and today entries from mocked live boundaries."""
    client = MagicMock(spec=WeatherClient)
    client.get_weather.side_effect = lambda args: result(args.period)
    tts = MagicMock(spec=TTS)
    tts.synthesize.return_value = (np.ones(10, dtype=np.float32), 24000)
    cache = WeatherCache(client)
    cache.refresh(tts, "now")
    cache.refresh(tts, "today")
    return cache, client, tts


def test_distinct_results_hits_and_misses(prepared: tuple) -> None:
    """Only home/current and home/today hit, with isolated result copies."""
    cache, client, tts = prepared
    now = cache.get_weather(WeatherArguments())
    today = cache.get_weather(WeatherArguments(period="today", city=" Home "))
    assert now["current_temperature_c"] == 12
    assert "current_temperature_c" not in today
    assert cache.entries["now"].text != cache.entries["today"].text
    now["city"] = "mutated"
    assert cache.get_weather(WeatherArguments())["city"] == "Home"
    assert client.get_weather.call_count == 2
    cache.get_weather(WeatherArguments(period="tomorrow"))
    cache.get_weather(WeatherArguments(city="Paris"))
    assert client.get_weather.call_count == 4
    assert tts.synthesize.call_count == 2


def test_age_and_midnight_expire(
    prepared: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enforce strictly less than an hour and invalidate at destination midnight."""
    cache, client, _ = prepared
    entry = cache.entries["now"]
    monkeypatch.setattr(
        "hoast.weather_cache.time.monotonic", lambda: entry.started + 3600
    )
    cache.get_weather(WeatherArguments())
    assert client.get_weather.call_count == 3
    monkeypatch.setattr("hoast.weather_cache.time.monotonic", lambda: entry.started + 5)
    entry.local_date = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    assert cache.valid("now") is None


def test_failed_refresh_does_not_publish(prepared: tuple) -> None:
    """A synthesis failure cannot pair new weather with the prior waveform."""
    cache, _, tts = prepared
    old = cache.entries["now"]
    tts.synthesize.side_effect = RuntimeError("offline synthesis failure")
    with pytest.raises(RuntimeError):
        cache.refresh(tts, "now")
    assert cache.entries["now"] is old


def test_voice_uses_only_single_weather_hit(prepared: tuple, tmp_path: Path) -> None:
    """Normal dispatch chooses prepared audio only for an exact single tool result."""
    cache, _, tts = prepared
    tool = replace(declare_weather_tool(), handler=cache.get_weather)
    model = MagicMock(spec=FunctionGemma)
    model.config = LLMConfig(tmp_path)
    model.tools = ToolRegistry([tool])
    app = VoiceApplication(
        Assistant(LocalAgent(Session(model)), None, cache),
        MagicMock(spec=STT),
        tts,
        lambda text: None,
    )
    event = threading.Event()
    app.respond(VoiceTurn("What is the weather now?", 0, 1, event))
    assert tts.play_samples.call_count == 1
    assert tts.play.call_count == 0
    model.generate_messages.return_value = Generation(
        "",
        (ToolCall("get_weather", {}), ToolCall("get_weather", {"period": "today"})),
        "",
        1,
        1,
        0.01,
        0.001,
    )
    app.respond(VoiceTurn("Compare the weather", 0, 1, event))
    assert tts.play_samples.call_count == 2
    assert tts.play.call_count == 0
    cache.entries["now"].started -= 3601
    app.respond(VoiceTurn("What is the weather now?", 0, 1, event))
    assert tts.play_samples.call_count == 2
    assert tts.play.call_count == 1
    event.set()
    app.respond(VoiceTurn("What is the weather now?", 0, 1, event))
    assert tts.play.call_count == 1


def test_refresh_cancel_joins_worker(prepared: tuple) -> None:
    """Cancellation waits for in-flight work before returning to engine cleanup."""
    cache, _, tts = prepared
    entered = threading.Event()
    release = threading.Event()

    def synthesize(text: str) -> tuple:
        entered.set()
        assert release.wait(5)
        return np.ones(10, dtype=np.float32), 24000

    tts.synthesize.side_effect = synthesize
    app = MagicMock(spec=VoiceApplication)
    app.assistant.weather_cache = cache
    app.tts = tts
    cache.entries.clear()

    async def run() -> None:
        with ThreadPoolExecutor(max_workers=1) as worker:
            task = asyncio.create_task(_refresh_weather(app, worker))
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_refresh_shares_one_fetch(prepared: tuple) -> None:
    """Prepare both distinct periods from one validated network result."""
    cache, client, tts = prepared
    cache.entries.clear()
    client.get_weather.reset_mock()
    completed = threading.Event()
    count = 0

    def synthesize(text: str) -> tuple:
        nonlocal count
        count += 1
        if count == 2:
            completed.set()
        return np.ones(10, dtype=np.float32), 24000

    tts.synthesize.side_effect = synthesize
    app = MagicMock(spec=VoiceApplication)
    app.assistant.weather_cache = cache
    app.tts = tts

    async def run() -> None:
        with ThreadPoolExecutor(max_workers=1) as worker:
            task = asyncio.create_task(_refresh_weather(app, worker))
            assert await asyncio.to_thread(completed.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert client.get_weather.call_count == 1
    assert set(cache.entries) == {"now", "today"}
    assert cache.entries["now"].text != cache.entries["today"].text


@pytest.mark.parametrize("offset", [-3601, 301])
def test_invalid_observation_rejected(offset: int) -> None:
    """Reject stale and excessively future provider observations.

    Args:
        offset:
            Observation offset in seconds relative to the current UTC time.

    """
    client = MagicMock(spec=WeatherClient)
    data = result("now")
    data["time"] = (datetime.now(UTC) + timedelta(seconds=offset)).isoformat()
    client.get_weather.return_value = data
    with pytest.raises(ValueError):
        WeatherCache(client).fetch("now")


def test_missing_observation_rejected() -> None:
    """Missing provider observation time never produces a cache entry."""
    client = MagicMock(spec=WeatherClient)
    data = result("now")
    del data["time"]
    client.get_weather.return_value = data
    with pytest.raises(ValueError):
        WeatherCache(client).fetch("now")


def test_local_naive_observation_uses_destination_timezone() -> None:
    """Interpret Open-Meteo's offset-free civil timestamp in its reported zone."""
    client = MagicMock(spec=WeatherClient)
    data = result("now")
    zone = ZoneInfo("Asia/Tokyo")
    local = datetime.now(UTC).astimezone(zone)
    data["timezone"] = "Asia/Tokyo"
    data["time"] = local.replace(tzinfo=None).isoformat()
    data["days"][0]["date"] = local.date().isoformat()
    client.get_weather.return_value = data
    _, _, _, _, observation = WeatherCache(client).fetch("now")
    assert abs((datetime.now(UTC) - observation).total_seconds()) < 2


def test_expired_during_turn_never_spoken(
    prepared: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache hit that expires during the turn is replaced, not synthesized."""
    cache, _, tts = prepared
    tool = replace(declare_weather_tool(), handler=cache.get_weather)
    model = MagicMock(spec=FunctionGemma)
    model.config = LLMConfig(tmp_path)
    model.tools = ToolRegistry([tool])
    app = VoiceApplication(
        Assistant(LocalAgent(Session(model)), None, cache),
        MagicMock(spec=STT),
        tts,
        lambda text: None,
    )
    original_reply = Assistant.reply

    def reply(self: Assistant, text: str, cancelled: Callable[[], bool] | None) -> str:
        answer = original_reply(self, text, cancelled)
        cache.entries["now"].started -= 3601
        return answer

    monkeypatch.setattr(Assistant, "reply", reply)
    app.respond(VoiceTurn("What is the weather now?", 0, 1, threading.Event()))
    assert tts.play_samples.call_count == 1
    assert "expired" in tts.synthesize.call_args.args[0]


def test_voice_home_miss_fetches_live(prepared: tuple) -> None:
    """An unavailable home entry uses the live weather client."""
    cache, client, _ = prepared
    cache.entries.clear()
    cache.voice_mode = True
    assert cache.get_weather(WeatherArguments())["city"] == "Home"
    assert client.get_weather.call_count == 3


def test_unavailable_fallback_and_text_provenance(prepared: tuple) -> None:
    """A failed stale-cache fallback is reported without accumulating text hits."""
    cache, client, _ = prepared
    cache.get_weather(WeatherArguments())
    assert cache.turn_hits == [] and cache.turn_weather == []
    cache.entries["now"].started -= 3601
    client.get_weather.side_effect = ValueError("stale provider observation")
    output = cache.get_weather(WeatherArguments())
    assert output == {"status": "unavailable"}
    assert cache.turn_hits == [] and cache.turn_weather == []
    assert render_weather(output) == "Weather is unavailable. Please try again."


def test_malformed_cached_result_falls_back_to_live(prepared: tuple) -> None:
    """Malformed prepared data cannot break normal live weather dispatch."""
    cache, client, _ = prepared
    del cache.entries["now"].result["days"]
    assert cache.get_weather(WeatherArguments())["city"] == "Home"
    assert client.get_weather.call_count == 3
    assert not cache.turn_hits


@pytest.mark.parametrize("expire", [False, True])
def test_mixed_hit_preserves_light_and_text(
    prepared: tuple, tmp_path: Path, expire: bool
) -> None:
    """Synthesize the original mixed answer, replacing only weather if it expires."""
    cache, _, tts = prepared
    weather = replace(declare_weather_tool(), handler=cache.get_weather)
    light = Tool(
        "set_light",
        "Light",
        LightArguments,
        lambda args: {"status": "confirmed", "on": args.on},
    )
    model = MagicMock(spec=FunctionGemma)
    model.config = LLMConfig(tmp_path)
    model.tools = ToolRegistry([weather, light])
    model.generate_messages.return_value = Generation(
        "",
        (ToolCall("get_weather", {}), ToolCall("set_light", {"on": True})),
        "",
        1,
        1,
        0.01,
        0.001,
    )
    output: list[str] = []
    app = VoiceApplication(
        Assistant(LocalAgent(Session(model)), None, cache),
        MagicMock(spec=STT),
        tts,
        output.append,
    )
    texts: list[str] = []

    def synthesize(text: str) -> tuple:
        texts.append(text)
        if expire and len(texts) == 1:
            cache.entries["now"].started -= 3601
        return np.ones(10, dtype=np.float32), 24000

    tts.synthesize.side_effect = synthesize
    app.respond(VoiceTurn("Weather and lights", 0, 1, threading.Event()))
    assert tts.play_samples.call_count == 1
    assert tts.play.call_count == 0
    assert output[-1] == "Assistant: " + texts[-1]
    assert "Light on." in texts[-1]
    assert ("expired" in texts[-1]) is expire
    assert len(texts) == (2 if expire else 1)


def test_cancel_during_cached_synthesis_suppresses_playback(
    prepared: tuple, tmp_path: Path
) -> None:
    """Cancellation after synthesis prevents both playback and answer output."""
    cache, _, tts = prepared
    weather = replace(declare_weather_tool(), handler=cache.get_weather)
    model = MagicMock(spec=FunctionGemma)
    model.config = LLMConfig(tmp_path)
    model.tools = ToolRegistry([weather])
    model.generate_messages.return_value = Generation(
        "",
        (ToolCall("get_weather", {}), ToolCall("get_weather", {"period": "today"})),
        "",
        1,
        1,
        0.01,
        0.001,
    )
    event = threading.Event()
    output: list[str] = []
    app = VoiceApplication(
        Assistant(LocalAgent(Session(model)), None, cache),
        MagicMock(spec=STT),
        tts,
        output.append,
    )

    def synthesize(text: str) -> tuple:
        event.set()
        return np.ones(10, dtype=np.float32), 24000

    tts.synthesize.side_effect = synthesize
    app.respond(VoiceTurn("Compare weather", 0, 1, event))
    tts.play_samples.assert_not_called()
    assert len(output) == 1


def test_network_refresh_does_not_block_voice(prepared: tuple) -> None:
    """Blocked network refresh leaves the inference executor free and is joined."""
    cache, client, tts = prepared
    cache.entries.clear()
    entered = threading.Event()
    release = threading.Event()

    def fetch(args: WeatherArguments) -> dict:
        entered.set()
        assert release.wait(5)
        return result(args.period)

    client.get_weather.side_effect = fetch
    app = MagicMock(spec=VoiceApplication)
    app.assistant.weather_cache = cache
    app.tts = tts

    async def run() -> None:
        with ThreadPoolExecutor(max_workers=1) as worker:
            task = asyncio.create_task(_refresh_weather(app, worker))
            assert await asyncio.to_thread(entered.wait, 5)
            assert (
                await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(worker, lambda: 42), 1
                )
                == 42
            )
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
