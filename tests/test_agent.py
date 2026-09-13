"""Offline grounded rendering and transactional weather/music agent integration."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from hoast.agent import LocalAgent, WeatherAgent, render_music, render_weather
from hoast.llm import (
    FunctionGemma,
    GeneratedCallError,
    Generation,
    LLMConfig,
    Tool,
    ToolArguments,
    ToolCall,
    ToolRegistry,
)
from hoast.music import MusicArguments, PlayMusicArguments, VolumeMusicArguments
from hoast.session import Session
from hoast.weather import WeatherArguments


def current() -> dict[str, JsonValue]:
    """Return a normalized observation with preceding-interval precipitation."""
    return {
        "city": "London, United Kingdom",
        "timezone": "Europe/London",
        "period": "current",
        "time": "2026-09-13T12:00",
        "temperature_c": 12.5,
        "condition": "clear sky",
        "precipitation_mm": 0.4,
        "interval_seconds": 900,
    }


def weather_call(city: str = "") -> ToolCall:
    """Construct a current-weather routing proposal.

    Args:
        city:
            Destination query; empty means home.

    """
    return ToolCall("get_weather", {"period": "current", "city": city})


@dataclass(slots=True)
class Routing:
    """Controlled inference and tool outcomes around a real session and registry."""

    calls: tuple[ToolCall, ...] = field(default_factory=lambda: (weather_call(),))
    """Next model-proposed batch."""

    text: str = "It is 99 degrees and raining everywhere."
    """Ungrounded model prose that must never reach agent speech."""

    results: list[JsonValue | Exception] = field(default_factory=lambda: [current()])
    """Ordered handler results or failures, consumed once."""

    seen: list[WeatherArguments] = field(default_factory=list)
    """Validated arguments dispatched to the fixture handler."""

    histories: list[list[dict[str, Any]]] = field(default_factory=list)
    """Native histories observed at each routing pass."""

    failure: Exception | None = None
    """Optional inference failure after incremental prose is delivered."""

    dispatched: list[ToolCall] = field(default_factory=list)
    """Ordered validated weather/music calls dispatched by the mixed fixture."""


@pytest.fixture
def agent_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[WeatherAgent, Routing]:
    """Replace inference only; retain real session transactions and strict tools.

    Args:
        monkeypatch:
            Installs weight-free generation for the duration of the test.

        tmp_path:
            Isolated unused model directory; loading is never requested.

    """
    state = Routing()

    def handler(args: WeatherArguments) -> JsonValue:
        """Record validated dispatch and consume one controlled outcome.

        Args:
            args:
                Weather arguments validated by the production registry.

        """
        state.seen.append(args)
        result = state.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def generate(
        model: FunctionGemma,
        messages: Sequence[dict[str, Any]],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Generation:
        """Emit untrusted prose and validate proposals without loading a backend.

        Args:
            model:
                Real owner supplying the strict weather registry.

            messages:
                Native conversation submitted by Session.

            on_text:
                Incremental raw-text callback used by Session's worker.

        """
        state.histories.append(deepcopy(list(messages)))
        if on_text is not None:
            on_text(state.text)
        if state.failure is not None:
            raise state.failure
        model.tools.validate(state.calls)
        return Generation(state.text, state.calls, state.text, 1, 1, 0.01, 0.001)

    monkeypatch.setattr(FunctionGemma, "generate_messages", generate)
    tools = ToolRegistry(
        [Tool("get_weather", "Fixture weather", WeatherArguments, handler)]
    )
    return WeatherAgent(Session(FunctionGemma(LLMConfig(tmp_path), tools))), state


@pytest.fixture
def mixed_runtime(
    agent_runtime: tuple[WeatherAgent, Routing], tmp_path: Path
) -> tuple[LocalAgent, Routing]:
    """Retain controlled inference with five real strict tool contracts and Session.

    Args:
        agent_runtime:
            Installs inference interception and supplies shared fixture outcomes.

        tmp_path:
            Unused weight-free model directory.

    """
    _, state = agent_runtime

    def tool(name: str, arguments: type[ToolArguments]) -> Tool[Any]:
        """Build a recording handler that consumes exactly one fixture outcome.

        Args:
            name:
                Production tool name to record.

            arguments:
                Production argument model enforced by the registry.

        """

        def handler(args: ToolArguments) -> JsonValue:
            """Record validated input and return or raise the next outcome.

            Args:
                args:
                    Arguments validated by the real registry.

            """
            state.dispatched.append(ToolCall(name, args.model_dump(mode="json")))
            result = state.results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        return Tool(name, f"Fixture {name}", arguments, handler)

    tools = ToolRegistry(
        [
            tool("get_weather", WeatherArguments),
            tool("pause_music", MusicArguments),
            tool("resume_music", MusicArguments),
            tool("play_music", PlayMusicArguments),
            tool("volume_music", VolumeMusicArguments),
        ]
    )
    return LocalAgent(Session(FunctionGemma(LLMConfig(tmp_path), tools))), state


@pytest.mark.parametrize("status", ["ambiguous_city", "city_not_found"])
def test_location_clarification_is_brief(status: str) -> None:
    """Ask for an exact location without reading any geocoding candidates.

    Args:
        status:
            Location resolution failure returned by the weather tool.

    """
    result: dict[str, JsonValue] = {
        "status": status,
        "period": "tomorrow",
        "city_query": "  Springfield\n ",
        "choices": list[JsonValue]([{"city": "Unspoken candidate " * 50}]) * 10,
    }
    assert render_weather(result) == (
        'I don\'t know "Springfield". Please tell me the exact location.'
    )
    result["city_query"] = "Long location " * 30
    assert render_weather(result) == (
        "I don't know that location. Please tell me the exact location."
    )


def test_long_metadata_is_not_read_aloud() -> None:
    """Bound location and music labels independently of provider metadata size."""
    weather_result = current()
    weather_result["city"] = "Very long administrative region, " * 30
    assert render_weather(weather_result).startswith("In that location, it's ")
    music_result: dict[str, JsonValue] = {
        "status": "started",
        "confirmation": "requested",
        "seed": {
            "title": "Long title " * 100,
            "artists": list[JsonValue](["Artist"]) * 100,
        },
    }
    assert render_music(music_result) == "Requested a mix based on your selection."
    assert (
        render_music(
            {
                "status": "player_required",
                "choices": list[JsonValue]([{"name": "Long player name " * 100}]) * 100,
            }
        )
        == "Please configure a music player."
    )
    assert render_music({"status": "cannot_resume", "reason": "Diagnostic " * 100}) == (
        "I can't control that playback. Please use the music app."
    )


@pytest.mark.parametrize(
    "invalid",
    [
        ToolCall("get_weather", {"city": "", "country": ""}),
        ToolCall("unknown_tool", {}),
        ToolCall("get_weather", {"period": 123}),
    ],
)
def test_invalid_call_repair(
    agent_runtime: tuple[WeatherAgent, Routing],
    monkeypatch: pytest.MonkeyPatch,
    invalid: ToolCall,
) -> None:
    """Repair a rejected batch with feedback before any handler executes.

    Args:
        agent_runtime:
            Real session and registry with controlled inference.

        monkeypatch:
            Installs a two-attempt backend validation simulation.

        invalid:
            Invalid second call; even the valid first call must wait for repair.

    """
    agent, state = agent_runtime
    original = FunctionGemma.generate_messages
    state.calls = (weather_call(), invalid)

    def generate(
        model: FunctionGemma,
        messages: Sequence[dict[str, Any]],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Generation:
        """Reject the first batch and accept corrected arguments on retry.

        Args:
            model:
                Model owning strict tool declarations.

            messages:
                Original request plus transient repair feedback when retrying.

            on_text:
                Callback receiving untrusted routing prose.

        """
        if state.histories:
            assert not state.seen
            assert "No tools ran" in messages[-1]["content"]
            assert "Validation error:" in messages[-1]["content"]
            assert "Rejected output:" in messages[-1]["content"]
            state.calls = (weather_call(),)
        try:
            return original(model, messages, on_text=on_text)
        except ValueError as error:
            raise GeneratedCallError(str(error), repr(state.calls)) from error

    monkeypatch.setattr(FunctionGemma, "generate_messages", generate)
    assert list(agent.stream("How's the weather?")) == [render_weather(current())]
    assert len(state.histories) == 2
    assert len(state.seen) == 1
    assert "Validation error:" not in str(agent.session.history)
    assert "unknown_tool" not in str(agent.session.history)


def test_exhausted_repair_preserves_session(
    agent_runtime: tuple[WeatherAgent, Routing],
) -> None:
    """Bound repairs, preserve follow-up history, and accept the next user turn.

    Args:
        agent_runtime:
            Real session with controlled repeated malformed generation failures.

    """
    agent, state = agent_runtime
    list(agent.stream("Weather today"))
    history = agent.session.history
    state.failure = GeneratedCallError("Malformed tool syntax", "get_weather(")
    reply = agent.stream("How about tomorrow?")
    assert next(reply) == "I couldn't understand that request. Please rephrase it."
    reply.close()
    assert agent.session.history == history
    assert len(state.histories) == 4
    assert len(state.seen) == 1
    assert not agent.session.pending_calls
    state.failure = None
    state.results = [current()]
    assert list(agent.stream("Weather now")) == [render_weather(current())]


def test_handler_failure_is_not_repaired(
    agent_runtime: tuple[WeatherAgent, Routing],
) -> None:
    """Even a generated-call exception from a handler must not trigger a retry.

    Args:
        agent_runtime:
            Real session with a handler that fails after dispatch starts.

    """
    agent, state = agent_runtime
    state.results = [GeneratedCallError("Handler failure", "not generated")]
    with pytest.raises(GeneratedCallError, match="Handler failure"):
        list(agent.stream("Weather now"))
    assert len(state.histories) == len(state.seen) == 1


def test_projection_failure_is_not_repaired(
    mixed_runtime: tuple[LocalAgent, Routing], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never regenerate or repeat side effects when compact rendering fails.

    Args:
        mixed_runtime:
            Real mixed session with offline handlers.

        monkeypatch:
            Inject a generated-call exception at the result projection boundary.

    """
    agent, state = mixed_runtime
    list(agent.stream("Weather first"))
    state.dispatched.clear()
    state.calls = (weather_call(), ToolCall("volume_music", {"action": "louder"}))
    state.results = [
        current(),
        {"status": "volume_set", "level": 40, "confirmation": "observed"},
    ]
    error = GeneratedCallError("projection failed", "not model output")

    def render(result: JsonValue) -> str:
        """Fail after the real music handler has completed its side effect.

        Args:
            result:
                Verified music result being projected into model history.

        """
        assert isinstance(result, dict) and result["status"] == "volume_set"
        raise error

    monkeypatch.setattr("hoast.agent.render_music", render)
    pieces: list[str] = []
    with pytest.raises(GeneratedCallError) as raised:
        pieces.extend(agent.stream("Turn it up and check weather"))
    assert raised.value is error
    assert pieces == []
    assert len(state.histories) == 2
    assert [call.name for call in state.dispatched] == ["get_weather", "volume_music"]
    assert agent.session.history == () and agent.session.pending_calls == ()
    state.calls, state.results = (weather_call(),), [current()]
    assert list(agent.stream("Fresh weather")) == [render_weather(current())]
    assert state.histories[-1] == [{"role": "user", "content": "Fresh weather"}]


def test_current_interval_and_nulls() -> None:
    """Render rounded current temperature without inferring rain from accumulation."""
    result = current()
    assert render_weather(result) == (
        "In London, United Kingdom, it's clear sky, 12 degrees outside now."
    )
    result.update(temperature_c=None, condition=None, precipitation_mm=None)
    assert render_weather(result) == (
        "In London, United Kingdom, it's conditions unavailable, temperature unavailable."
    )


def test_daily_snow_probability_and_nulls() -> None:
    """Describe a snowy dry-rain forecast without calling snow probability rain."""
    day: dict[str, JsonValue] = {
        "date": "2026-09-14",
        "temperature_min_c": -4,
        "temperature_max_c": -1,
        "condition": "snow",
        "rain_mm": 0,
        "precipitation_probability_max_pct": 80,
    }
    result: dict[str, JsonValue] = {"city": "Oslo", "period": "tomorrow", "days": [day]}
    assert render_weather(result) == (
        "For tomorrow, in Oslo, it's snow, -4 to -1 degrees, no rain."
    )
    day.update(
        temperature_min_c=None,
        condition=None,
        rain_mm=None,
        precipitation_probability_max_pct=None,
    )
    assert render_weather(result) == (
        "For tomorrow, in Oslo, it's conditions unavailable, temperature range "
        "unavailable, rain forecast unavailable."
    )


@pytest.mark.parametrize(
    ("dates", "expected"),
    [
        (["2026-09-15", "2026-09-19"], "90% chance of light rain."),
        ([], "no rain."),
        (None, "rain forecast unavailable."),
    ],
)
def test_weekly_summary(dates: JsonValue, expected: str) -> None:
    """Use weekly extrema and daily conditions, preserving incomplete rain data.

    Args:
        dates:
            Verified rainy dates, an empty dry-week list, or missing rain data.

        expected:
            Grounded rain sentence for that summary.

    """
    summary: dict[str, JsonValue] = {
        "temperature_min_c": -2,
        "temperature_max_c": 21,
        "precipitation_probability_max_pct": 90,
        "rainy_dates": dates,
    }
    result: dict[str, JsonValue] = {
        "city": "Oslo",
        "period": "next_week",
        "days": [
            {
                "date": f"2026-09-{day}",
                "condition": "light rain",
                "rain_mm": None if dates is None else 2 if dates else 0,
            }
            for day in range(14, 21)
        ],
        "summary": summary,
    }
    assert render_weather(result) == (
        "For next week, in Oslo, it's light rain, -2 to 21 degrees, " + expected
    )
    summary["temperature_max_c"] = None
    assert "temperature range unavailable" in render_weather(result)


def test_successful_multiturn_commit(
    agent_runtime: tuple[WeatherAgent, Routing],
) -> None:
    """Commit grounded answers and compact results with exactly one pass per turn.

    Args:
        agent_runtime:
            Real agent/session with controlled routing and weather results.

    """
    agent, state = agent_runtime
    expected = render_weather(current())
    assert list(agent.stream("Weather?")) == [expected]
    history = agent.session.history
    assert [message["role"] for message in history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history[-1] == {"role": "assistant", "content": expected}
    assert history[-2]["content"] == {"result": expected}
    state.results = [current()]
    assert list(agent.stream("And in London?")) == [expected]
    assert state.histories[1] == [
        *history,
        {"role": "user", "content": "And in London?"},
    ]
    assert len(state.histories) == len(state.seen) == 2
    assert agent.session.pending_calls == ()


@pytest.mark.parametrize("count", [4, 5])
def test_bounded_batch(agent_runtime: tuple[WeatherAgent, Routing], count: int) -> None:
    """Accept four ordered calls and reject five before any dispatch.

    Args:
        agent_runtime:
            Real agent/session with controlled routing and weather results.

        count:
            Proposed batch size at or above the dispatch limit.

    """
    agent, state = agent_runtime
    state.calls = tuple(weather_call(str(index)) for index in range(count))
    state.results = [current() for _ in range(count)]
    if count == 5:
        with pytest.raises(RuntimeError, match="at most four"):
            list(agent.stream("Compare cities"))
        assert state.seen == []
        assert agent.session.history == ()
    else:
        answer = render_weather(current())
        chunks = list(agent.stream("Compare cities"))
        assert chunks == [answer, " " + answer, " " + answer, " " + answer]
        assert [args.city for args in state.seen] == ["0", "1", "2", "3"]
        assert agent.session.history[-1]["content"] == "".join(chunks)
    assert len(state.histories) == 1
    assert agent.session.pending_calls == ()


@pytest.mark.parametrize(
    "failure_kind", ["generation", "validation", "handler", "render"]
)
def test_failures_reset_without_retry(
    agent_runtime: tuple[WeatherAgent, Routing], failure_kind: str
) -> None:
    """Withhold prose, propagate failures, clear prior history, and never retry tools.

    Args:
        agent_runtime:
            Real agent/session with controlled routing and weather results.

        failure_kind:
            Boundary at which the current turn fails.

    """
    agent, state = agent_runtime
    list(agent.stream("Successful first turn"))
    state.seen.clear()
    state.calls = (weather_call("first"), weather_call("second"), weather_call("third"))
    error = LookupError("fixture failure")
    error.add_note("controlled offline failure")
    expected_error: type[Exception]
    if failure_kind == "generation":
        state.failure = error
        expected_error = LookupError
    elif failure_kind == "validation":
        state.calls = (weather_call(), ToolCall("get_weather", {"period": 123}))
        expected_error = ValueError
    elif failure_kind == "handler":
        state.results = [current(), error, current()]
        expected_error = LookupError
    else:
        state.results = [current(), None, current()]
        expected_error = TypeError
    pieces: list[str] = []
    with pytest.raises(expected_error) as info:
        pieces.extend(agent.stream("Fail this turn"))
    if failure_kind in {"generation", "handler"}:
        assert info.value is error
        assert info.value.__notes__ == ["controlled offline failure"]
    assert pieces == []
    assert len(state.histories) == 2
    assert (
        len(state.seen)
        == {"generation": 0, "validation": 0, "handler": 2, "render": 3}[failure_kind]
    )
    assert agent.session.history == () and agent.session.pending_calls == ()
    state.failure = None
    state.calls, state.results = (weather_call(),), [current()]
    assert list(agent.stream("Fresh request")) == [render_weather(current())]
    assert state.histories[-1] == [{"role": "user", "content": "Fresh request"}]


def test_tool_free_hallucination_is_discarded(
    agent_runtime: tuple[WeatherAgent, Routing],
) -> None:
    """Replace ungrounded tool-free facts with clarification and discard context.

    Args:
        agent_runtime:
            Real agent/session with controlled routing and weather results.

    """
    agent, state = agent_runtime
    list(agent.stream("First turn"))
    state.calls = ()
    assert list(agent.stream("Weather?")) == ["Please rephrase your request."]
    assert len(state.seen) == 1 and len(state.histories) == 2
    assert agent.session.history == () and agent.session.pending_calls == ()


def test_abandoned_answer_resets(agent_runtime: tuple[WeatherAgent, Routing]) -> None:
    """Closing after the first grounded chunk clears even already committed history.

    Args:
        agent_runtime:
            Real agent/session with controlled routing and weather results.

    """
    agent, state = agent_runtime
    state.calls = (weather_call("first"), weather_call("second"))
    state.results = [current(), current()]
    stream = agent.stream("Compare")
    assert next(stream) == render_weather(current())
    assert len(state.seen) == 2
    assert agent.session.history[-1]["role"] == "assistant"
    stream.close()
    assert agent.session.history == () and agent.session.pending_calls == ()
    assert len(state.histories) == 1 and len(state.seen) == 2


@pytest.mark.parametrize(
    ("name", "result", "expected"),
    [
        (
            "pause_music",
            {"status": "paused", "confirmation": "observed"},
            "Music paused.",
        ),
        (
            "pause_music",
            {"status": "paused", "confirmation": "requested"},
            "Pause requested.",
        ),
        (
            "resume_music",
            {"status": "resumed", "confirmation": "observed"},
            "Music resumed.",
        ),
        (
            "resume_music",
            {"status": "resumed", "confirmation": "requested"},
            "Resume requested.",
        ),
        ("pause_music", {"status": "already_paused"}, "Music is already paused."),
        ("resume_music", {"status": "already_playing"}, "Music is already playing."),
        (
            "pause_music",
            {"status": "not_playing", "reason": "No retained source."},
            "I can't control that playback. Please use the music app.",
        ),
        (
            "resume_music",
            {"status": "cannot_resume", "reason": "Source changed; no retry."},
            "I can't control that playback. Please use the music app.",
        ),
        (
            "play_music",
            {"status": "not_found"},
            "Song not found. Please give the title and artist.",
        ),
        (
            "play_music",
            {"status": "ambiguous"},
            "Which recording? Please give the title and artist.",
        ),
        (
            "play_music",
            {
                "status": "player_required",
                "choices": [{"name": "Kitchen"}, {"name": "Study"}],
            },
            "Please configure a music player.",
        ),
        (
            "resume_music",
            {"status": "player_required", "choices": []},
            "Please configure a music player.",
        ),
        (
            "play_music",
            {
                "status": "started",
                "confirmation": "observed",
                "seed": {"title": "Verified Song", "artists": ["Artist A", "Artist B"]},
            },
            "Started a mix based on Verified Song by Artist A.",
        ),
        (
            "play_music",
            {
                "status": "started",
                "confirmation": "requested",
                "seed": {"title": "Verified Song", "artists": ["Artist A"]},
            },
            "Requested a mix based on Verified Song by Artist A.",
        ),
        (
            "play_music",
            {
                "status": "started",
                "confirmation": "observed",
                "seed": {"title": "", "artists": ["Artist A"]},
            },
            "Started a mix based on Artist A.",
        ),
    ],
)
def test_verified_music_outcomes(
    mixed_runtime: tuple[LocalAgent, Routing],
    name: str,
    result: JsonValue,
    expected: str,
) -> None:
    """Speak verified outcomes, distinguish acknowledgement, and never retry refusals.

    Args:
        mixed_runtime:
            Real mixed registry and transactional session with offline handlers.

        name:
            Music action proposed by the controlled router.

        result:
            Verified handler outcome, including unsuccessful control/search statuses.

        expected:
            Exact grounded speech, independent of model prose and requested metadata.

    """
    agent, state = mixed_runtime
    state.text = "Successfully playing the requested song right now."
    state.calls = (
        ToolCall(name, {"title": "Requested Song"} if name == "play_music" else {}),
    )
    state.results = [result]
    assert render_music(result) == expected
    assert list(agent.stream("Music please")) == [expected]
    assert len(state.histories) == len(state.dispatched) == 1
    assert state.dispatched[0].name == name
    assert agent.session.history[-2]["content"] == {"result": expected}
    assert agent.session.history[-1] == {"role": "assistant", "content": expected}
    assert agent.session.pending_calls == ()


@pytest.mark.parametrize(
    "first", ["pause_music", "resume_music", "play_music", "volume_music"]
)
@pytest.mark.parametrize(
    "second", ["pause_music", "resume_music", "play_music", "volume_music"]
)
def test_multiple_music_actions_rejected_before_weather_dispatch(
    mixed_runtime: tuple[LocalAgent, Routing], first: str, second: str
) -> None:
    """Reject every music-action pair, including duplicates, before any side effect.

    Args:
        mixed_runtime:
            Real mixed registry and transactional session with offline handlers.

        first:
            First proposed music action after a valid weather call.

        second:
            Second proposed music action in the same turn.

    """
    agent, state = mixed_runtime
    state.calls = (
        weather_call(),
        ToolCall(
            first,
            {"artist": "ABBA"}
            if first == "play_music"
            else {"action": "louder"}
            if first == "volume_music"
            else {},
        ),
        ToolCall(
            second,
            {"title": "Dancing Queen"}
            if second == "play_music"
            else {"action": "quieter"}
            if second == "volume_music"
            else {},
        ),
    )
    with pytest.raises(RuntimeError, match="one music action"):
        list(agent.stream("Weather and two music actions"))
    assert state.dispatched == [] and len(state.histories) == 1
    assert agent.session.history == () and agent.session.pending_calls == ()


@pytest.mark.parametrize(
    "query,name,status",
    [
        ("play", "resume_music", "resumed"),
        (" PlAy!? ", "resume_music", "resumed"),
        ("stop", "pause_music", "paused"),
        ("STOP.", "pause_music", "paused"),
    ],
)
@pytest.mark.parametrize("previous", [False, True])
def test_shortcuts_skip_generation(
    mixed_runtime: tuple[LocalAgent, Routing],
    query: str,
    name: str,
    status: str,
    previous: bool,
) -> None:
    """Route standalone controls explicitly even after unrelated conversation.

    Args:
        mixed_runtime:
            Offline agent with real session transactions.

        query:
            Standalone word with supported case and punctuation.

        name:
            Expected native tool.

        status:
            Observed handler outcome.

        previous:
            Whether to establish weather context first.

    """
    agent, state = mixed_runtime
    if previous:
        list(agent.stream("Weather first"))
    history = agent.session.history
    generations = len(state.histories)
    state.failure = AssertionError("Shortcut invoked inference")
    result: JsonValue = {"status": status, "confirmation": "observed"}
    state.results = [result]
    assert list(agent.stream(query)) == [render_music(result)]
    assert len(state.histories) == generations
    assert state.dispatched[-1] == ToolCall(name, {})
    assert agent.session.history[: len(history)] == history
    assert agent.session.history[-4] == {"role": "user", "content": query}
    assert agent.session.history[-3]["tool_calls"][0]["function"] == {
        "name": name,
        "arguments": {},
    }
    assert agent.session.history[-2]["content"] == {"result": render_music(result)}
    assert not agent.session.pending_calls


@pytest.mark.parametrize(
    "query",
    [
        "play title",
        "play music",
        "stop music",
        "playback",
        "stopping",
        "please play",
        "play,",
        "stop sign",
    ],
)
def test_shortcut_word_boundaries(
    mixed_runtime: tuple[LocalAgent, Routing], query: str
) -> None:
    """Send longer phrases and nonmatching words through model routing.

    Args:
        mixed_runtime:
            Offline mixed agent.

        query:
            Text that must not be treated as a standalone control.

    """
    agent, state = mixed_runtime
    state.calls = ()
    assert list(agent.stream(query)) == ["What would you like to do with the music?"]
    assert len(state.histories) == 1
    assert state.dispatched == []


@pytest.mark.parametrize(
    ("query", "action"),
    [("louder", "louder"), ("QUIETER!", "quieter"), (" quiter. ", "quieter")],
)
def test_volume_word_shortcuts(
    mixed_runtime: tuple[LocalAgent, Routing], query: str, action: str
) -> None:
    """Dispatch short volume intents without asking the model to infer a direction.

    Args:
        mixed_runtime:
            Real session and registry with offline handlers.

        query:
            Standalone direction, optional punctuation, or supported spelling alias.

        action:
            Exact relative adjustment that must reach the volume handler.

    """
    agent, state = mixed_runtime
    state.failure = AssertionError("Shortcut invoked inference")
    state.results = [{"status": "volume_set", "level": 35, "confirmation": "observed"}]
    assert list(agent.stream(query)) == ["Volume set to 35 percent."]
    assert not state.histories
    assert state.dispatched == [
        ToolCall("volume_music", {"action": action, "level": 0})
    ]
    assert agent.session.history[-4] == {"role": "user", "content": query}


@pytest.mark.parametrize("query", ["PLAY!", "stop?", "louder", "quieter", "quiter"])
def test_unconfigured_shortcut_preserves_context(
    agent_runtime: tuple[WeatherAgent, Routing], query: str
) -> None:
    """Explain missing music without generating or discarding weather context.

    Args:
        agent_runtime:
            Weather-only agent with controlled inference.

        query:
            Standalone music request.

    """
    agent, state = agent_runtime
    list(agent.stream("Weather first"))
    history = agent.session.history
    state.failure = AssertionError("Unconfigured shortcut invoked inference")
    assert list(agent.stream(query)) == ["Music isn't configured."]
    assert agent.session.history == history
    assert len(state.histories) == len(state.seen) == 1


@pytest.mark.parametrize(
    "result,expected",
    [
        (
            {"status": "volume_set", "level": 35, "confirmation": "observed"},
            "Volume set to 35 percent.",
        ),
        (
            {
                "status": "volume_set",
                "level": 35,
                "observed_level": 30,
                "confirmation": "requested",
            },
            "Requested volume 35 percent.",
        ),
        ({"status": "volume_unchanged", "level": 35}, "Volume is already 35 percent."),
        (
            {
                "status": "cannot_volume",
                "reason": "private diagnostic",
                "confirmation": "requested",
            },
            "Volume change requested, but not confirmed. Please check the music app.",
        ),
        (
            {"status": "cannot_volume", "reason": "unsupported"},
            "I can't adjust that volume. Please use the music app.",
        ),
    ],
)
def test_volume_voice_history(
    mixed_runtime: tuple[LocalAgent, Routing], result: JsonValue, expected: str
) -> None:
    """Keep volume acknowledgement semantics in speech and compact model history.

    Args:
        mixed_runtime:
            Offline mixed agent.

        result:
            Verified volume outcome.

        expected:
            Grounded spoken answer.

    """
    agent, state = mixed_runtime
    state.calls = (ToolCall("volume_music", {"action": "set", "level": 35}),)
    state.results = [result]
    assert list(agent.stream("Volume 35")) == [expected]
    assert agent.session.history[-2]["content"] == {"result": expected}
    assert len(state.dispatched) == len(state.histories) == 1


@pytest.mark.parametrize(
    "result",
    [
        {
            "status": "ambiguous_city",
            "city_query": "Springfield",
            "choices": [{"city": "PRIVATE CANDIDATE"}],
        },
        {**current(), "forecast_metadata": {"private": "PRIVATE FORECAST"}},
    ],
)
def test_details_excluded_from_followup_history(
    agent_runtime: tuple[WeatherAgent, Routing], result: JsonValue
) -> None:
    """Exclude provider details from both committed history and the next model input.

    Args:
        agent_runtime:
            Weather agent with captured native model inputs.

        result:
            Provider result containing unspoken details.

    """
    agent, state = agent_runtime
    state.results = [result, current()]
    answer = render_weather(result)
    assert list(agent.stream("Weather")) == [answer]
    assert agent.session.history[-2]["content"] == {"result": answer}
    list(agent.stream("And tomorrow?"))
    assert "PRIVATE" not in str(state.histories[-1])


@pytest.mark.parametrize("music_first", [False, True])
def test_weather_and_single_music_action(
    mixed_runtime: tuple[LocalAgent, Routing], music_first: bool
) -> None:
    """Allow a full four-call batch with one music action and preserve result order.

    Args:
        mixed_runtime:
            Real mixed registry and transactional session with offline handlers.

        music_first:
            Whether the music action precedes or follows three weather calls.

    """
    agent, state = mixed_runtime
    calls = [weather_call(str(index)) for index in range(3)]
    results: list[JsonValue | Exception] = [current(), current(), current()]
    answers = [render_weather(current())] * 3
    index = 0 if music_first else 3
    calls.insert(index, ToolCall("pause_music", {}))
    results.insert(index, {"status": "paused", "confirmation": "requested"})
    answers.insert(index, "Pause requested.")
    state.calls, state.results = tuple(calls), results
    assert list(agent.stream("Compare weather and pause music")) == [
        answer if index == 0 else " " + answer for index, answer in enumerate(answers)
    ]
    assert state.dispatched == [
        ToolCall("get_weather", {"period": "now", "city": call.arguments["city"]})
        if call.name == "get_weather"
        else call
        for call in calls
    ]
    assert len(state.histories) == 1
    assert agent.session.history[-1]["content"] == " ".join(answers)
    assert agent.session.pending_calls == ()


@pytest.mark.parametrize(
    ("call", "valid"),
    [
        (ToolCall("get_weather", {}), True),
        (ToolCall("pause_music", {}), True),
        (ToolCall("resume_music", {}), True),
        (ToolCall("play_music", {}), True),
        (ToolCall("play_music", {"title": "", "artist": ""}), True),
        (ToolCall("play_music", {"title": " \t", "artist": "\n"}), True),
        (ToolCall("play_music", {"title": None}), False),
        (ToolCall("play_music", {"artist": 42}), False),
        (ToolCall("play_music", {"title": "Song", "author": "Artist"}), False),
        (ToolCall("pause_music", {"title": "Song"}), False),
        (ToolCall("resume_music", {"artist": "Artist"}), False),
        (ToolCall("unknown_music", {}), False),
        (ToolCall("volume_music", {"action": "louder"}), True),
        (ToolCall("volume_music", {"action": "quieter", "level": 0}), True),
        (ToolCall("volume_music", {"action": "set", "level": 1}), True),
        (ToolCall("volume_music", {"action": "set", "level": 100}), True),
        (ToolCall("volume_music", {"action": "set"}), False),
        (ToolCall("volume_music", {"action": "set", "level": 0}), False),
        (ToolCall("volume_music", {"action": "set", "level": 101}), False),
        (ToolCall("volume_music", {"action": "set", "level": True}), False),
        (ToolCall("volume_music", {"action": "set", "level": "35"}), False),
        (ToolCall("volume_music", {"action": "set", "level": 35.0}), False),
        (ToolCall("volume_music", {"action": "louder", "level": 35}), False),
        (ToolCall("volume_music", {"action": "mute"}), False),
    ],
)
def test_mixed_argument_validation(
    mixed_runtime: tuple[LocalAgent, Routing], call: ToolCall, valid: bool
) -> None:
    """Validate empty calls and reject invalid batches before a preceding weather call.

    Args:
        mixed_runtime:
            Real mixed registry and transactional session with offline handlers.

        call:
            Proposed call whose argument contract is exercised.

        valid:
            Whether the strict production schema permits these arguments.

    """
    agent, state = mixed_runtime
    state.calls = (weather_call(), call)
    state.results = [
        current(),
        current() if call.name == "get_weather" else {"status": "already_paused"},
    ]
    if valid:
        list(agent.stream("Validate the request"))
        assert len(state.dispatched) == 2
        assert state.dispatched[-1].arguments == (
            {"period": "now", "city": ""}
            if call.name == "get_weather"
            else {"title": "", "artist": "", **call.arguments}
            if call.name == "play_music"
            else {"level": 0, **call.arguments}
            if call.name == "volume_music"
            else {}
        )
    else:
        pieces: list[str] = []
        with pytest.raises(ValueError):
            pieces.extend(agent.stream("Validate the request"))
        assert pieces == [] and state.dispatched == []
        assert agent.session.history == ()
    assert len(state.histories) == 1 and agent.session.pending_calls == ()


@pytest.mark.parametrize("failure_kind", ["handler", "scalar", "missing", "unknown"])
def test_music_failures_reset_without_success_or_retry(
    mixed_runtime: tuple[LocalAgent, Routing], failure_kind: str
) -> None:
    """Withhold all speech after partial dispatch and propagate music failures once.

    Args:
        mixed_runtime:
            Real mixed registry and transactional session with offline handlers.

        failure_kind:
            Handler exception or malformed verified result to reject.

    """
    agent, state = mixed_runtime
    list(agent.stream("First successful weather turn"))
    state.dispatched.clear()
    state.calls = (weather_call(), ToolCall("resume_music", {}), weather_call("last"))
    error = LookupError("Music acknowledgement failed")
    error.add_note("Do not retry an uncertain music action")
    outcomes: dict[str, tuple[JsonValue | Exception, type[Exception]]] = {
        "handler": (error, LookupError),
        "scalar": (None, TypeError),
        "missing": ({}, KeyError),
        "unknown": ({"status": "invented_success"}, ValueError),
    }
    outcome, exception = outcomes[failure_kind]
    state.results = [current(), outcome, current()]
    pieces: list[str] = []
    with pytest.raises(exception) as info:
        pieces.extend(agent.stream("Weather and resume music"))
    if failure_kind == "handler":
        assert info.value is error
        assert info.value.__notes__ == ["Do not retry an uncertain music action"]
    assert pieces == []
    assert [call.name for call in state.dispatched] == (
        ["get_weather", "resume_music"]
        if failure_kind == "handler"
        else ["get_weather", "resume_music", "get_weather"]
    )
    assert len(state.histories) == 2
    assert agent.session.history == () and agent.session.pending_calls == ()


def test_music_native_followups(mixed_runtime: tuple[LocalAgent, Routing]) -> None:
    """Retain compact play results across pause/resume without replaying prior calls.

    Args:
        mixed_runtime:
            Real mixed registry and transactional session with offline handlers.

    """
    agent, state = mixed_runtime
    for query, call, result in [
        (
            "Play Dancing Queen by ABBA",
            ToolCall("play_music", {"title": "Dancing Queen", "artist": "ABBA"}),
            {
                "status": "started",
                "confirmation": "observed",
                "seed": {"title": "Dancing Queen", "artists": ["ABBA"]},
            },
        ),
        (
            "Pause music",
            ToolCall("pause_music", {}),
            {"status": "paused", "confirmation": "observed"},
        ),
        (
            "Continue music",
            ToolCall("resume_music", {}),
            {"status": "resumed", "confirmation": "observed"},
        ),
    ]:
        history = agent.session.history
        state.calls, state.results = (call,), [result]
        assert list(agent.stream(query)) == [render_music(result)]
        assert state.histories[-1] == [*history, {"role": "user", "content": query}]
    assert [call.name for call in state.dispatched] == [
        "play_music",
        "pause_music",
        "resume_music",
    ]
    assert len(state.histories) == 3 and agent.session.pending_calls == ()


@pytest.mark.parametrize("music_mask", range(16))
@pytest.mark.parametrize("agent_type", [LocalAgent, WeatherAgent])
def test_optional_music_registry_is_complete(
    tmp_path: Path, music_mask: int, agent_type: type[LocalAgent]
) -> None:
    """Accept weather alone or all four music tools for both construction names.

    Args:
        tmp_path:
            Unused model path; registry construction never loads weights.

        music_mask:
            Bit mask selecting each possible subset of the four music tools.

        agent_type:
            Primary agent class or retained weather construction subclass.

    """
    tools: list[Tool[Any]] = [
        Tool("get_weather", "Fixture weather", WeatherArguments, lambda args: None)
    ]
    for index, (name, arguments) in enumerate(
        [
            ("pause_music", MusicArguments),
            ("resume_music", MusicArguments),
            ("play_music", PlayMusicArguments),
            ("volume_music", VolumeMusicArguments),
        ]
    ):
        if music_mask & (1 << index):
            tools.append(Tool(name, "Fixture music", arguments, lambda args: None))
    session = Session(FunctionGemma(LLMConfig(tmp_path), ToolRegistry(tools)))
    if music_mask in (0, 15):
        assert isinstance(agent_type(session), LocalAgent)
    else:
        with pytest.raises(ValueError, match="optional complete music tool set"):
            agent_type(session)


@pytest.mark.parametrize("query", ["", " \t\n"])
def test_empty_mixed_request_fails_before_inference(
    mixed_runtime: tuple[LocalAgent, Routing], query: str
) -> None:
    """Reject empty user turns without generation, dispatch, or invented success.

    Args:
        mixed_runtime:
            Real mixed registry and transactional session with offline handlers.

        query:
            Empty or whitespace-only user request.

    """
    agent, state = mixed_runtime
    with pytest.raises(ValueError, match="must not be empty"):
        list(agent.stream(query))
    assert state.histories == [] and state.dispatched == []
    assert agent.session.history == () and agent.session.pending_calls == ()


def test_music_tool_free_success_is_discarded(
    mixed_runtime: tuple[LocalAgent, Routing],
) -> None:
    """Replace unsupported playback claims with generic clarification and reset history.

    Args:
        mixed_runtime:
            Real mixed registry and transactional session with offline handlers.

    """
    agent, state = mixed_runtime
    list(agent.stream("First weather turn"))
    state.calls = ()
    state.text = "Music is playing successfully."
    assert list(agent.stream("Play music")) == [
        "What would you like to do with the music?"
    ]
    assert len(state.histories) == 2 and len(state.dispatched) == 1
    assert agent.session.history == () and agent.session.pending_calls == ()
