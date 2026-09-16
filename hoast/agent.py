"""Home Assistant routing with grounded, compact spoken answers."""

import re
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue

from .input_text import SENTENCE_PUNCTUATION, canonicalize_text
from .llm import GeneratedCallError, ToolCall
from .logging import get_logger
from .prompts import SYSTEM_PROMPT
from .session import Session

__all__ = [
    "SYSTEM_PROMPT",
    "LocalAgent",
    "WeatherAgent",
    "render_light",
    "render_music",
    "render_weather",
    "validate_action_batch",
]

logger = get_logger(__name__)


def validate_action_batch(calls: Sequence[ToolCall]) -> None:
    """Check application batch limits without executing tools or performing inference.

    Args:
        calls:
            Schema-validated proposed calls; at most four, one music mutation and
            one light action. Read-only playback queries are not music mutations.

    """
    if len(calls) > 4:
        raise RuntimeError("Local agent accepts at most four calls per turn")
    if (
        sum(call.name.endswith("_music") or call.name == "music_next" for call in calls)
        > 1
    ):
        raise RuntimeError("Request one music action at a time")
    if sum(call.name == "set_light" for call in calls) > 1:
        raise RuntimeError("Request one light action at a time")


def _temperature(low: float | None, high: float | None) -> str:
    """Render verified Celsius extrema, retaining missing-value semantics.

    Args:
        low:
            Minimum Celsius measurement or None from the weather tool.

        high:
            Maximum Celsius measurement or None from the weather tool.

    """
    if low is None or high is None:
        return "temperature range unavailable"
    return f"{low:.0f} to {high:.0f} degrees"


def _rain(days: list[dict[str, Any]], probability: Any) -> str:
    """Describe rain using forecast amounts and condition-derived intensity.

    The probability is the provider's all-precipitation daily maximum, including
    for a weekly forecast. Intensity is heavy if any rainy day says heavy, light
    if every rainy day says light or drizzle, and otherwise unspecified.

    Args:
        days:
            Selected daily forecasts with rain amounts in millimeters.

        probability:
            Maximum daily precipitation probability in percent, or None.

    """
    if any(day["rain_mm"] is None for day in days):
        return "rain forecast unavailable"
    rainy = [day for day in days if day["rain_mm"] > 0]
    if not rainy:
        return "no rain"
    if probability is None:
        return "rain chance unavailable"
    conditions = [(day["condition"] or "") for day in rainy]
    intensity = (
        "heavy "
        if any("heavy" in condition for condition in conditions)
        else "light "
        if all(
            "light" in condition or "drizzle" in condition for condition in conditions
        )
        else ""
    )
    return f"{probability:.0f}% chance of {intensity}rain"


def _voice_label(value: str, fallback: str) -> str:
    """Return a short single-line label or a natural spoken fallback.

    Args:
        value:
            Location or music metadata to mention, limited to 60 characters and
            eight words after whitespace normalization.

        fallback:
            Short phrase used when the label is empty or exceeds either limit.

    """
    label = " ".join(value.split())
    return label if label and len(label) <= 60 and len(label.split()) <= 8 else fallback


def render_weather(result: JsonValue) -> str:
    """Render one verified WeatherClient result without model-authored weather facts.

    Temperatures and percentages round to integers. Now forecasts
    include current temperature before today's range. Weekly conditions use the
    most frequent daily label (earliest on ties); probability is a daily maximum.
    Location failures ask for an exact location without reading candidate lists.
    Long location labels are replaced with a short spoken reference.

    Args:
        result:
            Normalized JSON result from the registered WeatherClient handler.

    """
    if not isinstance(result, dict):
        raise TypeError("Weather result must be an object")
    data: dict[str, Any] = result
    if data.get("status") in ("ambiguous_city", "city_not_found"):
        query = _voice_label(data["city_query"], "that location")
        reference = f'"{query}"' if query != "that location" else query
        return f"I don't know {reference}. Please tell me the exact location."
    location = _voice_label(data["city"], "that location")
    city = "" if data["city"].casefold() == "home" else f"in {location}, "
    if data["period"] == "current":
        temperature = data["temperature_c"]
        reading = (
            "temperature unavailable"
            if temperature is None
            else f"{temperature:.0f} degrees outside now"
        )
        condition = data["condition"] or "conditions unavailable"
        answer = f"{city}it's {condition}, {reading}."
        return answer[0].upper() + answer[1:]
    days = data["days"]
    if data["period"] == "next_week":
        summary = data["summary"]
        span = _temperature(summary["temperature_min_c"], summary["temperature_max_c"])
        conditions = [day["condition"] for day in days]
        condition = max(conditions, key=conditions.count) or "conditions unavailable"
        probability = summary["precipitation_probability_max_pct"]
    else:
        day = days[0]
        span = _temperature(day["temperature_min_c"], day["temperature_max_c"])
        condition = day["condition"] or "conditions unavailable"
        probability = day["precipitation_probability_max_pct"]
    prefix = f"For {data['period'].replace('_', ' ')}, "
    if "current_temperature_c" in data:
        prefix = ""
        temperature = data["current_temperature_c"]
        current = (
            "current temperature unavailable"
            if temperature is None
            else f"{temperature:.0f} degrees outside now"
        )
        span = f"{current}, {span}"
    answer = f"{prefix}{city}it's {condition}, {span}, {_rain(days, probability)}."
    return answer[0].upper() + answer[1:]


def render_music(result: JsonValue) -> str:
    """Describe verified music outcomes without claiming unobserved playback.

    Speak brief confirmations and next steps, without player lists or technical
    refusal reasons. Mix labels use a bounded title and first artist only.

    Args:
        result:
            Compact MusicClient result, including acknowledgement semantics.

    """
    if not isinstance(result, dict):
        raise TypeError("Music result must be an object")
    data: dict[str, Any] = result
    status = data["status"]
    if status == "playing":
        playing = data.get("now_playing")
        if playing is None:
            return "Music is playing, but track details are unavailable."
        title = _voice_label(playing["title"], "your selection")
        artist = _voice_label(playing["artist"], "that artist")
        return f"Now playing {title} by {artist}."
    if status == "nothing_playing":
        return (
            "Music is paused."
            if data["observed_state"] == "paused"
            else "Nothing is playing right now."
        )
    playback = data.get("playback")
    if (
        status in ("started", "resumed", "already_playing", "skipped")
        and data.get("confirmation") == "observed"
        and isinstance(playback, dict)
        and playback.get("status") == "playing"
    ):
        return render_music(playback)
    if (
        status in ("started", "resumed", "skipped")
        and data.get("confirmation") == "observed"
    ):
        playing = data.get("now_playing")
        if playing is not None:
            title = _voice_label(playing["title"], "your selection")
            artist = _voice_label(playing["artist"], "that artist")
            return f"Now playing {title} by {artist}."
    if status == "player_required":
        return "Please configure a music player."
    if status == "cannot_volume":
        if data.get("confirmation") == "requested":
            return "Volume change requested, but not confirmed. Please check the music app."
        return "I can't adjust that volume. Please use the music app."
    if status in ("volume_set", "volume_unchanged"):
        level = data["level"]
        if status == "volume_unchanged":
            return f"Volume is already {level} percent."
        if data["confirmation"] == "observed":
            return f"Volume set to {level} percent."
        return f"Requested volume {level} percent."
    if status in ("not_found", "ambiguous"):
        return (
            "Song not found. Please give the title and artist."
            if status == "not_found"
            else "Which recording? Please give the title and artist."
        )
    if status in ("cannot_resume", "not_playing"):
        return "I can't control that playback. Please use the music app."
    if status == "already_playing":
        return "Music is already playing."
    if status == "already_paused":
        return "Music is already paused."
    if status == "already_stopped":
        return "Music is already stopped."
    if status == "stopped":
        return (
            "Music stopped."
            if data["confirmation"] == "observed"
            else "Stop requested."
        )
    if status == "skipped":
        return (
            "Skipped to the next song."
            if data["confirmation"] == "observed"
            else "Next song requested."
        )
    if status in ("paused", "resumed"):
        if data["confirmation"] == "observed":
            return "Music paused." if status == "paused" else "Music resumed."
        return "Pause requested." if status == "paused" else "Resume requested."
    if status == "started":
        seed = data["seed"]
        artist = seed["artists"][0] if seed["artists"] else ""
        label = (
            f"{seed['title']} by {artist}"
            if seed["title"] and artist
            else seed["title"] or artist
        )
        label = _voice_label(label, "your selection")
        prefix = "Started" if data["confirmation"] == "observed" else "Requested"
        return f"{prefix} a mix based on {label}."
    raise ValueError(f"Unexpected music result status: {status}")


def render_light(result: JsonValue) -> str:
    """Confirm observed power only; describe failed control without claiming success.

    Args:
        result:
            Strict light-tool outcome containing status and requested boolean on.

    """
    if not isinstance(result, dict) or type(result.get("on")) is not bool:
        raise TypeError("Light result requires an object with boolean on")
    if result["status"] == "failed":
        return "I couldn't change the light. Please check the switch."
    if result["status"] != "confirmed":
        raise ValueError("Unexpected light result status")
    return "Light on." if result["on"] else "Light off."


@dataclass(slots=True)
class LocalAgent:
    """Route each request locally; render verified results in ordinary speech text.

    Standalone music controls bypass inference. Otherwise the model proposes one bounded
    tool batch per turn, with up to two repairs
    for invalid generated calls before dispatch. Application-rendered
    answers replace detailed tool results in native history for follow-ups. Tool-free model text
    is restricted to a neutral clarification rather than spoken as verified facts.
    """

    session: Session
    """Conversation with weather, optional music/light tools and routing instructions."""

    def __post_init__(self) -> None:
        """Validate renderer-compatible weather/core music and optional next/query/light.

        The guard checks an existing registry; it does not register tools. Custom
        registries may supply the four core music tools without next/query tools.
        These extensions require that core group; unrelated tool names are rejected.
        """
        names = {
            schema["function"]["name"] for schema in self.session.model.tools.schemas()
        }
        if names - {"set_light", "music_next", "what_is_playing"} not in (
            {"get_weather"},
            {
                "get_weather",
                "pause_music",
                "resume_music",
                "play_music",
                "volume_music",
            },
        ) or (names & {"music_next", "what_is_playing"} and "play_music" not in names):
            raise ValueError(
                "LocalAgent requires weather, an optional complete music tool set, optional next/query tools with music, and optional set_light"
            )

    def stream(
        self, user_text: str, *, cancelled: Callable[[], bool] | None = None
    ) -> Generator[str]:
        """Yield grounded answers after routing with at most two call repairs.

        Model routing prose is withheld. Errors and abandonment reset the session;
        errors propagate, and partially dispatched batches are never retried.
        Exhausted generated-call repairs yield a short clarification and preserve
        prior history so the next request can proceed. Standalone play/stop and
        louder/quieter optionally followed by an integer percentage use direct calls.
        Bare relative-volume words use five percentage points. Next/next song/next track and
        switch song/skip song/skip track similarly dispatch music_next directly.

        Args:
            user_text:
                Nonempty weather/music/light request or conversational follow-up.

            cancelled:
                Optional thread-safe cancellation predicate checked before inference,
                before the tool batch, and after dispatch. Dispatched effects persist.

        """
        if cancelled is not None and cancelled():
            self.session.reset()
            return
        user_text = canonicalize_text(user_text)
        shortcut_text = user_text.rstrip(SENTENCE_PUNCTUATION).rstrip().casefold()
        names = {
            schema["function"]["name"] for schema in self.session.model.tools.schemas()
        }
        shortcut = {
            "play": ToolCall("resume_music", {}),
            "stop": ToolCall("pause_music", {}),
            "louder": ToolCall("volume_music", {"action": "louder", "level": 5}),
            "quieter": ToolCall("volume_music", {"action": "quieter", "level": 5}),
            "quiter": ToolCall("volume_music", {"action": "quieter", "level": 5}),
            "raise volume": ToolCall("volume_music", {"action": "louder", "level": 5}),
            "lower volume": ToolCall("volume_music", {"action": "quieter", "level": 5}),
            "next": ToolCall("music_next", {}),
            "next song": ToolCall("music_next", {}),
            "next track": ToolCall("music_next", {}),
            "switch song": ToolCall("music_next", {}),
            "skip song": ToolCall("music_next", {}),
            "skip track": ToolCall("music_next", {}),
            "what is playing": ToolCall("what_is_playing", {}),
            "whats playing": ToolCall("what_is_playing", {}),
            "what's playing": ToolCall("what_is_playing", {}),
            "what song is playing": ToolCall("what_is_playing", {}),
            "lights on": ToolCall("set_light", {"on": True}),
            "light on": ToolCall("set_light", {"on": True}),
            "turn on the light": ToolCall("set_light", {"on": True}),
            "turn on the lights": ToolCall("set_light", {"on": True}),
            "lights off": ToolCall("set_light", {"on": False}),
            "light off": ToolCall("set_light", {"on": False}),
            "turn off the light": ToolCall("set_light", {"on": False}),
            "turn off the lights": ToolCall("set_light", {"on": False}),
        }.get(shortcut_text)
        volume = re.fullmatch(
            r"(louder|quieter|quiter)\s+(\d+)(?:\s+percent)?", shortcut_text
        )
        if volume:
            level = int(volume[2])
            if not 0 <= level <= 100:
                yield "Please use an integer percentage from zero to one hundred."
                return
            shortcut = ToolCall(
                "volume_music",
                {
                    "action": "louder" if volume[1] == "louder" else "quieter",
                    "level": level,
                },
            )
        if shortcut and shortcut.name not in names:
            logger.warning("Agent shortcut status=unconfigured tool=%s", shortcut.name)
            yield (
                "The light isn't configured."
                if shortcut.name == "set_light"
                else "Music isn't configured."
            )
            return
        try:
            if shortcut:
                logger.info(
                    "Agent shortcut tool=%s args=%r", shortcut.name, shortcut.arguments
                )
                self.session.request_tools(user_text, [shortcut])
            else:
                for _ in self.session.stream(user_text, max_repair_attempts=2):
                    pass
        except GeneratedCallError:
            yield "I couldn't understand that request. Please rephrase it."
            return
        except BaseException:
            self.session.reset()
            raise
        try:
            if cancelled is not None and cancelled():
                self.session.reset()
                return
            if not self.session.pending_calls:
                # Discard ungrounded model prose from both speech and future context.
                logger.warning(
                    "Agent routing status=no_calls; requesting clarification"
                )
                logger.debug("Tool-free routing history=%r", self.session.history)
                self.session.reset()
                has_music = any(
                    schema["function"]["name"] == "resume_music"
                    for schema in self.session.model.tools.schemas()
                )
                yield (
                    "What would you like to do with the music?"
                    if has_music
                    else "Please rephrase your request."
                )
                return
            calls = self.session.pending_calls
            logger.info(
                "Agent routing mode=%s tools=%s",
                "shortcut" if shortcut else "llm",
                [call.name for call in calls],
            )
            validate_action_batch(calls)
            answers: list[str] = []

            def compact_result(call: ToolCall, result: JsonValue) -> str:
                """Keep only the grounded spoken result in the model's history.

                Args:
                    call:
                        Validated tool call identifying the result renderer.

                    result:
                        Full isolated handler result, already logged by its client.

                """
                renderer = (
                    render_weather
                    if call.name == "get_weather"
                    else (render_light if call.name == "set_light" else render_music)
                )
                answer = renderer(result)
                answers.append(answer)
                return answer

            self.session.invoke_tools(compact_result)
            if cancelled is not None and cancelled():
                self.session.reset()
                return
            self.session.complete(" ".join(answers))
            for index, answer in enumerate(answers):
                yield (" " if index else "") + answer
        except BaseException:
            # GeneratorExit also resets when a speech consumer abandons the answer.
            self.session.reset()
            raise


class WeatherAgent(LocalAgent):
    """Weather-only construction name retained for existing session integrations."""
