"""Music Assistant 2.10.3 HTTP RPC and player-level transport controls.

Play, pause and next delegate queue/source/group handling to Music Assistant, as
its player controls do. Play may restore an idle queue or retained source; pause
may stop playback when MA cannot pause the output. Explicit directional commands
avoid reversing a user's play/pause intent. Commands are sent once and observations
are not atomic with dispatch; an acknowledgement is not proof of changed playback.
Explicit new music replaces a queue with a dynamic Endless Mix; its seed need not
play first and recommendation providers determine how long it can continue.
Blank play_music arguments use MA's player Play behavior without choosing new music.
"""

import json
import logging
import unicodedata
from collections.abc import Callable
from dataclasses import replace
from http.client import HTTPException
from traceback import TracebackException
from typing import Any, Literal, Self
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)

from hoast.config import MusicConfig
from hoast.llm import Tool, ToolArguments, declaration_only

_MAX_BYTES = 4 * 1024 * 1024
_SEARCH_LIMIT = 25
_LOGGER = logging.getLogger(__name__)


class MusicAssistantError(RuntimeError):
    """Transport, malformed response, or unsupported player operation error."""


class MusicArguments(ToolArguments):
    """No arguments for player transport controls and current-playback queries."""


class PlayMusicArguments(ToolArguments):
    """Optional new-music constraints; two blank fields resume existing playback."""

    title: str = Field(default="", max_length=300, description="Song title, if known")
    """Exact title constraint after Unicode, case, and whitespace normalization."""

    artist: str = Field(
        default="", max_length=300, description="Recording artist, if known"
    )
    """Exact recording artist constraint; blank with a blank title selects MA player Play."""


class VolumeMusicArguments(ToolArguments):
    """Explicit integer percentage setting or percentage-point adjustment."""

    action: Literal["set", "louder", "quieter"] = Field(
        description="set for an absolute percentage; louder/quieter to add/subtract level percentage points"
    )
    """Absolute setting or explicit relative adjustment direction."""

    level: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Integer 0–100: required absolute volume for set; louder/quieter change by this many percentage points, default 5",
    )
    """Integer in [0, 100]; relative requests default to five percentage points."""

    @model_validator(mode="after")
    def validate_level(self) -> Self:
        """Require an explicit absolute setting while permitting the relative default."""
        if self.action == "set" and "level" not in self.model_fields_set:
            raise ValueError("Set requires an explicit integer level from 0 to 100")
        return self


def declare_music_tools() -> list[Tool[Any]]:
    """Return the production music schemas with non-executable placeholder handlers."""
    return [
        Tool(
            "pause_music",
            "Pause music using Music Assistant's player controls.",
            MusicArguments,
            declaration_only,
        ),
        Tool(
            "resume_music",
            "Play or resume music, including an idle Music Assistant queue.",
            MusicArguments,
            declaration_only,
        ),
        Tool(
            "play_music",
            "No title or artist resumes existing playback; new explicit title/artist starts an endless mix.",
            PlayMusicArguments,
            declaration_only,
        ),
        Tool(
            "volume_music",
            "Set, raise, or lower volume with integer percentages 0–100. Set requires a level; louder/quieter default to 5 percentage points unless specified. Clamp relative results to 0–100.",
            VolumeMusicArguments,
            declaration_only,
        ),
        Tool(
            "music_next",
            "Skip to the next song. Use for next song, next track, or switch song.",
            MusicArguments,
            declaration_only,
        ),
        Tool(
            "what_is_playing",
            "Report the currently playing song and artist from Music Assistant.",
            MusicArguments,
            declaration_only,
        ),
    ]


class _Response(BaseModel):
    """Validate only the response fields consumed by this client."""

    model_config: ConfigDict = ConfigDict(strict=True, extra="ignore")
    """Reject coercion while ignoring unrelated external response fields."""


class _Source(_Response):
    """Native source pause capability."""

    id: str
    """Source identifier from the player's source list."""

    can_play_pause: bool = False
    """Whether the source advertises pause and unpause support."""


class _CurrentMedia(_Response):
    """Optional player media labels from the observed native playback snapshot."""

    title: str | None = None
    """Reported loaded title; absent does not imply an unknown track name."""

    artist: str | None = None
    """Reported artist display label, if available."""

    source_id: str | None = None
    """Optional source identity used to reject stale labels from another source."""

    uri: str | None = None
    """Optional media URI used for source and track identity, not shown to users."""

    queue_item_id: str | None = None
    """Optional queue item identity distinguishing repeated tracks at different positions."""


class _Player(_Response):
    """Small player state projection with optional currently loaded media labels."""

    player_id: str = Field(min_length=1)
    """Registered player identifier."""

    name: str
    """Human-readable player name."""

    type: str
    """MA player type, including protocol players excluded from selection."""

    available: bool
    """Whether the device is reachable."""

    enabled: bool = True
    """Whether MA enables this player."""

    playback_state: Literal["idle", "paused", "playing"] = "idle"
    """Observed native playback state."""

    active_source: str | None = None
    """Identifier of the currently retained source, if reported."""

    source_list: list[_Source] = Field(default_factory=list)
    """Source capabilities indexed locally by identifier."""

    supported_features: list[str] = Field(default_factory=list)
    """Advertised features; playback requires pause and volume requires volume_set."""

    volume_level: int | None = Field(default=None, ge=0, le=100)
    """Individual logical volume percentage; None means unavailable."""

    group_volume: int | None = Field(default=None, ge=0, le=100)
    """Group logical volume percentage; None means unavailable, not zero."""

    group_members: list[str] = Field(default_factory=list)
    """Group membership; a nonempty list identifies a native sync leader."""

    synced_to: str | None = None
    """Native sync leader, resolved before active_group."""

    active_group: str | None = None
    """Active group player whose source the member hears."""

    active_output_protocol: str | None = None
    """Active native/protocol output; volume checks use it to detect route changes."""

    current_media: _CurrentMedia | None = None
    """Optional loaded-media labels; meaningful for announcements only when playing."""

    @property
    def grouped(self) -> bool:
        """Identify dedicated groups and native sync leaders as MA does."""
        return self.type == "group" or bool(self.group_members)

    @property
    def current_volume(self) -> int | None:
        """Return group volume for grouped targets, otherwise individual volume."""
        return self.group_volume if self.grouped else self.volume_level

    def volume_route(self) -> tuple[str, str, tuple[str, ...], str | None, bool]:
        """Capture effective identity, membership, output, and volume capability."""
        return (
            self.player_id,
            self.type,
            tuple(sorted(self.group_members)),
            self.active_output_protocol,
            "volume_set" in self.supported_features,
        )

    def compact(self) -> dict[str, JsonValue]:
        """Return compact state with this target's group-aware volume, or None."""
        return {
            "player_id": self.player_id,
            "name": self.name,
            "available": self.available,
            "enabled": self.enabled,
            "state": self.playback_state,
            "source": self.active_source,
            "synced_to": self.synced_to,
            "active_group": self.active_group,
            "volume_level": self.current_volume,
        }


class _QueueSource(_Response):
    """URI-only projection of a queue's dynamic source."""

    uri: str | None = None
    """Container source URI, if reported by MA."""


class _ArtistName(_Response):
    """Structured current-track artist label."""

    name: str
    """Provider artist name in queue metadata."""


class _QueueMedia(_Response):
    """Structured current item; queue display labels are not parsed as track titles."""

    media_type: str | None = None
    """Expected track for a song announcement; other media types are not announced."""

    name: str | None = None
    """Actual current media title, if supplied."""

    artists: list[_ArtistName] = Field(default_factory=list)
    """Ordered artists; announcements mention only the first."""


class _QueueItem(_Response):
    """Current queue item metadata projection."""

    media_item: _QueueMedia | None = None
    """Structured media metadata, when available."""


class _Queue(_Response):
    """Queue identity and observable dynamic-playback state."""

    queue_id: str = Field(min_length=1)
    """Actual queue identifier, which can differ from the selected player."""

    available: bool
    """Whether this queue can receive playback."""

    state: Literal["idle", "paused", "playing"] = "idle"
    """Last reported queue playback state."""

    is_dynamic: bool = False
    """Whether MA reports a dynamic queue."""

    sources: list[_QueueSource] = Field(default_factory=list)
    """Small source-container list used to confirm the requested Endless Mix seed."""

    active: bool = False
    """Whether this queue is currently active on its player."""

    current_item: _QueueItem | None = None
    """Observed current track, which need not equal the requested mix seed."""


def _playing_labels(title: str | None, artist: str | None) -> dict[str, JsonValue]:
    """Return a normalized announcement field only when both labels are available.

    Args:
        title:
            Observed current media title, never the requested seed as a fallback.

        artist:
            First structured artist or the player's single artist display label.

    """
    if title is None or artist is None or not title.strip() or not artist.strip():
        return {}
    return {
        "now_playing": {
            "title": " ".join(title.split()),
            "artist": " ".join(artist.split()),
        }
    }


class _NoRedirect(HTTPRedirectHandler):
    """Refuse redirecting authenticated requests away from the configured API."""

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        """Reject every redirect via urllib's HTTPError path.

        Args:
            req:
                Original request.

            fp:
                Response stream.

            code:
                HTTP redirect status.

            msg:
                HTTP reason.

            headers:
                Response headers.

            newurl:
                Rejected redirect target.

        """
        return


def _object(value: JsonValue) -> dict[str, JsonValue]:
    """Require a response object without scanning unrelated nested arrays.

    Args:
        value:
            Decoded JSON value.

    """
    if not isinstance(value, dict):
        raise MusicAssistantError("Expected a Music Assistant response object")
    return value


def _array(value: JsonValue) -> list[JsonValue]:
    """Require a JSON array without validating all its elements.

    Args:
        value:
            Decoded JSON value.

    """
    if not isinstance(value, list):
        raise MusicAssistantError("Expected a Music Assistant response array")
    return value


def _text(value: JsonValue) -> str:
    """Require a nonblank string at the external boundary.

    Args:
        value:
            Decoded field value.

    """
    if not isinstance(value, str) or not value.strip():
        raise MusicAssistantError("Expected nonblank Music Assistant text")
    return value


def _boolean(value: JsonValue) -> bool:
    """Require a boolean without accepting integer coercion.

    Args:
        value:
            Decoded field value.

    """
    if not isinstance(value, bool):
        raise MusicAssistantError("Expected Music Assistant boolean")
    return value


def _parse[T: _Response](model: type[T], value: JsonValue) -> T:
    """Validate a small projection without exposing server data in errors.

    Args:
        model:
            Response projection class.

        value:
            Decoded JSON response.

    """
    try:
        return model.model_validate(value)
    except ValidationError:
        raise MusicAssistantError("Malformed Music Assistant state response") from None


def _normalize(value: str) -> str:
    """Compare catalog labels by letters/numbers, ignoring punctuation, case and spacing.

    Args:
        value:
            Title or artist text, not a numeric command. Spaced initials such as
            G E M compare equally to the catalog spelling G.E.M.

    """
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", value)
        if unicodedata.category(character)[0] in "LNM"
    ).casefold()


def _reject_constant(value: str) -> None:
    """Reject non-JSON numeric constants accepted by Python's decoder.

    Args:
        value:
            Invalid numeric constant spelling.

    """
    raise ValueError("Invalid JSON numeric constant")


class MusicClient:
    """Synchronous fixed-endpoint client; credentials are resolved by the caller."""

    _config: MusicConfig
    """Validated server and optional explicit player configuration."""

    _token: str
    """Bearer credential, never included in diagnostics or returned results."""

    _url: str
    """Configured server's fixed /api endpoint."""

    _opener: OpenerDirector
    """HTTP transport refusing all redirects."""

    def __init__(self, config: MusicConfig, token: str) -> None:
        """Construct the transport without making requests or reading environment.

        Args:
            config:
                Server URL and optional player identifier.

            token:
                Required bearer token resolved separately by the configuration helper.

        """
        if (
            not isinstance(token, str)
            or not token.strip()
            or any(ord(character) < 33 or ord(character) > 126 for character in token)
        ):
            raise ValueError("Music Assistant requires a nonblank bearer token")
        parts = urlsplit(config.server_url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or parts.path not in {"", "/"}
        ):
            raise ValueError("Music Assistant server_url must be an HTTP(S) origin")
        self._config = config
        self._token = token
        self._url = config.server_url.rstrip("/") + "/api"
        self._opener = build_opener(_NoRedirect())

    def _request(self, command: str, **args: JsonValue) -> JsonValue:
        """Log one RPC's arguments and selected response fields without credentials.

        Args:
            command:
                Fixed MA command name supplied by this client.

            **args:
                Command-specific JSON arguments, redacted before logging.

        """
        _LOGGER.debug("Music RPC %s args=%s", command, self._redact(json.dumps(args)))
        try:
            result = self._rpc(command, **args)
        except MusicAssistantError as error:
            self._log_failure(command, error)
            raise
        # Inspect only nonsecret state projections, never full provider metadata.
        summary: dict[str, JsonValue] = {"type": type(result).__name__}
        if isinstance(result, list):
            summary["count"] = len(result)
        elif isinstance(result, dict):
            for key in (
                "player_id",
                "queue_id",
                "playback_state",
                "state",
                "available",
                "active_source",
                "is_dynamic",
                "volume_level",
                "group_volume",
            ):
                value = result.get(key)
                if key in result and (
                    value is None or isinstance(value, (str, bool, int, float))
                ):
                    summary[key] = value
            for key in ("tracks", "artists"):
                value = result.get(key)
                if isinstance(value, list):
                    summary[f"{key}_count"] = len(value)
        _LOGGER.debug(
            "Music RPC %s status=ok response=%s",
            command,
            self._redact(json.dumps(summary)),
        )
        return result

    def _redact(self, text: str) -> str:
        """Remove the bearer token and its serialized spellings from diagnostic text.

        Args:
            text:
                Formatted diagnostic text, never a raw provider inventory.

        """
        for secret in (
            json.dumps(self._token)[1:-1],
            repr(self._token)[1:-1],
            self._token,
        ):
            text = text.replace(secret, "[redacted]")
        return text

    def _log_failure(self, operation: str, error: Exception) -> None:
        """Log complete sanitized traceback chains and notes without local variables.

        Args:
            operation:
                Public tool or RPC name whose execution failed.

            error:
                Failure, including contexts hidden from the user-facing exception.

        """
        trace = TracebackException.from_exception(error)
        current: TracebackException | None = trace
        while current is not None:
            current.__suppress_context__ = False
            current = current.__cause__ or current.__context__
        _LOGGER.error(
            "Music %s status=error\n%s",
            operation,
            self._redact("".join(trace.format(chain=True))),
        )

    def _run_tool(
        self,
        name: str,
        args: dict[str, JsonValue],
        operation: Callable[[], JsonValue],
    ) -> JsonValue:
        """Run a public music operation with consistent direct-call and tool diagnostics.

        Args:
            name:
                Public tool name.

            args:
                Explicit nonsecret tool arguments, redacted before logging.

            operation:
                Validated operation returning a compact outcome.

        """
        _LOGGER.info("Music tool %s args=%s", name, self._redact(json.dumps(args)))
        try:
            result = operation()
        except (MusicAssistantError, ValidationError) as error:
            self._log_failure(name, error)
            raise
        outcome = _object(result)
        status = outcome.get("status")
        _LOGGER.info(
            "Music tool %s status=%s confirmation=%s",
            name,
            self._redact(str(status)),
            outcome.get("confirmation"),
        )
        _LOGGER.debug(
            "Music tool %s result=%s", name, self._redact(json.dumps(outcome))
        )
        if status in {
            "not_playing",
            "cannot_resume",
            "cannot_volume",
            "player_required",
            "not_found",
            "ambiguous",
        }:
            reason = outcome.get("reason") or {
                "player_required": "Configure a unique available player.",
                "not_found": "No available match satisfies the title and artist constraints.",
                "ambiguous": "Matches have different recording artist sets.",
            }.get(str(status), "Existing playback cannot perform this operation.")
            _LOGGER.warning(
                "Music tool %s status=%s reason=%s",
                name,
                self._redact(str(status)),
                self._redact(str(reason)),
            )
        return result

    def _rpc(self, command: str, **args: JsonValue) -> JsonValue:
        """Execute one RPC, checking HTTP status and limiting responses to 4 MiB.

        No retries are performed, including after uncertain mutation failures.
        Server error bodies and underlying exception text are never exposed.

        Args:
            command:
                Fixed MA command name supplied by this client.

            **args:
                Command-specific JSON arguments.

        """
        request = Request(
            self._url,
            data=json.dumps({"command": command, "args": args}).encode(),
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=30) as response:
                if not 200 <= response.status < 300:
                    raise MusicAssistantError(f"Music Assistant HTTP {response.status}")
                body = response.read(_MAX_BYTES + 1)
        except HTTPError as error:
            code = error.code
            error.close()
            detail = " (check token and permissions)" if code in {401, 403} else ""
            raise MusicAssistantError(f"Music Assistant HTTP {code}{detail}") from None
        except URLError, OSError, ValueError, HTTPException:
            raise MusicAssistantError("Music Assistant transport failed") from None
        if len(body) > _MAX_BYTES:
            raise MusicAssistantError("Music Assistant response exceeds 4 MiB")
        try:
            result: JsonValue = json.loads(body, parse_constant=_reject_constant)
        except ValueError, UnicodeError, RecursionError:
            raise MusicAssistantError(
                "Music Assistant returned malformed JSON"
            ) from None
        return result

    def tools(self) -> list[Tool[Any]]:
        """Bind production music declarations to this client's service handlers."""
        handlers: dict[str, Callable[[Any], JsonValue]] = {
            "pause_music": lambda args: self.pause_music(),
            "resume_music": lambda args: self.resume_music(),
            "play_music": lambda args: self.play_music(args.title, args.artist),
            "volume_music": lambda args: self.volume_music(args.action, args.level),
            "music_next": lambda args: self.music_next(),
            "what_is_playing": lambda args: self.what_is_playing(),
        }
        declarations = declare_music_tools()
        assert set(handlers) == {tool.name for tool in declarations}
        return [replace(tool, handler=handlers[tool.name]) for tool in declarations]

    def require_airplay_2_ptp(self) -> None:
        """Require the selected Music Assistant player to use AirPlay 2 PTP timing.

        The setting is persisted on the selected player rather than inferred from
        transient playback state, which may be idle during application startup.

        Raises:
            MusicAssistantError: If no configured player is eligible or its configured
                streaming mode is not AirPlay 2 PTP.

        """
        selected = self._select()
        if isinstance(selected, dict):
            raise MusicAssistantError("Configure one available music player for AirPlay 2 PTP")
        mode = self._request(
            "config/players/get_value",
            player_id=selected.player_id,
            key="streaming_mode",
        )
        if mode != "ap2_ptp":
            _LOGGER.error(
                "music.airplay_ptp status=invalid player_id=%s configured_mode=%r",
                selected.player_id,
                mode,
            )
            raise MusicAssistantError(
                "Configured music player must use AirPlay 2 PTP timing"
            )
        _LOGGER.info("music.airplay_ptp status=ok player_id=%s", selected.player_id)

    def _players(self) -> list[_Player]:
        """Read registered nonprotocol players, including disabled/unavailable ones."""
        return [
            _parse(_Player, value)
            for value in _array(
                self._request(
                    "players/all",
                    return_unavailable=True,
                    return_disabled=True,
                    return_protocol_players=False,
                )
            )
        ]

    def players(self) -> JsonValue:
        """List compact targets with their own group-aware volume, without resolving members."""
        return [
            player.compact() for player in self._players() if player.type != "protocol"
        ]

    def _select(self) -> _Player | dict[str, JsonValue]:
        """Require a configured player or exactly one eligible nonprotocol player."""
        players = self._players()
        eligible = [
            p for p in players if p.available and p.enabled and p.type != "protocol"
        ]
        if self._config.player_id:
            selected = next(
                (p for p in players if p.player_id == self._config.player_id), None
            )
            if selected is None or selected not in eligible:
                raise MusicAssistantError(
                    "Configured music player is missing, unavailable, disabled, or protocol-only"
                )
            return selected
        if len(eligible) != 1:
            return {
                "status": "player_required",
                "choices": [p.compact() for p in eligible],
            }
        return eligible[0]

    def _effective(self, player_id: str) -> _Player:
        """Resolve the native sync leader and active group, rejecting routing cycles.

        Args:
            player_id:
                Selected player's identifier, re-read for every snapshot.

        """
        seen: set[str] = set()
        for _ in range(16):
            if player_id in seen:
                raise MusicAssistantError("Music Assistant player group cycle")
            seen.add(player_id)
            player = _parse(_Player, self._request("players/get", player_id=player_id))
            if player.player_id != player_id:
                raise MusicAssistantError("Music Assistant returned a different player")
            if not player.available or not player.enabled or player.type == "protocol":
                raise MusicAssistantError(
                    "Effective music player is unavailable or unsupported"
                )
            parent = next(
                (
                    p
                    for p in (player.synced_to, player.active_group)
                    if p and p != player_id
                ),
                None,
            )
            if parent is None:
                return player
            player_id = parent
        raise MusicAssistantError("Music Assistant group nesting exceeds 16 players")

    def status(self) -> JsonValue:
        """Read effective state, group-aware volume, and queue without changing playback."""
        selected = self._select()
        if isinstance(selected, dict):
            return selected
        player = self._effective(selected.player_id)
        result = player.compact()
        queue = self._request(
            "player_queues/get_active_queue", player_id=player.player_id
        )
        result["queue"] = (
            None if queue is None else _parse(_Queue, queue).model_dump(mode="json")
        )
        return result

    def prompt_context(self) -> dict[str, JsonValue]:
        """Read only on/off playback state for the next agent system prompt.

        Playing maps to on; paused and idle map to off. Player selection failures
        remain explicit. No title, artist, volume, source, or player details enter
        this background context; what_is_playing handles explicit metadata queries.
        """
        selected = self._select()
        if isinstance(selected, dict):
            _LOGGER.debug(
                "music.prompt_context selection=%s", self._redact(json.dumps(selected))
            )
            return {"status": "player_required"}
        player = self._effective(selected.player_id)
        result: dict[str, JsonValue] = {
            "status": "observed",
            "state": "on" if player.playback_state == "playing" else "off",
        }
        _LOGGER.debug(
            "music.prompt_context result=%s", self._redact(json.dumps(result))
        )
        return result

    def pause_music(self) -> JsonValue:
        """Request MA player pause, including its source routing and stop fallback."""
        return self._run_tool("pause_music", {}, lambda: self._control(resume=False))

    def resume_music(self) -> JsonValue:
        """Request MA player Play and include the canonical what_is_playing readback."""
        return self._run_tool("resume_music", {}, lambda: self._control(resume=True))

    def music_next(self) -> JsonValue:
        """Request the next song once; confirm only an observed media identity change."""
        return self._run_tool("music_next", {}, self._next)

    def what_is_playing(self, *, _player: _Player | None = None) -> JsonValue:
        """Return the canonical observed now-playing result without changing playback.

        The registered tool takes no arguments. Transport commands reuse their
        fresh post-command player observation through the internal-only keyword
        to avoid another identical RPC; ordinary callers query the selected player.

        Args:
            _player:
                Optional already-validated fresh player observation for internal
                transport integration. None performs live player selection/readback.

        """
        return self._run_tool("what_is_playing", {}, lambda: self._playing(_player))

    def _playing(self, player: _Player | None) -> JsonValue:
        """Describe observed playback with bounded-renderer metadata, never a mix seed.

        Args:
            player:
                Fresh validated effective player, or None to query it.

        """
        if player is None:
            selected = self._select()
            if isinstance(selected, dict):
                return selected
            player = self._effective(selected.player_id)
        playing: dict[str, JsonValue] = {}
        media = player.current_media
        if (
            player.playback_state == "playing"
            and media is not None
            and (
                media.source_id in (None, player.active_source)
                or (
                    player.active_source is not None
                    and media.uri == player.active_source
                )
            )
        ):
            playing = _playing_labels(media.title, media.artist)
        return {
            "status": "playing"
            if player.playback_state == "playing"
            else "nothing_playing",
            "player_id": player.player_id,
            "source": player.active_source,
            "observed_state": player.playback_state,
            "confirmation": "observed",
            **playing,
        }

    def _next(self) -> JsonValue:
        """Delegate next to the selected MA player, preserving source/group ownership.

        Next can be delayed or do nothing at queue end. Unchanged/missing identity
        is reported as requested, never retried or treated as proven advancement.
        """
        selected = self._select()
        if isinstance(selected, dict):
            return selected
        before = self._effective(selected.player_id)
        self._request("players/cmd/next", player_id=selected.player_id)
        after = self._effective(selected.player_id)
        old, new = before.current_media, after.current_media
        changed = False
        if old is not None and new is not None:
            if old.queue_item_id and new.queue_item_id:
                changed = old.queue_item_id != new.queue_item_id
            elif old.uri and new.uri:
                changed = old.uri != new.uri
        observed = (
            changed
            and after.playback_state == "playing"
            and after.player_id == before.player_id
            and after.active_source == before.active_source
        )
        return {
            "status": "skipped",
            "player_id": after.player_id,
            "source": after.active_source,
            "observed_state": after.playback_state,
            "confirmation": "observed" if observed else "requested",
            "playback": self.what_is_playing(_player=after),
        }

    def volume_music(
        self, action: Literal["louder", "quieter", "set"], level: int | None = None
    ) -> JsonValue:
        """Adjust idle or active volume without playback RPCs; observe once, never retry.

        Args:
            action:
                Louder/quieter adds/subtracts the requested percentage points;
                set uses an absolute percentage. Results clamp to [0, 100].

            level:
                Integer percentage or percentage-point delta in [0, 100]. None
                selects the relative default of five; set requires a value.

        """
        arguments: dict[str, JsonValue] = {"action": action}
        if level is not None:
            arguments["level"] = level
        return self._run_tool(
            "volume_music",
            arguments,
            lambda: self._volume_music(VolumeMusicArguments.model_validate(arguments)),
        )

    def _volume_music(self, args: VolumeMusicArguments) -> JsonValue:
        """Recheck routing and current volume before one native volume mutation.

        Snapshots are non-atomic. Group targets use group_volume, never a member's
        individual reading. Missing relative readings refuse dispatch; absolute
        settings need only advertised volume support. Readback lag is requested,
        while routing drift after dispatch reports cannot_volume and requested.

        Args:
            args:
                Validated absolute percentage or explicit percentage-point change.

        """
        selected = self._select()
        if isinstance(selected, dict):
            return {**selected, "reason": "Configure a unique available player."}
        before = self._effective(selected.player_id)
        current = before.current_volume
        target = args.level if args.action == "set" else current
        if current is not None and args.action != "set":
            target = (
                min(100, current + args.level)
                if args.action == "louder"
                else max(0, current - args.level)
            )
        base: dict[str, JsonValue] = {
            "player_id": before.player_id,
            "level": target,
        }
        if "volume_set" not in before.supported_features:
            return {
                **base,
                "status": "cannot_volume",
                "reason": "This player does not support volume control.",
            }
        if target is None:
            return {
                **base,
                "status": "cannot_volume",
                "reason": "Current volume is unavailable; specify an absolute level from 0 to 100.",
            }
        checked = self._effective(selected.player_id)
        if (
            checked.volume_route() != before.volume_route()
            or checked.current_volume != current
        ):
            return {
                **base,
                "status": "cannot_volume",
                "reason": "Player, group, capability, or volume changed; no command sent.",
            }
        if target == current:
            return {**base, "status": "volume_unchanged", "confirmation": "observed"}
        self._request(
            "players/cmd/group_volume" if before.grouped else "players/cmd/volume_set",
            player_id=before.player_id,
            volume_level=target,
        )
        after = self._effective(selected.player_id)
        if after.volume_route() != before.volume_route():
            return {
                **base,
                "status": "cannot_volume",
                "confirmation": "requested",
                "reason": "Player or group changed during dispatch; no retry attempted.",
                "observed_player_id": after.player_id,
            }
        return {
            **base,
            "status": "volume_set",
            "confirmation": "observed"
            if after.current_volume == target
            else "requested",
            "observed_level": after.current_volume,
        }

    def _control(self, *, resume: bool) -> JsonValue:
        """Send a directional MA player command once and report the immediate observation.

        MA owns idle-queue restoration, protocol/source capabilities, and group
        redirection. Source restoration is a normal play outcome. If the effective
        player changes, report only a request acknowledgement without retrying.

        Args:
            resume:
                True requests play/resume, False requests pause. MA may stop an
                output that cannot pause; observed idle is then reported as stopped.

        """
        selected = self._select()
        if isinstance(selected, dict):
            return selected
        before = self._effective(selected.player_id)
        base: dict[str, JsonValue] = {
            "player_id": before.player_id,
            "source": before.active_source,
        }
        if before.playback_state == ("playing" if resume else "paused"):
            return {
                **base,
                "status": "already_playing" if resume else "already_paused",
                "confirmation": "observed",
                **(
                    {"playback": self.what_is_playing(_player=before)} if resume else {}
                ),
            }
        if not resume and before.playback_state == "idle":
            return {**base, "status": "already_stopped", "confirmation": "observed"}
        self._request(
            "players/cmd/play" if resume else "players/cmd/pause",
            player_id=selected.player_id,
        )
        after = self._effective(selected.player_id)
        target = (
            "playing"
            if resume
            else ("idle" if after.playback_state == "idle" else "paused")
        )
        observed = (
            after.player_id == before.player_id and after.playback_state == target
        )
        return {
            **base,
            "status": "resumed"
            if resume
            else ("stopped" if target == "idle" else "paused"),
            "source": after.active_source,
            "confirmation": "observed" if observed else "requested",
            "observed_state": after.playback_state,
            "observed_player_id": after.player_id,
            **({"playback": self.what_is_playing(_player=after)} if resume else {}),
        }

    def _candidates(
        self,
        title: str,
        artist: str,
    ) -> tuple[list[dict[str, JsonValue]], list[JsonValue]]:
        """Return exact available matches in search order and up to eight suggestions.

        Search reads one bounded page and resolves matching track mappings for artist
        checks. Suggestions can lack artist details when MA returned only a mapping.

        Args:
            title:
                Nonblank title for track search, or empty for artist-only search.

            artist:
                Optional recording artist constraint or artist-only query.

        """
        kind = "track" if title else "artist"
        query = f"{artist} - {title}" if artist and title else title or artist
        response = _object(
            self._request(
                "music/search",
                search_query=query,
                media_types=[kind],
                limit=_SEARCH_LIMIT,
            )
        )
        values = _array(response.get("tracks" if title else "artists"))
        if len(values) > _SEARCH_LIMIT:
            raise MusicAssistantError(
                "Music Assistant exceeded the requested search limit"
            )
        matches: dict[tuple[str, tuple[str, ...]], dict[str, JsonValue]] = {}
        choices: list[JsonValue] = []
        for value in values:
            item = _object(value)
            name = _text(item.get("name"))
            uri = _text(item.get("uri"))
            preview: dict[str, JsonValue] = {
                "title": name if title else "",
                "uri": uri,
                "artists": [
                    _text(_object(entry).get("name"))
                    for entry in _array(item.get("artists", []))
                ]
                if title
                else [name],
            }
            if len(choices) < 8:
                choices.append(preview)
            if _normalize(name) != _normalize(title or artist):
                continue
            if title and "artists" not in item:
                item = _object(self._request("music/item_by_uri", uri=uri))
                name = _text(item.get("name"))
                if _normalize(name) != _normalize(title):
                    continue
                if _text(item.get("uri")) != uri:
                    raise MusicAssistantError(
                        "Music Assistant resolved a different media URI"
                    )
            names = (
                [
                    _text(_object(entry).get("name"))
                    for entry in _array(item.get("artists"))
                ]
                if title
                else [name]
            )
            if title and not names:
                raise MusicAssistantError(
                    "Music Assistant track has no recording artists"
                )
            preview["artists"] = list[JsonValue](names)
            if artist and _normalize(artist) not in {
                _normalize(name) for name in names
            }:
                continue
            if not _boolean(item.get("available", True)) or not _boolean(
                item.get("is_playable", True)
            ):
                continue
            if "provider_mappings" in item and not any(
                _boolean(_object(mapping).get("available", True))
                for mapping in _array(item["provider_mappings"])
            ):
                continue
            version = item.get("version", "")
            if not isinstance(version, str):
                raise MusicAssistantError("Music Assistant returned an invalid version")
            key = (
                _normalize(name),
                tuple(sorted({_normalize(n) for n in names})),
            )
            matches.setdefault(
                key,
                {
                    "title": name if title else "",
                    "artists": list(names),
                    "uri": uri,
                    "version": version,
                },
            )
        _LOGGER.debug(
            "Music search matches=%s choices=%s",
            self._redact(json.dumps(list(matches.values()))),
            self._redact(json.dumps(choices)),
        )
        return list(matches.values()), choices

    def play_music(self, title: str = "", artist: str = "") -> JsonValue:
        """Resume existing playback for blank fields, otherwise start a new Endless Mix.

        Provider duplicates and versions with identical normalized title and full
        artist set collapse to the first available ranked result; the API has no
        version selector. Title-only requests use the first available exact-title
        match in MA search order, even when other artists have recordings with the
        same title. Explicit artist constraints remain strict; different matching
        collaborator sets require clarification. Artist-only
        homonyms can collapse because names alone do not establish artist identity.
        Queue state distinguishes requested from observed dynamic playback; the
        what_is_playing result from a fresh player readback supplies rendering.
        These observations cannot prove audible playback or endless supply.

        Args:
            title:
                Song title, matched exactly after Unicode/case/whitespace normalization.
                Blank with a blank artist delegates to MA player Play without search.

            artist:
                Recording artist, matched exactly against a track's artists. Blank
                with a blank title lets MA restore its existing playback selection.

        """
        return self._run_tool(
            "play_music",
            {"title": title, "artist": artist},
            lambda: self._play_music(title, artist),
        )

    def _play_music(self, title: str, artist: str) -> JsonValue:
        """Validate play arguments and route blank requests before any search or queue RPC.

        Args:
            title:
                Optional exact song title; whitespace counts as blank.

            artist:
                Optional exact recording artist; whitespace counts as blank.

        """
        args = PlayMusicArguments(title=title, artist=artist)
        if not args.title.strip() and not args.artist.strip():
            return self.resume_music()
        selected = self._select()
        if isinstance(selected, dict):
            return selected
        candidates, choices = self._candidates(args.title.strip(), args.artist.strip())
        if not candidates:
            return {
                "status": "not_found",
                "title": args.title,
                "artist": args.artist,
                "choices": choices,
            }
        if len(candidates) != 1 and args.artist.strip():
            return {
                "status": "ambiguous",
                "choices": list[JsonValue](candidates[:8]),
                "count": len(candidates),
            }
        if len(candidates) > 1:
            _LOGGER.info(
                "Music title-only selection status=ranked_match candidates=%d",
                len(candidates),
            )
            _LOGGER.debug(
                "Music title-only selected=%s",
                self._redact(json.dumps(candidates[0])),
            )
        player = self._effective(selected.player_id)
        value = self._request(
            "player_queues/get_active_queue", player_id=player.player_id
        )
        if value is None:
            value = self._request("player_queues/get", queue_id=player.player_id)
        if value is None:
            raise MusicAssistantError("No existing queue is available for new music")
        queue = _parse(_Queue, value)
        if not queue.available:
            raise MusicAssistantError("Music Assistant queue is unavailable")
        seed = candidates[0]
        media = f"radio_playlist://playlist/{_text(seed['uri'])}"
        self._request(
            "player_queues/play_media",
            queue_id=queue.queue_id,
            media=media,
            option="replace",
        )
        observed = _parse(
            _Queue, self._request("player_queues/get", queue_id=queue.queue_id)
        )
        if observed.queue_id != queue.queue_id:
            raise MusicAssistantError(
                "Music Assistant returned a different queue after dispatch"
            )
        confirmed = (
            observed.available
            and observed.is_dynamic
            and observed.state == "playing"
            and any(source.uri == media for source in observed.sources)
        )
        playback = self.what_is_playing(_player=self._effective(selected.player_id))
        confirmed = confirmed and _object(playback)["status"] == "playing"
        return {
            "status": "started",
            "player_id": player.player_id,
            "queue_id": queue.queue_id,
            "seed": seed,
            "mode": "endless_mix",
            "seed_first": False,
            "confirmation": "observed" if confirmed else "requested",
            "observed_state": observed.state,
            "is_dynamic": observed.is_dynamic,
            "playback": playback,
        }
