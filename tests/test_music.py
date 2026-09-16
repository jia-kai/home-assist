"""Offline MA 2.10.3 fixtures and command traces; no live playback requests."""

import json
import logging
from copy import deepcopy
from email.message import Message
from http.client import IncompleteRead
from io import BytesIO
from typing import Any, Literal
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest
from pydantic import JsonValue, ValidationError

from hoast import music
from hoast.agent import render_music
from hoast.config import MusicConfig
from hoast.music import (
    MusicArguments,
    MusicAssistantError,
    MusicClient,
    PlayMusicArguments,
    VolumeMusicArguments,
)


def player(
    player_id: str = "speaker",
    state: str = "playing",
    source: str | None = "spotify",
    **overrides: JsonValue,
) -> dict[str, JsonValue]:
    """Build a minimal native-pause-capable player response.

    Args:
        player_id:
            Device or group identifier.

        state:
            Observed native playback state.

        source:
            Retained source identifier, including nullable idle sources.

        **overrides:
            Specific wire fields for malformed-state and routing fixtures.

    """
    return {
        "player_id": player_id,
        "name": player_id.title(),
        "type": "player",
        "available": True,
        "enabled": True,
        "playback_state": state,
        "active_source": source,
        "supported_features": ["pause"],
        "source_list": [{"id": source or "spotify", "can_play_pause": True}],
        **overrides,
    }


def track(
    name: str = "Song",
    artist: str = "Artist",
    uri: str = "spotify://track/1",
    **overrides: JsonValue,
) -> dict[str, JsonValue]:
    """Build a search track with an available provider mapping.

    Args:
        name:
            Search title.

        artist:
            Recording artist name.

        uri:
            Stable media identifier.

        **overrides:
            Extra wire fields or availability overrides.

    """
    return {
        "name": name,
        "uri": uri,
        "media_type": "track",
        "artists": [{"name": artist}],
        "provider_mappings": [{"available": True}],
        **overrides,
    }


def queue(
    queue_id: str = "group-queue", **overrides: JsonValue
) -> dict[str, JsonValue]:
    """Build a minimal queue whose identity is deliberately unlike a player ID.

    Args:
        queue_id:
            Actual queue identifier.

        **overrides:
            State and availability fields.

    """
    return {"queue_id": queue_id, "available": True, "state": "idle", **overrides}


class FixtureClient(MusicClient):
    """Scripted RPC replies with exact command/argument and exhaustion assertions."""

    replies: list[tuple[str, JsonValue]]
    """Expected commands and their response values, consumed in order."""

    requests: list[tuple[str, dict[str, JsonValue]]]
    """Complete ordered RPC trace, including mutation arguments."""

    def __init__(
        self,
        replies: list[tuple[str, JsonValue]],
        player_id: str = "speaker",
    ) -> None:
        """Initialize an offline client using the production configuration surface.

        Args:
            replies:
                Ordered expected command/response pairs.

            player_id:
                Configured player or blank for unique-player selection.

        """
        super().__init__(MusicConfig(player_id=player_id), "offline-token")
        self.replies = deepcopy(replies)
        self.requests = []

    def _request(self, command: str, **args: JsonValue) -> JsonValue:
        """Record every RPC and reject unexpected commands, especially mutations.

        Args:
            command:
                Actual MA RPC name.

            **args:
                Actual RPC arguments.

        """
        self.requests.append((command, args))
        assert self.replies, f"Unexpected RPC: {command}"
        expected, response = self.replies.pop(0)
        assert command == expected
        return response


def result_object(value: JsonValue) -> dict[str, JsonValue]:
    """Narrow a compact result for readable assertions.

    Args:
        value:
            Public client result.

    """
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"action": "up"},
        {"action": "set"},
        *(
            {"action": "set", "level": value}
            for value in [-1, 101, True, 5.0, "5", None]
        ),
        {"action": "louder", "level": -1},
        {"action": "quieter", "level": 5.0},
        {"action": "quieter", "extra": 1},
    ],
)
def test_volume_invalid_arguments(payload: dict[str, Any]) -> None:
    """Reject contradictory requests, coercion, and extra keys before any RPC.

    Args:
        payload:
            Invalid tool argument object.

    """
    with pytest.raises(ValidationError):
        VolumeMusicArguments.model_validate(payload)
    client = FixtureClient([])
    with pytest.raises(ValidationError):
        client.tools()[3].arguments.model_validate(payload)
    assert client.requests == []


@pytest.mark.parametrize(
    "action,current,amount,target",
    [
        ("louder", 0, None, 5),
        ("louder", 98, None, 100),
        ("louder", 100, None, 100),
        ("quieter", 0, None, 0),
        ("quieter", 1, None, 0),
        ("quieter", 3, None, 0),
        ("quieter", 50, None, 45),
        ("louder", 40, 20, 60),
        ("quieter", 40, 20, 20),
        ("louder", 40, 100, 100),
        ("quieter", 40, 100, 0),
        ("quieter", 40, 0, 40),
        ("set", None, 0, 0),
        ("set", None, 1, 1),
        ("set", 20, 100, 100),
        ("set", 100, 100, 100),
    ],
)
def test_volume_exact_trace(
    action: Literal["louder", "quieter", "set"],
    current: int | None,
    amount: int | None,
    target: int,
) -> None:
    """Bound percentage-point changes, preserve defaults, and permit an absolute zero.

    Args:
        action:
            Validated volume operation.

        current:
            Initial logical percentage or missing reading.

        amount:
            Explicit percentage value, or None for the five-point relative default.

        target:
            Expected bounded target percentage.

    """
    before = player(
        state="idle",
        source=None,
        volume_level=current,
        supported_features=["volume_set"],
    )
    replies: list[tuple[str, JsonValue]] = [
        ("players/all", [before]),
        ("players/get", before),
        ("players/get", before),
    ]
    expected: list[tuple[str, dict[str, JsonValue]]] = [
        (
            "players/all",
            {
                "return_unavailable": True,
                "return_disabled": True,
                "return_protocol_players": False,
            },
        ),
        ("players/get", {"player_id": "speaker"}),
        ("players/get", {"player_id": "speaker"}),
    ]
    if target != current:
        replies.extend(
            [
                ("players/cmd/volume_set", None),
                ("players/get", {**before, "volume_level": target}),
            ]
        )
        expected.extend(
            [
                (
                    "players/cmd/volume_set",
                    {"player_id": "speaker", "volume_level": target},
                ),
                ("players/get", {"player_id": "speaker"}),
            ]
        )
    client = FixtureClient(replies)
    result = result_object(client.volume_music(action, amount))
    assert result["status"] == (
        "volume_unchanged" if target == current else "volume_set"
    )
    assert result["level"] == target
    assert result["confirmation"] == "observed"
    assert client.requests == expected
    assert not client.replies


@pytest.mark.parametrize("group_type", ["group", "player"])
@pytest.mark.parametrize("phase", ["stable", "lag", "before", "after"])
def test_volume_group_routing(group_type: str, phase: str) -> None:
    """Use group readings and RPCs for groups and sync leaders, guarding membership.

    Args:
        group_type:
            Dedicated group or native sync leader wire type.

        phase:
            Stable state, lagging readback, or membership drift phase.

    """
    member = player(synced_to="leader", volume_level=90)
    leader = player(
        "leader",
        type=group_type,
        group_members=["speaker", "leader"],
        group_volume=30,
        volume_level=90,
        supported_features=["volume_set"],
    )
    drift = {**leader, "group_members": ["leader"]}
    replies: list[tuple[str, JsonValue]] = [
        ("players/all", [member, leader]),
        ("players/get", member),
        ("players/get", leader),
        ("players/get", member),
        ("players/get", drift if phase == "before" else leader),
    ]
    if phase != "before":
        observed = (
            drift
            if phase == "after"
            else {**leader, "group_volume": 30 if phase == "lag" else 35}
        )
        replies.extend(
            [
                ("players/cmd/group_volume", None),
                ("players/get", member),
                ("players/get", observed),
            ]
        )
    client = FixtureClient(replies)
    result = result_object(client.volume_music("louder"))
    assert result["level"] == 35
    assert result["player_id"] == "leader"
    assert result["status"] == (
        "cannot_volume" if phase in {"before", "after"} else "volume_set"
    )
    if phase != "before":
        assert result["confirmation"] == (
            "observed" if phase == "stable" else "requested"
        )
    assert [request for request in client.requests if "/cmd/" in request[0]] == (
        []
        if phase == "before"
        else [("players/cmd/group_volume", {"player_id": "leader", "volume_level": 35})]
    )
    assert not client.replies


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"volume_level": None},
        {"type": "group", "volume_level": 90},
        {"group_members": ["speaker"], "volume_level": 90},
        {"volume_level": 20, "supported_features": []},
    ],
)
def test_volume_refusal_logs(
    overrides: dict[str, JsonValue], caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse unknown readings and missing controls with actionable logged reasons.

    Args:
        overrides:
            Missing data or capability response fields.

        caplog:
            Captured public tool diagnostics.

    """
    before = player(**{"supported_features": ["volume_set"], **overrides})
    client = FixtureClient([("players/all", [before]), ("players/get", before)])
    with caplog.at_level(logging.DEBUG, logger="hoast.music"):
        result = result_object(client.volume_music("quieter"))
    assert result["status"] == "cannot_volume"
    assert result["reason"]
    assert "volume_music" in caplog.text and '"action": "quieter"' in caplog.text
    assert any(
        record.levelno == logging.WARNING and "reason=" in record.message
        for record in caplog.records
    )
    assert "result=" in caplog.text
    assert not client.replies


def test_volume_snapshot_and_selection() -> None:
    """Refuse concurrent volume edits and ambiguous player selection without writes."""
    before = player(volume_level=20, supported_features=["volume_set"])
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            ("players/get", {**before, "volume_level": 21}),
        ]
    )
    assert result_object(client.volume_music("set", 50))["status"] == "cannot_volume"
    assert not client.replies
    client = FixtureClient([("players/all", [before, player("other")])], player_id="")
    assert result_object(client.volume_music("louder"))["status"] == "player_required"
    assert not client.replies


@pytest.mark.parametrize("phase", ["before", "after"])
def test_volume_effective_target_drift(phase: str) -> None:
    """Refuse redirected effective targets before dispatch and qualify post-write drift.

    Args:
        phase:
            Snapshot at which a new effective group appears.

    """
    before = player(volume_level=20, supported_features=["volume_set"])
    member = {**before, "active_group": "group"}
    group = player(
        "group", type="group", group_volume=25, supported_features=["volume_set"]
    )
    replies: list[tuple[str, JsonValue]] = [
        ("players/all", [before]),
        ("players/get", before),
    ]
    if phase == "after":
        replies.extend([("players/get", before), ("players/cmd/volume_set", None)])
    replies.extend([("players/get", member), ("players/get", group)])
    client = FixtureClient(replies)
    result = result_object(client.volume_music("louder"))
    assert result["status"] == "cannot_volume"
    assert result["level"] == 25
    assert result.get("confirmation") == ("requested" if phase == "after" else None)
    assert not client.replies


def test_volume_compact_group_status() -> None:
    """Expose the effective group reading in status and each listed target's reading."""
    member = player(active_group="group", volume_level=90)
    group = player("group", type="group", group_volume=30, volume_level=90)
    client = FixtureClient([("players/all", [member, group])])
    listing = client.players()
    assert isinstance(listing, list)
    assert result_object(listing[0])["volume_level"] == 90
    assert result_object(listing[1])["volume_level"] == 30
    client = FixtureClient(
        [
            ("players/all", [member, group]),
            ("players/get", member),
            ("players/get", group),
            ("player_queues/get_active_queue", None),
        ]
    )
    assert result_object(client.status())["volume_level"] == 30
    assert not client.replies


def test_volume_direct_validation_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Direct calls reject coerced levels before RPC and log the validation traceback.

    Args:
        caplog:
            Captured diagnostic records.

    """
    client = FixtureClient([])
    with (
        caplog.at_level(logging.DEBUG, logger="hoast.music"),
        pytest.raises(ValidationError),
    ):
        client.volume_music("set", True)
    assert not client.requests
    assert "volume_music status=error" in caplog.text
    assert "ValidationError" in caplog.text


def test_tools_strict_and_blank_queries() -> None:
    """Expose optional play fields while rejecting unknown keys and coerced types."""
    client = FixtureClient([])
    tools = client.tools()
    assert [tool.name for tool in tools] == [
        "pause_music",
        "resume_music",
        "play_music",
        "volume_music",
        "music_next",
        "what_is_playing",
    ]
    for tool in tools:
        assert tool.schema()["function"]["parameters"]["additionalProperties"] is False
    assert tools[0].arguments is MusicArguments
    assert tools[2].arguments is PlayMusicArguments
    assert tools[3].arguments is VolumeMusicArguments
    for args in (
        {"title": 3},
        {"artist": None},
        {"title": "Song", "player_id": "other"},
    ):
        with pytest.raises(ValidationError):
            PlayMusicArguments.model_validate(args)
    with pytest.raises(ValidationError):
        MusicArguments.model_validate({"title": "Song"})
    invalid_title: Any = 3
    with pytest.raises(ValidationError):
        client.play_music(title=invalid_title)
    assert PlayMusicArguments() == PlayMusicArguments(title="", artist="")
    assert PlayMusicArguments(title=" \t", artist="\n").title == " \t"
    assert "resumes existing playback" in tools[2].description
    assert not client.requests


@pytest.mark.parametrize(
    "state,expected",
    [
        ("paused", "resumed"),
        ("playing", "already_playing"),
        ("idle", "resumed"),
    ],
)
@pytest.mark.parametrize("via_tool", [False, True])
@pytest.mark.parametrize("blank", ["", " \t\n"])
def test_blank_play_only_resumes_existing_stream(
    state: str,
    expected: str,
    via_tool: bool,
    blank: str,
) -> None:
    """Default and whitespace play delegate paused and idle playback to MA's Play command.

    Args:
        state:
            Retained native playback state before the request.

        expected:
            Observed resume or already-playing outcome.

        via_tool:
            Whether to invoke the registered tool instead of the direct CLI method.

        blank:
            Empty or whitespace-only query fields.

    """
    before = player(state=state)
    replies: list[tuple[str, JsonValue]] = [
        ("players/all", [before]),
        ("players/get", before),
    ]
    if state != "playing":
        replies.extend(
            [
                ("players/cmd/play", None),
                ("players/get", player()),
            ]
        )
    client = FixtureClient(replies)
    if via_tool:
        tool = client.tools()[2]
        args = {} if not blank else {"title": blank, "artist": blank}
        result = tool.handler(tool.arguments.model_validate(args))
    else:
        result = client.play_music(blank, blank) if blank else client.play_music()
    assert result_object(result)["status"] == expected
    assert {command for command, _ in client.requests} <= {
        "players/all",
        "players/get",
        "players/cmd/play",
    }
    assert not client.replies


@pytest.mark.parametrize("via_blank_play", [False, True])
@pytest.mark.parametrize("after_state", ["idle", "playing"])
def test_idle_external_source_resumes_native_session(
    via_blank_play: bool, after_state: str
) -> None:
    """Delegate an idle external source to MA without client-side queue/source selection.

    Args:
        via_blank_play:
            Whether the request uses blank play's existing-session delegation.

        after_state:
            Immediate server observation, which may lag behind the accepted command.

    """
    source = "spotify_connect://audio_source/speaker"
    before = player(state="idle", source=source)
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            ("players/cmd/play", None),
            ("players/get", player(state=after_state, source=source)),
        ]
    )
    result = result_object(
        client.play_music() if via_blank_play else client.resume_music()
    )
    assert result["status"] == "resumed"
    assert result["source"] == source
    assert result["confirmation"] == (
        "observed" if after_state == "playing" else "requested"
    )
    assert client.requests[2] == ("players/cmd/play", {"player_id": "speaker"})
    assert not client.replies


def test_idle_ma_queue_is_resumed_by_server() -> None:
    """An inactive retained MA queue is sent the same player Play command as the UI."""
    before = player(
        state="idle", source="speaker", supported_features=[], source_list=[]
    )
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            ("players/cmd/play", None),
            ("players/get", player(source="speaker")),
        ]
    )
    result = result_object(client.resume_music())
    assert result["status"] == "resumed"
    assert result["confirmation"] == "observed"
    assert client.requests[2] == ("players/cmd/play", {"player_id": "speaker"})
    assert not client.replies


@pytest.mark.parametrize("source", [None, "", "spotify_connect://audio_source/speaker"])
def test_idle_resume_delegates_missing_source_metadata(source: str | None) -> None:
    """MA can restore source/queue state even when the player's source metadata is absent.

    Args:
        source:
            Missing source or retained external source without advertised capabilities.

    """
    before = player(state="idle", source=source, source_list=[])
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            ("players/cmd/play", None),
            ("players/get", player(source="speaker")),
        ]
    )
    result = result_object(client.resume_music())
    assert result["status"] == "resumed" and result["confirmation"] == "observed"
    assert result["source"] == "speaker"
    assert not client.replies


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("source", ["spotify", "speaker"])
def test_player_controls_delegate_stream_handling(resume: bool, source: str) -> None:
    """Both external and MA sources use only one native control and read-only RPCs.

    Args:
        resume:
            Whether the requested operation is native unpause.

        source:
            External source or MA-owned source identifier.

    """
    before = player(state="paused" if resume else "playing", source=source)
    after = player(state="playing" if resume else "paused", source=source)
    command = "players/cmd/play" if resume else "players/cmd/pause"
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            (command, None),
            ("players/get", after),
        ]
    )
    result = result_object(client.resume_music() if resume else client.pause_music())
    assert result["status"] == ("resumed" if resume else "paused")
    assert result["confirmation"] == "observed"
    assert result["source"] == source
    assert client.requests[2] == (command, {"player_id": "speaker"})
    assert {name for name, _ in client.requests} == {
        "players/all",
        "players/get",
        command,
    }
    assert not client.replies


@pytest.mark.parametrize(
    "resume,state,expected",
    [
        (True, "playing", "already_playing"),
        (False, "paused", "already_paused"),
        (False, "idle", "already_stopped"),
    ],
)
def test_native_noops(resume: bool, state: str, expected: str) -> None:
    """Already-satisfied play/pause requests do not toggle playback in the wrong direction.

    Args:
        resume:
            Requested operation.

        state:
            Current player state.

        expected:
            Compact no-op or unsupported status.

    """
    before = player(state=state)
    client = FixtureClient([("players/all", [before]), ("players/get", before)])
    result = result_object(client.resume_music() if resume else client.pause_music())
    assert result["status"] == expected
    assert not client.replies


@pytest.mark.parametrize(
    "override",
    [
        {"active_source": None},
        {"active_source": ""},
        {"supported_features": []},
        {"source_list": []},
        {"source_list": [{"id": "spotify", "can_play_pause": False}]},
        {"source_list": [{"id": "different", "can_play_pause": True}]},
    ],
)
@pytest.mark.parametrize("resume", [False, True])
def test_capability_handling_is_delegated_to_ma(
    override: dict[str, JsonValue], resume: bool
) -> None:
    """Source/protocol capability metadata does not block MA's supported player routing.

    Args:
        override:
            Missing source or unsupported player/source capability.

        resume:
            Whether to try native resume instead of pause.

    """
    before = {**player(state="paused" if resume else "playing"), **override}
    command = "players/cmd/play" if resume else "players/cmd/pause"
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            (command, None),
            ("players/get", player(state="playing" if resume else "paused")),
        ]
    )
    result = result_object(client.resume_music() if resume else client.pause_music())
    assert result["status"] == ("resumed" if resume else "paused")
    assert result["confirmation"] == "observed"
    assert not client.replies


def test_source_restoration_is_a_valid_play_outcome() -> None:
    """MA may restore a queue/source and output protocol while starting playback."""
    before = player(state="idle", source=None)
    changed = player(
        state="playing", source="speaker", active_output_protocol="airplay"
    )
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            ("players/cmd/play", None),
            ("players/get", changed),
        ]
    )
    result = result_object(client.resume_music())
    assert result["status"] == "resumed" and result["confirmation"] == "observed"
    assert not client.replies


def test_idle_readback_is_requested_not_failure() -> None:
    """An acknowledged Play with delayed state change remains requested, without retry."""
    before = player(state="paused")
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            ("players/cmd/play", None),
            ("players/get", player(state="idle")),
        ]
    )
    result = result_object(client.resume_music())
    assert result["status"] == "resumed" and result["confirmation"] == "requested"
    assert not client.replies


def test_acknowledgement_does_not_claim_observed_pause() -> None:
    """A null command response followed by unchanged state is only a request acknowledgement."""
    before = player()
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            ("players/cmd/pause", None),
            ("players/get", before),
        ]
    )
    result = result_object(client.pause_music())
    assert result["status"] == "paused"
    assert result["confirmation"] == "requested"
    assert result["observed_state"] == "playing"


def test_pause_fallback_to_stop_is_reported_truthfully() -> None:
    """MA's pause command may stop an output without pause support."""
    before = player(
        state="playing", active_output_protocol="airplay-output", supported_features=[]
    )
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            ("players/cmd/pause", None),
            ("players/get", player(state="idle")),
        ]
    )
    result = result_object(client.pause_music())
    assert result["status"] == "stopped" and result["confirmation"] == "observed"
    assert render_music(result) == "Music stopped."
    assert not client.replies


def test_group_leader_native_control() -> None:
    """Observe the effective group but pass the selected player ID to MA for routing."""
    child = player(synced_to="leader")
    leader = player("leader", active_group="group")
    group = player("group", type="group")
    snapshot: list[tuple[str, JsonValue]] = [
        ("players/get", child),
        ("players/get", leader),
        ("players/get", group),
    ]
    client = FixtureClient(
        [
            ("players/all", [child]),
            *snapshot,
            ("players/cmd/pause", None),
            *snapshot,
        ]
    )
    assert result_object(client.pause_music())["player_id"] == "group"
    assert ("players/cmd/pause", {"player_id": "speaker"}) in client.requests
    assert not client.replies


def test_group_cycle_and_unavailable_leader() -> None:
    """Reject inconsistent group routing and inaccessible effective players."""
    child = player(synced_to="leader")
    for leader in (
        player("leader", synced_to="speaker"),
        player("leader", available=False),
    ):
        client = FixtureClient(
            [("players/all", [child]), ("players/get", child), ("players/get", leader)]
        )
        with pytest.raises(MusicAssistantError):
            client.pause_music()
        assert not client.replies


@pytest.mark.parametrize(
    "old,new,expected",
    [
        (
            {"queue_item_id": "one", "uri": "same"},
            {"queue_item_id": "two", "uri": "same"},
            "observed",
        ),
        ({"uri": "track:one"}, {"uri": "track:two"}, "observed"),
        ({"uri": "track:one"}, {"uri": "track:one"}, "requested"),
        ({}, {"uri": "track:two"}, "requested"),
    ],
)
def test_music_next_observes_identity(
    old: dict[str, JsonValue], new: dict[str, JsonValue], expected: str
) -> None:
    """Next sends one player command and confirms only comparable changed media identity.

    Args:
        old:
            Current media identity before the command.

        new:
            Current media identity in the immediate readback.

        expected:
            Whether advancement is observed or only requested.

    """
    before = player(current_media={**old, "title": "Before", "artist": "Artist"})
    after = player(current_media={**new, "title": "Next Song", "artist": "Artist"})
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            ("players/cmd/next", None),
            ("players/get", after),
        ]
    )
    result = result_object(client.music_next())
    assert result["status"] == "skipped" and result["confirmation"] == expected
    assert render_music(result) == (
        "Now playing Next Song by Artist."
        if expected == "observed"
        else "Next song requested."
    )
    assert client.requests[2] == ("players/cmd/next", {"player_id": "speaker"})
    assert not client.replies


def test_music_next_uses_selected_member_for_server_routing() -> None:
    """The player-level Next endpoint receives the selected member, not a queue ID."""
    child = player(synced_to="leader")
    leader = player("leader", current_media={"uri": "track:one"})
    after = player("leader", current_media={"uri": "track:two"})
    client = FixtureClient(
        [
            ("players/all", [child]),
            ("players/get", child),
            ("players/get", leader),
            ("players/cmd/next", None),
            ("players/get", child),
            ("players/get", after),
        ]
    )
    assert result_object(client.music_next())["confirmation"] == "observed"
    assert client.requests[3] == ("players/cmd/next", {"player_id": "speaker"})
    assert not client.replies


@pytest.mark.parametrize(
    "state,metadata,expected",
    [
        (
            "playing",
            {"title": "Actual Song", "artist": "Actual Artist"},
            "Now playing Actual Song by Actual Artist.",
        ),
        (
            "playing",
            {"title": "Actual Song"},
            "Music is playing, but track details are unavailable.",
        ),
        ("paused", {"title": "Cached Song", "artist": "Artist"}, "Music is paused."),
        (
            "idle",
            {"title": "Cached Song", "artist": "Artist"},
            "Nothing is playing right now.",
        ),
    ],
)
def test_what_is_playing_is_read_only(
    state: str, metadata: dict[str, JsonValue], expected: str
) -> None:
    """The query renders observed playback, without presenting cached paused/idle media as playing.

    Args:
        state:
            Current effective playback state.

        metadata:
            Current or cached media labels supplied by MA.

        expected:
            Brief spoken query result.

    """
    current = player(state=state, current_media=metadata)
    client = FixtureClient([("players/all", [current]), ("players/get", current)])
    tool = next(tool for tool in client.tools() if tool.name == "what_is_playing")
    result = tool.handler(MusicArguments())
    assert render_music(result) == expected
    assert not client.replies
    assert [command for command, _ in client.requests] == ["players/all", "players/get"]
    with pytest.raises(ValidationError):
        tool.arguments.model_validate({"_player": {"fake": True}})


@pytest.mark.parametrize("new_request", [False, True])
def test_start_implicitly_uses_query_result(new_request: bool) -> None:
    """Play/resume invoke the shared query and its returned text metadata is authoritative.

    Args:
        new_request:
            Whether to start a new explicit mix instead of restoring existing playback.

    """
    before = player(state="paused")
    after = player(current_media={"title": "Snapshot title", "artist": "Artist"})
    replies: list[tuple[str, JsonValue]] = [("players/all", [before])]
    if new_request:
        replies.extend(
            [
                ("music/search", {"tracks": [track()]}),
                ("players/get", before),
                ("player_queues/get_active_queue", queue()),
                ("player_queues/play_media", None),
                (
                    "player_queues/get",
                    queue(
                        state="playing",
                        is_dynamic=True,
                        sources=[
                            {"uri": "radio_playlist://playlist/spotify://track/1"}
                        ],
                    ),
                ),
            ]
        )
    else:
        replies.extend([("players/get", before), ("players/cmd/play", None)])
    replies.append(("players/get", after))
    client = FixtureClient(replies)
    query_result: JsonValue = {
        "status": "playing",
        "now_playing": {"title": "Query result title", "artist": "Query artist"},
    }
    with patch.object(client, "what_is_playing", return_value=query_result) as query:
        result = (
            client.play_music("Song", "Artist")
            if new_request
            else client.resume_music()
        )
    query.assert_called_once()
    assert query.call_args.kwargs["_player"].current_media.title == "Snapshot title"
    assert render_music(result) == "Now playing Query result title by Query artist."
    assert not client.replies


@pytest.mark.parametrize(
    "command,error",
    [
        ("play", "QueueEmpty"),
        ("pause", "Source cannot pause"),
        ("next", "Source cannot next"),
    ],
)
def test_player_command_failures_never_retry_or_change_route(
    command: str, error: str
) -> None:
    """MA failures propagate without client-side queue or source fallbacks.

    Args:
        command:
            Player endpoint operation to reject.

        error:
            Representative server-side queue/capability refusal.

    """
    before = player(state="idle" if command == "play" else "playing")
    client = MusicClient(MusicConfig(player_id="speaker"), "offline-token")
    action = {
        "play": client.resume_music,
        "pause": client.pause_music,
        "next": client.music_next,
    }[command]
    with (
        patch.object(
            client,
            "_request",
            side_effect=[[before], before, MusicAssistantError(error)],
        ) as request,
        pytest.raises(MusicAssistantError, match=error),
    ):
        action()
    assert [call.args[0] for call in request.call_args_list] == [
        "players/all",
        "players/get",
        f"players/cmd/{command}",
    ]


def test_player_selection_never_chooses_arbitrarily() -> None:
    """An empty configuration requires exactly one enabled available nonprotocol device."""
    for choices in ([], [player(), player("second")]):
        client = FixtureClient(
            [("players/all", list[JsonValue](choices))], player_id=""
        )
        assert result_object(client.pause_music())["status"] == "player_required"
        assert not client.replies
    selected = player(state="paused")
    client = FixtureClient(
        [
            (
                "players/all",
                [
                    selected,
                    player("offline", available=False),
                    player("disabled", enabled=False),
                    player("protocol", type="protocol"),
                ],
            ),
            ("players/get", selected),
        ],
        player_id="",
    )
    assert result_object(client.pause_music())["status"] == "already_paused"
    client = FixtureClient([("players/all", [player("other")])])
    with pytest.raises(MusicAssistantError, match="Configured music player"):
        client.resume_music()


def test_read_only_players_and_nullable_status_queue() -> None:
    """CLI reads compact state and accepts the null active queue of an external source."""
    current = player()
    client = FixtureClient(
        [("players/all", [current, player("protocol", type="protocol")])]
    )
    results = client.players()
    assert isinstance(results, list) and len(results) == 1
    assert "source_list" not in result_object(results[0])
    client = FixtureClient(
        [
            ("players/all", [current]),
            ("players/get", current),
            ("player_queues/get_active_queue", None),
        ]
    )
    result = result_object(client.status())
    assert result["queue"] is None
    assert result["source"] == "spotify"
    assert not client.replies


@pytest.mark.parametrize(
    "results,expected",
    [
        ([], "not_found"),
        ([track(name="Different")], "not_found"),
        ([track(artist="Other")], "not_found"),
        ([track(available=False)], "not_found"),
        ([track(provider_mappings=[{"available": False}])], "not_found"),
    ],
)
def test_search_constraints_and_availability(
    results: list[JsonValue], expected: str
) -> None:
    """Mismatches and unavailable search results never replace existing playback.

    Args:
        results:
            Bounded provider search results.

        expected:
            Not-found or clarification result.

    """
    client = FixtureClient(
        [("players/all", [player()]), ("music/search", {"tracks": results})]
    )
    assert result_object(client.play_music("Song", "Artist"))["status"] == expected
    assert client.requests[1] == (
        "music/search",
        {"search_query": "Artist - Song", "media_types": ["track"], "limit": 25},
    )
    assert not client.replies


@pytest.mark.parametrize("first_available", [True, False])
@pytest.mark.parametrize("artist", ["", " \t"])
def test_title_only_uses_first_available_exact_match(
    first_available: bool, artist: str
) -> None:
    """Use provider-ranked exact Chinese titles rather than asking about every cover.

    Args:
        first_available:
            Whether the top-ranked exact-title match can be played.

        artist:
            Omitted artist or whitespace-only equivalent; neither filters artists.

    """
    client = FixtureClient(
        [
            ("players/all", [player()]),
            (
                "music/search",
                {
                    "tracks": [
                        track(name="黑色猫衣", uri="spotify://track/different-title"),
                        track(
                            name="黑色毛衣",
                            artist="Jay Chou",
                            available=first_available,
                        ),
                        track(
                            name="黑色毛衣", artist="Choiyl", uri="spotify://track/2"
                        ),
                    ],
                },
            ),
            ("players/get", player()),
            ("player_queues/get_active_queue", queue()),
            ("player_queues/play_media", None),
            ("player_queues/get", queue()),
            ("players/get", player()),
        ]
    )
    result = result_object(client.play_music("黑色毛衣", artist))
    assert result["status"] == "started"
    seed = result_object(result["seed"])
    assert seed["artists"] == (["Jay Chou"] if first_available else ["Choiyl"])
    assert seed["title"] == "黑色毛衣"
    assert client.requests[4] == (
        "player_queues/play_media",
        {
            "queue_id": "group-queue",
            "media": "radio_playlist://playlist/spotify://track/1"
            if first_available
            else "radio_playlist://playlist/spotify://track/2",
            "option": "replace",
        },
    )
    assert not client.replies


def test_versions_use_first_available_ranked_match() -> None:
    """Six versions of the same title and full artist set need no impossible clarification."""
    versions: list[JsonValue] = [
        track(
            name="Never Gonna Give You Up",
            artist="Rick Astley",
            uri=f"spotify://track/{index}",
            version=version,
            available=index != 0,
        )
        for index, version in enumerate(
            ["Remastered", "Live", "", "Single", "Mix", "Radio"]
        )
    ]
    client = FixtureClient(
        [
            ("players/all", [player()]),
            ("music/search", {"tracks": versions}),
            ("players/get", player()),
            ("player_queues/get_active_queue", queue()),
            ("player_queues/play_media", None),
            ("player_queues/get", queue()),
            ("players/get", player()),
        ]
    )
    result = result_object(client.play_music("Never Gonna Give You Up", "Rick Astley"))
    assert result["status"] == "started"
    assert result_object(result["seed"])["version"] == "Live"
    assert (
        client.requests[4][1]["media"] == "radio_playlist://playlist/spotify://track/1"
    )
    assert not client.replies


def test_artist_constraint_preserves_full_artist_set_ambiguity() -> None:
    """A shared requested artist does not collapse recordings with different collaborators."""
    client = FixtureClient(
        [
            ("players/all", [player()]),
            (
                "music/search",
                {
                    "tracks": [
                        track(version="Live"),
                        track(
                            uri="spotify://track/duet",
                            artists=[
                                {"name": "Artist"},
                                {"name": "Collaborator"},
                            ],
                        ),
                    ]
                },
            ),
        ]
    )
    assert result_object(client.play_music("Song", "Artist"))["status"] == "ambiguous"
    assert not client.replies


def test_mismatch_returns_unselected_choices() -> None:
    """Near matches remain suggestions and never authorize replacing playback."""
    client = FixtureClient(
        [
            ("players/all", [player()]),
            ("music/search", {"tracks": [track(name="Different song")]}),
        ]
    )
    result = result_object(client.play_music("Song", "Artist"))
    assert result["status"] == "not_found"
    assert result["choices"] == [
        {
            "title": "Different song",
            "artists": ["Artist"],
            "uri": "spotify://track/1",
        }
    ]
    assert not client.replies


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("observed", [False, True])
def test_explicit_dynamic_mix_and_provider_duplicates(
    external: bool, observed: bool
) -> None:
    """Exact normalized duplicates use first available ranking and the actual active queue.

    Args:
        external:
            Whether active queue is null and the player's own existing queue is needed.

        observed:
            Whether the post-command state confirms dynamic playback.

    """
    selected = player()
    active = queue("speaker" if external else "group-queue")
    replies: list[tuple[str, JsonValue]] = [
        ("players/all", [selected]),
        (
            "music/search",
            {
                "tracks": [
                    track(available=False),
                    track(),
                    track(name="ＳＯＮＧ", artist=" artist ", uri="other://track/2"),
                ]
            },
        ),
        ("players/get", selected),
        ("player_queues/get_active_queue", None if external else active),
    ]
    if external:
        replies.append(("player_queues/get", active))
    replies.extend(
        [
            ("player_queues/play_media", None),
            (
                "player_queues/get",
                {
                    **active,
                    "is_dynamic": observed,
                    "state": "playing" if observed else "idle",
                    "sources": [{"uri": "radio_playlist://playlist/spotify://track/1"}],
                },
            ),
            ("players/get", player(state="playing" if observed else "idle")),
        ]
    )
    client = FixtureClient(replies)
    result = result_object(client.play_music(" song ", " ARTIST "))
    assert result["status"] == "started"
    assert result["confirmation"] == ("observed" if observed else "requested")
    assert result["seed_first"] is False
    assert (
        "player_queues/play_media",
        {
            "queue_id": active["queue_id"],
            "option": "replace",
            "media": "radio_playlist://playlist/spotify://track/1",
        },
    ) in client.requests
    assert not any(
        "autoplay" in command or "resume" in command for command, _ in client.requests
    )
    assert not client.replies


@pytest.mark.parametrize(
    "active,state", [(True, "playing"), (False, "playing"), (True, "idle")]
)
def test_now_playing_uses_current_track_not_seed(active: bool, state: str) -> None:
    """Announce structured current metadata only for an active observed playing mix.

    Args:
        active:
            Whether the queue is active on its player.

        state:
            Post-dispatch playback observation.

    """
    selected = player()
    observed = queue(
        state=state,
        active=active,
        is_dynamic=True,
        sources=[{"uri": "radio_playlist://playlist/spotify://track/1"}],
        current_item={
            "media_item": track(
                name="Actual Track",
                artist="First Artist",
                artists=[{"name": "First Artist"}, {"name": "Second Artist"}],
            )
        },
    )
    client = FixtureClient(
        [
            ("players/all", [selected]),
            ("music/search", {"tracks": [track()]}),
            ("players/get", selected),
            ("player_queues/get_active_queue", queue()),
            ("player_queues/play_media", None),
            ("player_queues/get", observed),
            (
                "players/get",
                player(
                    state="playing" if active and state == "playing" else "idle",
                    current_media={"title": "Actual Track", "artist": "First Artist"},
                ),
            ),
        ]
    )
    result = result_object(client.play_music("Song", "Artist"))
    if active and state == "playing":
        assert result_object(result["playback"])["now_playing"] == {
            "title": "Actual Track",
            "artist": "First Artist",
        }
        assert render_music(result) == "Now playing Actual Track by First Artist."
    else:
        assert "now_playing" not in result
        assert not render_music(result).startswith("Now playing")
    assert not client.replies


@pytest.mark.parametrize(
    "state,source,title",
    [
        ("playing", "spotify", "After"),
        ("paused", "spotify", "After"),
        ("playing", "other", "After"),
        ("playing", "spotify", ""),
    ],
)
def test_resume_metadata_is_observed_after_dispatch(
    state: str, source: str, title: str
) -> None:
    """Metadata refreshes permit native control but stale/missing labels cannot announce.

    Args:
        state:
            Native player state after the play command.

        source:
            Source identity attached to the observed media labels.

        title:
            Observed media title; blank data must not be invented.

    """
    before = player(
        state="paused", current_media={"title": "Before", "artist": "Artist"}
    )
    after = player(
        state=state,
        current_media={"title": title, "artist": "Artist", "source_id": source},
    )
    client = FixtureClient(
        [
            ("players/all", [before]),
            ("players/get", before),
            ("players/cmd/play", None),
            ("players/get", after),
        ]
    )
    result = result_object(client.resume_music())
    assert result["status"] == "resumed"
    if state == "playing" and source == "spotify" and title:
        assert render_music(result) == "Now playing After by Artist."
    else:
        assert "now_playing" not in result
    assert not client.replies


@pytest.mark.parametrize("state", ["playing", "paused", "idle"])
def test_prompt_context_uses_effective_observation(state: str) -> None:
    """Music context resolves the group but exposes only its binary playback state.

    Args:
        state:
            Native effective-group state mapped to on or off without media metadata.

    """
    member = player(active_group="group")
    group = player(
        player_id="group",
        state=state,
        type="group",
        current_media={"title": "Current", "artist": "Artist"},
        group_volume=35,
    )
    client = FixtureClient(
        [("players/all", [member]), ("players/get", member), ("players/get", group)]
    )
    assert client.prompt_context() == {
        "status": "observed",
        "state": "on" if state == "playing" else "off",
    }
    assert not client.replies


def test_artist_only_dynamic_mix() -> None:
    """Artist-only requests search artists and use their URI as the Endless Mix seed."""
    current = player()
    client = FixtureClient(
        [
            ("players/all", [current]),
            (
                "music/search",
                {
                    "artists": [
                        {
                            "name": "Artist",
                            "uri": "spotify://artist/1",
                            "available": True,
                        },
                    ]
                },
            ),
            ("players/get", current),
            ("player_queues/get_active_queue", queue()),
            ("player_queues/play_media", None),
            ("player_queues/get", queue()),
            ("players/get", current),
        ]
    )
    assert result_object(client.play_music(artist="Artist"))["status"] == "started"
    assert client.requests[1][1]["media_types"] == ["artist"]
    assert (
        client.requests[4][1]["media"] == "radio_playlist://playlist/spotify://artist/1"
    )


def test_prior_dynamic_mix_does_not_confirm_new_seed() -> None:
    """An earlier playing dynamic mix is insufficient confirmation of a new request."""
    client = FixtureClient(
        [
            ("players/all", [player()]),
            ("music/search", {"tracks": [track()]}),
            ("players/get", player()),
            ("player_queues/get_active_queue", queue()),
            ("player_queues/play_media", None),
            (
                "player_queues/get",
                queue(
                    state="playing",
                    is_dynamic=True,
                    sources=[
                        {"uri": "radio_playlist://playlist/spotify://track/earlier"},
                    ],
                ),
            ),
            ("players/get", player()),
        ]
    )
    assert result_object(client.play_music("Song"))["confirmation"] == "requested"


def test_track_mapping_resolves_artist_before_playback() -> None:
    """A title-only mapping must be resolved; a mismatched artist cannot be ignored."""
    client = FixtureClient(
        [
            ("players/all", [player()]),
            (
                "music/search",
                {
                    "tracks": [
                        {"name": "Song", "uri": "spotify://track/1"},
                    ]
                },
            ),
            ("music/item_by_uri", track(artist="Other")),
        ]
    )
    assert result_object(client.play_music("Song", "Artist"))["status"] == "not_found"
    assert client.requests[-1] == ("music/item_by_uri", {"uri": "spotify://track/1"})
    assert not client.replies


@pytest.mark.parametrize(
    "value",
    [None, {}, {"tracks": None}, {"tracks": [None]}, {"tracks": [track(artists=None)]}],
)
def test_malformed_search_fails_loudly(value: JsonValue) -> None:
    """Null objects, arrays, and required metadata do not silently become not-found.

    Args:
        value:
            Invalid external search payload.

    """
    client = FixtureClient([("players/all", [player()]), ("music/search", value)])
    with pytest.raises(MusicAssistantError):
        client.play_music("Song", "Artist")


def test_null_or_unavailable_queue_never_starts_music() -> None:
    """Explicit new play still requires a real available queue."""
    for value in (None, queue(available=False)):
        replies: list[tuple[str, JsonValue]] = [
            ("players/all", [player()]),
            ("music/search", {"tracks": [track()]}),
            ("players/get", player()),
            ("player_queues/get_active_queue", value),
        ]
        if value is None:
            replies.append(("player_queues/get", None))
        client = FixtureClient(replies)
        with pytest.raises(MusicAssistantError, match="queue"):
            client.play_music("Song")
        assert not client.replies


class Response(BytesIO):
    """Bytes-backed HTTP fixture retaining the requested read bound."""

    status: int
    """HTTP status code."""

    read_size: int | None
    """Last requested read bound in bytes."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        """Prepare an HTTP response without opening sockets.

        Args:
            body:
                Raw HTTP payload bytes.

            status:
                HTTP status returned by the fixture.

        """
        super().__init__(body)
        self.status = status
        self.read_size = None

    def read(self, size: int | None = -1) -> bytes:
        """Retain the byte bound and return fixture data.

        Args:
            size:
                Maximum response bytes requested.

        """
        self.read_size = size
        return super().read(size)


def install_transport(
    monkeypatch: pytest.MonkeyPatch,
    response: Response | Exception,
) -> tuple[MusicClient, list[Request]]:
    """Install an offline opener and retain production HTTP request objects.

    Args:
        monkeypatch:
            Scoped pytest attribute replacement.

        response:
            Raw HTTP response or transport exception.

    """
    client = MusicClient(MusicConfig(), "offline-token")
    requests: list[Request] = []

    def open_request(request: Request, *, timeout: int) -> Response:
        """Assert fixed timeout and return or raise the fixture.

        Args:
            request:
                Production urllib request.

            timeout:
                Network timeout in seconds.

        """
        assert timeout == 30
        requests.append(request)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(client._opener, "open", open_request)
    return client, requests


def test_http_raw_json_bearer_and_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP RPC consumes raw JSON, uses /api and bearer auth, and bounds the read.

    Args:
        monkeypatch:
            Scoped transport replacement.

    """
    response = Response(b"null")
    client, requests = install_transport(monkeypatch, response)
    assert client._request("players/cmd/pause", player_id="speaker") is None
    request = requests[0]
    assert request.full_url == "http://localhost:8095/api"
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer offline-token"
    assert isinstance(request.data, bytes)
    assert json.loads(request.data) == {
        "command": "players/cmd/pause",
        "args": {"player_id": "speaker"},
    }
    assert response.read_size == 4 * 1024 * 1024 + 1


@pytest.mark.parametrize(
    "body",
    [b"", b"{", b"<html>oops</html>", b"\xff", b"NaN", b"x" * (4 * 1024 * 1024 + 1)],
)
def test_http_invalid_payload(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    """Malformed and oversized HTTP bodies fail without leaking content.

    Args:
        monkeypatch:
            Scoped transport replacement.

        body:
            Invalid or oversized raw payload.

    """
    client, requests = install_transport(monkeypatch, Response(body))
    with pytest.raises(MusicAssistantError):
        client._request("players/all")
    assert len(requests) == 1


@pytest.mark.parametrize("code", [301, 302, 307, 401, 403, 500])
def test_http_errors_are_redacted_and_not_retried(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """HTTP failure diagnostics contain status, never headers, body, URL credentials, or token.

    Args:
        monkeypatch:
            Scoped transport replacement.

        code:
            Redirect, authorization, or server error status.

    """
    error = HTTPError(
        "http://offline-token/",
        code,
        "offline-token",
        Message(),
        BytesIO(b"offline-token"),
    )
    client, requests = install_transport(monkeypatch, error)
    with pytest.raises(MusicAssistantError, match=str(code)) as caught:
        client._request("players/cmd/pause", player_id="speaker")
    assert "offline-token" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert len(requests) == 1


@pytest.mark.parametrize(
    "error",
    [
        URLError("offline-token"),
        TimeoutError("offline-token"),
        IncompleteRead(b"offline-token"),
    ],
)
def test_transport_errors_do_not_leak_or_retry(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Network and truncated-response failures preserve uncertain mutation status without retries.

    Args:
        monkeypatch:
            Scoped transport replacement.

        error:
            Simulated network or HTTP framing failure.

    """
    client, requests = install_transport(monkeypatch, error)
    with pytest.raises(MusicAssistantError) as caught:
        client._request("players/cmd/play", player_id="speaker")
    assert "offline-token" not in str(caught.value)
    assert len(requests) == 1


def test_redirect_handler_refuses_every_destination() -> None:
    """Production opener installs a handler that cannot forward bearer credentials."""
    client = MusicClient(MusicConfig(), "offline-token")
    handlers = [
        handler
        for handler in vars(client._opener)["handlers"]
        if isinstance(handler, music._NoRedirect)
    ]
    assert len(handlers) == 1
    for code in (301, 302, 303, 307, 308):
        assert (
            handlers[0].redirect_request(
                Request("http://localhost:8095/api"),
                None,
                code,
                "redirect",
                {},
                "http://other/api",
            )
            is None
        )


def test_http_status_is_checked_even_if_opener_returns_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-2xx response cannot masquerade as successful raw JSON.

    Args:
        monkeypatch:
            Scoped transport replacement.

    """
    client, _ = install_transport(monkeypatch, Response(b"null", status=500))
    with pytest.raises(MusicAssistantError, match="500"):
        client._request("players/all")


@pytest.mark.parametrize(
    "payload", [None, {}, {"player_id": "speaker", "available": "yes"}]
)
def test_player_boundary_errors_are_sanitized(payload: Any) -> None:
    """Malformed player responses fail as MusicAssistantError rather than coercing state.

    Args:
        payload:
            Invalid player response.

    """
    client = FixtureClient([("players/all", [payload])])
    with pytest.raises(MusicAssistantError, match="Malformed"):
        client.players()


def test_tool_and_direct_outcomes_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Public and registered calls log names, arguments, and observed playback outcomes.

    Args:
        caplog:
            Captured module logger records, including debug candidate projections.

    """
    caplog.set_level(logging.DEBUG, logger="hoast.music")
    client = FixtureClient(
        [
            ("players/all", [player(state="idle")]),
            ("players/get", player(state="idle")),
            ("players/cmd/play", None),
            ("players/get", player()),
        ]
    )
    tool = client.tools()[2]
    assert result_object(tool.handler(PlayMusicArguments()))["status"] == "resumed"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        'play_music args={"title": "", "artist": ""}' in text for text in messages
    )
    assert any(
        "resume_music status=resumed confirmation=observed" in text for text in messages
    )
    assert any(
        "play_music status=resumed confirmation=observed" in text for text in messages
    )
    assert not any("Idle native resume" in text for text in messages)
    caplog.clear()
    client = FixtureClient(
        [
            ("players/all", [player(state="paused")]),
            ("players/get", player(state="paused")),
        ]
    )
    assert result_object(client.pause_music())["status"] == "already_paused"
    assert "pause_music status=already_paused confirmation=observed" in caplog.text


def test_candidates_and_results_logged_without_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Search clarification details live in debug logs, with any echoed bearer redacted.

    Args:
        caplog:
            Captured logger output.

    """
    caplog.set_level(logging.DEBUG, logger="hoast.music")
    client = FixtureClient(
        [
            ("players/all", [player()]),
            ("music/search", {"tracks": [track(name="offline-token")]}),
        ]
    )
    assert result_object(client.play_music("Song", "Artist"))["status"] == "not_found"
    assert "Music search matches=[] choices=" in caplog.text
    assert "play_music result=" in caplog.text
    assert "title and artist constraints" in caplog.text
    assert "offline-token" not in caplog.text
    assert "[redacted]" in caplog.text


def test_rpc_logs_only_selected_response_fields(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """RPC diagnostics retain command/arguments and compact state without raw private metadata.

    Args:
        monkeypatch:
            Offline transport replacement.

        caplog:
            Captured debug logs.

    """
    caplog.set_level(logging.DEBUG, logger="hoast.music")
    body = json.dumps(
        {
            "player_id": "offline-token",
            "playback_state": "paused",
            "credentials": "unrelated-secret",
            "provider_metadata": {"private": "hidden"},
        }
    ).encode()
    client, _ = install_transport(monkeypatch, Response(body))
    client._request("players/get", player_id="offline-token")
    assert "Music RPC players/get args=" in caplog.text
    assert '"playback_state": "paused"' in caplog.text
    assert "status=ok" in caplog.text
    for secret in ("offline-token", "unrelated-secret", "hidden", "credentials"):
        assert secret not in caplog.text


def test_failure_logs_full_redacted_traceback_and_notes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transport failures preserve sanitized original chains and notes before reaching callers.

    Args:
        monkeypatch:
            Offline transport replacement.

        caplog:
            Captured error diagnostics.

    """
    caplog.set_level(logging.DEBUG, logger="hoast.music")
    cause = ValueError("upstream offline-token")
    cause.add_note("original note offline-token")
    error = URLError("connection offline-token")
    error.__cause__ = cause
    error.add_note("transport note offline-token")
    client, requests = install_transport(monkeypatch, error)
    with pytest.raises(MusicAssistantError):
        client._request("players/cmd/play", player_id="speaker")
    assert len(requests) == 1
    assert "Traceback (most recent call last)" in caplog.text
    assert "URLError" in caplog.text and "ValueError" in caplog.text
    assert "original note [redacted]" in caplog.text
    assert "transport note [redacted]" in caplog.text
    assert "offline-token" not in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_direct_validation_failure_logs_sanitized_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Strict argument errors are logged even when direct CLI calls bypass the registry.

    Args:
        caplog:
            Captured public-operation diagnostic records.

    """
    caplog.set_level(logging.DEBUG, logger="hoast.music")
    client = FixtureClient([])
    invalid_title: Any = ["offline-token"]
    with pytest.raises(ValidationError):
        client.play_music(title=invalid_title)
    assert "play_music status=error" in caplog.text
    assert "ValidationError" in caplog.text
    assert "offline-token" not in caplog.text
    assert not client.requests


def test_redaction_handles_serialized_token_spellings() -> None:
    """JSON and exception repr escaping cannot reveal credentials in diagnostics."""
    token = 'token\\with"escapes'
    client = MusicClient(MusicConfig(), token)
    for text in (token, json.dumps({"value": token}), repr(token)):
        result = client._redact(text)
        assert "[redacted]" in result
        assert "token" not in result
