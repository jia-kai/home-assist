"""Offline CLI contracts using real registered handlers without model inference."""

import json
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import JsonValue, ValidationError

from hoast.config import MusicConfig
from hoast.llm import ToolCall, ToolRegistry
from hoast.music import MusicAssistantError, MusicClient
from hoast.music_cli import main


@pytest.fixture(autouse=True)
def isolated_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep credentials, network access, and diagnostic files isolated.

    Args:
        tmp_path:
            Working directory for CLI configuration and logs.

        monkeypatch:
            Override process environment, working directory, and transport.

    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MUSIC_ASSISTANT_TOKEN", "test-secret")
    monkeypatch.setattr(MusicClient, "_request", unexpected_request)


def unexpected_request(self: MusicClient, command: str, **args: JsonValue) -> JsonValue:
    """Reject accidental real RPCs in CLI tests.

    Args:
        self:
            Client under test.

        command:
            Unexpected RPC name.

        **args:
            Unexpected RPC arguments.

    """
    raise AssertionError(f"Unexpected RPC: {command}")


@pytest.mark.parametrize(
    "command,arguments,expected",
    [
        ("pause", [], {}),
        ("stop", [], {}),
        ("resume", [], {}),
        ("next", [], {}),
        ("now-playing", [], {}),
        ("volume", ["louder"], {"action": "louder"}),
        ("volume", ["QUIETER"], {"action": "quieter"}),
        ("volume", ["35"], {"action": "set", "level": 35}),
        ("volume", ["1"], {"action": "set", "level": 1}),
        ("volume", ["100"], {"action": "set", "level": 100}),
        ("play", [], {"title": "", "artist": ""}),
        (
            "play",
            ["--title", "One", "--artist", "Artist"],
            {"title": "One", "artist": "Artist"},
        ),
        ("play", ["--author", "Artist"], {"title": "", "artist": "Artist"}),
        ("play", ["--title", "One"], {"title": "One", "artist": ""}),
    ],
)
def test_registered_dispatch(
    command: str,
    arguments: list[str],
    expected: dict[str, JsonValue],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dispatch one validated call through the agent's real registration path.

    Args:
        command:
            Mutating CLI command.

        arguments:
            Command-specific options.

        expected:
            Expected tool arguments.

        capsys:
            Capture JSON output separately from diagnostics.

    """
    calls_seen: list[ToolCall] = []
    dispatch = ToolRegistry.dispatch
    tool_name = {"next": "music_next", "now-playing": "what_is_playing"}.get(
        command, f"{'pause' if command == 'stop' else command}_music"
    )

    def spy(registry: ToolRegistry, calls: Sequence[ToolCall]) -> list[JsonValue]:
        """Record calls while retaining actual validation and execution.

        Args:
            registry:
                Real music tool registry.

            calls:
                CLI calls to execute.

        """
        calls_seen.extend(calls)
        return dispatch(registry, calls)

    with (
        patch.object(ToolRegistry, "dispatch", spy),
        patch.object(
            MusicClient, tool_name, return_value={"status": "started"}
        ) as handler,
    ):
        assert main([command, *arguments]) == 0
    assert calls_seen == [ToolCall(tool_name, expected)]
    if command == "play":
        handler.assert_called_once_with(expected["title"], expected["artist"])
    elif command == "volume":
        handler.assert_called_once_with(expected["action"], expected.get("level", 0))
    else:
        handler.assert_called_once_with()
    output = capsys.readouterr()
    assert output.out == '{"status":"started"}\n'
    assert output.err == ""


@pytest.mark.parametrize(
    "command,result",
    [("players", [{"player_id": "speaker"}]), ("status", {"state": "paused"})],
)
def test_read_only(
    command: str, result: JsonValue, capsys: pytest.CaptureFixture[str]
) -> None:
    """Read player discovery/status directly without mutation dispatch.

    Args:
        command:
            Read-only command.

        result:
            Compact client response.

        capsys:
            Output capture.

    """
    with (
        patch.object(MusicClient, command, return_value=result) as read,
        patch.object(ToolRegistry, "dispatch") as dispatch,
    ):
        assert main([command]) == 0
    read.assert_called_once_with()
    dispatch.assert_not_called()
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize(
    "status,code",
    [
        ("cannot_resume", 2),
        ("not_found", 2),
        ("ambiguous", 2),
        ("not_playing", 2),
        ("player_required", 2),
        ("already_playing", 0),
        ("already_paused", 0),
        ("paused", 0),
        ("resumed", 0),
        ("stopped", 0),
        ("already_stopped", 0),
        ("skipped", 0),
        ("volume_set", 0),
        ("volume_unchanged", 0),
        ("cannot_volume", 2),
    ],
)
def test_result_exit_status(
    status: str, code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep actionable refusal results on stdout with a nonzero exit code.

    Args:
        status:
            Client outcome.

        code:
            Expected process exit code.

        capsys:
            Output capture.

    """
    with patch.object(MusicClient, "resume_music", return_value={"status": status}):
        assert main(["resume"]) == code
    assert json.loads(capsys.readouterr().out) == {"status": status}


@pytest.mark.parametrize(
    "use_config,override", [(False, False), (False, True), (True, False), (True, True)]
)
def test_configuration(
    use_config: bool, override: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply optional system TOML and CLI overrides, selecting the custom token.

    Args:
        use_config:
            Whether to supply explicit system TOML.

        override:
            Whether CLI player/server override defaults or TOML.

        tmp_path:
            Isolated configuration directory.

        monkeypatch:
            Isolate custom credential lookup.

    """
    argv: list[str] = []
    expected = MusicConfig()
    token = "test-secret"
    if use_config:
        path = tmp_path / "system.toml"
        path.write_text(
            '[weather]\nlatitude=1\nlongitude=2\n[music]\nserver_url="http://ma:8095"\nplayer_id="configured"\ntoken_env="CUSTOM_MA_TOKEN"\n'
        )
        env = tmp_path / "music.env"
        env.write_text("CUSTOM_MA_TOKEN=custom-secret\n")
        monkeypatch.delenv("CUSTOM_MA_TOKEN", raising=False)
        argv += ["--config", str(path), "--env-file", str(env)]
        expected = MusicConfig("http://ma:8095", "configured", "CUSTOM_MA_TOKEN")
        token = "custom-secret"
    if override:
        argv += ["--player", "override", "--server", "https://override.example"]
        expected = MusicConfig(
            "https://override.example", "override", expected.token_env
        )
    with patch("hoast.music_cli.MusicClient", autospec=True) as client:
        client.return_value.players.return_value = []
        assert main([*argv, "players"]) == 0
    client.assert_called_once_with(expected, token)


def test_weather_only_config(tmp_path: Path) -> None:
    """Allow direct music testing with an existing weather-only system config.

    Args:
        tmp_path:
            Isolated configuration directory.

    """
    path = tmp_path / "config.toml"
    path.write_text("[weather]\nlatitude=1\nlongitude=2\n")
    with patch("hoast.music_cli.MusicClient", autospec=True) as client:
        client.return_value.players.return_value = []
        assert main(["--config", str(path), "players"]) == 0
    client.assert_called_once_with(MusicConfig(), "test-secret")


@pytest.mark.parametrize(
    "arguments",
    [
        ["play"],
        ["play", "--title", " "],
        ["play", "--artist", "\t"],
        ["play", "--title", " \t", "--author", "\n "],
    ],
)
@pytest.mark.parametrize(
    "state,status,code,source",
    [
        ("paused", "resumed", 0, "spotify"),
        ("playing", "already_playing", 0, "spotify"),
        ("idle", "resumed", 0, "spotify"),
        ("idle", "resumed", 0, "spotify_connect://audio_source/speaker"),
    ],
)
def test_blank_play_resumes(
    arguments: list[str],
    state: str,
    status: str,
    code: int,
    source: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Blank play uses MA player Play for paused and idle states without selecting music.

    Args:
        arguments:
            Blank or whitespace-only play invocation.

        state:
            Initial native player state.

        status:
            Expected state-dependent outcome.

        code:
            Expected CLI exit code.

        source:
            Retained source identifier, including an idle Spotify Connect session.

        capsys:
            Output capture.

    """
    player: dict[str, JsonValue] = {
        "player_id": "speaker",
        "name": "Test speaker",
        "type": "player",
        "available": True,
        "enabled": True,
        "playback_state": state,
        "active_source": source,
        "supported_features": ["pause"],
        "source_list": [{"id": source, "can_play_pause": True}],
    }
    outputs: list[str] = []
    for invocation in (["resume"], arguments):
        responses: list[JsonValue] = [[player], player]
        expected_commands = ["players/all", "players/get"]
        if status == "resumed":
            responses.extend([None, {**player, "playback_state": "playing"}])
            expected_commands.extend(["players/cmd/play", "players/get"])
        with patch.object(MusicClient, "_request", side_effect=responses) as request:
            assert main(invocation) == code
        assert [call.args[0] for call in request.call_args_list] == expected_commands
        assert all(
            call.kwargs == {"player_id": "speaker"}
            for call in request.call_args_list[1:]
        )
        output = capsys.readouterr()
        assert json.loads(output.out)["status"] == status
        assert "ValidationError" not in output.err
        outputs.append(output.out)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "arguments",
    [{"title": None}, {"artist": 1}, {"title": []}, {"extra": "unexpected"}],
)
def test_invalid_play_tool_arguments(arguments: dict[str, JsonValue]) -> None:
    """Retain strict type and extra-field checks when dispatching play tools.

    Args:
        arguments:
            Malformed tool arguments; CLI options themselves are strings.

    """
    client = MusicClient(MusicConfig(), "test-secret")
    with (
        patch.object(MusicClient, "play_music") as handler,
        pytest.raises(ValidationError),
    ):
        ToolRegistry(client.tools()).dispatch([ToolCall("play_music", arguments)])
    handler.assert_not_called()


def test_durable_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Retain chained exceptions and notes with credentials redacted, without retry.

    Args:
        tmp_path:
            Isolated diagnostics root.

        capsys:
            Output capture.

    """
    cause = ValueError("underlying test-secret")
    error = MusicAssistantError("request failed test-secret")
    error.__cause__ = cause
    error.add_note("metadata test-secret")
    with patch.object(MusicClient, "resume_music", side_effect=error) as handler:
        assert main(["resume"]) == 1
    handler.assert_called_once_with()
    output = capsys.readouterr()
    assert output.out == ""
    log = (tmp_path / ".cache/hoast/diagnostics/music-cli.log").read_text()
    for text in (output.err, log):
        assert "Traceback" in text
        assert "underlying [REDACTED]" in text
        assert "metadata [REDACTED]" in text
        assert "test-secret" not in text


@pytest.mark.parametrize("validation_failure", [False, True])
def test_escaped_token_failure(
    validation_failure: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Redact accepted escaped credentials in chained and prehandler failures.

    Args:
        validation_failure:
            Exercise real registry validation instead of a synthetic exception.

        tmp_path:
            Isolated durable diagnostics directory.

        monkeypatch:
            Install a synthetic credential without reading production secrets.

        capsys:
            Capture stdout and stderr separately.

    """
    token = "a\\b\"'c"
    spellings = {token, json.dumps(token)[1:-1], repr(token)[1:-1]}
    assert len(spellings) == 3
    monkeypatch.setenv("MUSIC_ASSISTANT_TOKEN", token)
    embedded = " | ".join(sorted(spellings))
    cause = ValueError("cause: " + embedded)
    cause.add_note("cause note: " + embedded)
    error = MusicAssistantError("failure: " + embedded)
    error.__cause__ = cause
    error.add_note("failure note: " + embedded)
    with (
        patch.object(MusicClient, "_request") as request,
        patch.object(MusicClient, "play_music") as play,
        patch.object(MusicClient, "resume_music", side_effect=error) as resume,
    ):
        # A long title fails registry validation and renders its input using repr.
        arguments = (
            ["play", "--title", token + "x" * 301] if validation_failure else ["resume"]
        )
        assert main(arguments) == 1
    request.assert_not_called()
    play.assert_not_called()
    if validation_failure:
        resume.assert_not_called()
    else:
        resume.assert_called_once_with()
    output = capsys.readouterr()
    assert output.out == ""
    log = (tmp_path / ".cache/hoast/diagnostics/music-cli.log").read_text()
    for text in (output.err, log):
        assert "Traceback" in text
        assert "[REDACTED]" in text
        assert all(spelling not in text for spelling in spellings)
        if validation_failure:
            assert "ValidationError" in text
            assert "input_value=" in text
        else:
            assert "cause note:" in text
            assert "failure note:" in text
            assert "direct cause" in text


def test_missing_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail before constructing a client when no credential is configured.

    Args:
        monkeypatch:
            Remove the test credential.

        capsys:
            Output capture.

    """
    monkeypatch.delenv("MUSIC_ASSISTANT_TOKEN")
    with patch("hoast.music_cli.MusicClient") as client:
        assert main(["players"]) == 1
    client.assert_not_called()
    output = capsys.readouterr()
    assert output.out == ""
    assert "Set MUSIC_ASSISTANT_TOKEN" in output.err


@pytest.mark.parametrize(
    "arguments", [[], ["unknown"], ["status", "--player", "speaker"]]
)
def test_argument_errors(arguments: list[str]) -> None:
    """Require known subcommands and global options before the subcommand.

    Args:
        arguments:
            Invalid command-line syntax.

    """
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize(
    "value", ["0", "101", "-1", "35.0", "nan", "True", "set", "loud", ""]
)
def test_invalid_volume_before_rpc(value: str) -> None:
    """Reject invalid volume syntax before client construction or RPC dispatch.

    Args:
        value:
            Unsupported relative word or invalid absolute percentage.

    """
    with (
        patch("hoast.music_cli.MusicClient") as client,
        patch.object(ToolRegistry, "dispatch") as dispatch,
        pytest.raises(SystemExit) as raised,
    ):
        main(["volume", value])
    assert raised.value.code == 2
    client.assert_not_called()
    dispatch.assert_not_called()


@pytest.mark.parametrize("value,target", [("louder", 45), ("quieter", 35), ("35", 35)])
def test_native_volume_dispatch(
    value: str, target: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run real CLI validation and handlers through exactly one native volume command.

    Args:
        value:
            Relative word or absolute CLI percentage.

        target:
            Expected native target percentage from an initial level of forty.

        capsys:
            Capture the compact JSON outcome.

    """
    player: dict[str, JsonValue] = {
        "player_id": "speaker",
        "name": "Test speaker",
        "type": "player",
        "available": True,
        "enabled": True,
        "playback_state": "idle",
        "supported_features": ["volume_set"],
        "volume_level": 40,
    }
    responses: list[JsonValue] = [
        [player],
        player,
        player,
        None,
        {**player, "volume_level": target},
    ]
    with patch.object(MusicClient, "_request", side_effect=responses) as request:
        assert main(["volume", value]) == 0
    assert [(item.args, item.kwargs) for item in request.call_args_list] == [
        (
            ("players/all",),
            {
                "return_unavailable": True,
                "return_disabled": True,
                "return_protocol_players": False,
            },
        ),
        (("players/get",), {"player_id": "speaker"}),
        (("players/get",), {"player_id": "speaker"}),
        (("players/cmd/volume_set",), {"player_id": "speaker", "volume_level": target}),
        (("players/get",), {"player_id": "speaker"}),
    ]
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "volume_set"
    assert result["confirmation"] == "observed"
    assert result["level"] == result["observed_level"] == target
