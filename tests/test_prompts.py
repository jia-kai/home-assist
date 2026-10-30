"""Tests for privacy-preserving music routing context."""

import pytest
from pydantic import JsonValue

from hoast.prompts import build_system_prompt


@pytest.mark.parametrize(
    ("context", "state"),
    [
        ({"status": "observed", "state": "on", "title": "Private"}, "on"),
        ({"status": "observed", "state": "off"}, "off"),
        ({"status": "unavailable", "state": "on"}, "off"),
        ({"status": "player_required"}, "off"),
        ({"status": "not_configured"}, "off"),
    ],
)
def test_music_context_projection(context: dict[str, JsonValue], state: str) -> None:
    """Retain observed state and map nonobserved states to off, excluding metadata.

    Args:
        context:
            Provider context to project into routing instructions.

        state:
            Expected prompt playback state.

    """
    prompt = build_system_prompt(context, "Base")
    assert prompt == (
        "Base\nCurrent music playback state (on=playing, off=paused or idle): "
        f'{{"status": "observed", "state": "{state}"}}'
    )


@pytest.mark.parametrize(
    ("context", "error"),
    [
        ({}, KeyError),
        ({"status": "observed"}, KeyError),
        ({"status": "observed", "state": "unknown"}, ValueError),
        ({"status": "unknown"}, ValueError),
    ],
)
def test_invalid_music_context(
    context: dict[str, JsonValue], error: type[Exception]
) -> None:
    """Reject invalid provider contexts instead of silently mapping them to off.

    Args:
        context:
            Malformed provider context.

        error:
            Expected validation exception type.

    """
    with pytest.raises(error):
        build_system_prompt(context)
