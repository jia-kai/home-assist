"""Shared routing instructions and state-only context for runtime and fine-tuning."""

import json
from collections.abc import Mapping

from pydantic import JsonValue

SYSTEM_PROMPT = (
    "You are a home assistant. Use the available tools to fulfill clear requests. "
    "Distinguish requests to act from questions about the current state. "
    "Use only details supplied by the user or established in the conversation. "
    "Do not perform unrequested actions or invent arguments. "
    "If a request is ambiguous or unsupported, ask a brief clarification. "
    "Keep replies brief and do not claim success before a tool confirms it."
)


def build_system_prompt(
    context: Mapping[str, JsonValue], base_prompt: str = SYSTEM_PROMPT
) -> str:
    """Append only playback state or explicit unavailability to routing instructions.

    Extra provider metadata is deliberately excluded. Missing required state and
    unknown status values fail rather than being interpreted as music off.

    Args:
        context:
            Observed status with state on/off, or player_required, unavailable,
            or not_configured status. Song, artist, source and volume are ignored.

        base_prompt:
            Routing instructions; callers can retain a backend activation prefix.

    """
    if not base_prompt.strip():
        raise ValueError("Routing instructions must be nonempty")
    status = context["status"]
    if status not in ("observed", "player_required", "unavailable", "not_configured"):
        raise ValueError("Unexpected music prompt status")
    safe: dict[str, JsonValue] = {"status": status}
    if status == "observed":
        state = context["state"]
        if state not in ("on", "off"):
            raise ValueError("Music prompt state must be on or off")
        safe["state"] = state
    return (
        base_prompt
        + "\nCurrent music playback state (on=playing, off=paused or idle): "
        + json.dumps(safe, ensure_ascii=False)
    )
