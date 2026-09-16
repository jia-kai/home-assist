"""Agent light tools validate arguments and distinguish verified power from failure."""

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from hoast.agent import render_light
from hoast.config import SwitchConfig
from hoast.lights import LightArguments, LightClient


@pytest.mark.parametrize("on", [True, False])
def test_light_tool(on: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Render success only after the verified relay coroutine completes.

    Args:
        on:
            Explicit desired light state.

        monkeypatch:
            Replaces physical power control with an offline coroutine.

    """
    power = AsyncMock()
    monkeypatch.setattr("hoast.lights._set_power", power)
    client = LightClient(SwitchConfig("192.0.2.20"))
    result = client.tool().handler(LightArguments(on=on))
    power.assert_awaited_once_with(client.config, on)
    assert render_light(result) == ("Light on." if on else "Light off.")


def test_light_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unconfirmed power does not become a successful spoken action.

    Args:
        monkeypatch:
            Simulates a verified-relay state mismatch.

    """
    monkeypatch.setattr(
        "hoast.lights._set_power", AsyncMock(side_effect=RuntimeError("unconfirmed"))
    )
    result = LightClient(SwitchConfig("192.0.2.20")).set_light(LightArguments(on=True))
    assert "couldn't" in render_light(result)
    with pytest.raises(ValidationError):
        LightArguments.model_validate({"on": "true"})
    with pytest.raises(ValidationError):
        LightArguments.model_validate({})
