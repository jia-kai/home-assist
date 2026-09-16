"""Verified agent light control through the configured local Kasa relay."""

import asyncio
from dataclasses import dataclass, replace

from kasa import KasaException
from pydantic import Field, JsonValue

from .config import SwitchConfig
from .llm import Tool, ToolArguments, declaration_only
from .logging import get_logger
from .switch import _set_power

logger = get_logger(__name__)


class LightArguments(ToolArguments):
    """Explicit desired power state for the configured light."""

    on: bool = Field(description="True to turn the light on; false to turn it off.")
    """Required desired power state; implicit string/integer coercion is rejected."""


def declare_light_tool() -> Tool[LightArguments]:
    """Return the production light schema without a configured relay or device access."""
    return Tool(
        "set_light",
        "Turn the configured light on or off.",
        LightArguments,
        declaration_only,
    )


@dataclass(slots=True)
class LightClient:
    """Expose verified relay power control as a synchronous agent tool."""

    config: SwitchConfig
    """Validated IP address of the light's configured Kasa relay."""

    def tool(self) -> Tool[LightArguments]:
        """Bind the shared light declaration to verified relay control."""
        return replace(declare_light_tool(), handler=self.set_light)

    def set_light(self, args: LightArguments) -> JsonValue:
        """Set and verify the light relay; return a failure without claiming success.

        Call from a synchronous worker thread without a running event loop.

        Args:
            args:
                Validated explicit desired light power state.

        """
        logger.info("set_light ip=%s on=%s status=requested", self.config.ip, args.on)
        try:
            asyncio.run(_set_power(self.config, args.on))
        except KasaException, OSError, RuntimeError:
            logger.exception(
                "set_light ip=%s on=%s status=failed", self.config.ip, args.on
            )
            return {"status": "failed", "on": args.on}
        logger.info("set_light ip=%s on=%s status=confirmed", self.config.ip, args.on)
        return {"status": "confirmed", "on": args.on}
