"""Production-compatible tool declarations for dataset validation without live clients."""

from hoast.lights import declare_light_tool
from hoast.llm import ToolRegistry
from hoast.music import declare_music_tools
from hoast.weather import declare_weather_tool


def tool_registry() -> ToolRegistry:
    """Return all eight production schemas with handlers that reject execution."""
    return ToolRegistry(
        [declare_weather_tool(), *declare_music_tools(), declare_light_tool()]
    )
