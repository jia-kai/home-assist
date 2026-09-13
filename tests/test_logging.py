"""Validate colored console output and durable exception diagnostics."""

import io
import logging
from pathlib import Path

import pytest

from hoast.logging import ConsoleHandler, configure_logging, get_logger


class Terminal(io.StringIO):
    """In-memory stream advertising terminal color support."""

    def isatty(self) -> bool:
        """Advertise terminal capability without requiring a real TTY."""
        return True


def test_console_color_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Respect terminal capability, NO_COLOR, and explicit overrides.

    Args:
        monkeypatch:
            Isolates environment settings for automatic color selection.

    """
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert ConsoleHandler(Terminal()).color
    assert not ConsoleHandler(io.StringIO()).color
    monkeypatch.setenv("NO_COLOR", "")
    assert not ConsoleHandler(Terminal()).color
    assert ConsoleHandler(io.StringIO(), color=True).color


def test_plain_file_colored_console_and_reconfiguration(tmp_path: Path) -> None:
    """Keep full chained exceptions in plain files and avoid duplicate handlers.

    Args:
        tmp_path:
            Isolated destination for durable logging output.

    """
    stream = io.StringIO()
    path = tmp_path / "nested" / "application.log"
    root = logging.getLogger()
    original_level = root.level
    foreign_handler = logging.NullHandler()
    root.addHandler(foreign_handler)
    try:
        configure_logging(log_file=path, stream=stream, color=True)
        logger = get_logger("hoast.tests.logging")
        assert logger is get_logger("hoast.tests.logging")
        logger.debug("Detailed candidate fixture")
        assert "Detailed candidate fixture" in path.read_text()
        assert "Detailed candidate fixture" not in stream.getvalue()
        try:
            try:
                raise KeyError("required argument")
            except KeyError as cause:
                error = ValueError("invalid call")
                error.add_note("request=fixture-42")
                raise error from cause
        except ValueError:
            logger.exception("Handler failed")
        console = stream.getvalue()
        persisted = path.read_text()
        assert "\x1b[31m" in console and "\x1b[0m" in console
        assert "\x1b[" not in persisted
        for expected in (
            "KeyError",
            "ValueError",
            "direct cause",
            "request=fixture-42",
        ):
            assert expected in console and expected in persisted
        configure_logging(log_file=path, stream=stream, color=False)
        logger.info("Unique reconfigured record")
        assert stream.getvalue().count("Unique reconfigured record") == 1
        assert path.read_text().count("Unique reconfigured record") == 1
        assert foreign_handler in root.handlers
    finally:
        root.removeHandler(foreign_handler)
        configure_logging(level=original_level, stream=io.StringIO(), color=False)
