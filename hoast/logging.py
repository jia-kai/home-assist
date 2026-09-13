"""Package logging facade with colored consoles and plain durable log files."""

import logging as _logging
import os
import sys
import threading
from pathlib import Path
from typing import TextIO

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_COLORS: dict[int, str] = {
    _logging.DEBUG: "\x1b[36m",
    _logging.INFO: "\x1b[32m",
    _logging.WARNING: "\x1b[33m",
    _logging.ERROR: "\x1b[31m",
    _logging.CRITICAL: "\x1b[1;31m",
}
_managed_handlers: list[_logging.Handler] = []
_configuration_lock = threading.RLock()


def get_logger(name: str) -> _logging.Logger:
    """Return a standard named logger without configuring process logging.

    Args:
        name:
            Module name, normally `__name__`. Configure handlers once in the
            application entry point using `configure_logging`.

    """
    return _logging.getLogger(name)


class ConsoleFormatter(_logging.Formatter):
    """Color whole console records while leaving shared log records unmodified."""

    color: bool
    """Whether severity-specific ANSI color sequences are emitted."""

    def __init__(self, color: bool = True) -> None:
        """Use timestamp, severity, logger name, and the full formatted message.

        Args:
            color:
                Enable ANSI styling, including for rendered exception tracebacks.

        """
        super().__init__(_FORMAT, datefmt=_DATE_FORMAT)
        self.color = color

    def format(self, record: _logging.LogRecord) -> str:
        """Format one record, preserving exception chains and notes.

        Args:
            record:
                Shared record; ANSI codes are added only to the returned string,
                so file and other handlers receive uncolored content.

        """
        rendered = super().format(record)
        prefix = _COLORS.get(record.levelno, "") if self.color else ""
        return f"{prefix}{rendered}\x1b[0m" if prefix else rendered


class ConsoleHandler(_logging.StreamHandler):
    """Console output with automatic TTY and NO_COLOR detection."""

    color: bool
    """Resolved color setting for this output stream."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        color: bool | None = None,
    ) -> None:
        """Create a console handler without attaching it to a logger.

        Args:
            stream:
                Text destination; defaults to the current `sys.stderr`.

            color:
                Force colors on/off. None enables colors only on a TTY when
                `NO_COLOR` is absent from the environment.

        """
        destination = sys.stderr if stream is None else stream
        super().__init__(destination)
        self.color = (
            destination.isatty() and "NO_COLOR" not in os.environ
            if color is None
            else color
        )
        self.setFormatter(ConsoleFormatter(self.color))


def configure_logging(
    *,
    level: int | str = _logging.INFO,
    log_file: Path | None = None,
    color: bool | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure console severity and durable debug logging once in an entry point.

    Reconfiguration replaces and closes only handlers installed by this function,
    preserving application/framework handlers. File output is append-only UTF-8
    with plain formatting; full exception chains and notes are retained.

    Args:
        level:
            Standard numeric or uppercase console threshold. Files retain debug
            diagnostics regardless of the console threshold.

        log_file:
            Optional durable log file; missing parent directories are created.

        color:
            Console color override; None uses TTY and `NO_COLOR` detection.

        stream:
            Console text destination; defaults to the current `sys.stderr`.

    """
    console = ConsoleHandler(stream, color=color)
    console.setLevel(level)
    handlers: list[_logging.Handler] = [console]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = _logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(_logging.DEBUG)
        file_handler.setFormatter(_logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        handlers.append(file_handler)
    with _configuration_lock:
        root = _logging.getLogger()
        for handler in _managed_handlers:
            root.removeHandler(handler)
            handler.close()
        _managed_handlers.clear()
        for handler in handlers:
            root.addHandler(handler)
        _managed_handlers.extend(handlers)
        root.setLevel(min(console.level, _logging.DEBUG) if log_file else console.level)
