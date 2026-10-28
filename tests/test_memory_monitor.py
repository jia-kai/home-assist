"""Cadence and thread ownership of the slower periodic memory diagnostics."""

import asyncio
import threading
from unittest.mock import Mock

import pytest

from hoast import llm_cli


def test_diagnostics_cadence_and_worker_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scan mappings at startup and every 15 samples, outside the event-loop thread.

    Args:
        monkeypatch:
            Replaces diagnostic reads and advances virtual 60-second intervals.

    """
    loop_thread = threading.get_ident()
    detail_threads: list[int] = []
    details: list[str] = []
    mappings: list[str] = []
    calls = 0
    usage = Mock()

    def detail(boundary: str) -> None:
        """Capture the worker and boundary without accessing process memory.

        Args:
            boundary:
                Phase label supplied by the monitor.

        """
        detail_threads.append(threading.get_ident())
        details.append(boundary)

    async def sleep(delay: float) -> None:
        """Advance 15 intervals and then cancel the owner.

        Args:
            delay:
                Requested periodic interval in seconds.

        """
        nonlocal calls
        assert delay == 60
        calls += 1
        if calls == 16:
            raise asyncio.CancelledError

    monkeypatch.setattr(llm_cli, "log_memory_details", detail)
    monkeypatch.setattr(llm_cli, "log_memory_mappings", mappings.append)
    monkeypatch.setattr(llm_cli, "log_memory_usage", usage)
    monkeypatch.setattr(llm_cli.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(llm_cli._monitor_memory_usage())
    assert details == ["voice.monitor.startup"] + ["voice.periodic"] * 15
    assert mappings == ["voice.monitor.startup", "voice.periodic"]
    assert usage.call_count == 15
    assert set(detail_threads) != {loop_thread}
    assert len(set(detail_threads)) == 1
    assert not any(
        thread.name.startswith("memory-diagnostics") for thread in threading.enumerate()
    )
