"""Allocator-boundary checks without model inference or real heap trimming."""

import ctypes
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from hoast import memory


@pytest.fixture(autouse=True)
def clear_allocator_cache() -> None:
    """Keep process-symbol lookup independent across allocator-mocking tests."""
    memory._trim_function.cache_clear()


def test_reclaim_order_and_metrics(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Collect cycles before trimming and log signed RSS/cgroup observations.

    Args:
        monkeypatch:
            Replaces process counters, GC and allocator calls with ordered mocks.

        caplog:
            Captures the operational record without real model or prompt data.

    """
    events: list[str] = []
    snapshots = iter(
        [
            memory._MemorySnapshot(4096, 8192, 16384, 32768, 0),
            memory._MemorySnapshot(2048, 6144, 16384, 32768, 0),
        ]
    )

    def snapshot() -> memory._MemorySnapshot:
        """Record each lightweight measurement and return its fixture counters."""
        events.append("snapshot")
        return next(snapshots)

    def collect() -> int:
        """Record cycle collection without altering the test process heap."""
        events.append("collect")
        return 7

    def trim(pad: int) -> int:
        """Record a zero-padding trim request and simulate released heap pages.

        Args:
            pad:
                Allocator padding requested in bytes.

        """
        assert pad == 0
        events.append("trim")
        return 1

    monkeypatch.setattr(memory, "_snapshot", snapshot)
    monkeypatch.setattr(memory.gc, "collect", collect)
    monkeypatch.setattr(memory, "_trim_function", lambda: trim)
    with caplog.at_level(logging.INFO):
        memory.reclaim_memory("test.after")
    assert events == ["snapshot", "collect", "trim", "snapshot"]
    assert (
        "boundary=test.after status=completed gc_collected=7 trim_result=1"
        in caplog.text
    )
    assert "rss_reclaimed=2048" in caplog.text
    assert "cgroup_before=8192 cgroup_after=6144" in caplog.text
    assert "oom_kills=0" in caplog.text


def test_gc_without_supported_allocator(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Keep GC available when allocator trimming is unsupported.

    Args:
        monkeypatch:
            Supplies unavailable cgroup/allocator data and a collection mock.

        caplog:
            Captures explicit gc-only status and a non-guaranteed RSS outcome.

    """
    snapshots = [
        memory._MemorySnapshot(1024, None, None, None, None),
        memory._MemorySnapshot(2048, None, None, None, None),
    ]
    collect = Mock(return_value=0)
    monkeypatch.setattr(memory, "_snapshot", Mock(side_effect=snapshots))
    monkeypatch.setattr(memory, "_trim_function", lambda: None)
    monkeypatch.setattr(memory.gc, "collect", collect)
    with caplog.at_level(logging.INFO):
        memory.reclaim_memory("test.before")
    collect.assert_called_once_with()
    assert "status=gc_only" in caplog.text
    assert "rss_reclaimed=-1024" in caplog.text


def test_allocator_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the glibc signature once and treat zero return as a valid outcome.

    Args:
        monkeypatch:
            Supplies a process-library handle exposing a fake malloc_trim symbol.

    """
    trim = Mock(return_value=0)
    loader = Mock(return_value=SimpleNamespace(malloc_trim=trim))
    monkeypatch.setattr(memory.ctypes, "CDLL", loader)
    bound = memory._trim_function()
    assert bound is trim
    assert bound is not None
    assert bound(0) == 0
    assert trim.argtypes == [ctypes.c_size_t]
    assert trim.restype is ctypes.c_int
    assert memory._trim_function() is bound
    loader.assert_called_once_with(None)
    memory._trim_function.cache_clear()


def test_missing_allocator_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Make unsupported trim visible without repeating the warning per boundary.

    Args:
        monkeypatch:
            Supplies a process library that has no malloc_trim export.

        caplog:
            Captures the single capability warning.

    """
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda name: SimpleNamespace())
    with caplog.at_level(logging.WARNING):
        assert memory._trim_function() is None
        assert memory._trim_function() is None
    assert caplog.text.count("malloc_trim_unavailable") == 1
    memory._trim_function.cache_clear()


@pytest.mark.parametrize("maximum", ["max", "65536"])
def test_cgroup_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, maximum: str
) -> None:
    """Read complete optional cgroup counters and distinguish unlimited memory.

    Args:
        monkeypatch:
            Points cgroup lookup and process RSS at tiny independent fixtures.

        tmp_path:
            Temporary cgroup-like directory with text counter files.

        maximum:
            Linux unlimited marker or a finite byte limit.

    """
    for filename, value in {
        "memory.current": "8192\n",
        "memory.peak": "16384\n",
        "memory.max": maximum + "\n",
        "memory.events": "max 23\noom 2\noom_kill 1\n",
    }.items():
        (tmp_path / filename).write_text(value)
    monkeypatch.setattr(memory, "_cgroup_directory", lambda: tmp_path)
    monkeypatch.setattr(
        memory.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=4096)),
    )
    snapshot = memory._snapshot()
    assert snapshot.rss_bytes == 4096
    assert snapshot.cgroup_bytes == 8192
    assert snapshot.cgroup_peak_bytes == 16384
    assert snapshot.cgroup_limit_bytes == (None if maximum == "max" else 65536)
    assert snapshot.oom_kills == 1


def test_unavailable_rss_does_not_disable_collection(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Keep optional process diagnostics from blocking GC or allocator trimming.

    Args:
        monkeypatch:
            Makes RSS access fail while keeping garbage collection observable.

        caplog:
            Captures unavailable counters and the successful reclamation record.

    """
    monkeypatch.setattr(memory, "_cgroup_directory", lambda: None)
    monkeypatch.setattr(
        memory.psutil, "Process", Mock(side_effect=psutil.AccessDenied())
    )
    collect = Mock(return_value=0)
    trim = Mock(return_value=0)
    monkeypatch.setattr(memory.gc, "collect", collect)
    monkeypatch.setattr(memory, "_trim_function", lambda: trim)
    with caplog.at_level(logging.INFO):
        memory.reclaim_memory("test.unavailable")
    collect.assert_called_once_with()
    trim.assert_called_once_with(0)
    assert "memory.rss status=unavailable" in caplog.text
    assert "rss_before=None rss_after=None rss_reclaimed=None" in caplog.text
