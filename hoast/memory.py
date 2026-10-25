"""Lightweight memory observations and optional Python/glibc reclamation."""

import ctypes
import gc
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import psutil

from .logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class _MemorySnapshot:
    """Process RSS and optional Linux cgroup-v2 counters, all sizes in bytes."""

    rss_bytes: int | None
    """Resident process pages, including mapped models; None if counters are unavailable."""

    cgroup_bytes: int | None
    """Whole-cgroup memory.current, including charged page cache; None if unavailable."""

    cgroup_peak_bytes: int | None
    """Whole-cgroup lifetime high-water charge, not a per-call peak."""

    cgroup_limit_bytes: int | None
    """Finite memory.max; None for an unlimited or unavailable cgroup."""

    oom_kills: int | None
    """Cumulative memory.events oom_kill counter, not memory-limit reclaim attempts."""

    oom_events: int | None = None
    """Cumulative cgroup memory.events oom counter; None if unavailable."""

    anon_bytes: int | None = None
    """Whole-cgroup memory.stat anonymous charge in bytes."""

    anon_thp_bytes: int | None = None
    """Whole-cgroup anonymous transparent huge-page charge in bytes."""

    shmem_bytes: int | None = None
    """Whole-cgroup shared-memory charge in bytes, overlapping file accounting."""

    file_bytes: int | None = None
    """Whole-cgroup file charge in bytes, including shared memory."""


@lru_cache(maxsize=1)
def _trim_function() -> Callable[[int], int] | None:
    """Bind process malloc_trim once, or warn when the allocator lacks that API.

    The ctypes function retains its library owner. A zero return is a valid
    no-pages-released result, not an error. Non-glibc allocators and live native
    buffers need not be affected even when this process exports the symbol.
    """
    try:
        function = getattr(ctypes.CDLL(None), "malloc_trim", None)
    except OSError:
        logger.warning(
            "memory.trim status=unsupported reason=process_library_unavailable",
            exc_info=True,
        )
        return None
    if function is None:
        logger.warning("memory.trim status=unsupported reason=malloc_trim_unavailable")
        return None
    function.argtypes = [ctypes.c_size_t]
    function.restype = ctypes.c_int
    return function


@lru_cache(maxsize=1)
def _cgroup_directory() -> Path | None:
    """Resolve the current Linux cgroup-v2 directory without reading configuration.

    Both a container-root cgroup and a host process's absolute hierarchy path
    are supported. Optional diagnostic failures do not disable reclamation.
    """
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
                if directory.is_dir():
                    return directory
                break
    except OSError:
        logger.warning("memory.cgroup status=unavailable", exc_info=True)
        return None
    logger.warning("memory.cgroup status=unavailable reason=no_v2_directory")
    return None


def _snapshot() -> _MemorySnapshot:
    """Sample current RSS and cgroup counters without scanning process mappings."""
    current: int | None = None
    peak: int | None = None
    limit: int | None = None
    kills: int | None = None
    oom: int | None = None
    stats: dict[str, int] = {}
    directory = _cgroup_directory()
    if directory is not None:
        try:
            current = int((directory / "memory.current").read_text())
            peak_path = directory / "memory.peak"
            if peak_path.is_file():
                peak = int(peak_path.read_text())
            maximum = (directory / "memory.max").read_text().strip()
            limit = None if maximum == "max" else int(maximum)
            for line in (directory / "memory.events").read_text().splitlines():
                name, count = line.split()
                if name == "oom_kill":
                    kills = int(count)
                elif name == "oom":
                    oom = int(count)
        except OSError:
            logger.warning("memory.cgroup status=unavailable", exc_info=True)
        try:
            for line in (directory / "memory.stat").read_text().splitlines():
                name, count = line.split()
                if name in {"anon", "anon_thp", "shmem", "file"}:
                    stats[name] = int(count)
        except OSError:
            logger.warning("memory.cgroup_stat status=unavailable", exc_info=True)
    rss: int | None = None
    try:
        rss = psutil.Process().memory_info().rss
    except psutil.Error, OSError:
        logger.warning("memory.rss status=unavailable", exc_info=True)
    return _MemorySnapshot(
        rss_bytes=rss,
        cgroup_bytes=current,
        cgroup_peak_bytes=peak,
        cgroup_limit_bytes=limit,
        oom_kills=kills,
        oom_events=oom,
        anon_bytes=stats.get("anon"),
        anon_thp_bytes=stats.get("anon_thp"),
        shmem_bytes=stats.get("shmem"),
        file_bytes=stats.get("file"),
    )


def log_memory_usage(boundary: str) -> None:
    """Log current counters without collection, trimming, or mapping/heap scans.

    Sampling is synchronous and retains no history. RSS covers this process;
    cgroup counters include other processes and charged cache. Python counters
    describe interpreter allocation activity, not live object or native bytes.
    Elapsed seconds cover counter sampling, excluding emission of the log record.
    Optional OS-counter failures are logged and represented by None.

    Args:
        boundary:
            Short diagnostic phase label, never prompt content or credentials.

    """
    started = time.perf_counter()
    snapshot = _snapshot()
    allocated_blocks = sys.getallocatedblocks()
    gc_counts = gc.get_count()
    logger.info(
        "memory.usage boundary=%s rss_bytes=%s cgroup_bytes=%s "
        "cgroup_peak_bytes=%s cgroup_limit_bytes=%s oom_events=%s oom_kills=%s "
        "anon_bytes=%s anon_thp_bytes=%s shmem_bytes=%s file_bytes=%s "
        "python_allocated_blocks=%s gc_counts=%s sample_elapsed_seconds=%.6f",
        boundary,
        snapshot.rss_bytes,
        snapshot.cgroup_bytes,
        snapshot.cgroup_peak_bytes,
        snapshot.cgroup_limit_bytes,
        snapshot.oom_events,
        snapshot.oom_kills,
        snapshot.anon_bytes,
        snapshot.anon_thp_bytes,
        snapshot.shmem_bytes,
        snapshot.file_bytes,
        allocated_blocks,
        gc_counts,
        time.perf_counter() - started,
    )


def reclaim_memory(boundary: str) -> None:
    """Collect cycles, trim supported heaps and log observed memory/latency changes.

    Call only after synchronous inference is idle and borrowed temporary outputs
    are consumed or copied. This retains models, live tensors and native caches;
    it is not OpenVINO/GPU buffer release or a guarantee of reduced RSS. Full
    Python GC and allocator trimming can add latency and should be benchmarked.
    Measurements include both snapshots, GC and trimming. Cgroup figures cover
    all processes/cache in the cgroup and are not attributable to GC alone.

    Args:
        boundary:
            Short diagnostic phase label, never prompt content or credentials.

    """
    started = time.perf_counter()
    before = _snapshot()
    gc_started = time.perf_counter()
    collected = gc.collect()
    gc_seconds = time.perf_counter() - gc_started
    trim = _trim_function()
    trim_started = time.perf_counter()
    trim_result = None if trim is None else trim(0)
    trim_seconds = time.perf_counter() - trim_started
    after = _snapshot()
    rss_reclaimed = (
        before.rss_bytes - after.rss_bytes
        if before.rss_bytes is not None and after.rss_bytes is not None
        else None
    )
    logger.info(
        "memory.reclaim boundary=%s status=%s gc_collected=%d trim_result=%s "
        "rss_before=%s rss_after=%s rss_reclaimed=%s "
        "cgroup_before=%s cgroup_after=%s cgroup_peak=%s cgroup_limit=%s "
        "oom_kills=%s gc_seconds=%.6f trim_seconds=%.6f seconds=%.6f",
        boundary,
        "gc_only" if trim is None else "completed",
        collected,
        trim_result,
        before.rss_bytes,
        after.rss_bytes,
        rss_reclaimed,
        before.cgroup_bytes,
        after.cgroup_bytes,
        after.cgroup_peak_bytes,
        after.cgroup_limit_bytes,
        after.oom_kills,
        gc_seconds,
        trim_seconds,
        time.perf_counter() - started,
    )
