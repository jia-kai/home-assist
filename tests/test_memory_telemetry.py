"""Counter-only memory telemetry and model-free voice-monitor ownership checks."""

import asyncio
import logging
import threading
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import psutil
import pytest

from hoast import application, llm_cli, memory
from hoast.application import Assistant, VoiceApplication
from hoast.config import SatelliteConfig
from hoast.stt2 import STT
from hoast.tts2 import TTS
from hoast.voice import VoiceTurn


@pytest.mark.parametrize("missing", [False, True])
def test_details_rollup_and_pressure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    missing: bool,
) -> None:
    """Convert rollup kB to bytes and preserve PSI units and absent fields.

    Args:
        monkeypatch:
            Redirects proc rollup and cgroup resolution to fixtures.

        tmp_path:
            Holds a cgroup PSI fixture.

        caplog:
            Captures rollup and pressure records or optional-failure warnings.

        missing:
            Whether both optional diagnostic files are inaccessible.

    """
    rollup = (
        "0000-ffff ---p 00000000 00:00 0 [rollup]\n"
        "Rss: 1025 kB\nPss: 1000 kB\nAnonymous: 900 kB\n"
        "AnonHugePages: 512 kB\nPrivate_Clean: 3 kB\nPrivate_Dirty: 4 kB\n"
    )
    monkeypatch.setattr(memory, "_cgroup_directory", lambda: tmp_path)
    monkeypatch.setattr(
        Path,
        "open",
        Mock(side_effect=OSError("unavailable"))
        if missing
        else Mock(return_value=StringIO(rollup)),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        Mock(side_effect=OSError("unavailable"))
        if missing
        else Mock(
            return_value=(
                "some avg10=1.25 avg60=0.50 avg300=0.10 total=123456\n"
                "full avg10=0.00 avg60=0.00 avg300=0.00 total=7\n"
            )
        ),
    )
    with caplog.at_level(logging.INFO):
        memory.log_memory_details("test.detail")
    if missing:
        assert "memory.details boundary=test.detail status=unavailable" in caplog.text
        assert "memory.pressure boundary=test.detail status=unavailable" in caplog.text
    else:
        assert "rss_bytes=1049600 pss_bytes=1024000" in caplog.text
        assert "anonymous_bytes=921600 anon_huge_pages_bytes=524288" in caplog.text
        assert (
            "private_clean_bytes=3072 private_dirty_bytes=4096 swap_bytes=None"
            in caplog.text
        )
        assert "scope=cgroup kind=some avg10_percent=1.25" in caplog.text
        assert "avg300_percent=0.10 total_us=123456" in caplog.text
        assert "kind=full" in caplog.text


def test_smaps_units_and_missing_fields() -> None:
    """Keep absent fields distinct from zero and reject unexpected size units."""
    assert memory._smaps_sizes(["Swap: 0 kB\n", "Pss: 2 kB\n"]) == {
        "Swap": 0,
        "Pss": 2048,
    }
    with pytest.raises(ValueError, match="units"):
        memory._smaps_sizes(["Rss: 2 MB\n"])


def test_mappings_bounded_ranked_and_optional(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Bound mappings to eight, rank anonymous bytes and tolerate OS failures.

    Args:
        monkeypatch:
            Supplies smaps text without opening process or mapped file contents.

        caplog:
            Captures bounded mapping records and optional diagnostics warnings.

    """
    text = "".join(
        f"{index:08x}-{index + 1:08x} rw-p 00000000 00:00 0 /{'x' * 300}\n"
        f"Size: 100 kB\nRss: 90 kB\nAnonymous: {index} kB\n"
        for index in range(12)
    )
    opened = Mock(return_value=StringIO(text))
    monkeypatch.setattr(Path, "open", opened)
    with caplog.at_level(logging.INFO):
        memory.log_memory_mappings("test.maps")
    opened.assert_called_once_with()
    records = [
        r.message for r in caplog.records if r.message.startswith("memory.mapping ")
    ]
    assert len(records) == 8
    assert "address=0000000b-0000000c" in records[0]
    assert "size_bytes=102400 rss_bytes=92160 anonymous_bytes=11264" in records[0]
    assert "anon_huge_pages_bytes=None" in records[0]
    assert "anonymous_bytes=4096" in records[-1]
    assert "x" * 160 not in records[0]  # The leading slash occupies one character.
    assert "x" * 159 in records[0]
    monkeypatch.setattr(Path, "open", Mock(side_effect=OSError("unavailable")))
    memory.log_memory_mappings("test.missing")
    assert "memory.mappings boundary=test.missing status=unavailable" in caplog.text


def test_observation_is_counter_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a single sample and elapsed time without reclamation or mapping scans.

    Args:
        monkeypatch:
            Replaces counters and forbids GC, trimming, and full mapping reads.

        caplog:
            Captures the structured operational record.

    """
    snapshot = memory._MemorySnapshot(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    sample = Mock(return_value=snapshot)
    forbidden = Mock(side_effect=AssertionError("expensive operation"))
    monkeypatch.setattr(memory, "_snapshot", sample)
    monkeypatch.setattr(memory.gc, "collect", forbidden)
    monkeypatch.setattr(memory, "_trim_function", forbidden)
    monkeypatch.setattr(memory.psutil.Process, "memory_full_info", forbidden)
    monkeypatch.setattr(memory.sys, "getallocatedblocks", lambda: 11)
    monkeypatch.setattr(memory.gc, "get_count", lambda: (12, 13, 14))
    monkeypatch.setattr(memory.time, "perf_counter", Mock(side_effect=[1.0, 1.125]))
    with caplog.at_level(logging.INFO):
        memory.log_memory_usage("test.counter")
    sample.assert_called_once_with()
    forbidden.assert_not_called()
    records = [r for r in caplog.records if r.message.startswith("memory.usage")]
    assert len(records) == 1
    assert records[0].message == (
        "memory.usage boundary=test.counter rss_bytes=1 cgroup_bytes=2 "
        "cgroup_peak_bytes=3 cgroup_limit_bytes=4 oom_events=6 oom_kills=5 "
        "anon_bytes=7 anon_thp_bytes=8 shmem_bytes=9 file_bytes=10 "
        "max_events=None kernel_bytes=None pagetables_bytes=None slab_bytes=None "
        "thp_fault_alloc=None thp_collapse_alloc=None pgscan=None pgsteal=None "
        "pgscan_khugepaged=None threads=None "
        "python_allocated_blocks=11 gc_counts=(12, 13, 14) "
        "sample_elapsed_seconds=0.125000"
    )
    assert memory._MemorySnapshot(1, 2, 3, 4, 5).anon_bytes is None


@pytest.mark.parametrize("stat_available", [False, True])
def test_optional_cgroup_stat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    stat_available: bool,
) -> None:
    """Read small cgroup counters; an absent stat file preserves other counters.

    Args:
        monkeypatch:
            Redirects cgroup and process counters to fixture data.

        tmp_path:
            Independent cgroup-like directory, without machine-local dependencies.

        caplog:
            Captures a visible diagnostic when the optional file is unavailable.

        stat_available:
            Whether the optional memory.stat fixture exists.

    """
    for name, value in {
        "memory.current": "20\n",
        "memory.peak": "30\n",
        "memory.max": "max\n",
        "memory.events": "oom 2\noom_kill 1\nmax 9\n",
    }.items():
        (tmp_path / name).write_text(value)
    if stat_available:
        (tmp_path / "memory.stat").write_text(
            "anon 40\nfile 50\nshmem 60\nanon_thp 70\nfile_mapped 80\n"
            "kernel 81\npagetables 82\nslab 83\nthp_fault_alloc 84\n"
            "thp_collapse_alloc 85\npgscan 86\npgsteal 87\n"
        )
    monkeypatch.setattr(memory, "_cgroup_directory", lambda: tmp_path)
    monkeypatch.setattr(
        memory.psutil,
        "Process",
        lambda: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=10), num_threads=lambda: 8
        ),
    )
    result = memory._snapshot()
    assert result.rss_bytes == 10
    assert result.cgroup_bytes == 20
    assert result.cgroup_peak_bytes == 30
    assert result.cgroup_limit_bytes is None
    assert result.oom_events == 2 and result.oom_kills == 1
    assert result.anon_bytes == (40 if stat_available else None)
    assert result.file_bytes == (50 if stat_available else None)
    assert result.shmem_bytes == (60 if stat_available else None)
    assert result.anon_thp_bytes == (70 if stat_available else None)
    assert result.max_events == 9
    assert result.threads == 8
    for name, value in {
        "kernel_bytes": 81,
        "pagetables_bytes": 82,
        "slab_bytes": 83,
        "thp_fault_alloc": 84,
        "thp_collapse_alloc": 85,
        "pgscan": 86,
        "pgsteal": 87,
    }.items():
        assert getattr(result, name) == (value if stat_available else None)
    assert result.pgscan_khugepaged is None
    assert ("memory.cgroup_stat status=unavailable" in caplog.text) != stat_available


def test_sparse_counters_and_optional_khugepaged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep missing events/stats unknown and khugepaged counters in page units.

    Args:
        monkeypatch:
            Replaces cgroup resolution and process counters with test data.

        tmp_path:
            Holds sparse cgroup files representative of differing kernel support.

    """
    for name, value in {
        "memory.current": "20\n",
        "memory.max": "1024\n",
        "memory.events": "oom_kill 0\n",
        "memory.stat": "pgscan_khugepaged 17\n",
    }.items():
        (tmp_path / name).write_text(value)
    monkeypatch.setattr(memory, "_cgroup_directory", lambda: tmp_path)
    monkeypatch.setattr(
        memory.psutil,
        "Process",
        lambda: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=10),
            num_threads=Mock(side_effect=psutil.AccessDenied()),
        ),
    )
    result = memory._snapshot()
    assert result.rss_bytes == 10
    assert result.threads is None
    assert result.cgroup_peak_bytes is None
    assert result.cgroup_limit_bytes == 1024
    assert result.max_events is None
    assert result.oom_events is None
    assert result.oom_kills == 0
    assert result.pgscan_khugepaged == 17
    assert result.pgscan is None and result.pgsteal is None


@pytest.mark.parametrize("error", [OSError("unavailable"), psutil.AccessDenied()])
def test_counter_errors_are_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    """Keep OS diagnostic failures visible without failing the observed operation.

    Args:
        monkeypatch:
            Supplies an inaccessible process and absent optional cgroup files.

        tmp_path:
            Empty cgroup fixture directory.

        caplog:
            Captures counter-unavailable warnings and the final usage record.

        error:
            Expected OS or psutil diagnostic exception.

    """
    monkeypatch.setattr(memory, "_cgroup_directory", lambda: tmp_path)
    monkeypatch.setattr(memory.psutil, "Process", Mock(side_effect=error))
    with caplog.at_level(logging.INFO):
        memory.log_memory_usage("test.unavailable")
    assert "memory.rss status=unavailable" in caplog.text
    assert "memory.cgroup status=unavailable" in caplog.text
    assert "memory.cgroup_stat status=unavailable" in caplog.text
    assert "boundary=test.unavailable rss_bytes=None cgroup_bytes=None" in caplog.text


@pytest.mark.parametrize("fail", [False, True])
def test_transcribe_boundaries(monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    """Bracket recognition on both return and failure without extra inference.

    Args:
        monkeypatch:
            Captures memory boundary labels instead of sampling real counters.

        fail:
            Whether recognition raises its original exception.

    """
    observe = Mock()
    monkeypatch.setattr(application, "log_memory_usage", observe)
    stt = MagicMock(spec=STT)
    stt.transcribe_samples.return_value = "recognized"
    if fail:
        stt.transcribe_samples.side_effect = RuntimeError("recognition failed")
    assistant = MagicMock(spec=Assistant)
    tts = MagicMock(spec=TTS)
    app = VoiceApplication(assistant, stt, tts)
    if fail:
        with pytest.raises(RuntimeError, match="recognition failed"):
            app.transcribe(b"\0\0")
    else:
        assert app.transcribe(b"\0\0") == "recognized"
    assert [call.args[0] for call in observe.call_args_list] == [
        "voice.transcribe.before",
        "voice.transcribe.after",
    ]
    stt.transcribe_samples.assert_called_once()
    assistant.reply.assert_not_called()
    tts.play.assert_not_called()


@pytest.mark.parametrize(
    "outcome",
    ["success", "agent_error", "tts_error", "cancelled", "empty", "pre_cancelled"],
)
def test_turn_boundaries_and_reset(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Observe stage exits and post-reset state for failures and cancellation.

    Args:
        monkeypatch:
            Captures boundary labels and session-reset ordering.

        outcome:
            Completed, failed, empty, or cancelled turn scenario.

    """
    events: list[str] = []
    monkeypatch.setattr(application, "log_memory_usage", events.append)
    assistant = MagicMock(spec=Assistant)
    assistant.new_session.side_effect = lambda: events.append("reset")
    assistant.reply.return_value = "Done."
    tts = MagicMock(spec=TTS)
    cancelled = threading.Event()
    if outcome == "agent_error":
        assistant.reply.side_effect = RuntimeError("agent failure")
    elif outcome == "tts_error":
        tts.play.side_effect = RuntimeError("playback failure")
    elif outcome == "empty":
        assistant.reply.return_value = ""
    elif outcome == "cancelled":
        assistant.reply.side_effect = lambda *args: cancelled.set() or "Done."
    elif outcome == "pre_cancelled":
        cancelled.set()
    stt = MagicMock(spec=STT)
    app = VoiceApplication(assistant, stt, tts, Mock())
    turn = VoiceTurn("test", 0.0, 1.0, cancelled)
    if outcome == "tts_error":
        with pytest.raises(RuntimeError, match="playback failure"):
            app.respond(turn)
    else:
        app.respond(turn)
    if outcome == "pre_cancelled":
        assert events == []
        assistant.reply.assert_not_called()
        return
    assert events[:2] == ["reset", "voice.agent.before"]
    assert "voice.agent.after" in events
    assert events[-2:] == ["reset", "voice.turn.after"]
    assert ("voice.tts.after" in events) == (outcome not in {"cancelled", "empty"})
    assistant.reply.assert_called_once()
    stt.transcribe_samples.assert_not_called()


@pytest.mark.parametrize("outcome", ["normal", "failure", "cancelled"])
def test_run_voice_owns_idle_monitor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str
) -> None:
    """Run real voice orchestration and join its monitor on every listener exit.

    Args:
        monkeypatch:
            Replaces startup, listener, counters, and the monitor's 60-second sleep.

        tmp_path:
            Unused model/cache root forwarded to the mocked startup.

        outcome:
            Listener completion, listener failure, or owner-task cancellation.

    """
    observe = Mock()
    app = MagicMock(spec=VoiceApplication)
    prepare = Mock(return_value=app)
    monkeypatch.setattr(llm_cli, "prepare_voice", prepare)
    monkeypatch.setattr(llm_cli, "log_memory_usage", observe)
    monkeypatch.setattr(llm_cli, "log_memory_details", Mock())
    monkeypatch.setattr(llm_cli, "log_memory_mappings", Mock())

    async def scenario() -> None:
        """Coordinate one periodic sample then block until owner cleanup."""
        sleeping = asyncio.Event()
        stopped = asyncio.Event()
        listener_ready = asyncio.Event()
        monitor_tasks: list[asyncio.Task[object]] = []
        delays: list[float] = []

        async def sleep(delay: float) -> None:
            """Permit one sample then expose cancellation of the monitor sleep.

            Args:
                delay:
                    Requested interval in seconds, checked against the policy.

            """
            delays.append(delay)
            if len(delays) == 1:
                return
            task = asyncio.current_task()
            assert task is not None
            monitor_tasks.append(task)
            sleeping.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        async def listen(*args: object, **kwargs: object) -> None:
            """Wait for an idle observation, then exercise the requested exit.

            Args:
                args:
                    Native listener positional options, unused by this mock.

                kwargs:
                    Callback and playback options, unused by this mock.

            """
            await sleeping.wait()
            listener_ready.set()
            if outcome == "failure":
                raise RuntimeError("listener failure")
            if outcome == "cancelled":
                await asyncio.Future()

        def close() -> None:
            """Require monitor cancellation before releasing the speech engines."""
            assert stopped.is_set()

        monkeypatch.setattr(llm_cli.asyncio, "sleep", sleep)
        monkeypatch.setattr(llm_cli, "listen", listen)
        app.close.side_effect = close
        task = asyncio.create_task(
            llm_cli.run_voice(
                MagicMock(spec=Assistant), SatelliteConfig("mock"), None, tmp_path
            )
        )
        await asyncio.wait_for(listener_ready.wait(), timeout=5)
        if outcome == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "failure":
            with pytest.raises(RuntimeError, match="listener failure"):
                await task
        else:
            await task
        assert delays == [60, 60]
        assert len(monitor_tasks) == 1
        assert monitor_tasks[0].done() and monitor_tasks[0].cancelled()

    asyncio.run(scenario())
    assert [call.args[0] for call in observe.call_args_list] == [
        "voice.startup",
        "voice.idle",
    ]
    prepare.assert_called_once()
    app.close.assert_called_once_with()
    app.transcribe.assert_not_called()
    app.respond.assert_not_called()
    app.assistant.reply.assert_not_called()


def test_startup_failure_has_no_monitor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not start a monitor or listener when speech preparation fails.

    Args:
        monkeypatch:
            Supplies a startup exception and spies on monitor creation.

        tmp_path:
            Unused cache path passed through the startup interface.

    """
    monitor = Mock()
    listener = Mock()
    monkeypatch.setattr(
        llm_cli, "prepare_voice", Mock(side_effect=RuntimeError("startup"))
    )
    monkeypatch.setattr(llm_cli, "_monitor_memory_usage", monitor)
    monkeypatch.setattr(llm_cli, "listen", listener)
    with pytest.raises(RuntimeError, match="startup"):
        asyncio.run(
            llm_cli.run_voice(
                MagicMock(spec=Assistant), SatelliteConfig("mock"), None, tmp_path
            )
        )
    monitor.assert_not_called()
    listener.assert_not_called()


def test_startup_cancellation_has_no_monitor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Join cancelled preparation and close its application without a monitor.

    Args:
        monkeypatch:
            Blocks mocked startup while cancelling the owning voice task.

        tmp_path:
            Unused cache path forwarded without reading model artifacts.

    """
    started = threading.Event()
    release = threading.Event()
    app = MagicMock(spec=VoiceApplication)
    monitor = Mock()
    listener = Mock()

    def prepare(*args: object, **kwargs: object) -> MagicMock:
        """Hold the inference worker until the owner has requested cancellation.

        Args:
            args:
                Startup positional settings, unused by the model-free fixture.

            kwargs:
                Startup speech settings, unused by the model-free fixture.

        """
        started.set()
        assert release.wait(timeout=5)
        return app

    async def scenario() -> None:
        """Cancel startup, release its worker, and observe joined cleanup."""
        task = asyncio.create_task(
            llm_cli.run_voice(
                MagicMock(spec=Assistant), SatelliteConfig("mock"), None, tmp_path
            )
        )
        try:
            while not started.is_set():
                await asyncio.sleep(0)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

    monkeypatch.setattr(llm_cli, "prepare_voice", prepare)
    monkeypatch.setattr(llm_cli, "_monitor_memory_usage", monitor)
    monkeypatch.setattr(llm_cli, "listen", listener)
    asyncio.run(scenario())
    monitor.assert_not_called()
    listener.assert_not_called()
    app.close.assert_called_once_with()
