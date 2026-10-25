"""Model-free checks for release benchmark accounting, isolation and evidence."""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from tools import benchmark_memory_release as benchmark


@dataclass(slots=True)
class FakeConfig:
    """Minimal serializable benchmark configuration."""

    release_after_call: bool
    """Whether complete calls include cleanup."""


class FakeTTS:
    """Deterministic toy engine exposing the public benchmark contract."""

    config: FakeConfig
    """Configured release mode."""

    last_timings: dict[str, Any]
    """Seconds and nested nonnumeric stage metadata."""

    calls: int
    """Number of synthesis calls."""

    closed: bool
    """Whether close released the toy engine."""

    def __init__(self, release: bool) -> None:
        """Create an engine without external model resources.

        Args:
            release:
                Include the measured cleanup timing key when true.

        """
        self.config = FakeConfig(release)
        self.last_timings = {}
        self.calls = 0
        self.closed = False

    def synthesize(self, text: str) -> tuple[np.ndarray[Any, Any], int]:
        """Return owned mono float32 samples and nested stage timing metadata.

        Args:
            text:
                Nonempty input prompt, required by the real public API.

        """
        assert text
        self.calls += 1
        self.last_timings = {
            "total": 0.001,
            "language": "EN",
            "segments": [{"bert": 0.0002}],
        }
        if self.config.release_after_call:
            self.last_timings["release_memory"] = 0.0001
        return np.array([0.0, 0.25, -0.5, 1.0], dtype=np.float32), 16000

    def close(self) -> None:
        """Mark deterministic cleanup without affecting global CPU policy."""
        self.closed = True


def options(tmp_path: Path) -> argparse.Namespace:
    """Build isolated test paths and the full qualification counts.

    Args:
        tmp_path:
            Pytest-managed temporary directory, never deployed cache resources.

    """
    return argparse.Namespace(
        output=tmp_path,
        stt_root=tmp_path / "stt",
        tts_root=tmp_path / "tts",
        engine="tts",
        mode="retained",
        worker="measure",
        round=0,
        rounds=2,
        seed=7,
        warmups=2,
        repeats=8,
        idle_seconds=0,
        timeout=0.1,
    )


def test_distribution_and_policy() -> None:
    """Use interpolated p95 and require both absolute and relative latency bounds."""
    assert benchmark.distribution([1.0, 2.0, 3.0])["p95"] == pytest.approx(2.9)
    assert benchmark.comparison([1.0] * 8, [1.05] * 8)["latency_pass"]
    assert not benchmark.comparison([0.1] * 8, [0.12] * 8)["median_heuristic_pass"]
    assert not benchmark.comparison([2.0] * 8, [2.15] * 8)["median_heuristic_pass"]
    assert not benchmark.comparison([1.0] * 8, [1.0] * 7 + [2.0])["tail_screen_pass"]
    for values in ([], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="finite"):
            benchmark.distribution(values)


def test_waveform_hash_exactness() -> None:
    """Hash exact float32 output bytes, not rounded amplitudes or encoded WAVs."""
    samples = np.array([0.0, 0.1], dtype=np.float32)
    changed = samples.copy()
    changed[1] = np.nextafter(changed[1], np.float32(1))
    assert benchmark.waveform_hash(samples) == benchmark.waveform_hash(samples.copy())
    assert benchmark.waveform_hash(samples) != benchmark.waveform_hash(changed)


@pytest.mark.parametrize("engine", ["stt", "tts"])
@pytest.mark.parametrize("mode", ["retained", "release"])
def test_explicit_native_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str, mode: str
) -> None:
    """Forward exact roots, CPU mode, profiling policy and reproducible TTS seed.

    Args:
        tmp_path:
            Isolated fixture roots.

        monkeypatch:
            Replace optional heavyweight module imports with constructors.

        engine:
            Selected public speech engine.

        mode:
            Requested retained or release policy.

    """
    args = options(tmp_path)
    args.engine, args.mode = engine, mode
    module = ModuleType(f"hoast.{engine}2")
    config = MagicMock(return_value=object())
    constructor = MagicMock(return_value=object())
    setattr(module, engine.upper() + "Config", config)
    setattr(module, engine.upper(), constructor)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    benchmark.load_engine(args)
    kwargs = config.call_args.kwargs
    assert kwargs["model_path"] == getattr(args, f"{engine}_root")
    assert kwargs["backend"] == "cpu" and kwargs["threads"] == 2
    assert kwargs["profiling"] is False
    assert kwargs["release_after_call"] == (mode == "release")
    if engine == "tts":
        assert kwargs["seed"] == 7


def test_fixture_and_worker_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse exact FLOAT WAVs, retain nested profiles, and exclude warmups from aggregates.

    Args:
        tmp_path:
            Temporary diagnostics directory, containing only toy inputs.

        monkeypatch:
            Replace model loads, idle delays, memory and machine-specific affinity.

    """
    args = options(tmp_path)
    retained = FakeTTS(False)
    monkeypatch.setattr(benchmark, "load_engine", lambda _: retained)
    benchmark.generate_fixtures(args)
    assert retained.closed and retained.calls == len(benchmark.CASES)
    manifest = json.loads((tmp_path / "fixtures/manifest.json").read_text())
    assert manifest["seed"] == 7 and len(manifest["cases"]) == 6
    with pytest.raises(FileExistsError):
        benchmark.generate_fixtures(args)
    released = FakeTTS(True)
    args.mode = "release"
    monkeypatch.setattr(benchmark, "load_engine", lambda _: released)
    monkeypatch.setattr(
        benchmark, "affinity", lambda: {"process": [4, 5], "threads": []}
    )
    snapshot = {"rss": 100, "pss": 90, "uss": 80, "peak_rss": 120}
    monkeypatch.setattr(benchmark, "memory", lambda: snapshot)
    benchmark.run_worker(args)
    result = json.loads((tmp_path / "workers/tts-release-r0.json").read_text())
    assert result["status"] == "complete" and released.closed
    assert released.calls == 6 * (2 + 8)
    assert len([run for run in result["runs"] if run["phase"] == "warmup"]) == 12
    assert len([run for run in result["runs"] if run["phase"] == "measured"]) == 48
    assert all(run["release_seconds"] == 0.0001 for run in result["runs"])
    assert result["runs"][0]["timings"]["segments"] == [{"bert": 0.0002}]
    assert result["idle_after_warmup"] == snapshot
    assert len(list((tmp_path / "waveforms").glob("*.npy"))) == 6
    manifest["cases"][0]["file_sha256"] = "bad"
    benchmark.save_json(tmp_path / "fixtures/manifest.json", manifest)
    with pytest.raises(ValueError, match="Fixture changed"):
        benchmark.run_worker(args)


def raw_records() -> list[dict[str, Any]]:
    """Construct complete synthetic paired runs with deliberately slow warmups."""
    records = []
    for engine in ("stt", "tts"):
        for mode in ("retained", "release"):
            for round_number in range(2):
                runs = [
                    {
                        "case": name,
                        "phase": phase,
                        "total_seconds": seconds,
                        "release_seconds": 0.001 if mode == "release" else 0,
                        "output": {
                            "sha256": name,
                            "transcript": name,
                            "raw_transcript": name,
                        },
                    }
                    for name, _ in benchmark.CASES
                    for phase, seconds in (("warmup", 999.0), ("measured", 1.0))
                ]
                records.append(
                    {
                        "engine": engine,
                        "mode": mode,
                        "round": round_number,
                        "runs": runs,
                        "idle_after_repetitions": [
                            {
                                "phase": "measured",
                                "memory": {
                                    metric: 100 if mode == "retained" else 90
                                    for metric in ("rss", "pss", "uss", "peak_rss")
                                },
                            }
                        ],
                    }
                )
    return records


def test_aggregate_consistency_and_memory() -> None:
    """Qualify engines independently and preserve transcript/waveform disagreements."""
    records = raw_records()
    result = benchmark.aggregate(records)
    assert result["tts"]["qualified"] and result["stt"]["qualified"]
    assert result["stt"]["all_calls"]["retained_seconds"]["median"] == 1.0
    assert result["tts"]["cases"]["en-short"]["retained_seconds"]["count"] == 2
    assert result["stt"]["idle_saved_bytes"] == {"rss": 10.0, "pss": 10.0, "uss": 10.0}
    records[-1]["runs"][1]["output"]["sha256"] = "different"
    result = benchmark.aggregate(records)
    assert not result["tts"]["qualified"]
    assert result["stt"]["qualified"]
    assert result["tts"]["cases"]["en-short"]["output_identities"] == [
        "different",
        "en-short",
    ]


@pytest.mark.parametrize("resists_term", [False, True])
def test_worker_timeout_preserves_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resists_term: bool
) -> None:
    """Terminate only the owned group and save command/RNG metadata on timeout.

    Args:
        tmp_path:
            Isolated diagnostic destination.

        monkeypatch:
            Replace process creation and signals without starting real workers.

        resists_term:
            Require escalation to SIGKILL after the TERM grace period.

    """
    args = options(tmp_path)
    process = MagicMock(pid=123456)
    process.wait.side_effect = [subprocess.TimeoutExpired(["worker"], 0.1), 0]
    if resists_term:
        process.wait.side_effect = [
            subprocess.TimeoutExpired(["worker"], 0.1),
            subprocess.TimeoutExpired(["worker"], 10),
            -9,
        ]
    spawn = MagicMock(return_value=process)
    signal = MagicMock()
    monkeypatch.setattr(benchmark.subprocess, "Popen", spawn)
    monkeypatch.setattr(benchmark.os, "killpg", signal)
    with pytest.raises(subprocess.TimeoutExpired):
        benchmark.launch_worker(args, "stt", "release", 0)
    assert spawn.call_args.kwargs["start_new_session"]
    assert spawn.call_args.kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert spawn.call_args.kwargs["env"]["PYTHONHASHSEED"] == "7"
    assert signal.call_args_list[0].args == (123456, benchmark.signal.SIGTERM)
    assert signal.call_count == (2 if resists_term else 1)
    if resists_term:
        assert signal.call_args_list[1].args == (123456, benchmark.signal.SIGKILL)
    evidence = json.loads((tmp_path / "stt-release-r0-timeout.json").read_text())
    assert evidence["seed"] == 7 and "TimeoutExpired" in evidence["traceback"]
    assert (tmp_path / "stt-release-r0.log").exists()


def test_affinity_verifies_native_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a helper thread outside the exact budget rather than trust the leader.

    Args:
        monkeypatch:
            Supply synthetic native thread IDs and CPU masks without host dependencies.

    """
    process = MagicMock()
    process.threads.return_value = [MagicMock(id=11), MagicMock(id=12)]
    monkeypatch.setattr(benchmark.psutil, "Process", lambda: process)
    masks = {0: {4, 5}, 11: {4, 5}, 12: {4, 5}}
    monkeypatch.setattr(benchmark.os, "sched_getaffinity", lambda tid: masks[tid])
    assert benchmark.affinity()["process"] == [4, 5]
    masks[12] = {0, 1, 4, 5}
    with pytest.raises(RuntimeError, match="escaped affinity"):
        benchmark.affinity()


def test_failure_preserves_chains_and_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persist full chained tracebacks, notes and input/RNG metadata before rethrow.

    Args:
        tmp_path:
            Durable test diagnostics destination.

        monkeypatch:
            Replace CLI/resource setup and introduce a deliberate engine failure.

    """
    args = options(tmp_path)
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(benchmark.os, "sched_setaffinity", lambda *_: None)
    monkeypatch.setattr(benchmark, "configure_cpu_budget", lambda _: (4, 5))

    def fail(_: argparse.Namespace) -> None:
        """Raise a chained exception with a diagnostic note.

        Args:
            _:
                Unused worker arguments.

        """
        try:
            raise KeyError("original missing profile key")
        except KeyError as cause:
            error = ValueError("deliberate failure")
            error.add_note("retained TTS case and seed are relevant")
            raise error from cause

    monkeypatch.setattr(benchmark, "run_worker", fail)
    with pytest.raises(ValueError, match="deliberate"):
        benchmark.main()
    failure = json.loads(
        (tmp_path / "measure-tts-retained-r0-exception.json").read_text()
    )
    assert "KeyError" in failure["traceback"]
    assert "ValueError" in failure["traceback"]
    assert "retained TTS case and seed" in failure["traceback"]
    assert (
        failure["args"]["seed"] == 7
        and failure["rng_state"]["bit_generator"] == "PCG64"
    )
    assert len(failure["cases"]) == 6


@pytest.mark.parametrize(
    "option,value",
    [
        ("--warmups", "1"),
        ("--repeats", "7"),
        ("--rounds", "1"),
        ("--seed", "8"),
        ("--timeout", "nan"),
        ("--idle-seconds", "-1"),
    ],
)
def test_cli_rejects_invalid_budget(
    monkeypatch: pytest.MonkeyPatch, option: str, value: str
) -> None:
    """Reject insufficient repetitions, unsupported seeds and nonfinite deadlines.

    Args:
        monkeypatch:
            Replace process CLI arguments.

        option:
            CLI setting to invalidate.

        value:
            Invalid setting value.

    """
    monkeypatch.setattr(sys, "argv", ["benchmark", option, value])
    with pytest.raises(SystemExit):
        benchmark.parse_args()
