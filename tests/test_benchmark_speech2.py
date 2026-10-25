"""Model-free benchmark provenance, backend forwarding and owned-group cleanup tests."""

import argparse
import json
import os
import select
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import psutil
import pytest

from tools import benchmark_speech2 as benchmark


def _args(
    tmp_path: Path, model: str = "melo", backend: str = "cpu"
) -> argparse.Namespace:
    """Create bounded native options using only isolated fixture paths.

    Args:
        tmp_path:
            Pytest-owned cache/output root.

        model:
            Model label, without constructing any real engine.

        backend:
            Requested CPU/hybrid/GPU label.

    """
    return argparse.Namespace(
        model=model,
        backend=backend,
        threads=2,
        cache=tmp_path / "cache",
        model_dir=tmp_path / "models" / ("stt2" if model == "sensevoice" else "tts2"),
        profile=False,
        gpu_precision="f16",
        tag="fixture",
        repeats=2,
        timeout=5,
        memory_gib=1,
        worker=True,
    )


@pytest.mark.parametrize(
    "model,backend,valid",
    [
        ("whisper", "cpu", False),
        ("whisper", "GPU", False),
        ("whisper", "hybrid", False),
        ("kokoro", "cpu", False),
        ("kokoro", "hybrid", False),
        ("kokoro", "GPU", False),
        ("sensevoice", "cpu", True),
        ("sensevoice", "GPU", True),
        ("sensevoice", "hybrid", False),
        ("melo", "GPU", True),
    ],
)
def test_supported_combinations(
    tmp_path: Path, model: str, backend: str, valid: bool
) -> None:
    """Do not silently record a backend that the selected engine cannot execute.

    Args:
        tmp_path:
            Isolated option paths.

        model:
            Speech engine label.

        backend:
            Requested backend.

        valid:
            Whether the combination should pass static validation.

    """
    args = _args(tmp_path, model, backend)
    if valid:
        benchmark.validate_options(args)
    else:
        with pytest.raises(ValueError, match="Unsupported"):
            benchmark.validate_options(args)


def test_native_version_and_profile_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject alternate interpreters without constructing any speech model.

    Args:
        tmp_path:
            Isolated option paths.

        monkeypatch:
            Replaces only the benchmark's version metadata.

    """
    args = _args(tmp_path, "sensevoice")
    monkeypatch.setattr(benchmark.sys, "version_info", (3, 12, 0))
    with pytest.raises(RuntimeError, match="3.14"):
        benchmark.validate_options(args)


def test_sensevoice_backend_and_profile_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock optional imports and verify the engine receives the requested GPU/profile.

    Args:
        tmp_path:
            Isolated option paths.

        monkeypatch:
            Injects a tiny fake module, never importing OpenVINO model code.

    """
    module = ModuleType("hoast.stt2")
    config = MagicMock()
    engine = MagicMock()
    module.STTConfig = config  # pyright: ignore[reportAttributeAccessIssue]
    module.STT = engine  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "hoast.stt2", module)
    args = _args(tmp_path, "sensevoice", "GPU")
    args.profile = True
    result = benchmark.load_engine(args)
    config.assert_called_once_with(
        model_path=args.model_dir, threads=2, backend="GPU", profiling=True
    )
    assert result is engine.return_value


def test_artifact_identity_and_mutation_detection(tmp_path: Path) -> None:
    """Hash tiny model/frontend files and catch same-size content replacement metadata.

    Args:
        tmp_path:
            Isolated artifact directory; no external datasets or checkpoints.

    """
    first = tmp_path / "model.bin"
    second = tmp_path / "frontend.txt"
    first.write_bytes(b"abc")
    second.write_text("phonemes")
    paths = [first, second]
    identities, before = benchmark.hash_artifacts(paths)
    assert identities[str(first)]["bytes"] == 3
    assert len(identities[str(second)]["sha256"]) == 64
    previous = second.stat()
    second.write_text("phonemez")
    os.utime(second, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert benchmark.file_metadata(paths) != before


def test_sensevoice_inventory_ignores_unrelated_cache(tmp_path: Path) -> None:
    """Select native-model artifacts without capturing secret/cache neighbors.

    Args:
        tmp_path:
            Isolated cache with tiny embedded files and an unrelated secret sentinel.

    """
    args = _args(tmp_path, "sensevoice")
    root = args.model_dir
    root.mkdir(parents=True)
    args.cache.mkdir(parents=True)
    for name in ("model.bin", "model.xml", "tokenizer.model", "am.mvn"):
        (root / name).write_text("fixture")
    (args.cache / "credentials.json").write_text("do not inventory")
    files = benchmark.artifact_files(args)
    assert root / "model.bin" in files
    assert args.cache / "credentials.json" not in files
    args.model_dir = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        benchmark.artifact_files(args)


@dataclass(slots=True)
class _Config:
    """Minimal public speech settings for the fake waveform engine."""

    threads: int = 2
    """Fixture worker count."""

    seed: int = 1234
    """Fixture deterministic synthesis seed."""


class _Engine:
    """Return small finite audio while recording exact existing benchmark calls."""

    config: _Config
    """Public serializable fixture controls."""

    calls: list[str]
    """Every synthesis input in call order."""

    closed: bool
    """Whether deterministic cleanup ran."""

    def __init__(self) -> None:
        """Initialize model-free synthesis ownership."""
        self.config = _Config()
        self.calls = []
        self.closed = False

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return one millisecond of non-clipped mono FP32 audio.

        Args:
            text:
                Recorded benchmark sentence; pronunciation is deliberately not modeled.

        """
        self.calls.append(text)
        return np.full(24, 0.1, dtype=np.float32), 24000

    def close(self) -> None:
        """Mark fixture ownership released without touching any real runtime."""
        self.closed = True


@pytest.mark.parametrize("override", [False, True])
def test_worker_readiness_without_extra_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: bool
) -> None:
    """Readiness follows existing EN/ZH warm calls and provenance excludes timing.

    Args:
        tmp_path:
            Isolated output and tiny identity file.

        monkeypatch:
            Replaces engine, CPU policy and diagnostics with model-free fixtures.

        override:
            Whether to select an explicit model root instead of the permanent default.

    """
    args = _args(tmp_path)
    monkeypatch.chdir(tmp_path)
    if not override:
        args.model_dir = None
    model_root = args.model_dir or Path("models/tts2")
    model_root.mkdir(parents=True)
    manifest = model_root / "manifest.json"
    manifest.write_text("{}")
    identity = tmp_path / "model.bin"
    identity.write_bytes(b"fixture")
    engine = _Engine()
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "configure_cpu_budget", lambda threads: (0, 1))
    monkeypatch.setattr(benchmark, "artifact_files", lambda options: [identity])
    monkeypatch.setattr(benchmark, "load_engine", lambda options: engine)
    monkeypatch.setattr(benchmark, "memory", lambda: {"rss": 1, "uss": 1, "pss": 1})
    monkeypatch.setattr(benchmark, "masks", lambda: {"allowed_cpus": [0, 1]})
    benchmark.worker(args)
    record = json.loads((tmp_path / "results/speech2/fixture.json").read_text())
    assert len(engine.calls) == len(benchmark.CASES) * (args.repeats + 1)
    assert engine.closed and record["status"] == "completed"
    assert set(record["readiness"]) == {"en", "zh"}
    assert record["matched_en_zh_readiness"]["after_case"] == "zh-short"
    assert "construction_seconds" in record and "load_seconds" not in record
    assert record["artifacts_unchanged"] is True
    assert record["artifact_manifests"] == {
        "manifest.json": benchmark.hashlib.sha256(manifest.read_bytes()).hexdigest()
    }
    assert record["effective_config"] == {"threads": 2, "seed": 1234}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux owned sessions/subreaper")
def test_term_resistant_orphan_cleanup_without_unrelated_signals() -> None:
    """Reap an embedded TERM-resistant child after root exit, preserving another group."""
    previous = benchmark.set_subreaper(True)
    child_code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(30)"
    root_code = (
        "import subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}],stdout=subprocess.PIPE,text=True); "
        "child.stdout.readline(); print(child.pid,flush=True); time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", root_code],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    created = psutil.Process(process.pid).create_time()
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
    )
    try:
        assert process.stdout is not None
        assert select.select([process.stdout], [], [], 5)[0]
        child_pid = int(process.stdout.readline())
        process.terminate()
        process.wait(timeout=5)
        assert psutil.pid_exists(child_pid)
        code, cleaned, reaped = benchmark.cleanup_owned_group(
            process, created, grace_seconds=0.15
        )
        assert code == -15 and cleaned and reaped
        assert not psutil.pid_exists(child_pid)
        assert not benchmark.group_members(process.pid)
        assert unrelated.poll() is None
    finally:
        benchmark.cleanup_owned_group(process, created, grace_seconds=0.15)
        unrelated.terminate()
        unrelated.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        benchmark.set_subreaper(previous)


def test_cleanup_refuses_nonisolated_worker() -> None:
    """Never signal the caller's shared session/group for a nonisolated child."""
    process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    created = psutil.Process(process.pid).create_time()
    try:
        with pytest.raises(RuntimeError, match="session identity"):
            benchmark.cleanup_owned_group(process, created, grace_seconds=0.1)
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)
