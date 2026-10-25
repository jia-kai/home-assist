"""Benchmark native speech2 with bounded fresh-process CPU/RAM trials."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import soundfile as sf

from hoast.logging import configure_logging, get_logger
from hoast.runtime import configure_cpu_budget

logger = get_logger(__name__)
ROOT = Path(".cache/speech-bench")
CASES = (
    ("en-short", "The lights are on. What else can I help you with?"),
    (
        "en-long",
        "Tomorrow will be partly cloudy, with a high of twenty degrees and a ten percent chance of rain.",
    ),
    ("zh-short", "灯已经打开了。还有什么需要我帮忙的吗？"),
    ("zh-long", "明天多云，最高气温二十度，降雨概率百分之十。出门前请带好钥匙。"),
    ("mixed-short", "帮我打开 living room 的灯。"),
    ("mixed-long", "客厅的 lights 已经打开了，请把 kitchen 的灯也打开。"),
)
ARTIFACT_SUFFIXES = {
    ".bin",
    ".xml",
    ".onnx",
    ".json",
    ".txt",
    ".model",
    ".mvn",
    ".npy",
    ".npz",
    ".rep",
    ".en",
    ".py",
    ".cl",
    ".pickle",
    ".yaml",
    ".yml",
    ".so",
    ".dat",
}


def validate_options(args: argparse.Namespace) -> None:
    """Reject non-native execution and unsupported model/backend/profile combinations.

    Args:
        args:
            Parsed benchmark settings, also checked by direct worker callers.

    """
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("Speech benchmarks require native Python 3.14")
    supported = {
        "sensevoice": {"cpu", "GPU", "ort"},
        "melo": {"cpu", "hybrid", "GPU"},
    }
    if args.model not in supported or args.backend not in supported[args.model]:
        raise ValueError(f"Unsupported model/backend: {args.model}/{args.backend}")


def artifact_files(args: argparse.Namespace) -> list[Path]:
    """Select model, voice, frontend data/code files without scanning unrelated cache.

    Logs, credentials, outputs and bytecode are not inventoried.

    Args:
        args:
            Validated native model/cache/variant settings.

    """
    roots: list[Path] = []
    explicit: list[Path] = []
    project = Path(__file__).resolve().parents[1]
    package = project / "hoast"
    explicit.extend(
        package / name for name in ("speech_text.py", "input_text.py", "runtime.py")
    )
    explicit.append(Path(__file__).resolve())
    name = "stt2" if args.model == "sensevoice" else "tts2"
    root = args.model_dir or Path("models") / name
    if not root.is_dir():
        raise FileNotFoundError(root)
    roots.extend((root, package / name))
    selected = {path.absolute() for path in explicit}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        selected.update(
            path.absolute()
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in ARTIFACT_SUFFIXES
            and "__pycache__" not in path.parts
        )
    for path in selected:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(selected)


def file_metadata(paths: list[Path]) -> dict[str, dict[str, int]]:
    """Snapshot cheap size/timestamp/inode metadata without reading file contents.

    Args:
        paths:
            Explicit artifact inventory, whose absolute paths become evidence keys.

    """
    result = {}
    for path in paths:
        info = path.stat()
        result[str(path)] = {
            "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
            "device": info.st_dev,
            "inode": info.st_ino,
        }
    return result


def hash_artifacts(
    paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    """Hash selected artifacts once and reject mutation during the excluded-time scan.

    Args:
        paths:
            Model/voice/frontend inventory; hashing occurs before construction/inference.

    """
    before = file_metadata(paths)
    identity = {}
    for path in paths:
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        identity[str(path)] = {"bytes": before[str(path)]["bytes"], "sha256": checksum}
    if file_metadata(paths) != before:
        raise RuntimeError("Artifact changed while computing its identity")
    return identity, before


def effective_config(config: Any) -> dict[str, Any]:
    """Serialize only declared speech configuration fields, without ambient environment.

    Args:
        config:
            Speech engine dataclass containing public paths and synthesis/decoder controls.

    """
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError("Speech engine config must be a dataclass")
    return {
        field.name: str(value.absolute())
        if isinstance(value := getattr(config, field.name), Path)
        else value
        for field in fields(config)
    }


def load_engine(args: argparse.Namespace) -> Any:
    """Construct exactly the requested optional engine after excluded artifact hashing.

    Args:
        args:
            Validated backend/profile/resource settings, with no implicit fallback.

    """
    # Model dependencies remain lazy so policy and supervisor tests load no models.
    if args.model == "sensevoice":
        from hoast.stt2 import STT, STTConfig

        return STT(
            STTConfig(
                model_path=args.model_dir or Path("models/stt2"),
                threads=args.threads,
                backend=args.backend,
                profiling=args.profile,
            )
        )
    from hoast.tts2 import TTS, TTSConfig

    return TTS(
        TTSConfig(
            model_path=args.model_dir or Path("models/tts2"),
            threads=args.threads,
            backend=args.backend,
            seed=1234,
            profiling=args.profile,
            gpu_precision=args.gpu_precision,
        )
    )


def set_subreaper(enabled: bool) -> bool:
    """Set Linux orphan adoption for bounded owned-group cleanup, returning prior state.

    Args:
        enabled:
            Whether this supervisor should adopt descendants whose worker parent exits.

    """
    library = ctypes.CDLL(None, use_errno=True)
    current = ctypes.c_int()
    if library.prctl(37, ctypes.byref(current), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
    if library.prctl(36, int(enabled), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    return bool(current.value)


def group_members(group: int) -> list[psutil.Process]:
    """Return live/zombie members of exactly one session-owned process group.

    Args:
        group:
            Worker PID used as PGID and SID by Popen(start_new_session=True).

    """
    result = []
    for process in psutil.process_iter():
        try:
            if os.getpgid(process.pid) == group and os.getsid(process.pid) == group:
                result.append(process)
        except ProcessLookupError:
            continue
    return result


def cleanup_owned_group(
    process: subprocess.Popen[Any], leader_created: float, grace_seconds: float = 10
) -> tuple[int, bool, bool]:
    """Terminate/reap only a worker's isolated group, including TERM-resistant orphans.

    Caller enables subreaper before spawning. Negative-PGID waitpid reaps only
    adopted children from this owned group, never unrelated application children.
    PID reuse or a worker not created in its own session fails before signaling.

    Args:
        process:
            Owned Popen worker launched with start_new_session=True.

        leader_created:
            Worker creation timestamp captured immediately after Popen, before polling.

        grace_seconds:
            Positive TERM grace and post-KILL cleanup deadline, in seconds.

    """
    if not math.isfinite(grace_seconds) or grace_seconds <= 0:
        raise ValueError("Cleanup grace must be positive and finite")
    group = process.pid
    if group == os.getpgrp():
        raise RuntimeError("Refusing to signal supervisor's process group")
    try:
        leader = psutil.Process(group)
        if (
            leader.create_time() != leader_created
            or os.getpgid(group) != group
            or os.getsid(group) != group
        ):
            raise RuntimeError("Worker session identity changed; refusing group signal")
    except psutil.NoSuchProcess, ProcessLookupError:
        pass

    def signal_group(number: int) -> None:
        """Signal only the verified worker session if members remain.

        Args:
            number:
                POSIX signal sent to the owned process group.

        """
        if group_members(group):
            try:
                os.killpg(group, number)
            except ProcessLookupError:
                pass

    def reap() -> None:
        """Collect exited worker/adopted children while preserving Popen exit status."""
        process.poll()
        while True:
            try:
                pid, status = os.waitpid(-group, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            if pid == process.pid:
                process.returncode = os.waitstatus_to_exitcode(status)

    signal_group(signal.SIGTERM)
    for escalation in (False, True):
        if escalation:
            signal_group(signal.SIGKILL)
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            reap()
            if not group_members(group):
                code = process.wait()
                return code, True, True
            time.sleep(0.02)
    reap()
    code = process.poll()
    if code is None:
        raise RuntimeError("Owned worker did not exit after SIGKILL")
    return code, not group_members(group), False


def service_state() -> dict[str, Any]:
    """Read only the named Hoast container's identity/status, without credentials."""
    identity = subprocess.check_output(
        ["docker", "inspect", "hoast", "--format", "{{.Id}}"], text=True
    ).strip()
    state = json.loads(
        subprocess.check_output(
            ["docker", "inspect", identity, "--format", "{{json .State}}"], text=True
        )
    )
    return {
        "id": identity,
        "running": state["Running"],
        "started_at": state["StartedAt"],
    }


def memory() -> dict[str, int]:
    """Snapshot the current process's resident/private/proportional memory in bytes."""
    info = psutil.Process().memory_full_info()
    return {"rss": info.rss, "uss": info.uss, "pss": info.pss}


def masks() -> dict[str, Any]:
    """Inspect actual Linux native-worker affinity after lazy initialization."""
    allowed = set(os.sched_getaffinity(0))
    workers = []
    for thread in psutil.Process().threads():
        try:
            affinity = set(os.sched_getaffinity(thread.id))
        except ProcessLookupError:
            continue
        workers.append({"id": thread.id, "cpus": sorted(affinity)})
        if not affinity <= allowed:
            raise RuntimeError(f"Worker {thread.id} escaped CPU budget: {affinity}")
    return {"allowed_cpus": sorted(allowed), "workers": workers}


def worker(args: argparse.Namespace) -> None:
    """Load one engine, time full calls, and save incremental result/audio evidence.

    Args:
        args:
            Validated CLI options selecting model/backend, artifacts and trial label.
    """
    validate_options(args)
    configure_cpu_budget(args.threads)
    paths = artifact_files(args)
    started = time.perf_counter()
    identities, metadata_before = hash_artifacts(paths)
    provenance_seconds = time.perf_counter() - started
    started = time.perf_counter()
    engine = load_engine(args)
    construction_seconds = time.perf_counter() - started
    result_path = ROOT / "results/speech2" / f"{args.tag}.json"
    output = ROOT / "outputs/speech2" / args.tag
    output.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {
        "model": args.model,
        "backend": args.backend,
        "threads": args.threads,
        "schema_version": 2,
        "construction_seconds": construction_seconds,
        "construction_memory": memory(),
        "construction_semantics": "constructor only; lazy language graphs/frontend compilation is included in existing first calls",
        "provenance_scan_seconds": provenance_seconds,
        "construction_io_policy": "artifact hashing precedes construction and warms filesystem caches; not a disk-cold load metric",
        "cache": str(args.cache.absolute()),
        "effective_config": effective_config(engine.config),
        "artifact_identity": identities,
        "artifact_metadata_before": metadata_before,
        "artifact_scope": "selected runtime model/voice/frontend data and source; excludes secrets/logs/bytecode; installed dependency packages are not a full environment lock",
        "python_version": sys.version,
        "profile_enabled": args.profile,
        "gpu_precision": args.gpu_precision,
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "stochastic_policy": "Melo NumPy seed 1234 per utterance"
        if args.model == "melo"
        else None,
        "status": "incomplete",
        "runs": [],
        "inputs": [],
        "readiness": {},
    }
    record["native_package_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "openvino", "soundfile", "psutil")
    }
    if args.model in ("sensevoice", "melo"):
        artifact_root = args.model_dir or Path("models") / (
            "stt2" if args.model == "sensevoice" else "tts2"
        )
        record["artifact_manifests"] = {
            str(path.relative_to(artifact_root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(artifact_root.rglob("manifest.json"))
        }
    cases: list[tuple[str, Any, int | None]] = []
    if args.model == "sensevoice":
        for language in ("en", "zh"):
            path = ROOT / "inputs" / f"asr_{language}.wav"
            audio, rate = sf.read(path, dtype="float32")
            assert audio.ndim == 1 and audio.size > 0
            cases.append((language, audio, rate))
            record["inputs"].append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "sample_rate": rate,
                    "duration_seconds": audio.size / rate,
                }
            )
    else:
        cases.extend((name, text, None) for name, text in CASES)
    record["case_inventory"] = [name for name, _, _ in cases]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    inference_started = time.perf_counter()
    try:
        for name, value, rate in cases:
            for repetition in range(args.repeats + 1):
                started = time.perf_counter()
                if rate is not None:
                    transcript = engine.transcribe_samples(value, rate)
                    elapsed = time.perf_counter() - started
                    run: dict[str, Any] = {
                        "case": name,
                        "repetition": repetition,
                        "warm": repetition > 0,
                        "seconds": elapsed,
                        "transcript": transcript,
                        "raw_text": engine.last_raw_text,
                    }
                else:
                    audio, audio_rate = engine.synthesize(value)
                    elapsed = time.perf_counter() - started
                    if audio.dtype != np.float32 or audio.ndim != 1 or not audio.size:
                        raise RuntimeError("Invalid waveform shape/dtype")
                    if not np.isfinite(audio).all():
                        raise RuntimeError("Nonfinite waveform")
                    rms = float(np.sqrt(np.mean(audio**2)))
                    clipped = float(np.mean(np.abs(audio) >= 0.999))
                    if rms <= 1e-4 or clipped >= 0.001:
                        raise RuntimeError(
                            f"Waveform gate failed rms={rms} clipping={clipped}"
                        )
                    wav = output / f"{name}-{repetition}.wav"
                    sf.write(wav, audio, audio_rate, subtype="PCM_16")
                    duration = audio.size / audio_rate
                    run = {
                        "case": name,
                        "text": value,
                        "repetition": repetition,
                        "warm": repetition > 0,
                        "seconds": elapsed,
                        "sample_rate": audio_rate,
                        "audio_seconds": duration,
                        "rtf": elapsed / duration,
                        "wav": str(wav),
                        "rms": rms,
                        "clipped_fraction": clipped,
                        "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                    }
                run["memory"] = memory()
                run["stage_profile"] = getattr(engine, "last_timings", None)
                if args.model == "sensevoice" and args.profile:
                    run["node_profile"] = engine.last_profile
                run["affinity"] = masks()
                record["runs"].append(run)
                language = name.split("-", 1)[0]
                if repetition == 1 and language in {"en", "zh"}:
                    record["readiness"].setdefault(
                        language,
                        {
                            "after_case": name,
                            "after_repetition": repetition,
                            "memory": run["memory"],
                            "elapsed_since_first_case_seconds": time.perf_counter()
                            - inference_started,
                        },
                    )
                    if set(record["readiness"]) == {"en", "zh"}:
                        record.setdefault(
                            "matched_en_zh_readiness",
                            {
                                "after_case": name,
                                "after_repetition": repetition,
                                "memory": run["memory"],
                                "semantics": "resident after existing first warm EN and ZH calls; includes earlier measured calls, no added workload",
                            },
                        )
                record["peak_rss_kib"] = resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                result_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n"
                )
                logger.info(
                    "benchmark model=%s case=%s repetition=%d seconds=%.4f",
                    args.model,
                    name,
                    repetition,
                    elapsed,
                )
        after_paths = artifact_files(args)
        record["artifact_metadata_after"] = file_metadata(after_paths)
        record["artifacts_unchanged"] = (
            paths == after_paths
            and metadata_before == record["artifact_metadata_after"]
        )
        if not record["artifacts_unchanged"]:
            raise RuntimeError(
                "Runtime artifact inventory or metadata changed during trial"
            )
        record["status"] = "completed"
    except BaseException:
        record["status"] = "failed"
        record["exception"] = traceback.format_exc()
        raise
    finally:
        try:
            if hasattr(engine, "close"):
                engine.close()
        finally:
            record["final_memory"] = memory()
            record["final_memory_semantics"] = (
                "after optional engine.close; allocator retention and models without close differ"
            )
            result_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n"
            )


def supervise(args: argparse.Namespace) -> None:
    """Bound worker group time/RAM, sample process-tree RSS, and always reap it.

    Args:
        args:
            Validated CLI options shared with the child worker.
    """
    validate_options(args)
    result_dir = ROOT / "results/speech2"
    log_dir = ROOT / "logs/speech2"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{args.tag}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    before_service = service_state()
    if before_service["running"]:
        raise RuntimeError("Stop Hoast and its systemd restarter before measurement")
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    command = [
        sys.executable,
        "-m",
        "tools.benchmark_speech2",
        "--worker",
        "--model",
        args.model,
        "--backend",
        args.backend,
        "--threads",
        str(args.threads),
        "--cache",
        str(args.cache),
        "--tag",
        args.tag,
        "--repeats",
        str(args.repeats),
        "--gpu-precision",
        args.gpu_precision,
    ]
    if args.model_dir is not None:
        command.extend(("--model-dir", str(args.model_dir)))
    if args.profile:
        command.append("--profile")
    peak = 0
    available = psutil.virtual_memory().available
    reason = "completed"
    started = time.monotonic()
    group_cleaned = False
    descendants_reaped = False
    with (log_dir / f"{args.tag}.log").open("w") as log:
        previous_subreaper = set_subreaper(True)
        try:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            set_subreaper(previous_subreaper)
            raise
        leader_created = psutil.Process(process.pid).create_time()
        try:
            root = psutil.Process(process.pid)
            while process.poll() is None:
                if time.monotonic() - started > args.timeout:
                    reason = "timeout"
                    break
                try:
                    members = [root, *root.children(recursive=True)]
                    rss = sum(member.memory_info().rss for member in members)
                except psutil.NoSuchProcess:
                    time.sleep(0.05)
                    continue
                peak = max(peak, rss)
                available = min(available, psutil.virtual_memory().available)
                if rss > args.memory_gib * 1024**3 or available < 2 * 1024**3:
                    reason = "RAM protection"
                    break
                time.sleep(0.05)
        finally:
            try:
                code, group_cleaned, descendants_reaped = cleanup_owned_group(
                    process, leader_created
                )
            finally:
                set_subreaper(previous_subreaper)
    record = json.loads(result_path.read_text()) if result_path.exists() else {}
    after_service = service_state()
    service_isolated = before_service == after_service and not after_service["running"]
    record.update(
        exit_code=code,
        process_reaped=True,
        process_group_cleaned=group_cleaned,
        descendants_reaped=descendants_reaped,
        peak_tree_rss_bytes=peak,
        min_available_bytes=available,
        process_wall_seconds=time.monotonic() - started,
        supervisor_outcome=reason,
        hoast_before=before_service,
        hoast_after=after_service,
        hoast_unchanged_and_stopped=service_isolated,
    )
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    if (
        code != 0
        or reason != "completed"
        or record.get("status") != "completed"
        or not service_isolated
        or not group_cleaned
        or not descendants_reaped
    ):
        raise RuntimeError(
            f"Benchmark failed {args.tag}: {reason}, exit={code}; see durable log"
        )


def main() -> None:
    """Parse bounded benchmark settings; refuse accidental artifact overwrites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("sensevoice", "melo"), required=True)
    parser.add_argument(
        "--backend", choices=("cpu", "hybrid", "GPU", "ort"), default="cpu"
    )
    parser.add_argument("--threads", type=int, choices=(1, 2), required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/hoast"))
    parser.add_argument(
        "--model-dir", type=Path, help="Explicit prepared speech2 variant directory"
    )
    parser.add_argument("--gpu-precision", choices=("f16", "f32"), default="f16")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable node counters; exclude from headline timing",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not args.tag or any(not c.isalnum() and c not in "-_" for c in args.tag):
        raise ValueError(
            "Tag must contain only letters, numbers, hyphens and underscores"
        )
    if args.repeats < 2 or not all(
        math.isfinite(v) and v > 0 for v in (args.timeout, args.memory_gib)
    ):
        raise ValueError(
            "At least two warm calls and finite positive budgets are required"
        )
    validate_options(args)
    if args.worker:
        worker(args)
    else:
        supervise(args)


if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except Exception:
        logger.exception("speech2.benchmark status=failed")
        raise
