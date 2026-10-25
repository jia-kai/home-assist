"""Reproducible, offline CPU speech intermediate-memory release qualification."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import soundfile as sf

from hoast.runtime import configure_cpu_budget

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT = Path(".cache/hoast/diagnostics/memory-release-20261007")
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


def save_json(path: Path, value: Any) -> None:
    """Atomically preserve JSON evidence, including incomplete worker progress.

    Args:
        path:
            Destination in the diagnostics directory.

        value:
            JSON-safe record; nonfinite numbers are rejected.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def file_hash(path: Path) -> str:
    """Return SHA256 of a complete file outside inference timing.

    Args:
        path:
            Existing fixture, model or source file.

    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def waveform_hash(samples: np.ndarray[Any, Any]) -> str:
    """Hash exact C-order waveform bytes, with dtype/shape recorded separately.

    Args:
        samples:
            Mono float32 waveform shaped (samples,), amplitudes dimensionless.

    """
    return hashlib.sha256(samples.tobytes(order="C")).hexdigest()


def memory() -> dict[str, int]:
    """Return Linux RSS/PSS/USS and lifetime ru_maxrss peak in bytes."""
    info = psutil.Process().memory_full_info()
    return {
        "rss": info.rss,
        "pss": info.pss,
        "uss": info.uss,
        "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def affinity() -> dict[str, Any]:
    """Verify every live native thread inherited the exact two-CPU budget."""
    masks = []
    for thread in psutil.Process().threads():
        try:
            cpus = sorted(os.sched_getaffinity(thread.id))
        except ProcessLookupError:
            continue
        if cpus != [4, 5]:
            raise RuntimeError(f"Native thread escaped affinity: {thread.id}: {cpus}")
        masks.append({"tid": thread.id, "cpus": cpus})
    return {"process": sorted(os.sched_getaffinity(0)), "threads": masks}


def native_environment(seed: int) -> dict[str, str]:
    """Construct explicit inherited CPU/RNG/offline budgets before worker imports.

    Args:
        seed:
            Nonnegative reproducible per-utterance RNG and Python hash seed.

    """
    return {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "MAX_JOBS": "1",
        "PYTHONHASHSEED": str(seed),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def load_engine(args: argparse.Namespace) -> Any:
    """Load only the selected native CPU engine with explicit opt-in cleanup.

    Args:
        args:
            Validated worker paths, engine, mode and seed; profiles stay disabled.

    """
    # Optional heavyweight speech dependencies stay lazy for model-free tests.
    if args.engine == "stt":
        from hoast.stt2 import STT, STTConfig

        return STT(
            STTConfig(
                model_path=args.stt_root,
                threads=2,
                backend="cpu",
                profiling=False,
                release_after_call=args.mode == "release",
            )
        )
    from hoast.tts2 import TTS, TTSConfig

    return TTS(
        TTSConfig(
            model_path=args.tts_root,
            threads=2,
            backend="cpu",
            seed=args.seed,
            profiling=False,
            release_after_call=args.mode == "release",
        )
    )


def idle_memory(seconds: float) -> dict[str, int]:
    """Wait without trimming allocators or forcing GC, then snapshot memory.

    Args:
        seconds:
            Nonnegative idle interval excluded from inference latency.

    """
    time.sleep(seconds)
    return memory()


def generate_fixtures(args: argparse.Namespace) -> None:
    """Generate durable FLOAT WAV inputs once with retained seeded TTS.

    Args:
        args:
            Fixture worker settings; a preexisting manifest is never overwritten.

    """
    manifest_path = args.output / "fixtures/manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    engine = load_engine(args)
    records = []
    try:
        for name, text in CASES:
            save_json(
                args.output / "fixture-progress.json",
                {"case": name, "text": text, "seed": args.seed},
            )
            samples, rate = engine.synthesize(text)
            path = args.output / "fixtures" / f"{name}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, samples, rate, subtype="FLOAT")
            reread, reread_rate = sf.read(path, dtype="float32")
            if rate != reread_rate or not np.array_equal(samples, reread):
                raise RuntimeError("FLOAT WAV fixture did not preserve exact samples")
            records.append(
                {
                    "case": name,
                    "text": text,
                    "path": str(path.absolute()),
                    "samples": int(samples.size),
                    "sample_rate": rate,
                    "duration_seconds": samples.size / rate,
                    "dtype": str(samples.dtype),
                    "waveform_sha256": waveform_hash(samples),
                    "file_sha256": file_hash(path),
                }
            )
            save_json(
                args.output / "fixture-progress.json",
                {"completed": records, "seed": args.seed},
            )
        save_json(
            manifest_path, {"seed": args.seed, "mode": "retained", "cases": records}
        )
    finally:
        engine.close()


def run_worker(args: argparse.Namespace) -> None:
    """Time complete calls including cleanup/reallocation and preserve raw outputs.

    Args:
        args:
            One engine/mode/round with identical saved WAVs and per-case repetitions.

    """
    if args.worker == "fixtures":
        generate_fixtures(args)
        return
    fixture = json.loads((args.output / "fixtures/manifest.json").read_text())
    if fixture["seed"] != args.seed:
        raise ValueError("Fixture seed differs from worker seed")
    inputs = {}
    for case in fixture["cases"]:
        path = Path(case["path"])
        if file_hash(path) != case["file_sha256"]:
            raise ValueError(f"Fixture changed: {path}")
        samples, rate = sf.read(path, dtype="float32")
        if waveform_hash(samples) != case["waveform_sha256"]:
            raise ValueError(f"Fixture waveform changed: {path}")
        inputs[case["case"]] = (samples, rate)
    started = time.perf_counter()
    engine = load_engine(args)
    record: dict[str, Any] = {
        "schema_version": 1,
        "engine": args.engine,
        "mode": args.mode,
        "round": args.round,
        "seed": args.seed,
        "status": "incomplete",
        "pid": os.getpid(),
        "python": sys.version,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(engine.config).items()
        },
        "constructor_seconds": time.perf_counter() - started,
        "baseline_after_constructor": memory(),
        "baseline_semantics": "STT graph loaded; TTS language graphs are lazy and included in first warmups",
        "runs": [],
        "idle_after_repetitions": [],
        "fixtures": fixture,
    }
    result_path = (
        args.output / "workers" / f"{args.engine}-{args.mode}-r{args.round}.json"
    )
    evidence = args.output / "waveforms"
    evidence.mkdir(exist_ok=True)
    save_json(result_path, record)
    try:
        for phase, count in (("warmup", args.warmups), ("measured", args.repeats)):
            for repetition in range(count):
                for name, text in CASES:
                    record["active_input"] = {
                        "case": name,
                        "text": text,
                        "phase": phase,
                        "repetition": repetition,
                        "seed": args.seed,
                    }
                    save_json(result_path, record)
                    started = time.perf_counter()
                    if args.engine == "tts":
                        output, rate = engine.synthesize(text)
                    else:
                        samples, rate = inputs[name]
                        output = engine.transcribe_samples(samples, rate)
                    elapsed = time.perf_counter() - started
                    timings = json.loads(json.dumps(engine.last_timings))
                    if args.mode == "release" and "release_memory" not in timings:
                        raise KeyError("release_memory timing is missing")
                    if "total" not in timings:
                        raise KeyError("total timing is missing")
                    if args.engine == "tts":
                        digest = waveform_hash(output)
                        waveform_path = evidence / f"{name}-{digest}.npy"
                        if not waveform_path.exists():
                            np.save(waveform_path, output, allow_pickle=False)
                        identity = {
                            "sha256": digest,
                            "samples": int(output.size),
                            "sample_rate": rate,
                            "dtype": str(output.dtype),
                            "waveform_path": str(waveform_path.absolute()),
                        }
                    else:
                        identity = {
                            "transcript": output,
                            "raw_transcript": engine.last_raw_text,
                        }
                    del output
                    record["runs"].append(
                        {
                            "case": name,
                            "phase": phase,
                            "repetition": repetition,
                            "total_seconds": elapsed,
                            "release_seconds": timings.get("release_memory", 0.0),
                            "timings": timings,
                            "output": identity,
                        }
                    )
                    logger.info(
                        "call engine=%s mode=%s round=%d phase=%s repetition=%d case=%s seconds=%.4f",
                        args.engine,
                        args.mode,
                        args.round,
                        phase,
                        repetition,
                        name,
                        elapsed,
                    )
                    save_json(result_path, record)
                snapshot = {
                    "phase": phase,
                    "repetition": repetition,
                    "memory": idle_memory(args.idle_seconds),
                    "affinity": affinity(),
                }
                record["idle_after_repetitions"].append(snapshot)
                save_json(result_path, record)
            if phase == "warmup":
                record["idle_after_warmup"] = idle_memory(args.idle_seconds)
        record["idle_after_measurements"] = idle_memory(args.idle_seconds)
        record["affinity"] = affinity()
        record["status"] = "complete"
        record.pop("active_input")
        save_json(result_path, record)
    finally:
        engine.close()


def distribution(values: list[float]) -> dict[str, float | int]:
    """Summarize finite samples with median and linearly interpolated p95.

    Args:
        values:
            Nonempty observations in caller-declared units.

    """
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Distribution requires nonempty finite observations")
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
        "min": min(values),
        "max": max(values),
    }


def comparison(retained: list[float], released: list[float]) -> dict[str, Any]:
    """Apply both median limits and an explicit 10%/100ms p95 tail screen.

    Args:
        retained:
            Complete-call seconds without cleanup, including reusable storage.

        released:
            Complete-call seconds including release and next-call reallocation.

    """
    base, candidate = distribution(retained), distribution(released)
    median_added = float(candidate["median"]) - float(base["median"])
    p95_added = float(candidate["p95"]) - float(base["p95"])
    median_ratio = float(candidate["median"]) / float(base["median"])
    p95_ratio = float(candidate["p95"]) / float(base["p95"])
    median_pass = median_ratio <= 1.10 and median_added <= 0.100
    tail_pass = p95_ratio <= 1.10 and p95_added <= 0.100
    return {
        "retained_seconds": base,
        "release_seconds": candidate,
        "added_median_seconds": median_added,
        "median_ratio": median_ratio,
        "added_p95_seconds": p95_added,
        "p95_ratio": p95_ratio,
        "median_heuristic_pass": median_pass,
        "tail_screen_pass": tail_pass,
        "latency_pass": median_pass and tail_pass,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw per-case calls, exact outputs and steady idle memory.

    Args:
        records:
            Completed fresh-worker records for both modes over all rounds.

    """
    result: dict[str, Any] = {}
    for engine in ("stt", "tts"):
        selected = [record for record in records if record["engine"] == engine]
        modes = {
            mode: [
                run
                for record in selected
                if record["mode"] == mode
                for run in record["runs"]
                if run["phase"] == "measured"
            ]
            for mode in ("retained", "release")
        }
        entry: dict[str, Any] = {"cases": {}}
        for name, _ in CASES:
            groups = {
                mode: [run for run in runs if run["case"] == name]
                for mode, runs in modes.items()
            }
            stats = comparison(
                [run["total_seconds"] for run in groups["retained"]],
                [run["total_seconds"] for run in groups["release"]],
            )
            outputs = [run["output"] for group in groups.values() for run in group]
            key = "sha256" if engine == "tts" else "transcript"
            identities = sorted({output[key] for output in outputs})
            stats["exact_consistency"] = len(identities) == 1
            stats["output_identities"] = identities
            if engine == "stt":
                stats["raw_transcript_identities"] = sorted(
                    {output["raw_transcript"] for output in outputs}
                )
            stats["release_call_seconds"] = distribution(
                [run["release_seconds"] for run in groups["release"]]
            )
            entry["cases"][name] = stats
        entry["all_calls"] = comparison(
            [run["total_seconds"] for run in modes["retained"]],
            [run["total_seconds"] for run in modes["release"]],
        )
        entry["idle_memory_bytes"] = {
            mode: {
                metric: distribution(
                    [
                        float(snapshot["memory"][metric])
                        for record in selected
                        if record["mode"] == mode
                        for snapshot in record["idle_after_repetitions"]
                        if snapshot["phase"] == "measured"
                    ]
                )
                for metric in ("rss", "pss", "uss", "peak_rss")
            }
            for mode in ("retained", "release")
        }
        entry["idle_saved_bytes"] = {
            metric: float(entry["idle_memory_bytes"]["retained"][metric]["median"])
            - float(entry["idle_memory_bytes"]["release"][metric]["median"])
            for metric in ("rss", "pss", "uss")
        }
        entry["qualified"] = all(
            case["latency_pass"] and case["exact_consistency"]
            for case in entry["cases"].values()
        )
        result[engine] = entry
    return result


def service_state() -> dict[str, Any]:
    """Inspect only Hoast status; never start, stop or modify the container."""
    state = json.loads(
        subprocess.check_output(
            ["docker", "inspect", "hoast", "--format", "{{json .State}}"], text=True
        )
    )
    if state["Running"]:
        raise RuntimeError("Hoast must be stopped before benchmarking")
    return {
        key: state[key]
        for key in ("Running", "Status", "ExitCode", "StartedAt", "FinishedAt")
    }


def launch_worker(
    args: argparse.Namespace,
    engine: str,
    mode: str,
    round_number: int,
    fixture: bool = False,
) -> None:
    """Bound one fresh worker and preserve full stdout/stderr before any retry.

    Args:
        args:
            Supervisor's validated diagnostics and native resource settings.

        engine:
            stt or tts.

        mode:
            retained or release.

        round_number:
            Zero-based paired round.

        fixture:
            Generate shared inputs rather than timing benchmark calls.

    """
    tag = "fixtures" if fixture else f"{engine}-{mode}-r{round_number}"
    command = [
        sys.executable,
        "-m",
        "tools.benchmark_memory_release",
        "--worker",
        "fixtures" if fixture else "measure",
        "--engine",
        engine,
        "--mode",
        mode,
        "--round",
        str(round_number),
        "--output",
        str(args.output),
        "--stt-root",
        str(args.stt_root),
        "--tts-root",
        str(args.tts_root),
        "--seed",
        str(args.seed),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--idle-seconds",
        str(args.idle_seconds),
    ]
    save_json(
        args.output / "commands" / f"{tag}.json",
        {
            "command": command,
            "environment": native_environment(args.seed),
            "affinity": [4, 5],
            "timeout_seconds": args.timeout,
        },
    )
    with (args.output / f"{tag}.log").open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ | native_environment(args.seed),
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            save_json(
                args.output / f"{tag}-timeout.json",
                {
                    "command": command,
                    "seed": args.seed,
                    "timeout_seconds": args.timeout,
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    if code != 0:
        raise RuntimeError(
            f"Worker {tag} failed with {code}; full exception in {tag}.log"
        )


def supervise(args: argparse.Namespace) -> None:
    """Generate shared inputs then run reversed paired rounds with raw provenance.

    Args:
        args:
            Validated benchmark settings; outputs cannot replace completed evidence.

    """
    if (args.output / "results.json").exists():
        raise FileExistsError("Choose a new output directory; results already exist")
    args.output.mkdir(parents=True, exist_ok=True)
    before = service_state()
    paths = [
        path
        for root in (args.stt_root, args.tts_root)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix
        in {
            ".xml",
            ".bin",
            ".json",
            ".model",
            ".txt",
            ".rep",
            ".npy",
            ".npz",
            ".yaml",
            ".yml",
        }
    ]
    source_paths = [Path(__file__)] + [
        path
        for root in (Path("hoast/stt2"), Path("hoast/tts2"))
        for path in root.rglob("*.py")
    ]
    environment: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": Path("/proc/cpuinfo").read_text(),
        "affinity": affinity(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "openvino",
                "numpy",
                "soundfile",
                "psutil",
                "scipy",
                "torch",
                "sentencepiece",
            )
        },
        "native_environment": native_environment(args.seed),
        "service_before": before,
        "artifacts": {
            str(path.absolute()): {
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for path in paths
        },
        "sources": {str(path.absolute()): file_hash(path) for path in source_paths},
        "image_python_reference": "3.14.2; measurements use host .venv Python 3.14.7",
    }
    save_json(args.output / "environment.json", environment)
    if not (args.output / "fixtures/manifest.json").exists():
        launch_worker(args, "tts", "retained", 0, fixture=True)
    records = []
    order = []
    for round_number in range(args.rounds):
        for engine in ("stt", "tts"):
            for mode in (
                ("retained", "release")
                if round_number % 2 == 0
                else ("release", "retained")
            ):
                logger.info(
                    "worker engine=%s mode=%s round=%d", engine, mode, round_number
                )
                launch_worker(args, engine, mode, round_number)
                order.append({"engine": engine, "mode": mode, "round": round_number})
                records.append(
                    json.loads(
                        (
                            args.output
                            / "workers"
                            / f"{engine}-{mode}-r{round_number}.json"
                        ).read_text()
                    )
                )
    environment["service_after"] = service_state()
    source_changes = {
        str(path.absolute()): {
            "before_sha256": environment["sources"][str(path.absolute())],
            "after_sha256": after_hash,
        }
        for path in source_paths
        if (after_hash := file_hash(path))
        != environment["sources"][str(path.absolute())]
    }
    summaries = aggregate(records)
    for summary in summaries.values():
        summary["measured_heuristic_pass"] = summary["qualified"]
        summary["qualified"] = summary["qualified"] and not source_changes
    result = {
        "schema_version": 1,
        "status": "source-drift" if source_changes else "complete",
        "qualification_valid": not source_changes,
        "source_changes": source_changes,
        "units": {"latency": "seconds", "memory": "bytes"},
        "seed": args.seed,
        "warmups_per_case_per_worker": args.warmups,
        "measured_per_case_per_worker": args.repeats,
        "rounds": args.rounds,
        "idle_seconds": args.idle_seconds,
        "cases": [{"name": name, "text": text} for name, text in CASES],
        "order": order,
        "environment": environment,
        "workers": records,
        "aggregate": summaries,
        "policy": "Per-case median <=10% AND <=100ms added; p95 screened with same limits; exact output required. No defaults changed.",
        "limitations": [
            "TTS constructor baseline excludes lazy graph loading; after-warmup snapshots include both languages.",
            "No allocator trim/forced GC; OpenVINO release may not return pages to OS.",
            "No playback/network; generated fixture transcripts are consistency, not accuracy evaluation.",
            "Warm filesystem caches; full artifact hashing excluded from measured timings.",
            "Isolated engines do not measure combined application/LLM resident memory.",
        ],
    }
    save_json(args.output / "results.json", result)
    if source_changes:
        raise RuntimeError(
            "Runtime source changed during benchmark; "
            "raw results preserved but qualification invalid"
        )
    logger.info(
        "complete result=%s qualified=%s",
        args.output / "results.json",
        {engine: value["qualified"] for engine, value in result["aggregate"].items()},
    )


def parse_args() -> argparse.Namespace:
    """Parse and reject underpowered qualification or unbounded worker settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stt-root", type=Path, default=Path("models/stt2"))
    parser.add_argument("--tts-root", type=Path, default=Path("models/tts2"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--idle-seconds", type=float, default=0.2)
    parser.add_argument("--worker", choices=("fixtures", "measure"))
    parser.add_argument("--engine", choices=("stt", "tts"), default="tts")
    parser.add_argument("--mode", choices=("retained", "release"), default="retained")
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 14):
        parser.error("native Python 3.14 required")
    if (
        args.seed not in (7, 42)
        or args.warmups < 2
        or args.repeats < 8
        or args.rounds < 2
    ):
        parser.error(
            "seed 7/42, >=2 warmups, >=8 measured repetitions and >=2 rounds required"
        )
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("worker timeout must be positive and finite")
    if not math.isfinite(args.idle_seconds) or args.idle_seconds < 0:
        parser.error("idle seconds must be nonnegative and finite")
    return args


def main() -> None:
    """Apply affinity before model load and durably capture complete failures."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        os.sched_setaffinity(0, {4, 5})
        configure_cpu_budget(2)
        if args.worker:
            run_worker(args)
        else:
            supervise(args)
    except Exception:
        tag = (
            "supervisor"
            if not args.worker
            else f"{args.worker}-{args.engine}-{args.mode}-r{args.round}"
        )
        save_json(
            args.output / f"{tag}-exception.json",
            {
                "traceback": traceback.format_exc(),
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "rng_state": np.random.default_rng(args.seed).bit_generator.state,
                "cases": [{"case": name, "text": text} for name, text in CASES],
            },
        )
        logger.exception("benchmark failed; complete metadata preserved")
        raise


if __name__ == "__main__":
    main()
