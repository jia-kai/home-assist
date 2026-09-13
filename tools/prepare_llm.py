"""Prepare FunctionGemma or LFM2.5 artifacts and OpenVINO CPU/GPU caches."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

from hoast.lfm import LFM2, LFMConfig
from hoast.llm import DEFAULT_CACHE, FunctionGemma, LLMConfig, ToolRegistry
from hoast.logging import configure_logging, get_logger

logger = get_logger(__name__)
FUNCTIONGEMMA_ID = "google/functiongemma-270m-it"
LFM_ID = "LiquidAI/LFM2.5-350M"
LFM_REPOSITORY = "OpenVINO/LFM2.5-350M-int8-ov"
LFM_REVISION = "b6a4a9c42aa2dc4bacefa45717befc45acc3a1bf"
type Model = Literal["functiongemma", "lfm"]
type Device = Literal["CPU", "GPU"]


def versions() -> dict[str, str]:
    """Return versions governing conversion and runtime compatibility."""
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "optimum-intel",
            "openvino",
            "openvino-genai",
            "openvino-tokenizers",
            "nncf",
        )
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically publish a UTF-8 preparation manifest.

    Args:
        path:
            Destination manifest; parent directories are created as needed.

        value:
            JSON-serializable provenance or preparation settings.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(cache: Path, model: Model, revision: str | None) -> Path:
    """Download an immutable checkpoint/export and record its provenance.

    Args:
        cache:
            Root for downloaded artifacts and selection manifests.

        model:
            FunctionGemma source checkpoint or public LFM OpenVINO export.

        revision:
            Requested commit/branch; defaults to main for FunctionGemma or the
            pinned public export commit for LFM. Branches resolve once to a SHA.

    """
    # Initialize Hugging Face only after main configures its local caches.
    from huggingface_hub import HfApi, snapshot_download

    repository = FUNCTIONGEMMA_ID if model == "functiongemma" else LFM_REPOSITORY
    requested = revision or ("main" if model == "functiongemma" else LFM_REVISION)
    token = False if model == "lfm" else None
    info = HfApi().model_info(repository, revision=requested, token=token)
    if info.sha is None:
        raise RuntimeError("Hugging Face did not return an immutable revision")
    patterns = ["*.json", "*.jinja", "tokenizer.model", "README.md"]
    patterns += (
        ["*.safetensors"]
        if model == "functiongemma"
        else ["openvino_*.xml", "openvino_*.bin"]
    )
    logger.info("Downloading %s at %s", repository, info.sha)
    path = Path(
        snapshot_download(
            repository,
            revision=info.sha,
            token=token,
            cache_dir=cache / "hub",
            allow_patterns=patterns,
        )
    )
    if model == "functiongemma":
        manifest = {
            "model_id": repository,
            "revision": info.sha,
            "path": str(path.resolve()),
        }
        destination = cache / "model.json"
    else:
        manifest = {
            "model_id": LFM_ID,
            "export_repository": repository,
            "export_revision": info.sha,
            "path": str(path.resolve()),
            "quantization": {"mode": "INT8_ASYM", "group_size": -1},
            "upstream_weight_revision": None,
            "provenance_note": (
                "Public OpenVINO conversion linked from LiquidAI's model card; "
                "upstream weight commit not specified by export"
            ),
        }
        destination = cache / "lfm/model.json"
    write_json(destination, manifest)
    return path


def checkpoint(cache: Path) -> tuple[Path, dict[str, Any]]:
    """Read the official FunctionGemma download selection without network access.

    Args:
        cache:
            Root containing model.json and its referenced checkpoint.

    """
    manifest = json.loads((cache / "model.json").read_text(encoding="utf-8"))
    if manifest["model_id"] != FUNCTIONGEMMA_ID:
        raise ValueError("Export requires the official Google checkpoint")
    path = Path(manifest["path"])
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {path}")
    return path, manifest


def compile_model(
    cache: Path, model: Model, source: Path, threads: int, device: Device
) -> None:
    """Load the selected IR to populate device caches without generating text.

    FunctionGemma prepares PA and SDPA; LFM uses its native hybrid pipeline.
    Runtime errors propagate rather than falling back to another device.

    Args:
        cache:
            Root for device-specific compiled artifacts.

        model:
            Model family whose runtime loads the IR.

        source:
            Prepared OpenVINO model directory.

        threads:
            Positive CPU inference thread count; not a GPU thread setting.

        device:
            Explicit OpenVINO CPU or GPU target.

    """
    if threads < 1:
        raise ValueError("Threads must be positive")
    tools = ToolRegistry([])
    if model == "functiongemma":
        for attention in ("SDPA", "PA"):
            config = LLMConfig(
                source,
                cache_dir=cache,
                threads=threads,
                attention_backend=attention,
                device=device,
            )
            with FunctionGemma(config, tools):
                pass
    else:
        with LFM2(
            LFMConfig(source, cache_dir=cache, threads=threads, device=device), tools
        ):
            pass
    logger.info("Prepared %s on %s: %s", model, device, source)


def export_model(cache: Path, precision: str, threads: int, device: Device) -> Path:
    """Export stateful FunctionGemma IR and compile the selected device pipelines.

    Identity includes revision, dependency versions and compression settings.
    Completion is published only after compilation succeeds. Subprocess output
    and command metadata are retained in diagnostics, including full failures.

    Args:
        cache:
            Root containing the checkpoint and generated artifacts.

        precision:
            Stored weight format: fp32, symmetric int8, or symmetric int4.

        threads:
            Positive CPU inference thread count for precompilation.

        device:
            Explicit CPU or GPU compilation target.

    """
    if precision not in ("fp32", "int8", "int4") or threads < 1:
        raise ValueError("Invalid weight precision or thread count")
    source, model_manifest = checkpoint(cache)
    settings: dict[str, Any] = {
        "model": model_manifest,
        "versions": versions(),
        "precision": precision,
        "stateful": True,
        "symmetric": precision != "fp32",
        "group_size": 128,
        "ratio": 1.0,
    }
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    output = cache / "exports" / f"{precision}-{key}"
    manifest_path = output / "hoast-manifest.json"
    started = time.perf_counter()
    if not manifest_path.exists():
        output.mkdir(parents=True, exist_ok=True)
        log_path = cache / "diagnostics" / f"export-{precision}-{time.time_ns()}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "optimum.commands.optimum_cli",
            "export",
            "openvino",
            "--model",
            str(source),
            "--task",
            "text-generation-with-past",
            "--weight-format",
            precision,
        ]
        if precision != "fp32":
            command += ["--sym"]
        if precision == "int4":
            command += ["--group-size", "128", "--ratio", "1.0"]
        command += [str(output)]
        logger.info("Exporting %s; full output: %s", precision, log_path)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(json.dumps({"command": command, "settings": settings}) + "\n")
            log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    compile_model(cache, "functiongemma", output, threads, device)
    settings["compilation"] = {
        "attention_backends": ["SDPA", "PA"],
        "cache_mode": "OPTIMIZE_SPEED",
        "threads": threads,
        "device": device,
    }
    settings["preparation_seconds"] = time.perf_counter() - started
    write_json(manifest_path, settings)
    write_json(cache / f"{precision}.json", {"path": str(output.resolve()), "key": key})
    return output


def main() -> None:
    """Run download, export or compile with durable metadata and failure logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["functiongemma", "lfm"], default="lfm")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser(
        "download", help="Download checkpoint or public LFM INT8 IR"
    )
    get.add_argument("--revision", default=None)
    export = commands.add_parser(
        "export", help="Export FunctionGemma IR and compile it"
    )
    compile_command = commands.add_parser(
        "compile", help="Compile downloaded/exported IR"
    )
    for command in (export, compile_command):
        command.add_argument(
            "--precision", choices=["fp32", "int8", "int4"], default="int8"
        )
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    diagnostics = cache / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    log_path = diagnostics / f"prepare-{args.model}-{args.command}-{time.time_ns()}.log"
    configure_logging(log_file=log_path)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "torchinductor")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(getattr(args, "threads", 4))
    logger.info("Preparation arguments: %s; Python: %s", vars(args), sys.version)
    try:
        if args.command == "download":
            download(cache, args.model, args.revision)
        elif args.command == "export":
            if args.model != "functiongemma":
                raise ValueError(
                    "LFM uses published INT8 IR; use download then compile"
                )
            export_model(cache, args.precision, args.threads, args.device)
        else:
            if args.model == "lfm":
                if args.precision != "int8":
                    raise ValueError("The public LFM export supports only INT8 weights")
                source = LFMConfig.from_cache(cache).model_path
            else:
                source = LLMConfig.from_cache(
                    cache, precision=args.precision
                ).model_path
            compile_model(cache, args.model, source, args.threads, args.device)
    except BaseException:
        logger.exception("Preparation failed; complete diagnostics: %s", log_path)
        raise


if __name__ == "__main__":
    main()
