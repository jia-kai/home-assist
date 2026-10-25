"""Prepare a pinned local SenseVoice checkpoint using native Python 3.14 only.

This tool never downloads anything or invokes another interpreter. Fetch the
explicitly pinned upstream snapshot separately, then pass its directory as
--source. Output is assembled in a sibling staging directory and published with
one atomic rename; an existing output directory is never overwritten.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from hoast.logging import configure_logging
from hoast.runtime import configure_cpu_budget
from hoast.stt2 import DEFAULT_MODEL
from hoast.stt2.artifacts import (
    ARCHITECTURE_SPEC,
    ARTIFACT_NAMES,
    FORMAT,
    FRONTEND_SPEC,
    file_record,
    validate_artifacts,
)
from hoast.stt2.frontend import CMVN
from tools.stt2_provenance import bundle_provenance, validate_license_inputs

logger = logging.getLogger(__name__)
REPOSITORY = "FunAudioLLM/SenseVoiceSmall"
REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
CHECKPOINT_SHA256 = "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"
TOKENIZER_NAME = "chn_jpn_yue_eng_ko_spectok.bpe.model"
SOURCE_NAMES = ("model.pt", "config.yaml", "am.mvn", TOKENIZER_NAME)


def validate_source_config(path: Path) -> None:
    """Reject source configuration incompatible with the fixed inference architecture.

    Args:
        path:
            Audited upstream YAML containing model, encoder and frontend settings.

    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("SenseVoice source config must be a YAML mapping")
    required: dict[str, Any] = {
        "model": "SenseVoiceSmall",
        "encoder": "SenseVoiceEncoderSmall",
        "frontend": "WavFrontend",
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported SenseVoice source setting: {key}")
    sections: dict[str, dict[str, Any]] = {
        "encoder_conf": {
            "output_size": 512,
            "attention_heads": 4,
            "linear_units": 2048,
            "num_blocks": 50,
            "tp_blocks": 20,
            "normalize_before": True,
            "kernel_size": 11,
            "sanm_shfit": 0,
            "input_layer": "pe",
            "pos_enc_class": "SinusoidalPositionEncoder",
            "selfattention_layer_type": "sanm",
        },
        "frontend_conf": {
            "fs": 16000,
            "window": "hamming",
            "n_mels": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "lfr_m": 7,
            "lfr_n": 6,
        },
    }
    for section, settings in sections.items():
        actual = config.get(section)
        if not isinstance(actual, dict):
            raise TypeError(f"Missing SenseVoice source config mapping: {section}")
        for key, value in settings.items():
            if actual.get(key) != value:
                raise ValueError(
                    f"Unsupported SenseVoice source setting: {section}.{key}"
                )
    if config["encoder_conf"].get("concat_after", False):
        raise ValueError("SenseVoice concat_after is not supported")


def load_checkpoint(path: Path, threads: int) -> dict[str, NDArray[np.float32]]:
    """Load only native Torch tensor weights, without importing upstream model code.

    Args:
        path:
            Local hash-verified model.pt checkpoint containing a plain state dict.

        threads:
            One or two native Torch preparation threads.

    """
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("STT2 checkpoint loading requires native Python 3.14")
    if type(threads) is not int or threads not in (1, 2):
        raise ValueError("Preparation threads must be one or two")
    # Torch is heavy and is needed for preparation only, not runtime or CLI help.
    import torch

    torch.set_num_threads(threads)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise TypeError("SenseVoice checkpoint must contain a plain state dictionary")
    arrays: dict[str, NDArray[np.float32]] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(
                "SenseVoice checkpoint state must map string keys to tensors"
            )
        if value.dtype != torch.float32:
            raise ValueError(f"SenseVoice checkpoint tensor must be float32: {key}")
        arrays[key] = np.asarray(value.detach().numpy(), dtype=np.float32)
    return arrays


def prepare(
    source: Path,
    output: Path,
    threads: int = 2,
    revision: str = REVISION,
    checkpoint_sha256: str = CHECKPOINT_SHA256,
    output_logits: bool = False,
    precision: str = "FP32",
    license_dir: Path | None = None,
) -> None:
    """Build direct OpenVINO IR and publish a complete integrity manifest atomically.

    A full graph shape/key audit runs during construction. This tool does not
    compile or infer the real model; numerical/latency qualification is a separate
    serialized benchmark. Failed preparation removes its staging files, preserving
    existing artifacts. The source checkpoint is read, never changed.

    Args:
        source:
            Existing pinned upstream snapshot containing all SOURCE_NAMES files.

        output:
            Destination directory, which must not already exist. Publication is
            an atomic rename on the same filesystem as its sibling staging path.

        threads:
            Native preparation CPU thread budget, one or two.

        revision:
            Explicit full 40-character upstream commit recorded in the manifest.

        checkpoint_sha256:
            Explicit 64-character checkpoint digest verified before Torch load.
            The default pins the audited upstream checkpoint at REVISION.

        output_logits:
            Include the large float32 logits output for numerical qualification;
            false emits just per-frame argmax IDs for production inference.

        precision:
            FP32 storage or FP16 compressed constants with FP32 graph tensors;
            neither option implies a qualified execution backend or quality gate.

        license_dir:
            Offline upstream legal texts and fetch provenance; None selects
            source/licenses. The official model card is also bundled verbatim.

    """
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError(
            "STT2 preparation requires the existing Python 3.14 environment"
        )
    if type(threads) is not int or threads not in (1, 2):
        raise ValueError("Preparation threads must be one or two")
    if precision not in ("FP32", "FP16"):
        raise ValueError("Expected FP32 or FP16 compressed storage")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be an explicit full lowercase commit hash")
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
        raise ValueError(
            "checkpoint_sha256 must be an explicit lowercase SHA256 digest"
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite STT2 artifacts: {output}")
    for name in SOURCE_NAMES:
        if not (source / name).is_file():
            raise FileNotFoundError(
                f"Missing pinned SenseVoice source artifact: {source / name}"
            )
    validate_license_inputs(source, license_dir)
    validate_source_config(source / "config.yaml")
    CMVN.load(source / "am.mvn")
    source_records = {name: file_record(source / name) for name in SOURCE_NAMES}
    if source_records["model.pt"]["sha256"] != checkpoint_sha256:
        raise ValueError("Pinned SenseVoice checkpoint SHA256 mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        configure_cpu_budget(threads)
        # OpenVINO graph construction is heavy and only requested after pin checks.
        import openvino as ov

        from hoast.stt2.graph import build_graph

        logger.info(
            "stt2.prepare source=%s output=%s revision=%s threads=%d",
            source,
            output,
            revision,
            threads,
        )
        weights = load_checkpoint(source / "model.pt", threads)
        model = build_graph(weights, output_logits=output_logits)
        ov.save_model(
            model, staging / "model.xml", compress_to_fp16=precision == "FP16"
        )
        shutil.copyfile(source / "config.yaml", staging / "config.yaml")
        shutil.copyfile(source / "am.mvn", staging / "am.mvn")
        shutil.copyfile(source / TOKENIZER_NAME, staging / "tokenizer.model")
        manifest: dict[str, Any] = {
            "format": FORMAT,
            "repository": REPOSITORY,
            "revision": revision,
            "source_files": source_records,
            "architecture": ARCHITECTURE_SPEC,
            "frontend": FRONTEND_SPEC,
            "precision": precision,
            "outputs": ["frame_ids", "ctc_logits"] if output_logits else ["frame_ids"],
            "preparation": {
                "python": sys.version.split()[0],
                "openvino": ov.__version__,
                "method": "direct_checkpoint_graph",
                "weight_precision": "float32"
                if precision == "FP32"
                else "float16-compressed-storage",
            },
            "files": {name: file_record(staging / name) for name in ARTIFACT_NAMES},
            "licenses": bundle_provenance(source, staging, license_dir),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        validate_artifacts(staging, verify_hashes=False)
        if output.exists():
            raise FileExistsError(f"STT2 output appeared during preparation: {output}")
        os.rename(staging, output)
        logger.info("stt2.prepare status=ok output=%s", output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    """Prepare explicit local pinned artifacts and retain full failure tracebacks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Existing pinned upstream snapshot"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL,
        help="Permanent artifact root (default: models/stt2)",
    )
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--precision", choices=("FP32", "FP16"), default="FP32")
    parser.add_argument("--license-dir", type=Path)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument(
        "--output-logits",
        action="store_true",
        help="Include logits for numerical qualification",
    )
    args = parser.parse_args()
    configure_logging(log_file=Path(".cache/hoast/diagnostics/prepare-stt2.log"))
    try:
        prepare(
            args.source,
            args.output,
            args.threads,
            args.revision,
            args.checkpoint_sha256,
            args.output_logits,
            args.precision,
            args.license_dir,
        )
    except Exception:
        logger.exception("stt2.prepare status=failed output=%s", args.output)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
