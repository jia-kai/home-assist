"""Offline artifact format and streaming integrity validation for native STT."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

FORMAT = "hoast-sensevoice-openvino-v1"
ARTIFACT_NAMES = ("model.xml", "model.bin", "am.mvn", "tokenizer.model", "config.yaml")
ONNX_ARTIFACT_NAMES = ("model.onnx", "am.mvn", "tokenizer.model", "config.yaml")
FRONTEND_SPEC: dict[str, Any] = {
    "sample_rate": 16000,
    "sample_scale": 32768,
    "frame_length": 400,
    "frame_shift": 160,
    "fft_length": 512,
    "remove_dc_offset": True,
    "preemphasis": 0.97,
    "window": "nonperiodic_hamming",
    "mel_bins": 80,
    "low_frequency": 20,
    "high_frequency": 8000,
    "log_floor": "float32_epsilon",
    "lfr_width": 7,
    "lfr_stride": 6,
    "cmvn": "signed_addshift_then_rescale",
    "dither": 0.0,
    "short_input": "reject_below_400_samples",
}
ARCHITECTURE_SPEC: dict[str, int] = {
    "input_size": 560,
    "hidden_size": 512,
    "heads": 4,
    "feedforward_size": 2048,
    "encoder_blocks": 50,
    "tp_blocks": 20,
    "kernel_size": 11,
    "vocab_size": 25055,
    "embedding_count": 16,
}


def sha256_file(path: Path) -> str:
    """Hash a local artifact with bounded 1 MiB memory rather than reading it all.

    Args:
        path:
            Existing regular file to validate; I/O failures propagate.

    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str | int]:
    """Return a file's size in bytes and lowercase SHA256 digest.

    Args:
        path:
            Existing local artifact; hashing streams the complete file once.

    """
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def artifact_names(backend: str = "openvino") -> tuple[str, ...]:
    """Return the required files for an explicit manifest backend.

    Args:
        backend:
            openvino (also the default for manifests without this field) or
            onnxruntime; unsupported backend names raise ValueError.

    """
    if backend == "openvino":
        return ARTIFACT_NAMES
    if backend == "onnxruntime":
        return ONNX_ARTIFACT_NAMES
    raise ValueError(f"Unsupported STT2 artifact backend: {backend}")


def validate_artifacts(directory: Path, verify_hashes: bool = True) -> dict[str, Any]:
    """Validate required local files and their explicit inference contract.

    No network lookup, download or repair occurs. Hash verification streams each
    artifact once at startup (including the weights). Size checks always run.

    Args:
        directory:
            Prepared directory with manifest.json and backend-specific files.

        verify_hashes:
            Compare SHA256 in addition to file sizes; true is the runtime default.
            False is useful for preparation's already-hashed staged directory.

    """
    if not (directory / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing STT2 artifact: {directory / 'manifest.json'}")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("Unsupported STT2 artifact manifest format")
    names = artifact_names(manifest.get("backend", "openvino"))
    for name in names:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing STT2 artifact: {directory / name}")
    if manifest.get("frontend") != FRONTEND_SPEC:
        raise ValueError("STT2 artifact frontend contract does not match this runtime")
    if manifest.get("architecture") != ARCHITECTURE_SPEC:
        raise ValueError("STT2 artifact architecture does not match SenseVoiceSmall")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(names):
        raise ValueError("STT2 manifest must describe exactly the required artifacts")
    for name in names:
        record = files[name]
        if not isinstance(record, dict):
            raise TypeError(f"Invalid STT2 artifact record: {name}")
        size = record.get("bytes")
        digest = record.get("sha256")
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid STT2 artifact size: {name}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Invalid STT2 artifact digest: {name}")
        path = directory / name
        if path.stat().st_size != size:
            raise ValueError(f"STT2 artifact size mismatch: {path}")
        if verify_hashes and sha256_file(path) != digest:
            raise ValueError(f"STT2 artifact SHA256 mismatch: {path}")
    return manifest
