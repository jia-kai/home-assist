"""Capture CPU OpenVINO LFM checkpoints for offline GPU parity investigations.

Example: ``python tools/lfm_gpu_reference.py MODEL.xml OUT.npz --xml-sha256
HEX --bin-sha256 HEX --chunks 1,2,3;4`` (quote the chunk argument in a shell).
Each semicolon-separated chunk is a nonempty sequence of decimal token IDs;
the request starts with reset state and each chunk advances the same state.
Hashes must be independently established for the pinned export, not calculated
from an untrusted artifact and immediately accepted. No tokenizer is involved.

The archive has keys ``stepN/layerNN`` for post-FFN residual activations,
``stepN/logits``, ``stepN/state/NAME`` (OpenVINO variable states), and
``metadata`` (JSON string). Shapes and state names are those of the IR/runtime;
logits and layers normally have axes (batch, current tokens, features). States
are implementation-specific, not automatically equivalent to GPU cache layouts.
Capture changes graph outputs and may inhibit CPU fusion; use it for parity, not
performance. CPU inference precision is explicitly requested as f32, but the IR's
quantization and f16 operations remain part of the graph.

Compare matching chunks, tokens, positions, masks, dtypes and state semantics.
Use ``compare`` with explicit tolerances: elementwise abs(a-b) <= atol +
rtol*abs(reference). Start with measured errors, e.g. atol=1e-3, rtol=1e-3 for
f32 paths, then justify changes for f16/quantized GPU arithmetic; these are
investigation thresholds, not universal guarantees. Check first divergent layer,
max absolute error and final top-token/margin; a passing tolerance alone does not
prove generation or tool-call parity. Chunked and one-shot CPU execution may
also differ numerically; measure that baseline separately. Never compare
incompatible cache layouts.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

LAYER_COUNT = 16


def digest(path: Path) -> str:
    """Return the SHA-256 of a file, streaming its bytes.

    Args:
        path:
            File to hash.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def parse_chunks(value: str) -> list[list[int]]:
    """Parse semicolon-separated token chunks and reject invalid token IDs.

    Args:
        value:
            Decimal IDs separated by commas and chunks by semicolons.
    """
    chunks: list[list[int]] = []
    for chunk in value.split(";"):
        parts = chunk.split(",")
        if any(
            not part.strip().isascii() or not part.strip().isdigit() for part in parts
        ):
            raise ValueError("Chunks require nonnegative decimal token IDs")
        tokens = [int(part) for part in parts]
        if any(token >= 65536 for token in tokens):
            raise ValueError("Token ID outside LFM vocabulary")
        chunks.append(tokens)
    return chunks


def compare(
    reference: NDArray[Any], actual: NDArray[Any], *, atol: float, rtol: float
) -> dict[str, Any]:
    """Compare finite, nonempty tensors without broadcasting.

    Args:
        reference:
            Trusted tensor of any numeric dtype and shape.

        actual:
            Candidate tensor with exactly the reference shape.

        atol:
            Nonnegative finite absolute tolerance.

        rtol:
            Nonnegative finite relative tolerance against the reference.
    """
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("Tolerances must be finite and nonnegative")
    if reference.shape != actual.shape or reference.size == 0:
        raise ValueError("Tensor shapes must match and be nonempty")
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(actual, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite tensor values")
    error = np.abs(a - b)
    return {
        "passed": bool(np.all(error <= atol + rtol * np.abs(a))),
        "max_abs": float(error.max()),
        "max_index": [
            int(i) for i in np.unravel_index(int(error.argmax()), error.shape)
        ],
    }


def logit_summary(logits: NDArray[Any]) -> dict[str, Any]:
    """Return the last-position top token and its margin over the runner-up.

    Args:
        logits:
            Nonempty logits with vocabulary on the last axis, at least two entries.
    """
    if logits.ndim == 0 or logits.size == 0 or logits.shape[-1] < 2:
        raise ValueError("Logits need at least two vocabulary entries")
    row = np.asarray(logits, dtype=np.float64).reshape(-1, logits.shape[-1])[-1]
    if not np.isfinite(row).all():
        raise ValueError("Nonfinite logits")
    order = np.argsort(-row, kind="stable")
    return {"top_token": int(order[0]), "margin": float(row[order[0]] - row[order[1]])}


def capture(
    xml: Path, xml_sha256: str, bin_sha256: str, chunks: list[list[int]]
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Run a fresh stateful CPU request and collect layer outputs and state copies.

    Args:
        xml:
            IR XML path; its same-stem BIN is required.

        xml_sha256:
            Independently pinned lowercase hexadecimal XML digest.

        bin_sha256:
            Independently pinned lowercase hexadecimal BIN digest.

        chunks:
            Ordered nonempty token chunks; no padding or beam search is used.
    """
    # OpenVINO is a heavy optional dependency for comparison-only callers.
    import openvino as ov

    bin_path = xml.with_suffix(".bin")
    if digest(xml) != xml_sha256 or digest(bin_path) != bin_sha256:
        raise ValueError("IR digest mismatch")
    if not chunks or any(
        not chunk or any(type(t) is not int or t < 0 or t >= 65536 for t in chunk)
        for chunk in chunks
    ):
        raise ValueError("Expected nonempty chunks of valid integer token IDs")
    core = ov.Core()
    model = core.read_model(str(xml), str(bin_path))
    operations = {node.get_friendly_name(): node for node in model.get_ordered_ops()}
    names = [
        f"__module.model.model.layers.{i}/aten::add/Add_1" for i in range(LAYER_COUNT)
    ]
    for name in names:
        if name not in operations or operations[name].get_type_name() != "Add":
            raise ValueError(f"Missing expected LFM layer checkpoint: {name}")
        model.add_outputs(operations[name].output(0))
    compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    request = compiled.create_infer_request()
    request.reset_state()
    arrays: dict[str, NDArray[Any]] = {}
    length = 0
    summaries: list[dict[str, Any]] = []
    for step, tokens in enumerate(chunks):
        length += len(tokens)
        request.infer(
            {
                "input_ids": np.array([tokens], dtype=np.int64),
                "attention_mask": np.ones((1, length), dtype=np.int64),
                "beam_idx": np.array([0], dtype=np.int32),
            }
        )
        logits = np.array(request.get_tensor("logits").data, copy=True)
        arrays[f"step{step}/logits"] = logits
        for index in range(LAYER_COUNT):
            arrays[f"step{step}/layer{index:02}"] = np.array(
                request.get_output_tensor(index + 1).data, copy=True
            )
        for state in request.query_state():
            arrays[f"step{step}/state/{state.name}"] = np.array(
                state.state.data, copy=True
            )
        summaries.append(logit_summary(logits))
    metadata = {
        "xml_sha256": xml_sha256,
        "bin_sha256": bin_sha256,
        "openvino_version": importlib.metadata.version("openvino"),
        "device": "CPU",
        "inference_precision_hint": "f32",
        "chunks": chunks,
        "layer_nodes": names,
        "logit_summaries": summaries,
    }
    return arrays, metadata


def main() -> None:
    """Capture a pinned IR to an NPZ archive from explicitly supplied tokens."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--xml-sha256", required=True)
    parser.add_argument("--bin-sha256", required=True)
    parser.add_argument("--chunks", required=True)
    args = parser.parse_args()
    arrays, metadata = capture(
        args.xml, args.xml_sha256, args.bin_sha256, parse_chunks(args.chunks)
    )
    np.savez_compressed(
        args.output,
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
        **arrays,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
