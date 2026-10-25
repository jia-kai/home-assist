"""Import the sherpa SenseVoice UINT8 graph without its runtime or Python bridge.

Metadata, CMVN and complete token ordering are checked against local official
resources. Dynamic-quantized matrix arithmetic is retained in ONNX or OpenVINO
IR; GPU execution is experimental and does not imply integer dot-product acceleration.
"""

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

from hoast.logging import configure_logging
from hoast.runtime import configure_cpu_budget
from hoast.stt2.artifacts import (
    ARCHITECTURE_SPEC,
    FORMAT,
    FRONTEND_SPEC,
    artifact_names,
    file_record,
    sha256_file,
    validate_artifacts,
)
from hoast.stt2.frontend import CMVN
from tools.stt2_provenance import bundle_provenance, validate_license_inputs

logger = logging.getLogger(__name__)
COMMUNITY_SHA256 = "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"


def prepare(
    community: Path,
    official: Path,
    output: Path,
    threads: int,
    backend: Literal["openvino", "onnxruntime"] = "openvino",
    license_dir: Path | None = None,
) -> Path:
    """Verify local community data and publish a native batch-one graph atomically.

    Args:
        community:
            Extracted sherpa 2024-07-17 archive with model.int8.onnx and tokens.txt.

        official:
            Official snapshot containing CMVN, SentencePiece and config.yaml.

        output:
            New artifact directory; existing paths are never overwritten.

        threads:
            One or two allowed CPU cores/native workers during conversion.

        backend:
            openvino converts to IR; onnxruntime retains the transformed ONNX
            graph and quantized equations, without importing OpenVINO.

        license_dir:
            Offline upstream legal texts and fetch provenance; None selects
            official/licenses. Community attribution is preserved verbatim.

    """
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    names = artifact_names(backend)
    validate_license_inputs(official, license_dir)
    configure_cpu_budget(threads)
    # Heavy preparation dependencies are loaded only after CPU budgeting.
    import onnx
    import sentencepiece as sp

    source = community / "model.int8.onnx"
    if sha256_file(source) != COMMUNITY_SHA256:
        raise ValueError("Community SenseVoice graph differs from the audited release")
    model = onnx.load(source)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    if (
        metadata.get("model_type") != "sense_voice_ctc"
        or metadata.get("vocab_size") != "25055"
        or metadata.get("lfr_window_size") != "7"
        or metadata.get("lfr_window_shift") != "6"
        or metadata.get("normalize_samples") != "0"
    ):
        raise ValueError("Unsupported community SenseVoice metadata")
    cmvn = CMVN.load(official / "am.mvn")
    np.testing.assert_allclose(
        np.fromstring(metadata["neg_mean"], sep=",", dtype=np.float32),
        cmvn.shift,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.fromstring(metadata["inv_stddev"], sep=",", dtype=np.float32),
        cmvn.scale,
        rtol=0,
        atol=1e-6,
    )
    tokenizer = sp.SentencePieceProcessor(
        model_file=str(official / "chn_jpn_yue_eng_ko_spectok.bpe.model")
    )
    lines = (community / "tokens.txt").read_text().splitlines()
    if len(lines) != tokenizer.vocab_size():
        raise ValueError(
            "Community vocabulary size differs from official SentencePiece"
        )
    for index, line in enumerate(lines):
        token, identifier = line.rsplit(" ", 1)
        if int(identifier) != index or tokenizer.id_to_piece(index) != token:
            raise ValueError(f"Community tokenizer ordering differs at {index}")
    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    if not op_counts.get("MatMulInteger") or not op_counts.get("DynamicQuantizeLinear"):
        raise ValueError("Community graph lacks dynamic integer MatMul")
    if {p.name for p in model.graph.input} != {
        "x",
        "x_length",
        "language",
        "text_norm",
    }:
        raise ValueError("Community graph input contract mismatch")
    # Rewrite only the public contract at ONNX level; retain quantized equations.
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in {"x", "language", "text_norm"}:
                node.input[index] = {
                    "x": "speech",
                    "language": "_language32",
                    "text_norm": "_textnorm32",
                }[name]
    helper = onnx.helper
    front = [
        helper.make_node("Shape", ["speech"], ["_speech_shape"]),
        helper.make_node(
            "Constant",
            [],
            ["_frame_axis"],
            value=helper.make_tensor("axis", onnx.TensorProto.INT64, [1], [1]),
        ),
        helper.make_node(
            "Gather", ["_speech_shape", "_frame_axis"], ["_length64"], axis=0
        ),
        helper.make_node(
            "Cast", ["_length64"], ["x_length"], to=onnx.TensorProto.INT32
        ),
        helper.make_node(
            "Cast", ["language"], ["_language32"], to=onnx.TensorProto.INT32
        ),
        helper.make_node(
            "Cast", ["textnorm"], ["_textnorm32"], to=onnx.TensorProto.INT32
        ),
    ]
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(front + original_nodes)
    del model.graph.input[:]
    model.graph.input.extend(
        [
            helper.make_tensor_value_info(
                "speech", onnx.TensorProto.FLOAT, [1, "T", 560]
            ),
            helper.make_tensor_value_info("language", onnx.TensorProto.INT64, [1]),
            helper.make_tensor_value_info("textnorm", onnx.TensorProto.INT64, [1]),
        ]
    )
    # Prefix casts keep upstream INT32 arithmetic while public IDs stay INT64.
    model.graph.node.append(
        helper.make_node(
            "ArgMax", [model.graph.output[0].name], ["frame_ids"], axis=-1, keepdims=0
        )
    )
    del model.graph.output[:]
    model.graph.output.extend(
        [helper.make_tensor_value_info("frame_ids", onnx.TensorProto.INT64, [1, "T4"])]
    )
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".community-stt2-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        onnx_path = staging / "model.onnx"
        onnx.save(model, onnx_path)
        del model
        if backend == "openvino":
            # Conversion is an optional heavy dependency, not needed for ONNX output.
            import openvino as ov

            native = ov.Core().read_model(onnx_path)
            ov.save_model(native, staging / "model.xml", compress_to_fp16=False)
            onnx_path.unlink()
        for original, name in (
            ("am.mvn", "am.mvn"),
            ("config.yaml", "config.yaml"),
            ("chn_jpn_yue_eng_ko_spectok.bpe.model", "tokenizer.model"),
        ):
            shutil.copyfile(official / original, staging / name)
        manifest: dict[str, Any] = {
            "format": FORMAT,
            "frontend": FRONTEND_SPEC,
            "architecture": ARCHITECTURE_SPEC,
            "precision": "UINT8-dynamic-MatMul",
            "backend": backend,
            "source": {
                "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
                "onnx_sha256": sha256_file(source),
                "tokens": file_record(community / "tokens.txt"),
                "metadata": metadata,
                "ops": op_counts,
            },
            "files": {name: file_record(staging / name) for name in names},
            "licenses": bundle_provenance(official, staging, license_dir, community),
            "official_resources": {
                "repository": "FunAudioLLM/SenseVoiceSmall",
                "revision_verification": "No repository revision lookup; local resources checked against community metadata/token ordering",
                "files": {
                    name: file_record(official / name)
                    for name in (
                        "am.mvn",
                        "config.yaml",
                        "chn_jpn_yue_eng_ko_spectok.bpe.model",
                        "README.md",
                    )
                },
            },
            "qualification": "metadata/token/CMVN checked; real inference qualification required",
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        validate_artifacts(staging)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.rename(output)
    logger.info(
        "stt2.community status=prepared output=%s precision=dynamic-UINT8", output
    )
    return output


def main() -> None:
    """Parse explicit offline locations and retain full conversion tracebacks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--community", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Explicit permanent candidate root, e.g. models/stt2-uint8-cpu",
    )
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--license-dir", type=Path)
    parser.add_argument(
        "--backend", choices=("openvino", "onnxruntime"), default="openvino"
    )
    args = parser.parse_args()
    configure_logging(
        log_file=Path(".cache/hoast/diagnostics/prepare-stt2-community.log")
    )
    try:
        prepare(
            args.community,
            args.official,
            args.output,
            args.threads,
            args.backend,
            args.license_dir,
        )
    except Exception:
        logger.exception("stt2.community status=failed output=%s", args.output)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
