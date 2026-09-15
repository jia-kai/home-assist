"""Export the pinned official Chinese Kokoro checkpoint in the supported speech environment."""

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from kokoro_onnx.tokenizer import Tokenizer

from hoast.logging import configure_logging, get_logger

logger = get_logger(__name__)
REPOSITORY = "hexgrad/Kokoro-82M-v1.1-zh"
REVISION = "01e7505bd6a7a2ac4975463114c3a7650a9f7218"
WEIGHT_SHA256 = "b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b"


def _digest(path: Path) -> str:
    """Return a file's SHA256 without loading the full model into memory.

    Args:
        path:
            Existing model or metadata artifact.

    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cached(output: Path, voice: str) -> bool:
    """Reuse a complete matching export after verifying all recorded artifact hashes.

    Missing readiness metadata or artifacts requests regeneration. Corrupt recorded
    artifacts raise rather than being overwritten implicitly.

    Args:
        output:
            Prepared Chinese artifact directory.

        voice:
            Requested official voice identifier.

    """
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["repository"] != REPOSITORY
        or manifest["revision"] != REVISION
        or manifest["voice"] != voice
        or "artifacts" not in manifest
    ):
        return False
    if (
        manifest["checkpoint_sha256"] != WEIGHT_SHA256
        or manifest["voice_index"] != "phoneme_count_minus_one"
        or manifest["g2p_version"] != "1.1"
    ):
        raise ValueError("Unexpected Chinese export provenance or frontend convention")
    artifacts = manifest["artifacts"]
    expected = {"model.xml", "model.bin", "config.json", f"voices/{voice}.npy"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ValueError("Invalid Chinese artifact manifest")
    for name, digest in artifacts.items():
        path = output / name
        if not path.is_file():
            logger.warning("Chinese artifact missing; regenerating: %s", path)
            return False
        if _digest(path) != digest:
            raise ValueError(
                f"Chinese artifact checksum mismatch: {path}; remove manifest.json to re-export"
            )
    logger.info("Chinese TTS artifacts verified and reused: %s", output)
    return True


def publish(staging: Path, output: Path, manifest: dict[str, Any]) -> None:
    """Publish validated artifacts, with the readiness manifest replaced last.

    Export/validation failures leave existing artifacts intact. Publication failure
    leaves no readiness manifest, preventing later runtime loading of a partial set.

    Args:
        staging:
            Private directory containing the complete validated export.

        output:
            Destination directory; its source downloads and other voices are retained.

        manifest:
            Export provenance including the selected voice.

    """
    names = ("model.xml", "model.bin", "config.json", f"voices/{manifest['voice']}.npy")
    manifest["artifacts"] = {name: _digest(staging / name) for name in names}
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output / "voices").mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").unlink(missing_ok=True)
    for name in (*names, "validation.wav"):
        (staging / name).replace(output / name)
    (staging / "manifest.json").replace(output / "manifest.json")


class ExportModel(torch.nn.Module):
    """Tensor-only export wrapper around the upstream Chinese model."""

    model: Any
    """Upstream KModel, loaded only inside the optional export environment."""

    def __init__(self, model: Any) -> None:
        """Retain the upstream model.

        Args:
            model:
                Evaluation-mode KModel with local checkpoint weights.

        """
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, ref_s: torch.Tensor, speed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return waveform and integer durations from tensor inputs.

        Args:
            input_ids:
                Int64 IDs shaped (1, phonemes + 2), with boundary zeros.

            ref_s:
                Float32 voice style shaped (1, 256).

            speed:
                Float32 speed multiplier shaped (1,).

        """
        return self.model.forward_with_tokens(input_ids, ref_s, speed)


def main() -> None:
    """Download verified Chinese artifacts, export IR and verify CPU synthesis."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/hoast/tts/chinese"),
        help="Chinese runtime artifact directory",
    )
    parser.add_argument(
        "--voice", default="zf_001", help="Official Chinese voice identifier"
    )
    parser.add_argument(
        "--threads",
        type=int,
        choices=(1, 2),
        default=2,
        help="CPU export and validation workers",
    )
    args = parser.parse_args()
    configure_logging(log_file=Path(".cache/hoast/diagnostics/export-chinese-tts.log"))
    staging: Path | None = None
    try:
        if re.fullmatch(r"[A-Za-z0-9_]+", args.voice) is None:
            raise ValueError("Voice must be an official artifact identifier")
        if cached(args.output, args.voice):
            return
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        source = args.output / "source"
        args.output.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".chinese-export-", dir=args.output))
        snapshot_download(
            REPOSITORY,
            revision=REVISION,
            local_dir=source,
            token=False,
            allow_patterns=[
                "config.json",
                "kokoro-v1_1-zh.pth",
                f"voices/{args.voice}.pt",
            ],
        )
        weight_path = source / "kokoro-v1_1-zh.pth"
        with weight_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != WEIGHT_SHA256:
            raise ValueError("Chinese Kokoro checkpoint checksum mismatch")
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        # These optional packages intentionally load only in the Python 3.12 exporter.
        factory = importlib.import_module("kokoro").KModel
        english = Tokenizer()

        def english_phones(text: str) -> str:
            """Provide English phonemes for bilingual Chinese frontend input.

            Args:
                text:
                    English span.

            """
            return english.phonemize(text, "en-us")

        g2p = importlib.import_module("misaki.zh").ZHG2P(
            version="1.1", en_callable=english_phones
        )
        phonemes, _ = g2p("你好，今天天气很好。")
        if "❓" in phonemes:
            raise ValueError("Chinese frontend produced unknown phonemes")
        ids = [config["vocab"][character] for character in phonemes]
        voice = torch.load(
            source / "voices" / f"{args.voice}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if (
            voice.ndim != 3
            or tuple(voice.shape[1:]) != (1, 256)
            or voice.dtype != torch.float32
        ):
            raise ValueError("Unexpected Chinese voice table format")
        inputs = (
            torch.tensor([[0, *ids, 0]], dtype=torch.int64),
            voice[len(ids) - 1].clone(),
            torch.tensor([1.0], dtype=torch.float32),
        )
        model = factory(
            repo_id=REPOSITORY,
            config=str(source / "config.json"),
            model=str(weight_path),
        ).eval()
        wrapped = ExportModel(model).eval()
        torch.manual_seed(20260914)
        converted = ov.convert_model(
            wrapped,
            example_input=inputs,
            input=[
                ov.PartialShape([1, ov.Dimension(2, -1)]),
                ov.PartialShape([1, 256]),
                ov.PartialShape([1]),
            ],
        )
        for port, name in zip(
            converted.inputs, ("input_ids", "ref_s", "speed"), strict=True
        ):
            port.get_tensor().set_names({name})
        for port, name in zip(converted.outputs, ("audio", "durations"), strict=True):
            port.get_tensor().set_names({name})
        ov.save_model(converted, staging / "model.xml", compress_to_fp16=True)
        (staging / "voices").mkdir(exist_ok=True)
        np.save(
            staging / "voices" / f"{args.voice}.npy",
            voice.numpy(),
            allow_pickle=False,
        )
        (staging / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        numpy_inputs = {
            name: tensor.numpy()
            for name, tensor in zip(
                ("input_ids", "ref_s", "speed"), inputs, strict=True
            )
        }
        del converted, wrapped, model, voice
        gc.collect()
        compiled = ov.Core().compile_model(
            str(staging / "model.xml"),
            "CPU",
            {
                "INFERENCE_NUM_THREADS": args.threads,
                "NUM_STREAMS": "1",
                "INFERENCE_PRECISION_HINT": "f32",
            },
        )
        output = compiled(numpy_inputs)
        samples = np.asarray(
            output[compiled.output("audio")], dtype=np.float32
        ).reshape(-1)
        if not samples.size or not np.isfinite(samples).all():
            raise RuntimeError("Chinese model returned invalid audio")
        sf.write(staging / "validation.wav", samples, 24000, subtype="FLOAT")
        manifest = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "checkpoint_sha256": digest,
            "voice": args.voice,
            "voice_index": "phoneme_count_minus_one",
            "g2p_version": "1.1",
            "g2p_python": sys.executable,
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "openvino",
                    "kokoro",
                    "misaki",
                    "pypinyin",
                    "jieba",
                )
            },
        }
        publish(staging, args.output, manifest)
        logger.info(
            "Chinese TTS prepared path=%s voice=%s samples=%d",
            args.output,
            args.voice,
            samples.size,
        )
    except Exception:
        logger.exception("Chinese TTS export failed")
        raise SystemExit(1) from None
    finally:
        if staging is not None:
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
