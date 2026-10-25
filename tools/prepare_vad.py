"""Acquire the hash-pinned Silero v6 streaming ONNX asset and MIT provenance.

Network access is explicit preparation only. Existing output or --source files
must match the pinned SHA256; --offline never opens a network connection. Run
``python -m tools.prepare_vad`` from the project root before starting Hoast.
"""

import argparse
import hashlib
import json
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)
DEFAULT_MODEL = Path("models/vad/silero_vad_v6.onnx")
REPOSITORY = "SYSTRAN/faster-whisper"
REVISION = "65882eee9f5cdbeeb2d877f1131d48cf241b327d"
MODEL_URL = (
    f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/"
    "faster_whisper/assets/silero_vad_v6.onnx"
)
MODEL_SHA256 = "4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2"
SILERO_REVISION = "fba061dc5559f696e62171e9a0741782b0fdc23c"
SILERO_LICENSE_URL = (
    f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_REVISION}/LICENSE"
)
SYSTRAN_LICENSE_URL = (
    f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/LICENSE"
)
MIT_TERMS = """Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
SILERO_LICENSE = "MIT License\n\nCopyright (c) 2020-present Silero Team\n\n" + MIT_TERMS
SYSTRAN_LICENSE = "MIT License\n\nCopyright (c) 2023 SYSTRAN\n\n" + MIT_TERMS


def validate_model(path: Path) -> None:
    """Reject absent or changed model bytes, including corrupted existing output.

    Args:
        path:
            Local ONNX file to hash; filesystem failures propagate to the caller.

    """
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != MODEL_SHA256:
        raise ValueError(
            f"Silero VAD SHA256 mismatch for {path}: "
            f"expected {MODEL_SHA256}, got {actual}"
        )


def download_model(path: Path) -> None:
    """Download the pinned HTTPS asset into staging; failures are never suppressed.

    Args:
        path:
            Temporary destination; the caller validates its digest before publishing.

    """
    with (
        urllib.request.urlopen(MODEL_URL, timeout=60) as response,
        path.open("wb") as stream,
    ):
        shutil.copyfileobj(response, stream)


def write_provenance(model_path: Path) -> None:
    """Materialize pinned source metadata and full upstream MIT license notices.

    Args:
        model_path:
            Validated model whose parent receives provenance.json and license files.

    """
    provenance = {
        "model": model_path.name,
        "sha256": MODEL_SHA256,
        "size_bytes": model_path.stat().st_size,
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "source_release": "v1.2.1",
        "source_url": MODEL_URL,
        "model_family": "Silero VAD v6, faster-whisper streaming ONNX export",
        "model_upstream_repository": "snakers4/silero-vad",
        "model_upstream_release": "v6.0",
        "license": "MIT",
        "silero_license_url": SILERO_LICENSE_URL,
        "systran_license_url": SYSTRAN_LICENSE_URL,
        "interface": {
            "sample_rate_hz": 16000,
            "frame_samples": 512,
            "context_samples": 64,
            "inputs": ["input", "h", "c"],
            "outputs": ["speech_probs", "hn", "cn"],
            "recurrent_shape": [1, 1, 128],
        },
    }
    (model_path.parent / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (model_path.parent / "LICENSE.silero").write_text(SILERO_LICENSE, encoding="utf-8")
    (model_path.parent / "LICENSE.faster-whisper").write_text(
        SYSTRAN_LICENSE, encoding="utf-8"
    )


def prepare_model(
    model_path: Path = DEFAULT_MODEL,
    *,
    source: Path | None = None,
    offline: bool = False,
) -> Path:
    """Reuse validated output or atomically publish a validated local/downloaded asset.

    A mismatched existing file is an error, not an invitation to replace it.
    Provenance is restored on reuse without network access.

    Args:
        model_path:
            Output ONNX path; its parent is created if acquisition is needed.

        source:
            Optional local asset to validate and copy when output does not exist.

        offline:
            Forbid downloads; missing output requires a supplied local source.

    Returns:
        The validated output model path.
    """
    if model_path.exists():
        validate_model(model_path)
    else:
        if source is not None:
            validate_model(source)
        elif offline:
            raise FileNotFoundError(
                f"Silero VAD asset is unavailable offline: {model_path}; "
                "supply --source with the pinned ONNX file"
            )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".prepare-vad-", dir=model_path.parent
        ) as directory:
            staged = Path(directory) / model_path.name
            if source is not None:
                shutil.copyfile(source, staged)
            else:
                download_model(staged)
            validate_model(staged)
            staged.replace(model_path)
    write_provenance(model_path)
    logger.info("Validated Silero v6 VAD asset: %s", model_path)
    return model_path


def main() -> None:
    """Prepare the CLI-selected asset, logging progress and propagating all failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, help="Local pinned ONNX file to copy")
    parser.add_argument("--offline", action="store_true", help="Forbid network access")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare_model(args.output, source=args.source, offline=args.offline)


if __name__ == "__main__":
    main()
