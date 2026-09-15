"""Download matched Kokoro artifacts and warm up CPU/GPU hybrid synthesis."""

import argparse
import hashlib
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from hoast.chinese_g2p import DEFAULT_SPEECH_PYTHON
from hoast.logging import configure_logging, get_logger
from hoast.runtime import configure_cpu_budget
from hoast.tts import (
    DEFAULT_CHINESE_MODEL,
    DEFAULT_MODEL,
    DEFAULT_VOICES,
    TTS,
    TTSConfig,
)
from tools.build_speech_kernels import build as build_cpu_kernels

logger = get_logger(__name__)
BASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
SHA256 = {
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}


def download(name: str, output: Path) -> None:
    """Download a pinned release asset atomically and verify its SHA256.

    Existing files are verified before reuse. Interrupted downloads leave a .part
    file overwritten on retry; incomplete files are never published as assets.
    A corrupt existing file raises an error; remove that file before retrying.

    Args:
        name:
            Asset filename from SHA256 within the Kokoro v1.0 release.

        output:
            Destination path whose parent is created as needed.

    """
    expected = SHA256[name]
    url = f"{BASE_URL}/{name}"
    output.parent.mkdir(parents=True, exist_ok=True)
    source = output
    if not output.is_file():
        temporary = output.with_suffix(output.suffix + ".part")
        logger.info("tts.download url=%s output=%s", url, output)
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        source = temporary
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(
            f"SHA256 mismatch for {source}: expected {expected}, got {digest}"
        )
    if source != output:
        source.replace(output)
    logger.info(
        "tts.download status=ok url=%s output=%s sha256=%s", url, output, digest
    )


def prepare(output: Path, threads: int) -> None:
    """Download source artifacts, build CPU fusion and warm up hybrid inference.

    Args:
        output:
            Directory for the matched ONNX model and voice archive.

        threads:
            One or two CPU inference threads.

    """
    name = DEFAULT_MODEL.name
    config = TTSConfig(
        model_path=output / name,
        voices_path=output / DEFAULT_VOICES.name,
        threads=threads,
    )
    download(name, config.model_path)
    download(DEFAULT_VOICES.name, config.voices_path)
    build_cpu_kernels()
    engine = TTS(config)
    try:
        engine.synthesize("Ready.")
    finally:
        engine.close()
    logger.info("tts.prepare status=ok output=%s threads=%d", output, threads)


def _prepare_chinese(output: Path, voice: str, threads: int) -> None:
    """Provision the pinned Python 3.12 frontend/export environment and Chinese IR.

    Args:
        output:
            Destination for Chinese model, vocabulary and voice artifacts.

        voice:
            Official Chinese voice identifier.

        threads:
            One or two CPU export/validation workers.

    """
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError(
            "uv is required to prepare the Chinese speech environment"
        )
    environment = os.environ.copy()
    environment["UV_PROJECT_ENVIRONMENT"] = str(
        DEFAULT_SPEECH_PYTHON.parents[1].resolve()
    )
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    subprocess.run(
        [
            uv,
            "sync",
            "--project",
            str(Path(__file__).with_name("speech_env").resolve()),
            "--locked",
            "--python",
            "3.12",
        ],
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            str(DEFAULT_SPEECH_PYTHON.absolute()),
            "-m",
            "tools.export_chinese_tts",
            "--output",
            str(output),
            "--voice",
            voice,
            "--threads",
            str(threads),
        ],
        env=environment,
        check=True,
    )


def main() -> None:
    """Prepare local Kokoro artifacts and retain complete failure diagnostics."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL.parent,
        help="Non-Chinese model and voice directory",
    )
    parser.add_argument(
        "--threads",
        type=int,
        choices=(1, 2),
        default=2,
        help="CPU worker/core budget for compilation and validation",
    )
    parser.add_argument(
        "--chinese",
        action="store_true",
        help="Also provision Chinese v1.1 artifacts and its supported G2P environment",
    )
    parser.add_argument(
        "--chinese-model-dir",
        type=Path,
        default=DEFAULT_CHINESE_MODEL,
        help="Chinese artifact directory",
    )
    parser.add_argument(
        "--chinese-voice", default="zf_001", help="Official Chinese voice to prepare"
    )
    args = parser.parse_args()
    configure_logging(log_file=Path(".cache/hoast/diagnostics/prepare-tts.log"))
    try:
        configure_cpu_budget(args.threads)
        prepare(args.output, args.threads)
        if args.chinese:
            _prepare_chinese(args.chinese_model_dir, args.chinese_voice, args.threads)
    except Exception:
        logger.exception("tts.prepare status=failed output=%s", args.output)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
