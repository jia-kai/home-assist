"""Download the pinned faster-whisper small checkpoint and initialize INT8 CPU."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from hoast.logging import configure_logging, get_logger
from hoast.runtime import configure_cpu_budget
from hoast.stt import DEFAULT_MODEL, STT, STTConfig

logger = get_logger(__name__)
REPOSITORY = "Systran/faster-whisper-small"
REVISION = "536b0662742c02347bc0e980a01041f333bce120"


def prepare(output: Path, threads: int) -> None:
    """Resume a pinned download and verify it loads with CPU INT8 compute.

    Args:
        output:
            Local model directory, created if needed; HF tracks download integrity.

        threads:
            One or two CPU inference threads for initialization.

    """
    config = STTConfig(model_path=output, threads=threads)
    logger.info(
        "stt.prepare repository=%s revision=%s output=%s threads=%d",
        REPOSITORY,
        REVISION,
        output,
        threads,
    )
    snapshot_download(
        REPOSITORY,
        revision=REVISION,
        local_dir=output,
        token=False,
        allow_patterns=["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"],
    )
    STT(config)
    logger.info("stt.prepare status=ok")


def main() -> None:
    """Download and initialize STT, retaining complete failure diagnostics."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL,
        help="Local small-model artifact directory",
    )
    parser.add_argument(
        "--threads",
        type=int,
        choices=(1, 2),
        default=2,
        help="CPU worker/core budget for validation",
    )
    args = parser.parse_args()
    configure_logging(log_file=Path(".cache/hoast/diagnostics/prepare-stt.log"))
    try:
        configure_cpu_budget(args.threads)
        prepare(args.output, args.threads)
    except Exception:
        logger.exception("stt.prepare status=failed output=%s", args.output)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
