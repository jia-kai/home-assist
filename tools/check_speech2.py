"""Offline native Melo/SenseVoice round-trip check with durable audio and failures."""

import argparse
import json
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from hoast.logging import configure_logging, get_logger
from hoast.runtime import configure_cpu_budget
from hoast.stt2 import STT, STTConfig
from hoast.tts2 import TTS, TTSConfig

logger = get_logger(__name__)


def canonical(text: str) -> str:
    """Compare spoken characters case-insensitively, ignoring punctuation and spacing.

    Args:
        text:
            Expected or recognized words; digits and letters are never discarded.

    """
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", text).casefold()
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def check_round_trip(
    output: Path, stt_config: STTConfig, tts_config: TTSConfig, chinese: bool
) -> None:
    """Synthesize complete utterances and require exact normalized native transcripts.

    Args:
        output:
            Existing directory for float WAVs and incrementally saved JSON evidence.

        stt_config:
            Explicit local artifacts/backend and bounded recognition settings.

        tts_config:
            Explicit local artifacts/backend and bounded synthesis settings.

        chinese:
            Include Mandarin and mixed-language smoke fixtures.

    """
    texts = ["Hello world.", "The weather is sunny."]
    if chinese:
        texts.extend(["你好，今天天气很好。", "Hello world. 你好，欢迎回家。"])
    records: list[dict[str, Any]] = []
    with TTS(tts_config) as tts, STT(stt_config) as stt:
        for index, text in enumerate(texts):
            samples, rate = tts.synthesize(text)
            if (
                samples.ndim != 1
                or samples.dtype != np.float32
                or rate != 44100
                or not samples.size
                or not np.isfinite(samples).all()
            ):
                raise ValueError("Melo returned invalid mono FP32 44100 Hz audio")
            wav = output / f"speech-{index}.wav"
            sf.write(wav, samples, rate, subtype="FLOAT")
            transcript = stt.transcribe_samples(samples, rate)
            matched = canonical(text) == canonical(transcript)
            records.append(
                {
                    "text": text,
                    "transcript": transcript,
                    "matched": matched,
                    "wav": str(wav),
                    "tts_timings": tts.last_timings,
                    "stt_timings": stt.last_timings,
                }
            )
            (output / "roundtrip.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n"
            )
            logger.info(
                "native.roundtrip text=%r transcript=%r matched=%s",
                text,
                transcript,
                matched,
            )
            if not matched:
                raise ValueError(
                    f"Native round-trip mismatch for fixture {index}; see {output}"
                )


def main() -> None:
    """Run explicitly selected local native backends without downloading or playback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stt-model-dir", type=Path, default=STTConfig().model_path)
    parser.add_argument("--tts-model-dir", type=Path, default=TTSConfig().model_path)
    parser.add_argument("--stt-backend", choices=("cpu", "GPU", "ort"), default="cpu")
    parser.add_argument(
        "--tts-backend", choices=("cpu", "hybrid", "GPU"), default="cpu"
    )
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--chinese", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/hoast/diagnostics/speech2-check")
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_logging(log_file=args.output / "run.log", color=False)
    try:
        if sys.version_info[:2] != (3, 14):
            raise RuntimeError("Use the project Python 3.14 environment")
        configure_cpu_budget(args.threads)
        check_round_trip(
            args.output,
            STTConfig(
                model_path=args.stt_model_dir,
                backend=args.stt_backend,
                threads=args.threads,
            ),
            TTSConfig(
                model_path=args.tts_model_dir,
                backend=args.tts_backend,
                threads=args.threads,
                seed=42,
            ),
            args.chinese,
        )
    except Exception:
        logger.exception("Native speech check failed; evidence=%s", args.output)
        raise SystemExit(1) from None
    logger.info("Native speech checks passed; evidence=%s", args.output)


if __name__ == "__main__":
    main()
