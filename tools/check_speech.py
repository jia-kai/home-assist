"""Compact real-device kernel correctness and public TTS-to-STT smoke check."""

import argparse
import json
import sys
import unicodedata
from dataclasses import replace
from pathlib import Path

import numpy as np
import openvino as ov
import soundfile as sf
from openvino import opset13 as ops

from hoast.logging import configure_logging, get_logger
from hoast.runtime import configure_cpu_budget
from hoast.stt import STT, STTConfig
from hoast.tts import TTS
from hoast.tts_gpu import CPUConvOffload, SharedGPU

logger = get_logger(__name__)


def check_kernel(output: Path) -> None:
    """Compare the final shared-memory kernel against FP32 convolution on tiny fixtures.

    Fixtures have odd channel counts, padded boundaries, temporal tails, dilation
    and stride. Power-of-two scales make weight quantization exact; relative RMS
    arithmetic error must stay below 0.3% across both lengths.

    Args:
        output:
            Existing directory retaining the last complete fixture before assertions.

    """
    rng = np.random.default_rng(20260915)
    gpu = SharedGPU()
    try:
        for stride, dilation in ((1, 1), (1, 3), (2, 1)):
            weights = rng.integers(-127, 128, (17, 7, 3)).astype(np.float32) / 128
            weights[:, 0, 0] = 127 / 128
            data = ops.parameter([1, 7, -1], ov.Type.f32)
            reference = ops.convolution(
                data,
                ops.constant(weights),
                [stride],
                [dilation],
                [dilation],
                [dilation],
            )
            offload = CPUConvOffload(data.output(0), gpu, gpu.prepare(reference))
            model = ov.Model([offload, reference], [data])  # pyright: ignore[reportGeneralTypeIssues]
            compiled = ov.Core().compile_model(
                model,
                "CPU",
                {
                    "INFERENCE_NUM_THREADS": 2,
                    "NUM_STREAMS": "1",
                    "INFERENCE_PRECISION_HINT": "f32",
                    "ENABLE_CPU_PINNING": False,
                },
            )
            programs = len(gpu.programs)
            for width in (31, 49):
                values = (
                    rng.normal(0, 0.2, (1, 7, width))
                    .astype(np.float16)
                    .astype(np.float32)
                )
                np.savez(
                    output / "kernel-case.npz",
                    inputs=values,
                    weights=weights,
                    stride=stride,
                    dilation=dilation,
                    width=width,
                )
                result = compiled([values])
                actual, expected = (
                    result[compiled.output(0)],
                    result[compiled.output(1)],
                )
                np.savez(
                    output / "kernel-case.npz",
                    inputs=values,
                    weights=weights,
                    actual=actual,
                    expected=expected,
                    stride=stride,
                    dilation=dilation,
                )
                error = float(
                    np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                )
                logger.info(
                    "check.kernel stride=%d dilation=%d width=%d relative_rms=%g",
                    stride,
                    dilation,
                    width,
                    error,
                )
                assert np.isfinite(actual).all() and error < 0.003
                assert len(gpu.programs) == programs
            del compiled, model, offload, reference, data
    finally:
        gpu.close()


def canonical(text: str) -> str:
    """Normalize case, punctuation and explicit script-equivalent fixture characters.

    Args:
        text:
            Expected or recognized fixture text; all spoken words remain significant.

    """
    text = (
        unicodedata.normalize("NFKC", text)
        .casefold()
        .translate(str.maketrans("氣歡", "气欢"))
    )
    return "".join(character for character in text if character.isalnum())


def check_round_trip(output: Path, chinese: bool) -> None:
    """Synthesize and recognize short novel-length utterances through the public API.

    Args:
        output:
            Existing directory for generated audio and complete transcript results.

        chinese:
            Include prepared Mandarin/mixed resources and check lazy cached reuse.

    """
    records: list[dict[str, str]] = []
    tts = TTS()
    try:
        stt = STT(STTConfig())
        texts = ["Hello world.", "The weather is sunny."]
        if chinese:
            texts.extend(["你好，今天天气很好。", "Hello world. 你好，欢迎回家。"])
        cached_chinese = None
        for index, text in enumerate(texts):
            if index < 2:
                assert tts._chinese is None
            else:
                stt.config = replace(stt.config, language="zh")
            samples, rate = tts.synthesize(text)
            assert rate == 24000 and samples.ndim == 1 and samples.dtype == np.float32
            sf.write(output / f"speech-{index}.wav", samples, rate, subtype="FLOAT")
            transcript = stt.transcribe_samples(samples, rate)
            records.append({"text": text, "transcript": transcript})
            logger.info("check.roundtrip expected=%r transcript=%r", text, transcript)
            assert canonical(text) == canonical(transcript)
            if index == 2:
                cached_chinese = tts._chinese
                assert cached_chinese is not None
            if index == 3:
                assert tts._chinese is cached_chinese
    finally:
        try:
            (output / "roundtrip.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        finally:
            tts.close()


def main() -> None:
    """Run prepared-device checks and retain full errors while keeping stdout compact."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--chinese",
        action="store_true",
        help="Also verify prepared Chinese and mixed speech",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/hoast/diagnostics/speech-check"),
        help="Audio, fixture and diagnostic directory",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=args.output / "run.log", level="WARNING")
    try:
        configure_cpu_budget(2)
        check_kernel(args.output)
        check_round_trip(args.output, args.chinese)
    except Exception:
        logger.exception("Speech correctness check failed seed=20260915")
        raise SystemExit(1) from None
    sys.stdout.write("Speech checks passed.\n")


if __name__ == "__main__":
    main()
