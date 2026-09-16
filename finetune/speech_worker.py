"""Isolated native Kokoro synthesis and batched Whisper inference for augmentation."""

import argparse
import ast
import hashlib
import json
import logging
import math
import re
import shutil
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer as WhisperTokenizer
from faster_whisper.transcribe import get_compression_ratio, get_suppressed_tokens
from huggingface_hub import snapshot_download
from kokoro import KModel
from kokoro_onnx.tokenizer import Tokenizer as KokoroTokenizer
from loguru import logger as kokoro_logger
from numpy.typing import NDArray
from opencc import OpenCC
from scipy.signal import resample_poly

from hoast.chinese import contains_han, phoneme_batches
from hoast.chinese_g2p import ChineseG2P
from hoast.input_text import canonicalize_text
from hoast.speech_text import prepare_speech_text

logger = logging.getLogger(__name__)
MODELS = {
    "en": {
        "repository": "hexgrad/Kokoro-82M",
        "revision": "f3ff3571791e39611d31c381e3a41a3af07b4987",
        "weight": "kokoro-v1_0.pth",
        "voice": "af_heart",
    },
    "zh": {
        "repository": "hexgrad/Kokoro-82M-v1.1-zh",
        "revision": "01e7505bd6a7a2ac4975463114c3a7650a9f7218",
        "weight": "kokoro-v1_1-zh.pth",
        "voice": "zf_001",
    },
}
STT_REPOSITORY = "Systran/faster-whisper-small"
STT_REVISION = "536b0662742c02347bc0e980a01041f333bce120"


def inference_signature(source: str) -> str:
    """Fingerprint model execution and retry code independently of cache migration.

    Args:
        source:
            Complete verified worker source. Model class, inference loop, referenced
            imports and module assignments must match; source locations are ignored.

    """
    tree = ast.parse(source)
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SpeechWorker"
    )
    entry = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    loop = next(
        node
        for node in ast.walk(entry)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "offset"
    )
    selected: list[ast.stmt] = [model, loop]
    names = {
        node.id
        for item in selected
        for node in ast.walk(item)
        if isinstance(node, ast.Name)
    }
    for item in tree.body:
        if isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in item.targets
        ):
            selected.append(item)
            names.update(
                node.id for node in ast.walk(item) if isinstance(node, ast.Name)
            )
    imports = [
        item
        for item in tree.body
        if isinstance(item, (ast.Import, ast.ImportFrom))
        and any(
            (alias.asname or alias.name.split(".")[0]) in names for alias in item.names
        )
    ]
    representation = ast.dump(
        ast.Module(body=[*imports, *selected], type_ignores=[]),
        include_attributes=False,
    )
    return hashlib.sha256(representation.encode()).hexdigest()


def migrate_record(
    record: dict[str, Any],
    job: dict[str, Any],
    settings: dict[str, Any],
    previous_settings: dict[str, Any],
    script: Any,
) -> dict[str, Any]:
    """Re-normalize a verified raw transcript without claiming fresh speech inference.

    Args:
        record:
            Cached speech result with original raw transcript and audio provenance.

        job:
            Exact synthesis text and source split requested for this observation.

        settings:
            Current settings; only worker/normalizer implementation hashes may differ.

        previous_settings:
            Recorded inference settings belonging to the cached speech result.

        script:
            Script-only audit converter exposing convert(text).

    """
    excluded = {"worker_sha256", "normalizer_sha256", "inference_sha256"}
    if {k: v for k, v in settings.items() if k not in excluded} != {
        k: v for k, v in previous_settings.items() if k not in excluded
    }:
        raise ValueError("Cached synthesis/ASR settings are incompatible")
    previous_hash = hashlib.sha256(
        json.dumps(previous_settings, sort_keys=True).encode()
    ).hexdigest()
    if record["job"] != job or record["settings_sha256"] != previous_hash:
        raise ValueError("Cached speech provenance does not match the requested job")
    if record["status"] not in ("ok", "empty", "invalid_input"):
        raise ValueError("Only completed recognizer outputs can be re-normalized")
    result = dict(record)
    error: str | None = None
    try:
        suppressed = (
            record["no_speech_probability"] > 0.6 and record["avg_logprob"] < -1
        )
        text = "" if suppressed else canonicalize_text(record["raw_transcript"])
    except ValueError as exc:
        text, error = "", str(exc)
    result.update(
        transcript=text,
        canonicalization_error=error,
        audit_transcript=script.convert(text),
        audit_source=script.convert(canonicalize_text(job["text"])),
        status="invalid_input" if error else "ok" if text else "empty",
        settings_sha256=hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest(),
        reused_inference=True,
        inference_provenance={
            "settings": previous_settings,
            "settings_sha256": previous_hash,
            "prior_provenance": record.get("inference_provenance"),
        },
    )
    return result


class SpeechWorker:
    """Own GPU/CPU speech models; decode independent utterances in actual ASR batches."""

    device: str
    """Explicit torch/CTranslate2 target, cuda or cpu."""

    models: dict[str, Any]
    """Native FP32 Kokoro models keyed by English/Mandarin routing."""

    voices: dict[str, torch.Tensor]
    """Style tables shaped (phoneme_lengths, 1, 256), in FP32 on the target device."""

    english: Any
    """Production eSpeak-based English Kokoro frontend."""

    chinese: ChineseG2P
    """Production Mandarin frontend, including English insertions, in its own process."""

    recognizer: Any
    """Whisper-small CTranslate2 model with no cross-utterance decoder context."""

    audit_script: Any
    """Traditional-to-simplified comparison only; returned model input retains its script."""

    def __init__(self, assets: Path, device: str, threads: int) -> None:
        """Load pinned speech assets and allocate inference on the requested device.

        Args:
            assets:
                Download/cache directory containing pinned models and voice tables.

            device:
                Explicit cuda or cpu execution; unavailable CUDA raises.

            threads:
                Positive host inference thread budget for each backend.

        """
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_num_threads(threads)
        self.device = device
        self.audit_script = OpenCC("t2s")
        self.models = {}
        self.voices = {}
        for language, settings in MODELS.items():
            root = assets / language
            snapshot_download(
                settings["repository"],
                revision=settings["revision"],
                local_dir=root,
                allow_patterns=[
                    "config.json",
                    settings["weight"],
                    f"voices/{settings['voice']}.pt",
                ],
                token=False,
            )
            self.models[language] = (
                KModel(
                    repo_id=settings["repository"],
                    config=str(root / "config.json"),
                    model=str(root / settings["weight"]),
                )
                .to(device)
                .eval()
            )
            self.voices[language] = torch.load(
                root / "voices" / f"{settings['voice']}.pt",
                map_location=device,
                weights_only=True,
            ).to(torch.float32)
        self.english = KokoroTokenizer()
        self.chinese = ChineseG2P(
            python=Path(sys.executable), log_file=assets / "g2p.log"
        )
        whisper = assets / "whisper-small"
        snapshot_download(
            STT_REPOSITORY,
            revision=STT_REVISION,
            local_dir=whisper,
            allow_patterns=[
                "model.bin",
                "config.json",
                "tokenizer.json",
                "vocabulary.txt",
            ],
            token=False,
        )
        self.recognizer = WhisperModel(
            str(whisper),
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
            cpu_threads=threads,
            num_workers=1,
            local_files_only=True,
        )

    def synthesize(self, text: str, seed: int) -> NDArray[np.float32]:
        """Synthesize one mono 24 kHz float32 waveform without silent phoneme dropping.

        Kokoro's duration expansion supports one utterance at a time. Synthesis is
        on the selected GPU/CPU; batching applies to the subsequent Whisper step.

        Args:
            text:
                Original clean command, allowing punctuation to inform TTS prosody.

            seed:
                Stable per-utterance seed for Kokoro's stochastic excitation.

        """
        torch.manual_seed(seed)
        text = prepare_speech_text(text)
        language = "zh" if contains_han(text) else "en"
        phonemes = (
            self.chinese.phonemize(text)
            if language == "zh"
            else self.english.phonemize(text, "en-us")
        )
        model = self.models[language]
        unknown = set(phonemes) - model.vocab.keys()
        if unknown or "❓" in phonemes:
            raise ValueError(f"Text contains unsupported phonemes: {sorted(unknown)!r}")
        parts: list[NDArray[np.float32]] = []
        for batch in phoneme_batches(phonemes, limit=120 if language == "zh" else 450):
            ids = [model.vocab[character] for character in batch]
            if not ids or len(ids) > self.voices[language].shape[0]:
                raise ValueError("Phoneme sequence exceeds voice context")
            with torch.inference_mode():
                audio, _ = model.forward_with_tokens(
                    torch.tensor([[0, *ids, 0]], device=self.device, dtype=torch.long),
                    self.voices[language][len(ids) - 1],
                    1.0,
                )
            parts.append(audio.detach().float().cpu().numpy().reshape(-1))
        if not parts:
            raise ValueError("Text produced no speakable phonemes")
        samples = np.concatenate(parts).astype(np.float32, copy=False)
        if samples.size >= 29.5 * 24000:
            raise ValueError("Synthesized command exceeds the bounded ASR window")
        return samples

    def recognize(
        self, waveforms: list[NDArray[np.float32]], full_context: bool = False
    ) -> list[dict[str, Any]]:
        """Decode independently prompted utterances in one batched encoder/decoder call.

        Args:
            waveforms:
                Nonempty list of mono float32 24 kHz waveforms. No audio from one
                utterance is concatenated with another; language is detected per item.

            full_context:
                Use 30-second padding for a low-confidence retry; otherwise use at
                least eight seconds and enough context for the longest item plus 0.5 s.

        """
        audio = [
            resample_poly(samples, 2, 3).astype(np.float32) for samples in waveforms
        ]
        frames = (
            3000
            if full_context
            else min(
                3000, 2 * math.ceil(max(800, max(len(a) / 160 + 50 for a in audio)) / 2)
            )
        )
        features = np.stack(
            [
                pad_or_trim(self.recognizer.feature_extractor(a), length=frames)
                for a in audio
            ]
        )
        encoder = self.recognizer.encode(features)
        languages = self.recognizer.model.detect_language(encoder)
        tokenizers = [
            WhisperTokenizer(
                self.recognizer.hf_tokenizer,
                True,
                task="transcribe",
                language=choices[0][0][2:-2],
            )
            for choices in languages
        ]
        prompts = [
            self.recognizer.get_prompt(tokenizer, [], without_timestamps=True)
            for tokenizer in tokenizers
        ]
        results = self.recognizer.model.generate(
            encoder,
            prompts,
            beam_size=1,
            patience=1,
            length_penalty=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
            max_length=448,
            suppress_blank=True,
            suppress_tokens=get_suppressed_tokens(tokenizers[0], (-1,)),
            sampling_temperature=0,
            return_scores=True,
            return_no_speech_prob=True,
        )
        output: list[dict[str, Any]] = []
        for result, tokenizer, choices in zip(
            results, tokenizers, languages, strict=True
        ):
            tokens = result.sequences_ids[0]
            raw = tokenizer.decode(tokens).strip()
            logprob = result.scores[0] * len(tokens) / (len(tokens) + 1)
            canonicalization_error: str | None = None
            try:
                text = (
                    ""
                    if result.no_speech_prob > 0.6 and logprob < -1
                    else canonicalize_text(raw)
                )
            except ValueError as error:
                text = ""
                canonicalization_error = str(error)
            output.append(
                {
                    "raw_transcript": raw,
                    "transcript": text,
                    "canonicalization_error": canonicalization_error,
                    "audit_transcript": self.audit_script.convert(text),
                    "language": choices[0][0][2:-2],
                    "language_probability": choices[0][1],
                    "avg_logprob": logprob,
                    "no_speech_probability": result.no_speech_prob,
                    "compression_ratio": get_compression_ratio(raw),
                    "encoder_frames": frames,
                    "asr_batch_size": len(waveforms),
                    "full_context_retry": full_context,
                }
            )
        return output

    def close(self) -> None:
        """Release the owned Chinese phonemizer subprocess."""
        self.chinese.close()


def main() -> None:
    """Process hash-addressed speech jobs, saving audio and resumable transcript records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--assets", type=Path, default=Path(".cache/hoast/finetune/speech-assets")
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--reuse-from",
        type=Path,
        help="Verified speech directory whose raw outputs may be re-normalized",
    )
    parser.add_argument("--prior-worker-source", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.output / "worker.log"),
            logging.StreamHandler(),
        ],
    )
    kokoro_logger.remove()
    kokoro_logger.add(sys.stderr, level="WARNING")
    worker: SpeechWorker | None = None
    try:
        if args.batch_size < 1 or args.threads < 1:
            raise ValueError("Batch size and thread budget must be positive")
        jobs = [json.loads(line) for line in args.jobs.read_text().splitlines()]
        if any(re.fullmatch(r"[0-9a-f]{24}", job["id"]) is None for job in jobs):
            raise ValueError("Speech job IDs must be 24 hexadecimal characters")
        if len({job["id"] for job in jobs}) != len(jobs):
            raise ValueError("Speech jobs must have unique IDs")
        settings = {
            "tts": MODELS,
            "stt_repository": STT_REPOSITORY,
            "stt_revision": STT_REVISION,
            "device": args.device,
            "batch_size": args.batch_size,
            "tts_dtype": "float32",
            "stt_compute_type": "float16" if args.device == "cuda" else "int8",
            "beam_size": 1,
            "language": "auto_per_utterance",
            "sample_rate": 24000,
            "stt_sample_rate": 16000,
            "speed": 1.0,
            "tts_batch_size": 1,
            "versions": {
                name: version(name)
                for name in (
                    "torch",
                    "kokoro",
                    "kokoro-onnx",
                    "misaki",
                    "faster-whisper",
                    "ctranslate2",
                    "opencc-python-reimplemented",
                )
            },
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inference_sha256": inference_signature(Path(__file__).read_text()),
            "normalizer_sha256": hashlib.sha256(
                (
                    Path(__file__).resolve().parent.parent / "hoast/input_text.py"
                ).read_bytes()
            ).hexdigest(),
            "speech_frontend_sha256": {
                name: hashlib.sha256(
                    (
                        Path(__file__).resolve().parent.parent / "hoast" / name
                    ).read_bytes()
                ).hexdigest()
                for name in ("chinese_g2p_worker.py", "speech_text.py")
            },
        }
        fingerprint = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()
        manifest = args.output / "settings.json"
        if manifest.exists() and json.loads(manifest.read_text()) != settings:
            raise ValueError("Speech settings changed; use a new output directory")
        manifest.write_text(json.dumps(settings, indent=2))
        (args.output / "worker-source.py").write_text(Path(__file__).read_text())
        reused = 0
        if args.reuse_from is not None:
            if args.reuse_from.resolve() == args.output.resolve():
                raise ValueError(
                    "Re-normalization requires a separate output directory"
                )
            previous_settings = json.loads(
                (args.reuse_from / "settings.json").read_text()
            )
            prior_source = args.prior_worker_source or (
                args.reuse_from / "worker-source.py"
            )
            source_text = prior_source.read_text()
            if (
                hashlib.sha256(source_text.encode()).hexdigest()
                != previous_settings["worker_sha256"]
            ):
                raise ValueError(
                    "Prior worker source does not match the cached inference"
                )
            if inference_signature(source_text) != settings["inference_sha256"]:
                raise ValueError(
                    "Speech inference implementation changed; generate fresh speech"
                )
            script = OpenCC("t2s")
            for job in jobs:
                destination = args.output / f"{job['id']}.json"
                source = args.reuse_from / destination.name
                if destination.exists() or not source.is_file():
                    continue
                record = json.loads(source.read_text())
                if record["status"] == "tts_rejected":
                    continue
                migrated = migrate_record(
                    record, job, settings, previous_settings, script
                )
                if record["audio_file"] != f"{job['id']}.wav":
                    raise ValueError("Unexpected cached audio filename")
                audio = args.reuse_from / record["audio_file"]
                if (
                    hashlib.sha256(audio.read_bytes()).hexdigest()
                    != record["audio_sha256"]
                ):
                    raise ValueError("Cached audio checksum mismatch")
                shutil.copy2(audio, args.output / audio.name)
                destination.write_text(
                    json.dumps(migrated, ensure_ascii=False, indent=2)
                )
                reused += 1
            logger.info("speech reused_raw_outputs=%d", reused)
        pending: list[dict[str, Any]] = []
        for job in jobs:
            path = args.output / f"{job['id']}.json"
            if path.exists():
                cached = json.loads(path.read_text())
                if cached["job"] != job or cached["settings_sha256"] != fingerprint:
                    raise ValueError("Cached speech job does not match its inputs")
                continue
            pending.append(job)
        logger.info(
            "speech jobs=%d pending=%d device=%s asr_batch_size=%d",
            len(jobs),
            len(pending),
            args.device,
            args.batch_size,
        )
        if pending:
            worker = SpeechWorker(args.assets, args.device, args.threads)
        for offset in range(0, len(pending), args.batch_size):
            assert worker is not None
            batch = pending[offset : offset + args.batch_size]
            waves: list[NDArray[np.float32]] = []
            records: list[dict[str, Any]] = []
            for job in batch:
                record: dict[str, Any] = {
                    "job": job,
                    "settings_sha256": fingerprint,
                    "audit_source": worker.audit_script.convert(
                        canonicalize_text(job["text"])
                    ),
                    "synthesis_text": prepare_speech_text(job["text"]),
                }
                try:
                    samples = worker.synthesize(job["text"], int(job["id"][:8], 16))
                except ValueError as error:
                    logger.exception("speech status=tts_rejected id=%s", job["id"])
                    record.update(
                        status="tts_rejected",
                        reason=str(error),
                        raw_transcript="",
                        transcript="",
                    )
                    (args.output / f"{job['id']}.json").write_text(
                        json.dumps(record, ensure_ascii=False, indent=2)
                    )
                    continue
                path = args.output / f"{job['id']}.wav"
                sf.write(path, samples, 24000, subtype="FLOAT")
                record.update(
                    audio_file=path.name,
                    audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    audio_seconds=len(samples) / 24000,
                )
                waves.append(samples)
                records.append(record)
            if records:
                transcriptions = worker.recognize(waves)
                retry = [
                    i
                    for i, item in enumerate(transcriptions)
                    if item["encoder_frames"] < 3000
                    and (
                        not item["transcript"]
                        or item["avg_logprob"] < -1
                        or item["compression_ratio"] > 2.4
                    )
                ]
                if retry:
                    corrected = worker.recognize(
                        [waves[i] for i in retry], full_context=True
                    )
                    for index, item in zip(retry, corrected, strict=True):
                        transcriptions[index] = item
                for record, transcript in zip(records, transcriptions, strict=True):
                    record.update(transcript)
                    record["status"] = (
                        "invalid_input"
                        if record["canonicalization_error"] is not None
                        else "ok"
                        if record["transcript"]
                        else "empty"
                    )
                    path = args.output / f"{record['job']['id']}.json"
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(record, ensure_ascii=False, indent=2)
                    )
                    temporary.replace(path)
                    logger.info(
                        "speech status=%s id=%s", record["status"], record["job"]["id"]
                    )
            logger.info(
                "speech completed=%d/%d",
                min(offset + args.batch_size, len(pending)),
                len(pending),
            )
        counts = Counter(
            json.loads((args.output / f"{job['id']}.json").read_text())["status"]
            for job in jobs
        )
        logger.info("speech status=complete outcomes=%s", dict(counts))
    except Exception:
        logger.exception("speech worker status=failed")
        raise
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
