"""Local faster-whisper small INT8 transcription and an audio-file test CLI."""

import argparse
import math
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import (
    Segment,
    TranscriptionInfo,
    TranscriptionOptions,
    get_suppressed_tokens,
    restore_speech_timestamps,
)
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps
from numpy.typing import NDArray
from scipy.signal import resample_poly

from .input_text import canonicalize_text
from .logging import configure_logging, get_logger
from .runtime import configure_cpu_budget

logger = get_logger(__name__)
DEFAULT_MODEL = Path(".cache/hoast/stt/small")
ENGLISH_PREFERENCE_MARGIN = 0.05


def _select_preferred_language(probabilities: Sequence[tuple[str, float]]) -> tuple[str, float]:
    """Select English or Chinese, choosing English when their scores are close.

    Args:
        probabilities:
            Whisper language-code probabilities from one detected audio window.

    Returns:
        The selected language code and its unmodified Whisper probability.

    Raises:
        ValueError: If Whisper omits English or Chinese from its language scores.

    """
    scores = dict(probabilities)
    try:
        english = scores["en"]
        chinese = scores["zh"]
    except KeyError as error:
        raise ValueError("Whisper language detection must score English and Chinese") from error
    if english + ENGLISH_PREFERENCE_MARGIN >= chinese:
        return "en", english
    return "zh", chinese


class _WindowedWhisper(WhisperModel):
    """Use CTranslate2's variable-length encoder without discarding real audio."""

    encoder_frame_limit: int = 3000
    """Maximum 10 ms feature frames for the current input, bounded by 3000."""

    def encode(self, features: NDArray[np.float32]) -> ctranslate2.StorageView:
        """Remove only the encoder padding beyond the caller's conservative limit.

        Args:
            features:
                Float32 log-mel features shaped (mel_bins, frames) or
                (batch, mel_bins, frames), including faster-whisper's padding.

        """
        return super().encode(features[..., : self.encoder_frame_limit])


@dataclass(slots=True, frozen=True)
class STTConfig:
    """Settings for a single CPU transcription worker."""

    model_path: Path = DEFAULT_MODEL
    """Prepared CTranslate2 small model directory, including its tokenizer."""

    threads: int = 2
    """One or two CPU inference threads; the model has one worker."""

    language: str | None = "en"
    """Whisper language code; None enables automatic language detection."""

    beam_size: int = 1
    """Positive decoding beam width; one minimizes command latency."""

    without_timestamps: bool = True
    """Decode text tokens only; disable to include internal timestamp tokens."""

    encoder_min_seconds: float = 8.0
    """Minimum encoder context for short input; 30 disables adaptive padding removal."""

    def __post_init__(self) -> None:
        """Reject invalid thread, decoding, context and language settings."""
        if self.threads not in (1, 2) or self.beam_size < 1:
            raise ValueError(
                "threads must be one or two and beam_size must be positive"
            )
        if self.language is not None and not self.language.strip():
            raise ValueError("language must be a language code or None")
        if (
            not math.isfinite(self.encoder_min_seconds)
            or not 4 <= self.encoder_min_seconds <= 30
        ):
            raise ValueError("encoder_min_seconds must be finite and between 4 and 30")


class STT:
    """Reusable offline recognizer; serialize calls on each instance."""

    config: STTConfig
    """Validated CPU and decoder settings."""

    model: _WindowedWhisper
    """Loaded small checkpoint with one INT8 CPU worker."""

    last_raw_text: str | None
    """Unnormalized text from the last successful decode, or None before transcription."""

    def __init__(self, config: STTConfig | None = None) -> None:
        """Load prepared artifacts without downloading missing resources.

        Args:
            config:
                Local model location and inference settings; None uses defaults.

        """
        config = STTConfig() if config is None else config
        logger.debug("stt.load settings=%r", config)
        for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
            if not (config.model_path / name).is_file():
                raise FileNotFoundError(
                    f"Missing STT artifact: {config.model_path / name}"
                )
        self.config = config
        self.last_raw_text = None
        start = time.perf_counter()
        self.model = _WindowedWhisper(
            str(config.model_path),
            device="cpu",
            compute_type="int8",
            cpu_threads=config.threads,
            num_workers=1,
            local_files_only=True,
        )
        logger.info(
            "stt.load status=ok threads=%d seconds=%.3f model=%s",
            config.threads,
            time.perf_counter() - start,
            config.model_path,
        )

    def transcribe(self, audio: Path) -> str:
        """Decode to canonical text with basic punctuation; return empty for no speech.

        PyAV downmixes and resamples supported audio to 16 kHz mono. Silero VAD
        uses faster-whisper's single-thread CPU session. Segment iteration is
        completed here, so timings include inference rather than generator setup.

        Args:
            audio:
                Existing audio file, such as WAV, FLAC or MP3.

        """
        if not audio.is_file():
            raise FileNotFoundError(audio)
        start = time.perf_counter()
        samples = np.asarray(decode_audio(str(audio)), dtype=np.float32)
        logger.debug(
            "stt.decode audio=%s samples=%d seconds=%.6f",
            audio,
            samples.size,
            time.perf_counter() - start,
        )
        return self._transcribe(samples, str(audio))

    def transcribe_samples(self, samples: NDArray[np.float32], sample_rate: int) -> str:
        """Return canonical text from mono audio, resampling to 16 kHz without WAV I/O.

        Float32 input at 16 kHz is passed through without copying. Other rates
        use polyphase resampling. A finite-value scan at this external audio
        boundary rejects NaN/Inf before they can reach the inference backend.

        Args:
            samples:
                Nonempty mono float32 waveform shaped (samples,), conventionally
                normalized to [-1, 1]. The caller must not mutate it during inference.

            sample_rate:
                Positive integer sample rate in Hz; Kokoro returns 24000.

        """
        if samples.dtype != np.float32 or samples.ndim != 1 or not samples.size:
            raise ValueError("samples must be a nonempty mono float32 array")
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise ValueError("sample_rate must be a positive integer")
        if not np.isfinite(samples).all():
            raise ValueError("samples must contain only finite values")
        start = time.perf_counter()
        if sample_rate != 16000:
            divisor = math.gcd(sample_rate, 16000)
            audio = np.asarray(
                resample_poly(samples, 16000 // divisor, sample_rate // divisor),
                dtype=np.float32,
            )
        else:
            audio = samples
        logger.debug(
            "stt.resample samples=%d sample_rate=%d output_samples=%d seconds=%.6f",
            samples.size,
            sample_rate,
            audio.size,
            time.perf_counter() - start,
        )
        return self._transcribe(audio, "memory")

    def _transcribe(self, audio: NDArray[np.float32], label: str) -> str:
        """Decode and canonicalize text, retaining raw output for roundtrip auditing.

        The frame bound includes all original audio plus 0.5 seconds of padding;
        VAD can only shorten it. Inputs at least 29.5 seconds retain the full
        encoder context. Low-confidence, repetitive or unexpectedly empty results
        retry the unshortened encoder before any text is returned.

        Args:
            audio:
                Mono float32 samples shaped (samples,) at 16 kHz.

            label:
                Short source description for logs, avoiding raw waveform output.

        """
        start = time.perf_counter()
        self.model.encoder_frame_limit = min(
            3000,
            2
            * math.ceil(
                max(self.config.encoder_min_seconds * 100, audio.size / 160 + 50) / 2
            ),
        )
        segments, info = self._decode(audio)
        completed = list(segments)
        uncertain = (not completed and info.duration_after_vad > 0) or any(
            segment.avg_logprob < -1.0 or segment.compression_ratio > 2.4
            for segment in completed
        )
        if self.model.encoder_frame_limit < 3000 and uncertain:
            logger.warning(
                "stt.context retry=full audio=%s frames=%d reason=uncertain_decode",
                label,
                self.model.encoder_frame_limit,
            )
            self.model.encoder_frame_limit = 3000
            segments, info = self._decode(audio)
            completed = list(segments)
        self.last_raw_text = " ".join(
            segment.text.strip() for segment in completed
        ).strip()
        text = canonicalize_text(self.last_raw_text)
        elapsed = time.perf_counter() - start
        logger.info(
            "stt.transcribe status=ok audio=%s seconds=%.3f duration=%.3f "
            "rtf=%.3f language=%s characters=%d encoder_frames=%d",
            label,
            elapsed,
            info.duration,
            elapsed / info.duration if info.duration else 0.0,
            info.language,
            len(text),
            self.model.encoder_frame_limit,
        )
        logger.debug(
            "stt.transcribe raw=%r result=%r info=%r", self.last_raw_text, text, info
        )
        return text

    def _decode(
        self, audio: NDArray[np.float32]
    ) -> tuple[Iterable[Segment], TranscriptionInfo]:
        """Create a lazy transcription with fixed deterministic decoder settings.

        Args:
            audio:
                Mono float32 waveform shaped (samples,) at 16 kHz.

        """
        if self.config.language is None:
            return self._decode_preferred_language(audio)
        return self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            without_timestamps=self.config.without_timestamps,
        )

    def _decode_preferred_language(
        self, audio: NDArray[np.float32]
    ) -> tuple[Iterable[Segment], TranscriptionInfo]:
        """Decode auto-selected English or Chinese while preferring English near a tie.

        The implementation uses faster-whisper internals to give its first decoder
        window the encoder output used for language detection, avoiding a second
        encoder execution for the short voice-command inputs used by this service.

        Args:
            audio:
                Mono float32 waveform shaped (samples,) at 16 kHz.

        """
        sampling_rate = self.model.feature_extractor.sampling_rate
        vad_options = VadOptions(min_silence_duration_ms=300)
        speech_chunks = get_speech_timestamps(audio, vad_options)
        audio_chunks, _ = collect_chunks(audio, speech_chunks)
        filtered_audio = np.concatenate(audio_chunks, axis=0)
        features = self.model.feature_extractor(filtered_audio)
        initial_features = features[
            ..., : self.model.feature_extractor.nb_max_frames
        ]
        encoder_output = self.model.encode(pad_or_trim(initial_features))
        language_probabilities = [
            (token[2:-2], probability)
            for token, probability in self.model.model.detect_language(encoder_output)[0]
        ]
        language, language_probability = _select_preferred_language(
            language_probabilities
        )
        logger.debug(
            "stt.language selected=%s probability=%.3f english=%.3f chinese=%.3f",
            language,
            language_probability,
            dict(language_probabilities)["en"],
            dict(language_probabilities)["zh"],
        )
        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language=language,
        )
        options = TranscriptionOptions(
            beam_size=self.config.beam_size,
            best_of=1,
            patience=1,
            length_penalty=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
            prompt_reset_on_temperature=0.5,
            temperatures=[0.0],
            initial_prompt=None,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=get_suppressed_tokens(tokenizer, (-1,)),
            without_timestamps=self.config.without_timestamps,
            max_initial_timestamp=1.0,
            word_timestamps=False,
            prepend_punctuations="\"'“¿([{-",
            append_punctuations="\"'.。,，!！?？:：”)]}、",
            multilingual=False,
            max_new_tokens=None,
            clip_timestamps="0",
            hallucination_silence_threshold=None,
            hotwords=None,
        )
        segments = self.model.generate_segments(
            features, tokenizer, options, False, encoder_output
        )
        return (
            restore_speech_timestamps(segments, speech_chunks, sampling_rate),
            TranscriptionInfo(
                language=language,
                language_probability=language_probability,
                duration=audio.size / sampling_rate,
                duration_after_vad=filtered_audio.size / sampling_rate,
                transcription_options=options,
                vad_options=vad_options,
                all_language_probs=language_probabilities,
            ),
        )


def main() -> None:
    """Transcribe one audio file to stdout; retain diagnostics and failures on disk."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio", type=Path, help="Input WAV, FLAC or MP3; decoded to mono 16 kHz"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Prepared faster-whisper small checkpoint directory",
    )
    parser.add_argument(
        "--threads",
        type=int,
        choices=(1, 2),
        default=2,
        help="CPU inference worker/core budget",
    )
    parser.add_argument("--language", default="en", help="Language code, or auto")
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Decoding beam width; one minimizes latency",
    )
    parser.add_argument(
        "--encoder-min-seconds",
        type=float,
        default=8.0,
        help="Minimum short-audio encoder context; 30 keeps full padding",
    )
    parser.add_argument(
        "--timestamp-decoding",
        action="store_true",
        help="Decode timestamp tokens internally; stdout still contains only text",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(".cache/hoast/diagnostics/stt.log"),
        help="Durable timing and diagnostic log",
    )
    args = parser.parse_args()
    configure_logging(log_file=args.log_file, level="WARNING")
    try:
        configure_cpu_budget(args.threads)
        config = STTConfig(
            args.model,
            args.threads,
            None if args.language == "auto" else args.language,
            args.beam_size,
            without_timestamps=not args.timestamp_decoding,
            encoder_min_seconds=args.encoder_min_seconds,
        )
        sys.stdout.write(STT(config).transcribe(args.audio) + "\n")
    except Exception:
        logger.exception(
            "stt.cli status=failed audio=%s threads=%s", args.audio, args.threads
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
