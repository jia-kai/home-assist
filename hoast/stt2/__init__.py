"""Offline native SenseVoiceSmall recognition with deterministic audio features.

No FunASR runtime, framework checkpoint load, downloader or subprocess bridge is
used. Prepare the pinned local checkpoint with tools/prepare_stt2.py first.
Runtime supports complete utterances in one batch, without VAD or truncation.
Attention is quadratic in utterance length; max_audio_seconds defaults to 60.
Source rates must be integer 8000..192000 Hz with reduced resampling factors
at most 4000, bounding both output expansion and polyphase FIR allocation.
"""

import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Protocol, Self

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

from hoast.input_text import canonicalize_text
from hoast.runtime import configure_cpu_budget

from .artifacts import validate_artifacts
from .frontend import CMVN, SAMPLE_RATE, Frontend
from .settings import Backend, NodeProfile, validate_settings

logger = logging.getLogger(__name__)
DEFAULT_MODEL = Path("models/stt2")
LANGUAGE_IDS: dict[str, int] = {
    "auto": 0,
    "zh": 3,
    "en": 4,
    "yue": 7,
    "ja": 11,
    "ko": 12,
}
_METADATA = re.compile(r"<\|[^<>]*\|>")
_MIN_SOURCE_RATE = 8000
_MAX_SOURCE_RATE = 192000
_MAX_RESAMPLE_FACTOR = 4000


class FrameEngine(Protocol):
    """Minimal synchronous inference contract shared by native and mock engines."""

    profile_path: Path | None
    """Durable native trace path when available; None when disabled or unfinalized."""

    def infer(
        self, features: NDArray[np.float32], language_id: int, textnorm_id: int
    ) -> NDArray[np.int64]:
        """Return int64 IDs shaped (speech frames+4,), including metadata frames.

        Args:
            features:
                Float32 normalized feature matrix shaped (speech frames, 560).

            language_id:
                Language prefix embedding index.

            textnorm_id:
                Text normalization embedding index, 14 or 15.

        """
        ...

    def close(self) -> None:
        """Release resources; repeated calls must be safe."""
        ...

    def get_profile(self) -> list[NodeProfile]:
        """Copy JSON-safe counters; ORT makes these available only after close."""
        ...

    def release_memory(self) -> None:
        """Release idle intermediate storage; unsupported backends raise explicitly."""
        ...


class Tokenizer(Protocol):
    """SentencePiece decoding contract; token string concatenation is not equivalent."""

    def decode(self, ids: list[int], /) -> str:
        """Decode collapsed vocabulary IDs with SentencePiece's actual semantics.

        Args:
            ids:
                Positional-only ordered nonblank IDs from the complete CTC
                sequence; native SentencePiece calls this parameter input.

        """
        ...


def make_engine(
    model_path: Path,
    threads: int,
    backend: Backend = "cpu",
    profiling: bool = False,
) -> FrameEngine:
    """Load only the explicitly requested native backend, without fallback.

    Args:
        model_path:
            Validated local model.xml (OpenVINO) or model.onnx (ORT) artifact path.

        threads:
            One or two CPU inference/host threads within the process CPU budget.

        backend:
            cpu/GPU selects OpenVINO; ort selects ONNX Runtime CPUExecutionProvider.
            No automatic selection or fallback is supported.

        profiling:
            Whether to enable per-node execution counters at compilation time.

    """
    if backend == "ort":
        # Heavy optional runtimes stay lazy and mutually independent.
        from .ort_engine import ORTEngine

        return ORTEngine(model_path, threads, profiling)
    # Keep the heavy optional backend lazy for frontend-only tools and tests.
    from .engine import OpenVINOEngine

    return OpenVINOEngine(model_path, threads, backend, profiling)


def make_tokenizer(model_path: Path) -> Tokenizer:
    """Load a native SentencePiece model and validate its complete vocabulary.

    Args:
        model_path:
            Local tokenizer.model file from the pinned checkpoint snapshot.

    """
    # Native SentencePiece is optional until runtime construction is requested.
    import sentencepiece as sp

    tokenizer = sp.SentencePieceProcessor(model_file=str(model_path))
    if tokenizer.vocab_size() != 25055:
        raise ValueError("STT2 SentencePiece vocabulary must contain 25055 pieces")
    return tokenizer


def collapse_ctc(frame_ids: NDArray[np.int64]) -> list[int]:
    """Collapse adjacent IDs before removing blank zero, preserving blank-separated repeats.

    Args:
        frame_ids:
            One-dimensional int64 IDs for every valid frame, including metadata.

    """
    if frame_ids.ndim != 1 or frame_ids.dtype != np.int64:
        raise ValueError("CTC frame IDs must be a one-dimensional int64 array")
    if not frame_ids.size:
        return []
    keep = np.concatenate((np.array([True]), frame_ids[1:] != frame_ids[:-1]))
    return frame_ids[keep & (frame_ids != 0)].tolist()


def canonical_text(raw_text: str) -> str:
    """Strip SenseVoice metadata and apply the application's canonical text contract.

    Args:
        raw_text:
            Actual SentencePiece decode, potentially including <|metadata|> tags.
            Tags are removed without adding spaces, preserving Chinese continuity.

    """
    return canonicalize_text(_METADATA.sub("", raw_text))


@dataclass(slots=True, frozen=True)
class STTConfig:
    """Offline model location and a bounded single-worker inference policy.

    Args:
        model_path:
            Prepared local artifact directory; model loading never downloads files.

        threads:
            One or two CPU workers/allowed CPU cores. Construction applies this
            process-wide budget before loading the inference plugin.

        language:
            None/auto detects language; zh/en/yue/ja/ko select explicit controls.

        use_itn:
            True selects ITN prefix row 14; false uses reference default row 15.

        backend:
            cpu is the default. GPU is an explicit FP32 experimental candidate;
            ort selects the transformed ONNX graph on CPUExecutionProvider only.
            Unsupported graph/device compilation raises without fallback.

        profiling:
            Enable native instrumentation. OpenVINO snapshots follow inference;
            ORT finalizes a durable trace and last_profile only on close. Stage
            wall timings are retained independently of this optional setting.

        max_audio_seconds:
            Finite positive maximum source-audio duration in seconds, inclusive.
            Over-limit input raises before decoding files or resampling arrays;
            increase explicitly for deliberate longer diagnostics, never truncate.

        max_input_samples:
            Positive integer inclusive source scalar-sample budget. File frames
            times channels are checked before decoding; mono arrays before scans.

        max_output_tokens:
            Positive integer inclusive collapsed nonblank CTC token budget,
            including metadata tokens, checked before SentencePiece decoding.

        max_output_chars:
            Positive integer inclusive character budget for both raw decoded
            text and canonical text; excess raises without truncating either.

        release_after_call:
            Release OpenVINO intermediate buffers after success or failure while
            retaining weights. ONNX Runtime rejects this option.

    """

    model_path: Path = DEFAULT_MODEL
    """Prepared backend-specific artifacts; no resources are downloaded at runtime."""

    threads: int = 2
    """Exactly one or two CPU threads for one synchronous inference worker."""

    language: str | None = None
    """None/auto enables model language detection; zh/en/yue/ja/ko force a language."""

    use_itn: bool = False
    """Select ITN embedding 14 when true; reference inference defaults to non-ITN 15."""

    backend: Backend = "cpu"
    """cpu/GPU selects OpenVINO; ort explicitly selects ONNX Runtime CPU only."""

    profiling: bool = False
    """Enable plugin execution counters; off for uninstrumented latency comparisons."""

    max_audio_seconds: float = 60.0
    """Inclusive source-duration limit in seconds, checked before neural preparation."""

    release_after_call: bool = False
    """Release OpenVINO intermediate storage after each call, retaining model weights."""

    max_input_samples: int = 2_880_000
    """Inclusive source scalar-sample limit, including every file channel."""

    max_output_tokens: int = 1024
    """Inclusive collapsed nonblank CTC token limit before tokenizer decoding."""

    max_output_chars: int = 4096
    """Inclusive raw and canonical Unicode character limit; excess is rejected."""

    def __post_init__(self) -> None:
        """Reject invalid settings instead of falling back to another language/backend."""
        if not isinstance(self.model_path, Path):
            raise TypeError("model_path must be a pathlib.Path")
        validate_settings(self.backend, self.threads, self.profiling)
        if self.language is not None and self.language not in LANGUAGE_IDS:
            raise ValueError("STT2 language must be auto, zh, en, yue, ja, ko or None")
        if type(self.use_itn) is not bool:
            raise TypeError("use_itn must be a boolean")
        if type(self.release_after_call) is not bool:
            raise TypeError("release_after_call must be a boolean")
        if self.release_after_call and self.backend == "ort":
            raise ValueError("Post-call memory release requires OpenVINO")
        if type(self.max_audio_seconds) not in (int, float):
            raise TypeError("max_audio_seconds must be a number")
        if not math.isfinite(self.max_audio_seconds) or self.max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be finite and positive")
        for name in ("max_input_samples", "max_output_tokens", "max_output_chars"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class STT:
    """Reusable offline recognizer; transcriptions and resource release serialize.

    Owns one compiled model/request and one tokenizer. Use close() or a with
    block for deterministic release. Construction applies a process-wide CPU
    budget shared by other inference engines; close does not restore affinity or
    native pool settings. Loading differently budgeted instances is therefore a
    process-level policy decision, not an independent pool reservation.
    """

    config: STTConfig
    """Validated artifact, CPU and language settings."""

    frontend: Frontend | None
    """Cached deterministic feature frontend; None before load or after close."""

    engine: FrameEngine
    """One synchronous explicitly selected native worker, released by close()."""

    tokenizer: Tokenizer | None
    """Actual native SentencePiece decoder; None before load or after close."""

    cpu_affinity: tuple[int, ...]
    """Allowed process CPU IDs selected by the shared one/two-core budget helper."""

    last_raw_text: str | None
    """Bounded successful SentencePiece output; cleared when a new request starts."""

    last_timings: dict[str, float]
    """Successful call's stage seconds; cleared when a new request starts."""

    last_profile: list[NodeProfile]
    """Successful call's counters or finalized ORT nodes; cleared on a new request."""

    _lock: RLock
    """Reentrant lock serializing each instance's complete transcription and release."""

    _closed: bool
    """Whether close has released this instance's backend."""

    def __init__(self, config: STTConfig | None = None) -> None:
        """Validate offline artifacts, apply the CPU budget and load the requested plugin.

        CPU affinity and existing native pools are bounded before lazy backend
        import/plugin initialization. Missing artifacts fail before changing the
        process budget. Failed inference construction retains no native worker.

        Args:
            config:
                Explicit settings; None uses STTConfig defaults. Manifest SHA256
                verification streams artifacts at startup and is included in load timing.

        """
        start = time.perf_counter()
        self.config = STTConfig() if config is None else config
        self.last_raw_text = None
        self.last_timings = {}
        self.last_profile = []
        self._lock = RLock()
        self._closed = False
        self.frontend = None
        self.tokenizer = None
        manifest = validate_artifacts(self.config.model_path)
        expected_backend = "onnxruntime" if self.config.backend == "ort" else "openvino"
        if manifest.get("backend", "openvino") != expected_backend:
            raise ValueError(
                "STT2 artifact backend does not match the requested runtime"
            )
        self.cpu_affinity = configure_cpu_budget(self.config.threads)
        self.frontend = Frontend(CMVN.load(self.config.model_path / "am.mvn"))
        self.tokenizer = make_tokenizer(self.config.model_path / "tokenizer.model")
        try:
            self.engine = make_engine(
                self.config.model_path
                / ("model.onnx" if self.config.backend == "ort" else "model.xml"),
                self.config.threads,
                self.config.backend,
                self.config.profiling,
            )
        except Exception:
            # Release non-engine native storage when plugin construction fails.
            self.frontend = None
            self.tokenizer = None
            self._closed = True
            raise
        logger.info(
            "stt2.load status=ok backend=%s device=%s threads=%d profiling=%s affinity=%s seconds=%.6f model=%s",
            expected_backend,
            self.config.backend,
            self.config.threads,
            self.config.profiling,
            self.cpu_affinity,
            time.perf_counter() - start,
            self.config.model_path,
        )

    def transcribe(self, path: Path) -> str:
        """Decode an existing libsndfile-supported audio file to canonical text.

        Multi-channel files are averaged to mono in float32 and resampled to
        16 kHz. No VAD, truncation or silence heuristic changes the model input.
        File frame/rate metadata and frames times channels are checked against
        duration and scalar-sample limits before decoding. Decoded geometry is
        checked before averaging. Starting a request clears prior audit outputs.
        Source rates must be 8000..192000 Hz and both reduced factors for
        resampling to 16000 Hz must be at most 4000, checked before file decoding.

        Args:
            path:
                Existing WAV, FLAC, OGG or another format supported by libsndfile.
                Its source rate must satisfy the supported resampling bounds.

        """
        with self._lock:
            self._ensure_open()
            self._clear_audit()
            started = time.perf_counter()
            if not path.is_file():
                raise FileNotFoundError(path)
            metadata = sf.info(path)
            self._validate_duration(metadata.frames, metadata.samplerate)
            if (
                type(metadata.channels) is not int
                or not 1 <= metadata.channels <= self.config.max_input_samples
            ):
                raise ValueError(
                    "STT2 audio channels must be a bounded positive integer"
                )
            self._validate_input_samples(metadata.frames * metadata.channels)
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            if (
                samples.shape != (metadata.frames, metadata.channels)
                or sample_rate != metadata.samplerate
            ):
                raise ValueError("STT2 decoded audio does not match file metadata")
            self._validate_input_samples(samples.size)
            mono = np.asarray(samples.mean(axis=1, dtype=np.float32), dtype=np.float32)
            decoded = time.perf_counter()
            result = self.transcribe_samples(mono, int(sample_rate))
            self.last_timings["file_decode"] = decoded - started
            self.last_timings["total"] = time.perf_counter() - started
            return result

    def _ensure_open(self) -> None:
        """Raise RuntimeError after resource release instead of recreating a worker."""
        if self._closed:
            raise RuntimeError("STT2 recognizer is closed")

    def _clear_audit(self) -> None:
        """Drop prior text, timings and counters before accepting a new request."""
        self.last_raw_text = None
        self.last_timings = {}
        self.last_profile = []

    def _validate_input_samples(self, sample_count: int) -> None:
        """Reject source scalar-sample counts above the inclusive allocation budget.

        Args:
            sample_count:
                Total source samples across all channels, before resampling.

        """
        if sample_count > self.config.max_input_samples:
            raise ValueError(
                f"STT2 source samples {sample_count} exceeds "
                f"max_input_samples={self.config.max_input_samples}"
            )

    def _validate_duration(self, frame_count: int, sample_rate: int) -> None:
        """Reject unsupported resampling geometry and excessive source duration.

        Cheap scalar checks run before file decoding, full-array scans,
        resampling and feature extraction. The rate range bounds output
        expansion to 2x; reduced factors bound polyphase FIR filter allocation.
        Supplied audio is never changed or truncated.

        Args:
            frame_count:
                Number of source audio frames (samples per channel), nonnegative.

            sample_rate:
                Exact integer source sampling rate, inclusive 8000..192000 Hz.
                Reduced up/down factors to 16000 Hz must each be at most 4000.

        """
        if type(frame_count) is not int or frame_count < 0:
            raise ValueError("STT2 audio frame count must be a nonnegative integer")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("STT2 sample_rate must be a positive integer")
        if not _MIN_SOURCE_RATE <= sample_rate <= _MAX_SOURCE_RATE:
            raise ValueError("STT2 sample_rate must be in [8000,192000] Hz")
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        if max(SAMPLE_RATE // divisor, sample_rate // divisor) > _MAX_RESAMPLE_FACTOR:
            raise ValueError("STT2 reduced resampling factors must be at most 4000")
        duration = frame_count / sample_rate
        if duration > self.config.max_audio_seconds:
            raise ValueError(
                f"STT2 audio duration {duration:.6f}s exceeds "
                f"max_audio_seconds={self.config.max_audio_seconds}"
            )

    def transcribe_samples(self, samples: NDArray[np.float32], sample_rate: int) -> str:
        """Transcribe audio and optionally release intermediate memory even on failure.

        Cleanup runs under the inference lock after outputs and counters are
        copied. Successful timing totals include cleanup; weights remain loaded.
        Starting a request clears prior audit outputs, including on failure.

        Args:
            samples:
                Nonempty finite mono float32 waveform shaped (samples,).

            sample_rate:
                Exact integer source rate in 8000..192000 Hz with reduced
                resampling factors at most 4000; checked before scanning samples.

        """
        with self._lock:
            self._ensure_open()
            self._clear_audit()
            started = time.perf_counter()
            try:
                text = self._transcribe_samples(samples, sample_rate)
            finally:
                before_release = time.perf_counter()
                if self.config.release_after_call:
                    self.release_memory()
            if self.config.release_after_call:
                self.last_timings["release_memory"] = (
                    time.perf_counter() - before_release
                )
            self.last_timings["total"] = time.perf_counter() - started
            return text

    def release_memory(self) -> None:
        """Serialize release of idle OpenVINO storage without unloading the recognizer.

        ONNX Runtime cannot release its arena through this interface and raises.
        Native allocators may keep freed memory resident for subsequent reuse.
        """
        with self._lock:
            self._ensure_open()
            self.engine.release_memory()

    def _transcribe_samples(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> str:
        """Return canonical text from finite mono float32 audio without WAV I/O.

        The input boundary performs a full finite-value scan. At least 400
        resampled samples (25 ms) are required; shorter/empty input raises
        ValueError explicitly. All input audio is retained; silence can decode
        to metadata only, returning an empty canonical string.
        Source samples and duration are bounded before scanning values or
        resampling. Collapsed tokens are bounded before decoding; raw and
        canonical text are both bounded before retention. Limits are inclusive
        and excess raises rather than truncates.

        Args:
            samples:
                Nonempty mono float32 waveform shaped (samples,), conventionally
                in [-1,1]; caller must not mutate it while inference is running.

            sample_rate:
                Exact integer input rate in 8000..192000 Hz with reduced
                resampling factors at most 4000. Polyphase resampling converts
                supported rates to 16000 Hz before feature extraction.

        """
        with self._lock:
            self._ensure_open()
            start = time.perf_counter()
            frontend = self.frontend
            tokenizer = self.tokenizer
            if frontend is None or tokenizer is None:
                raise RuntimeError("STT2 recognizer resource invariant is broken")
            if samples.dtype != np.float32 or samples.ndim != 1 or not samples.size:
                raise ValueError("STT2 samples must be a nonempty mono float32 array")
            self._validate_duration(samples.shape[0], sample_rate)
            self._validate_input_samples(samples.size)
            if not np.isfinite(samples).all():
                raise ValueError("STT2 samples must contain only finite values")
            validated = time.perf_counter()
            if sample_rate != SAMPLE_RATE:
                divisor = math.gcd(sample_rate, SAMPLE_RATE)
                audio = np.asarray(
                    resample_poly(
                        samples, SAMPLE_RATE // divisor, sample_rate // divisor
                    ),
                    dtype=np.float32,
                )
            else:
                audio = samples
            resampled = time.perf_counter()
            features = frontend.extract(audio)
            extracted = time.perf_counter()
            frame_ids = self.engine.infer(
                features,
                LANGUAGE_IDS[self.config.language or "auto"],
                14 if self.config.use_itn else 15,
            )
            inferred = time.perf_counter()
            profile = self.engine.get_profile() if self.config.profiling else []
            profiled = time.perf_counter()
            if frame_ids.shape != (features.shape[0] + 4,):
                raise ValueError("STT2 engine returned an incorrect number of frames")
            ids = collapse_ctc(frame_ids)
            if len(ids) > self.config.max_output_tokens:
                raise ValueError("STT2 output exceeds max_output_tokens")
            raw = tokenizer.decode(ids)
            if len(raw) > self.config.max_output_chars:
                raise ValueError("STT2 raw output exceeds max_output_chars")
            text = canonical_text(raw)
            if len(text) > self.config.max_output_chars:
                raise ValueError("STT2 canonical output exceeds max_output_chars")
            self.last_raw_text = raw
            completed = time.perf_counter()
            self.last_profile = profile
            self.last_timings = {
                "validation": validated - start,
                "resample": resampled - validated,
                "frontend": extracted - resampled,
                "inference": inferred - extracted,
                "decode": completed - profiled,
                "total": completed - start,
            }
            if self.config.profiling:
                self.last_timings["profiling"] = profiled - inferred
            logger.debug(
                "stt2.transcribe status=ok samples=%d sample_rate=%d feature_frames=%d timings=%s raw=%r text=%r",
                samples.size,
                sample_rate,
                features.shape[0],
                self.last_timings,
                raw,
                text,
            )
            return text

    def close(self) -> None:
        """Wait for any active call and release model/tokenizer/frontend idempotently.

        Raw text and timing/profile snapshots remain available for auditing. The
        process-wide affinity/native-pool CPU policy is not restored or enlarged.
        """
        with self._lock:
            if not self._closed:
                try:
                    self.engine.close()
                    if self.config.backend == "ort" and self.config.profiling:
                        self.last_profile = self.engine.get_profile()
                finally:
                    self.tokenizer = None
                    self.frontend = None
                    self._closed = True

    @property
    def profile_path(self) -> Path | None:
        """Return the durable ORT trace path after close, or None if unavailable."""
        with self._lock:
            return self.engine.profile_path

    def __enter__(self) -> Self:
        """Return this open recognizer for deterministic with-block resource ownership."""
        with self._lock:
            self._ensure_open()
            return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release resources at with-block exit without suppressing exceptions.

        Args:
            exception_type:
                Exception class from the with block, or None for a normal exit.

            exception:
                Exception instance from the block; it is propagated unchanged.

            traceback:
                Original exception traceback, or None; never retained by STT.

        """
        self.close()
