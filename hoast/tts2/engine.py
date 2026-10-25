"""Resident Melo inference with native text, explicit noise, and sparse frame expansion."""

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import numpy as np
import openvino as ov
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

from hoast.chinese import contains_han
from hoast.logging import get_logger
from hoast.runtime import configure_cpu_budget
from hoast.speech_text import prepare_speech_text
from hoast.tts_audio import AudioPlayback

from .frontend import Frontend

logger = get_logger(__name__)
DEFAULT_MODEL = Path("models/tts2")


@dataclass(slots=True, frozen=True)
class TTSConfig:
    """Prepared native Melo artifacts and explicit CPU/backend controls."""

    model_path: Path = DEFAULT_MODEL
    """Root containing separate EN/ZH graphs, tokenizers and frontend resources."""

    threads: int = 2
    """One or two CPU inference threads; one execution stream per stage."""

    backend: Literal["cpu", "hybrid", "GPU"] = "cpu"
    """CPU stages, GPU decoder only, or full GPU stages; never silently changed."""

    voice: str = "EN-Default"
    """Official English speaker name, resolved from the checkpoint configuration."""

    chinese_voice: str = "ZH"
    """Official Chinese speaker name; its checkpoint ID is one."""

    speed: float = 1.0
    """Inverse duration scale, constrained to [0.5, 2.0]."""

    sdp_ratio: float = 0.2
    """Stochastic/deterministic log-duration blend, in [0, 1]."""

    noise_scale: float = 0.6
    """Nonnegative acoustic latent Gaussian multiplier."""

    noise_scale_w: float = 0.8
    """Nonnegative Gaussian multiplier for the stochastic duration flow."""

    seed: int | None = None
    """Fixed per-request NumPy seed for diagnostics; None samples fresh noise."""

    max_frames: int = 4000
    """Maximum generated latent frames, a validation limit rather than truncation."""

    max_total_frames: int = 12000
    """Utterance-wide frame limit (about 139 seconds), bounding retained waveform RAM."""

    max_segments: int = 64
    """Maximum sentence chunks per utterance; excess raises rather than truncates."""

    max_phone_positions: int = 2048
    """Per-segment encoder phone bound, limiting quadratic attention allocation."""

    profiling: bool = False
    """Enable OpenVINO node-level counters; disabled for headline timings."""

    gpu_precision: Literal["f16", "f32"] = "f16"
    """GPU arithmetic hint; CPU retains FP32 accumulation even with packed weights."""

    release_after_call: bool = False
    """Release all loaded stage intermediates after synthesis, retaining weights."""

    def __post_init__(self) -> None:
        """Reject invalid resource, precision-independent synthesis and RNG settings."""
        if type(self.threads) is not int or self.threads not in (1, 2):
            raise ValueError("threads must be one or two")
        if self.backend not in ("cpu", "hybrid", "GPU"):
            raise ValueError("Unknown synthesis backend")
        if self.gpu_precision not in ("f16", "f32"):
            raise ValueError("GPU precision must be f16 or f32")
        if type(self.release_after_call) is not bool:
            raise TypeError("release_after_call must be a boolean")
        if not math.isfinite(self.speed) or not 0.5 <= self.speed <= 2:
            raise ValueError("speed must be finite and in [0.5, 2]")
        if not math.isfinite(self.sdp_ratio) or not 0 <= self.sdp_ratio <= 1:
            raise ValueError("sdp_ratio must be finite and in [0, 1]")
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.noise_scale, self.noise_scale_w)
        ):
            raise ValueError("Noise scales must be finite and nonnegative")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        if type(self.max_frames) is not int or self.max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        if type(self.max_total_frames) is not int or self.max_total_frames < 1:
            raise ValueError("max_total_frames must be a positive integer")
        if type(self.max_segments) is not int or self.max_segments < 1:
            raise ValueError("max_segments must be a positive integer")
        if type(self.max_phone_positions) is not int or self.max_phone_positions < 1:
            raise ValueError("max_phone_positions must be a positive integer")
        if not self.voice or not self.chinese_voice:
            raise ValueError("Speaker names must be nonempty")


class Stage:
    """One synchronous graph request retaining outputs only until the next call."""

    device: str
    """Explicit CPU/GPU execution device, with no implicit fallback."""

    precision_hint: str
    """Requested arithmetic hint, not proof of kernel precision or packed residency."""

    compiled: ov.CompiledModel
    """Resident graph, constants and explicit single-stream execution backend."""

    request: ov.InferRequest
    """Owned serialized request with reusable activation storage."""

    inputs: set[str]
    """Required port names in the exported tensor contract."""

    def __init__(
        self, core: ov.Core, path: Path, config: TTSConfig, device: str
    ) -> None:
        """Compile a local graph, preserving FP32 precision and bounded CPU streams.

        Args:
            core:
                TTS-owned OpenVINO core shared across stages/languages.

            path:
                Existing XML artifact, with its matching binary in the same directory.

            config:
                Validated CPU budget and profiling selection.

            device:
                Explicit CPU or GPU device; failure propagates without fallback.
        """
        properties: dict[str, Any] = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_PRECISION_HINT": "f32"
            if device == "CPU"
            else config.gpu_precision,
            "PERF_COUNT": config.profiling,
        }
        if device == "CPU":
            properties.update(
                INFERENCE_NUM_THREADS=config.threads, ENABLE_CPU_PINNING=False
            )
        self.device = device
        self.precision_hint = properties["INFERENCE_PRECISION_HINT"]
        graph = core.read_model(path)
        self.compiled = core.compile_model(graph, device, properties)
        self.request = self.compiled.create_infer_request()
        self.inputs = {port.get_any_name() for port in self.compiled.inputs}

    def run(self, values: dict[str, NDArray[Any]]) -> list[NDArray[np.float32]]:
        """Execute one stage, returning borrowed output arrays in declared port order.

        Args:
            values:
                Contiguous owned input arrays keyed by exact exported input names.
                They stay alive until this synchronous call completes.
        """
        if set(values) != self.inputs:
            raise ValueError(
                f"Graph inputs mismatch: required {self.inputs}, supplied {set(values)}"
            )
        result = self.request.infer(values, share_inputs=True, share_outputs=True)
        return [
            np.asarray(result[port], dtype=np.float32) for port in self.compiled.outputs
        ]


@dataclass(slots=True)
class Resources:
    """One language's checkpoint-specific text frontend and neural stages."""

    frontend: Frontend
    """Native phoneme/tokenizer implementation with explicit prepared local assets."""

    stages: dict[str, Stage]
    """Resident bert, encoder, flow and decoder requests."""

    speakers: dict[str, int]
    """Checkpoint-defined names and integer IDs; language vocabularies stay separate."""

    rate: int
    """Checkpoint waveform sample rate in Hz."""

    hop: int
    """Waveform samples generated per latent frame."""

    manifest: dict[str, Any]
    """Preparation provenance and tensor contracts used for offline validation."""


def expand_prior(
    mean: NDArray[np.float32],
    log_scale: NDArray[np.float32],
    durations: NDArray[np.int64],
    max_frames: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Expand priors with exact duration gather, avoiding quadratic dense alignment.

    Args:
        mean:
            FP32 prior tensor shaped (1, 192, phone_positions).

        log_scale:
            FP32 prior log standard deviations with the same shape as mean.

        durations:
            Nonnegative INT64 durations shaped (phone_positions,).

        max_frames:
            Positive upper bound on total frames; excess raises instead of truncating.
    """
    if mean.ndim != 3 or mean.shape[:2] != (1, 192) or mean.shape != log_scale.shape:
        raise ValueError("Invalid prior tensor geometry")
    if durations.dtype != np.int64 or durations.shape != (mean.shape[2],):
        raise ValueError("Invalid duration tensor geometry")
    if np.any(durations < 0):
        raise ValueError("Durations must be nonnegative")
    total = int(durations.sum())
    if not 0 < total <= max_frames:
        raise ValueError(f"Generated frames {total} outside [1,{max_frames}]")
    indices = np.repeat(np.arange(durations.size), durations)
    return np.ascontiguousarray(mean[:, :, indices]), np.ascontiguousarray(
        log_scale[:, :, indices]
    )


class TTS:
    """Native bilingual Melo inference and compatible ordered playback ownership."""

    config: TTSConfig
    """Validated graph, voice, noise and CPU budget configuration."""

    last_timings: dict[str, Any]
    """Last full synthesis stage wall times, geometry and optional per-node counters."""

    _core: ov.Core | None
    """Shared core, absent for playback-only instances or after cleanup."""

    _resources: dict[str, Resources]
    """Lazily loaded language resources; serialized and retained between requests."""

    _lock: threading.RLock
    """Protects inference, lazy initialization, playback submission and close."""

    _closed: bool
    """Whether inference ownership ended; close is idempotent."""

    _playback_only: bool
    """Whether construction intentionally omitted all model/graph access."""

    _output: AudioPlayback | None
    """Shared 24-kHz ordered playback queue, absent while idle."""

    _buffer_seconds: float | None
    """Capacity of the active playback queue, or None while idle."""

    def __init__(
        self, config: TTSConfig | None = None, *, playback_only: bool = False
    ) -> None:
        """Initialize ownership without loading unrelated languages or starting workers.

        Args:
            config:
                Prepared model root and validated controls; None uses defaults.

            playback_only:
                Support raw playback without graph, tokenizer or model requirements.
        """
        self.config = TTSConfig() if config is None else config
        if not playback_only:
            configure_cpu_budget(self.config.threads)
        self.last_timings = {}
        self._resources = {}
        self._lock = threading.RLock()
        self._closed = False
        self._playback_only = playback_only
        self._output = None
        self._buffer_seconds = None
        self._core = None if playback_only else ov.Core()

    def _load(self, language: Literal["EN", "ZH"]) -> Resources:
        """Load and validate only the requested checkpoint once, without downloads.

        Args:
            language:
                EN or ZH checkpoint selector determined by text routing.
        """
        if language in self._resources:
            return self._resources[language]
        assert self._core is not None
        root = self.config.model_path / language
        for name in (
            "manifest.json",
            "config.json",
            "tokenizer.json",
            "bert.xml",
            "bert.bin",
            "encoder.xml",
            "encoder.bin",
            "flow.xml",
            "flow.bin",
            "decoder.xml",
            "decoder.bin",
        ):
            if not (root / name).is_file():
                raise FileNotFoundError(
                    f"Prepare native Melo artifacts first: {root / name}"
                )
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest.get("version") != 1
            or manifest.get("language") != language
            or manifest.get("backend") != "openvino"
            or manifest.get("precision") not in ("FP32", "FP16", "INT8")
            or manifest.get("source", {}).get("revision")
            != "209145371cff8fc3bd60d7be902ea69cbdb7965a"
        ):
            raise ValueError("Unsupported native Melo artifact schema")
        required = {
            "config.json",
            "tokenizer.json",
            "frontend-assets.json",
            "cmudict.rep",
            "g2p/g2p.npz",
            "g2p/homographs.en",
            "g2p/cmudict.json",
            "g2p/tagger/weights.json",
            "g2p/tagger/tagdict.json",
            "g2p/tagger/classes.json",
            *(
                f"{stage}.{suffix}"
                for stage in ("bert", "encoder", "flow", "decoder")
                for suffix in ("xml", "bin")
            ),
        }
        if language == "ZH":
            required.add("opencpop-strict.txt")
        if not required <= manifest["files"].keys():
            raise ValueError("Melo artifact integrity inventory is incomplete")
        for name, metadata in manifest["files"].items():
            path = root / name
            if (
                not path.resolve().is_relative_to(root.resolve())
                or path.stat().st_size != metadata["bytes"]
            ):
                raise ValueError(f"Invalid Melo artifact: {name}")
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != metadata["sha256"]:
                raise ValueError(f"Corrupt Melo artifact: {name}")
        configuration = json.loads((root / "config.json").read_text())
        data = configuration["data"]
        if data["sampling_rate"] != 44100 or data["hop_length"] != 512:
            raise ValueError("Unexpected Melo waveform geometry")
        # Validate/load offline text resources before allocating neural execution memory.
        english = None
        for existing in self._resources.values():
            english_files = {
                name: metadata
                for name, metadata in manifest["files"].items()
                if name == "cmudict.rep" or name.startswith("g2p/")
            }
            existing_files = {
                name: metadata
                for name, metadata in existing.manifest["files"].items()
                if name == "cmudict.rep" or name.startswith("g2p/")
            }
            if english_files == existing_files:
                english = existing.frontend.english
                break
        # The enclosing inference lock also serializes the shared OOV cache.
        frontend = Frontend(root, language, english=english)
        stages = {}
        for name in ("bert", "encoder", "flow", "decoder"):
            device = (
                "GPU"
                if self.config.backend == "GPU"
                or (self.config.backend == "hybrid" and name == "decoder")
                else "CPU"
            )
            stages[name] = Stage(self._core, root / f"{name}.xml", self.config, device)
        resources = Resources(
            frontend,
            stages,
            dict(data["spk2id"]),
            data["sampling_rate"],
            data["hop_length"],
            manifest,
        )
        self._resources[language] = resources
        logger.info(
            "tts2.load language=%s threads=%d backend=%s",
            language,
            self.config.threads,
            self.config.backend,
        )
        return resources

    def synthesize(self, text: str) -> tuple[NDArray[np.float32], int]:
        """Return owned speech and optionally release intermediate storage on exit.

        All stages finish consuming borrowed tensors before cleanup runs under
        the inference lock. Cleanup also runs after failures; weights stay loaded.
        Successful timing totals include release, but exclude audio playback.

        Args:
            text:
                Nonempty speakable text; Han-containing speech uses the ZH frontend.

        """
        with self._lock:
            if self._closed or self._playback_only:
                raise RuntimeError("TTS is closed or playback-only")
            started = time.perf_counter()
            try:
                result = self._synthesize(text)
            finally:
                before_release = time.perf_counter()
                if self.config.release_after_call:
                    self.release_memory()
            if self.config.release_after_call:
                self.last_timings["release_memory"] = (
                    time.perf_counter() - before_release
                )
            self.last_timings["total"] = time.perf_counter() - started
            return result

    def release_memory(self) -> None:
        """Serialize release of every loaded stage's idle intermediate storage.

        Outputs must be consumed or copied before calling this method. Native
        allocators may retain freed pages; this is not a guaranteed RSS reduction.
        Every loaded stage is attempted even if native release fails. Failures
        carry language/stage notes; one is re-raised unchanged, and multiple
        failures are propagated together as an ExceptionGroup.
        """
        with self._lock:
            if self._closed or self._playback_only:
                raise RuntimeError("TTS is closed or playback-only")
            errors: list[Exception] = []
            for language, resources in self._resources.items():
                for name, stage in resources.stages.items():
                    try:
                        stage.compiled.release_memory()
                    except Exception as error:  # noqa: BLE001 - aggregate native failures
                        error.add_note(
                            f"TTS release_memory language={language} stage={name}"
                        )
                        errors.append(error)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise ExceptionGroup("TTS stage memory release failed", errors)

    def _synthesize(self, text: str) -> tuple[NDArray[np.float32], int]:
        """Return native-rate mono FP32 speech with language routing and explicit noise.

        Args:
            text:
                Nonempty speakable text; Han-containing speech uses the ZH mixed frontend.
        """
        with self._lock:
            if self._closed or self._playback_only:
                raise RuntimeError("TTS is closed or playback-only")
            if not text.strip():
                raise ValueError("text must be nonempty")
            started = time.perf_counter()
            text = prepare_speech_text(text)
            language: Literal["EN", "ZH"] = "ZH" if contains_han(text) else "EN"
            load_started = time.perf_counter()
            resources = self._load(language)
            load = time.perf_counter() - load_started
            speaker = (
                self.config.chinese_voice if language == "ZH" else self.config.voice
            )
            if speaker not in resources.speakers:
                raise ValueError(f"Unknown {language} speaker: {speaker}")
            rng = np.random.default_rng(self.config.seed)
            parts = []
            total_frames = 0
            sentences = resources.frontend.split(text)
            if len(sentences) > self.config.max_segments:
                raise ValueError("Utterance exceeds the configured segment limit")
            timings: dict[str, Any] = {
                "language": language,
                "load": load,
                "segments": [],
                "weight_precision": resources.manifest["precision"],
                "stage_execution": {
                    name: {
                        "device": stage.device,
                        "precision_hint": stage.precision_hint,
                    }
                    for name, stage in resources.stages.items()
                },
            }
            for sentence in sentences:
                segment_started = time.perf_counter()
                prepared = resources.frontend.prepare(sentence)
                if prepared.input_ids.shape[-1] > 512:
                    raise ValueError("Segment exceeds BERT's 512-token position limit")
                if prepared.phones.shape[-1] > self.config.max_phone_positions:
                    raise ValueError(
                        "Segment exceeds the configured phone-position limit"
                    )
                frontend_seconds = time.perf_counter() - segment_started
                stage = resources.stages["bert"]
                inputs = {"input_ids": prepared.input_ids}
                if "attention_mask" in stage.inputs:
                    inputs["attention_mask"] = np.ones_like(prepared.input_ids)
                if "token_type_ids" in stage.inputs:
                    inputs["token_type_ids"] = np.zeros_like(prepared.input_ids)
                before = time.perf_counter()
                features = stage.run(inputs)[0]
                bert_seconds = time.perf_counter() - before
                if features.shape != (1, prepared.word2ph.size, 768):
                    raise RuntimeError("BERT/token alignment mismatch")
                before = time.perf_counter()
                repeated = np.repeat(features[0], prepared.word2ph, axis=0).T[None]
                if repeated.shape[-1] != prepared.phones.shape[-1]:
                    raise RuntimeError("BERT/phone alignment mismatch")
                noise = rng.standard_normal(
                    (1, 2, prepared.phones.shape[-1]), dtype=np.float32
                )
                encoded = resources.stages["encoder"].run(
                    {
                        "phones": prepared.phones,
                        "tones": prepared.tones,
                        "languages": prepared.languages,
                        "phone_features": np.ascontiguousarray(repeated),
                        "speaker_id": np.array(
                            [resources.speakers[speaker]], dtype=np.int64
                        ),
                        # Encoder artifacts multiply their noise input by the pinned 0.8.
                        "duration_noise": noise * (self.config.noise_scale_w / 0.8),
                    }
                )
                mean, log_scale, log_dp, log_sdp, condition = encoded
                encoder_seconds = time.perf_counter() - before
                before = time.perf_counter()
                log_duration = (
                    self.config.sdp_ratio * log_sdp
                    + (1 - self.config.sdp_ratio) * log_dp
                )
                if not np.isfinite(log_duration).all() or np.any(
                    log_duration > math.log(self.config.max_frames)
                ):
                    raise RuntimeError("Invalid generated duration")
                durations = np.ceil(
                    np.exp(log_duration.reshape(-1)) / self.config.speed
                ).astype(np.int64)
                expanded_mean, expanded_log_scale = expand_prior(
                    mean, log_scale, durations, self.config.max_frames
                )
                total_frames += expanded_mean.shape[-1]
                if total_frames > self.config.max_total_frames:
                    raise ValueError(
                        "Utterance exceeds the configured total-frame limit"
                    )
                epsilon = rng.standard_normal(expanded_mean.shape, dtype=np.float32)
                latent = (
                    expanded_mean
                    + epsilon * np.exp(expanded_log_scale) * self.config.noise_scale
                )
                expansion_seconds = time.perf_counter() - before
                before = time.perf_counter()
                transformed = resources.stages["flow"].run(
                    {"latent": latent, "speaker_condition": condition}
                )[0]
                flow_seconds = time.perf_counter() - before
                before = time.perf_counter()
                output = resources.stages["decoder"].run(
                    {"latent": transformed, "speaker_condition": condition}
                )[0]
                decoder_seconds = time.perf_counter() - before
                expected = expanded_mean.shape[-1] * resources.hop
                if output.shape != (1, 1, expected):
                    raise RuntimeError(
                        f"Unexpected decoder waveform shape {output.shape}"
                    )
                parts.extend(
                    (
                        output.reshape(-1).copy(),
                        np.zeros(
                            int(resources.rate * 0.05 / self.config.speed),
                            dtype=np.float32,
                        ),
                    )
                )
                profile: dict[str, Any] = {
                    "frontend": frontend_seconds,
                    "bert": bert_seconds,
                    "encoder": encoder_seconds,
                    "expand": expansion_seconds,
                    "flow": flow_seconds,
                    "decoder": decoder_seconds,
                    "bert_tokens": prepared.word2ph.size,
                    "phone_positions": durations.size,
                    "latent_frames": expanded_mean.shape[-1],
                }
                if self.config.profiling:
                    profile["nodes"] = {
                        name: [
                            {
                                "name": row.node_name,
                                "type": row.node_type,
                                "exec_type": row.exec_type,
                                "real_microseconds": row.real_time.total_seconds()
                                * 1e6,
                            }
                            for row in stage.request.get_profiling_info()
                        ]
                        for name, stage in resources.stages.items()
                    }
                    profile["compiled_nodes"] = {
                        name: [
                            {
                                "name": node.get_friendly_name(),
                                "type": node.get_type_name(),
                                "runtime_info": {
                                    key: str(value)
                                    for key, value in node.get_rt_info().items()
                                },
                            }
                            for node in stage.compiled.get_runtime_model().get_ordered_ops()
                        ]
                        for name, stage in resources.stages.items()
                    }
                timings["segments"].append(profile)
            if not parts:
                raise ValueError("Text produced no speakable segments")
            audio = np.concatenate(parts)
            if not np.isfinite(audio).all():
                raise RuntimeError("Melo returned nonfinite audio")
            elapsed = time.perf_counter() - started
            timings.update(total=elapsed, audio_seconds=audio.size / resources.rate)
            self.last_timings = timings
            logger.info(
                "tts2.synthesize language=%s backend=%s threads=%d seconds=%.4f duration=%.3f",
                language,
                self.config.backend,
                self.config.threads,
                elapsed,
                audio.size / resources.rate,
            )
            return audio, resources.rate

    def play(
        self,
        text: str,
        *,
        blocking: bool = True,
        buffer_seconds: float = 1.0,
        cancelled: threading.Event | None = None,
    ) -> None:
        """Synthesize on the caller and submit to the compatible bounded playback queue.

        Args:
            text:
                Speakable text passed to synthesize.

            blocking:
                Drain accepted playback before returning when True.

            buffer_seconds:
                Positive capacity in seconds; must match an active queue.

            cancelled:
                Optional event checked before synthesis and at the playback boundary;
                cancellation skips new submissions but drains accepted audio.
        """
        with self._lock:
            if cancelled is not None and cancelled.is_set():
                return
            samples, rate = self.synthesize(text)
            self.play_samples(
                samples,
                rate,
                blocking=blocking,
                buffer_seconds=buffer_seconds,
                cancelled=cancelled,
            )

    def play_samples(
        self,
        samples: NDArray[np.float32],
        sample_rate: int,
        *,
        blocking: bool = True,
        buffer_seconds: float = 1.0,
        cancelled: threading.Event | None = None,
    ) -> None:
        """Resample once to 24 kHz; cancellation drains already-accepted ordered audio.

        Args:
            samples:
                Nonempty finite mono FP32 samples; not mutated by playback.

            sample_rate:
                Positive integer source rate in Hz; 24 kHz bypasses resampling.

            blocking:
                Drain and close the queue after submission when True.

            buffer_seconds:
                Finite queue capacity holding at least one 24-kHz sample.

            cancelled:
                Optional event checked before resampling and each 100-ms submission.
        """
        if (
            samples.dtype != np.float32
            or samples.ndim != 1
            or not samples.size
            or not np.isfinite(samples).all()
        ):
            raise ValueError("Expected finite nonempty mono float32 samples")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        with self._lock:
            if self._closed:
                raise RuntimeError("TTS is closed")
            if not math.isfinite(buffer_seconds) or buffer_seconds < 1 / 24000:
                raise ValueError("buffer_seconds must hold at least one frame")
            if self._output is not None and buffer_seconds != self._buffer_seconds:
                raise ValueError("Drain playback before changing queue capacity")
            if cancelled is not None and cancelled.is_set():
                return
            if sample_rate != 24000:
                divisor = math.gcd(sample_rate, 24000)
                samples = np.asarray(
                    resample_poly(samples, 24000 // divisor, sample_rate // divisor),
                    dtype=np.float32,
                )
            try:
                if self._output is None:
                    self._output = AudioPlayback(24000, buffer_seconds)
                    self._buffer_seconds = buffer_seconds
                for offset in range(0, samples.size, 2400):
                    if cancelled is not None and cancelled.is_set():
                        break
                    self._output.submit(samples[offset : offset + 2400])
                if blocking:
                    self.wait_playback()
            except Exception:
                logger.exception("tts2.playback status=failed")
                output, self._output = self._output, None
                self._buffer_seconds = None
                if output is not None:
                    try:
                        output.close()
                    except Exception:
                        logger.exception("tts2.playback status=cleanup_failed")
                raise

    def wait_playback(self) -> None:
        """Drain and release the shared playback queue, including after cancellation."""
        with self._lock:
            output, self._output = self._output, None
            self._buffer_seconds = None
            if output is not None:
                output.close()

    def write_wav(self, text: str, output: Path) -> None:
        """Write native-rate mono PCM16 WAV, preserving the returned waveform geometry.

        Args:
            text:
                Nonempty text to synthesize.

            output:
                Destination WAV in an existing directory; an existing file is replaced.
        """
        audio, rate = self.synthesize(text)
        sf.write(output, audio, rate, subtype="PCM_16")

    def close(self) -> None:
        """Drain accepted playback and release every graph; idempotent after failure."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.wait_playback()
            finally:
                self._resources.clear()
                self._core = None

    def __enter__(self) -> Self:
        """Retain this engine's inference/playback ownership until context exit."""
        if self._closed:
            raise RuntimeError("TTS is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release ownership without suppressing the calling block's failure.

        Args:
            exc_type:
                Exception class from the block, or None for normal completion.

            exc_value:
                Block exception instance, which remains unsuppressed.

            traceback:
                Block traceback; cleanup retains no references to it.

        """
        self.close()
