"""Kokoro speech synthesis with hybrid CPU/GPU inference and WAV or system audio output."""

import argparse
import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import soundfile as sf
from kokoro_onnx import Kokoro
from numpy.typing import NDArray

from .chinese import contains_han
from .chinese import phoneme_batches as chinese_batches
from .chinese_g2p import DEFAULT_SPEECH_PYTHON, ChineseG2P
from .logging import configure_logging, get_logger
from .runtime import configure_cpu_budget
from .tts_audio import AudioPlayback
from .tts_gpu import SharedGPU, replace_convolutions
from .tts_kernels import DEFAULT_KERNEL, fuse_cpu_activations

logger = get_logger(__name__)
DEFAULT_MODEL = Path(".cache/hoast/tts/kokoro-v1.0.onnx")
DEFAULT_VOICES = Path(".cache/hoast/tts/voices-v1.0.bin")
DEFAULT_CHINESE_MODEL = Path(".cache/hoast/tts/chinese")


@dataclass(slots=True, frozen=True)
class _ModelInput:
    """Input name consumed by the pinned Kokoro frontend's session interface."""

    name: str
    """Model tensor name."""


class _OpenVINOSession:
    """Hybrid OpenVINO adapter for Kokoro's narrow session interface."""

    _model_path: str
    """Local model filename required by Kokoro's artifact validation."""

    _compiled: ov.CompiledModel | None
    """CPU graph with GPU decoder convolutions, absent after close."""

    _request: ov.InferRequest | None
    """Reusable synchronous request, absent after close; callers serialize inference."""

    _inputs: list[_ModelInput]
    """Names used by Kokoro to select the export's token input convention."""

    _input_aliases: dict[str, str]
    """Map frontend tensor names to ONNX or native OpenVINO IR input names."""

    def __init__(
        self,
        model_path: Path,
        threads: int,
        activation_kernel: Path | None = DEFAULT_KERNEL,
        *,
        gpu: SharedGPU,
    ) -> None:
        """Compile a local CPU graph with verified decoder-only GPU convolution.

        Args:
            model_path:
                Existing Kokoro ONNX graph or native OpenVINO XML IR.

            threads:
                One or two CPU inference threads; one stream is used.

            activation_kernel:
                Optional prepared AVX2 activation library; None disables fusion.

            gpu:
                TTS-owned GPU executor, shared by English and lazy Chinese sessions.

        """
        self._model_path = str(model_path)
        core = ov.Core()
        graph = core.read_model(model_path)
        replace_convolutions(graph, gpu)
        fuse_cpu_activations(core, graph, activation_kernel)
        self._compiled = core.compile_model(
            graph,
            "CPU",
            {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
                "INFERENCE_NUM_THREADS": threads,
                "INFERENCE_PRECISION_HINT": "f32",
                "ENABLE_CPU_PINNING": False,
            },
        )
        self._request = self._compiled.create_infer_request()
        names = [port.get_any_name() for port in self._compiled.inputs]
        if set(names) == {"input_ids", "ref_s", "speed"}:
            # Present the float-speed convention used by native PyTorch exports.
            self._input_aliases = {
                "tokens": "input_ids",
                "style": "ref_s",
                "speed": "speed",
            }
        else:
            self._input_aliases = {name: name for name in names}
        self._inputs = [_ModelInput(alias) for alias in self._input_aliases]

    def get_inputs(self) -> list[_ModelInput]:
        """Return frontend input names, including native IR aliases."""
        return self._inputs

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, Any]
    ) -> list[NDArray[np.float32]]:
        """Run one request and copy outputs so later batches cannot overwrite them.

        Args:
            output_names:
                Must be None; Kokoro requests the model's full output list.

            input_feed:
                Kokoro token, style and speed tensors keyed by frontend name;
                native IR aliases are applied before inference.

        """
        if output_names is not None:
            raise ValueError("Kokoro OpenVINO inference requires all outputs")
        if self._request is None or self._compiled is None:
            raise RuntimeError("TTS model session is closed")
        result = self._request.infer(
            {self._input_aliases[name]: value for name, value in input_feed.items()}
        )
        return [
            np.array(result[port], dtype=np.float32, copy=True)
            for port in self._compiled.outputs
        ]

    def close(self) -> None:
        """Release compiled model ownership before its shared GPU executor is closed."""
        self._request = None
        self._compiled = None


def phoneme_batches(phonemes: str) -> list[str]:
    """Split phonemes below Kokoro's limit without dropping unpunctuated text.

    Args:
        phonemes:
            Phonemizer output. Prefer word boundaries; split oversized words
            at 450 characters to stay below the model's 510-token limit.

    """
    remaining = phonemes.strip()
    batches: list[str] = []
    while len(remaining) > 450:
        boundary = remaining.rfind(" ", 0, 451)
        if boundary <= 0:
            boundary = 450
        batches.append(remaining[:boundary])
        remaining = remaining[boundary:].lstrip()
    if remaining:
        batches.append(remaining)
    return batches


@dataclass(slots=True, frozen=True)
class TTSConfig:
    """Local Kokoro artifacts and bounded CPU/GPU hybrid synthesis settings."""

    model_path: Path = DEFAULT_MODEL
    """Kokoro ONNX or OpenVINO XML model; defaults to the v1.0 FP32 ONNX export."""

    voices_path: Path = DEFAULT_VOICES
    """Matching v1.0 NumPy voice archive."""

    threads: int = 2
    """One or two CPU inference threads; each engine uses one execution stream."""

    voice: str = "af_heart"
    """Kokoro voice identifier; defaults to American English."""

    language: str = "en-us"
    """eSpeak phonemizer language matching the selected voice."""

    speed: float = 1.0
    """Speech speed multiplier in [0.5, 2.0]."""

    activation_kernel: Path | None = DEFAULT_KERNEL
    """Optional prepared AVX2 fusion library; missing files use native CPU operations."""

    chinese_model_dir: Path = DEFAULT_CHINESE_MODEL
    """Prepared official Chinese v1.1 model, configuration, manifest and voices."""

    chinese_voice: str = "zf_001"
    """Chinese voice used for Han-containing utterances, including English insertions."""

    chinese_python: Path = DEFAULT_SPEECH_PYTHON
    """Prepared Python 3.12 worker interpreter for official Chinese phonemization."""

    def __post_init__(self) -> None:
        """Reject invalid thread counts, labels, speeds and voice identifiers."""
        if self.threads not in (1, 2):
            raise ValueError("threads must be one or two")
        if not math.isfinite(self.speed) or not 0.5 <= self.speed <= 2.0:
            raise ValueError("speed must be finite and between 0.5 and 2.0")
        if not self.voice.strip() or not self.language.strip():
            raise ValueError("voice and language must be nonempty")
        if re.fullmatch(r"[A-Za-z0-9_]+", self.chinese_voice) is None:
            raise ValueError("Chinese voice must be an artifact identifier")


@dataclass(slots=True)
class _ChineseResources:
    """Cached Chinese inference, vocabulary, speaker table and official G2P worker."""

    session: _OpenVINOSession
    """Loaded CPU model with canonical input aliases."""

    vocab: dict[str, int]
    """Pinned v1.1 phoneme-to-token mapping."""

    voice: NDArray[np.float32]
    """Original style table shaped (payload_lengths, 1, 256), indexed by N-1."""

    phonemizer: ChineseG2P
    """Lazily started, reusable official Chinese frontend."""


class TTS:
    """Reusable synthesizer owning one shared GPU executor and serialized model sessions."""

    config: TTSConfig
    """Validated synthesis settings."""

    model: Kokoro
    """Kokoro frontend with an explicitly bounded CPU inference session."""

    _gpu: SharedGPU
    """One GPU context and kernel cache shared by both language models."""

    _session: _OpenVINOSession | None
    """English model session, absent during initialization or after close."""

    _closed: bool
    """Whether all owned inference resources have been released."""

    _output: AudioPlayback | None
    """Bounded playback queue, absent until buffered playback starts."""

    _output_buffer_seconds: float | None
    """Queue capacity in seconds for the active playback worker."""

    _chinese: _ChineseResources | None
    """Chinese resources, absent until the first Han-containing utterance."""

    _chinese_lock: threading.RLock
    """Serializes synthesis, lazy initialization and resource release."""

    def __init__(self, config: TTSConfig | None = None) -> None:
        """Load local English artifacts and initialize the UHD 630 hybrid runtime.

        Args:
            config:
                Prepared artifact paths, thread budget and voice settings;
                None uses defaults.

        """
        config = TTSConfig() if config is None else config
        logger.debug("tts.load settings=%r", config)
        for path in (config.model_path, config.voices_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing TTS artifact: {path}")
        self.config = config
        self._chinese = None
        self._chinese_lock = threading.RLock()
        self._session = None
        self._closed = False
        self._output = None
        self._output_buffer_seconds = None
        start = time.perf_counter()
        self._gpu = SharedGPU()
        try:
            self._session = _OpenVINOSession(
                config.model_path,
                config.threads,
                config.activation_kernel,
                gpu=self._gpu,
            )
            # Kokoro uses a structural session interface; its annotation names ORT.
            self.model = Kokoro.from_session(self._session, str(config.voices_path))  # pyright: ignore[reportArgumentType]
            if config.voice not in self.model.get_voices():
                raise ValueError(f"Unknown Kokoro voice: {config.voice}")
        except BaseException:
            self.close()
            raise
        logger.info(
            "tts.load status=ok threads=%d seconds=%.3f model=%s runtime=hybrid",
            config.threads,
            time.perf_counter() - start,
            config.model_path,
        )

    def synthesize(self, text: str) -> tuple[NDArray[np.float32], int]:
        """Synthesize text, routing Han-containing utterances to cached Chinese v1.1.

        Mixed text uses the Chinese model with English phoneme insertions to keep
        one voice throughout. English-only text retains the configured voice. Returned
        samples are mono float32 shaped (samples,), at 24 kHz.

        Args:
            text:
                Nonempty text to speak; language routing inspects original characters.

        """
        with self._chinese_lock:
            if self._closed:
                raise RuntimeError("TTS is closed")
            if not text.strip():
                raise ValueError("text must be nonempty")
            if not contains_han(text):
                return self._synthesize_english(text)
            return self._synthesize_chinese(text), 24000

    def _load_chinese(self) -> _ChineseResources:
        """Load prepared Chinese resources once without network access."""
        if self._chinese is not None:
            return self._chinese
        root = self.config.chinese_model_dir
        voice_path = root / "voices" / f"{self.config.chinese_voice}.npy"
        for path in (
            root / "model.xml",
            root / "model.bin",
            root / "config.json",
            root / "manifest.json",
            voice_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Prepare Chinese TTS first; missing {path}")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["repository"] != "hexgrad/Kokoro-82M-v1.1-zh"
            or manifest["voice_index"] != "phoneme_count_minus_one"
        ):
            raise ValueError("Unexpected Chinese model or voice-index convention")
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        vocab = config["vocab"]
        if not isinstance(vocab, dict) or any(
            not isinstance(key, str) or not isinstance(value, int)
            for key, value in vocab.items()
        ):
            raise ValueError("Invalid Chinese vocabulary")
        voice = np.load(voice_path, allow_pickle=False)
        if voice.dtype != np.float32 or voice.ndim != 3 or voice.shape[1:] != (1, 256):
            raise ValueError("Invalid Chinese voice table")
        session = _OpenVINOSession(
            root / "model.xml",
            self.config.threads,
            self.config.activation_kernel,
            gpu=self._gpu,
        )
        self._chinese = _ChineseResources(
            session,
            vocab,
            voice,
            ChineseG2P(self.config.chinese_python, self.config.language),
        )
        logger.info(
            "tts.chinese status=loaded model=%s voice=%s",
            root,
            self.config.chinese_voice,
        )
        return self._chinese

    def _synthesize_chinese(self, text: str) -> NDArray[np.float32]:
        """Synthesize official Chinese phonemes without discarding unknown symbols.

        Args:
            text:
                Han-containing utterance, possibly with English insertions.

        """
        with self._chinese_lock:
            start = time.perf_counter()
            resources = self._load_chinese()
            phonemes = resources.phonemizer.phonemize(text)
            unknown = set(phonemes) - resources.vocab.keys()
            if unknown or "❓" in phonemes:
                logger.debug("tts.chinese unknown_phonemes=%r", sorted(unknown))
                raise ValueError("Chinese text contains unsupported phonemes")
            parts: list[NDArray[np.float32]] = []
            for batch in chinese_batches(phonemes):
                ids = [resources.vocab[character] for character in batch]
                if not 0 < len(ids) <= 450 or len(ids) > resources.voice.shape[0]:
                    raise ValueError(
                        "Chinese phoneme batch exceeds model or voice context"
                    )
                output = resources.session.run(
                    None,
                    {
                        "tokens": np.array([[0, *ids, 0]], dtype=np.int64),
                        "style": resources.voice[len(ids) - 1],
                        "speed": np.array([self.config.speed], dtype=np.float32),
                    },
                )[0]
                if output.ndim != 1 or not output.size:
                    raise RuntimeError("Chinese model returned invalid audio")
                parts.append(output)
            samples = np.concatenate(parts)
            logger.info(
                "tts.chinese status=ok characters=%d seconds=%.3f audio_seconds=%.3f",
                len(text),
                time.perf_counter() - start,
                samples.size / 24000,
            )
            return samples

    def close(self) -> None:
        """Drain playback and release inference resources; safe to repeat."""
        with self._chinese_lock:
            if self._closed:
                return
            self._closed = True
            chinese, self._chinese = self._chinese, None
            try:
                try:
                    self.wait_playback()
                finally:
                    if chinese is not None:
                        try:
                            chinese.phonemizer.close()
                        finally:
                            chinese.session.close()
            finally:
                session, self._session = self._session, None
                try:
                    if session is not None:
                        session.close()
                finally:
                    self._gpu.close()

    def _synthesize_english(self, text: str) -> tuple[NDArray[np.float32], int]:
        """Return mono float32 samples shaped (samples,) and sample rate in Hz.

        Long phoneme sequences are split without truncation. Trimming is disabled
        to preserve the model's boundary pauses and skip waveform postprocessing.

        Args:
            text:
                Nonempty text to speak using the configured voice and language.

        """
        if not text.strip():
            raise ValueError("text must be nonempty")
        start = time.perf_counter()
        phonemes = self.model.tokenizer.phonemize(text, self.config.language)
        batches = phoneme_batches(phonemes)
        if not batches:
            raise ValueError("text produced no speakable phonemes")
        parts: list[NDArray[np.float32]] = []
        rate = 24000
        for batch in batches:
            samples, rate = self.model.create(
                batch,
                voice=self.config.voice,
                speed=self.config.speed,
                lang=self.config.language,
                is_phonemes=True,
                trim=False,
            )
            samples = np.asarray(samples, dtype=np.float32)
            if samples.ndim != 1 or samples.size == 0 or rate != 24000:
                raise RuntimeError("Kokoro returned invalid mono 24 kHz audio")
            parts.append(samples)
        samples = np.concatenate(parts)
        elapsed = time.perf_counter() - start
        logger.info(
            "tts.synthesize status=ok characters=%d voice=%s seconds=%.3f "
            "duration=%.3f rtf=%.3f",
            len(text),
            self.config.voice,
            elapsed,
            samples.size / rate,
            elapsed / (samples.size / rate),
        )
        return samples, rate

    def play(
        self, text: str, *, blocking: bool = True, buffer_seconds: float = 1.0
    ) -> None:
        """Synthesize on the caller's thread and submit ordered mono 24 kHz audio.

        A playback-only worker feeds a low-latency device from a bounded PCM queue.
        Submission blocks when the queue fills; playback starts without waiting
        for it to fill. A full utterance is synthesized before submission and is
        outside the queue's memory bound. Device buffering is additional and small.
        Calls are serialized, including synthesis and submission. To serialize LLM
        and TTS compute, call this method directly from the LLM's text consumer.
        Audio-device failures propagate; failed playback is discarded, not retried.

        Args:
            text:
                Nonempty text to synthesize and play through system audio.

            blocking:
                If True, drain all submitted audio before returning. If False,
                return once samples are accepted; call wait_playback or close to
                drain at the end of a turn.

            buffer_seconds:
                Positive finite queued audio capacity in seconds, at least one
                24 kHz frame. Independent of device latency. Must stay unchanged
                until the active stream is drained.

        """
        if not math.isfinite(buffer_seconds) or buffer_seconds < 1 / 24000:
            raise ValueError(
                "buffer_seconds must be finite and hold at least one frame"
            )
        with self._chinese_lock:
            if (
                self._output is not None
                and buffer_seconds != self._output_buffer_seconds
            ):
                raise ValueError("Drain playback before changing buffer_seconds")
            samples, rate = self.synthesize(text)
            try:
                if self._output is None:
                    self._output = AudioPlayback(rate, buffer_seconds)
                    self._output_buffer_seconds = buffer_seconds
                self._output.submit(samples)
                logger.debug(
                    "tts.play status=submitted samples=%d rate=%d", samples.size, rate
                )
                if blocking:
                    self.wait_playback()
            except Exception:
                logger.exception("tts.play status=failed blocking=%s", blocking)
                output, self._output = self._output, None
                self._output_buffer_seconds = None
                if output is not None:
                    try:
                        output.close()
                    except Exception:
                        logger.exception("tts.play status=cleanup_failed")
                raise

    def wait_playback(self) -> None:
        """Drain submitted audio and close the device; safe when no stream is active.

        Blocks until pending samples finish playing. Device failures propagate;
        the stream is closed even when draining fails.
        """
        with self._chinese_lock:
            output, self._output = self._output, None
            self._output_buffer_seconds = None
            if output is None:
                return
            try:
                output.close()
                logger.info("tts.play status=completed")
            except Exception:
                logger.exception("tts.wait_playback status=failed")
                raise

    def write_wav(self, text: str, output: Path) -> None:
        """Synthesize and write a mono 24 kHz PCM16 WAV, replacing an existing file.

        Args:
            text:
                Nonempty text to synthesize.

            output:
                Destination file in an existing directory.

        """
        samples, rate = self.synthesize(text)
        sf.write(output, samples, rate, format="WAV", subtype="PCM_16")
        logger.info("tts.write_wav status=ok output=%s", output)


def main() -> None:
    """Speak text or piped stdin to a WAV or system audio, retaining durable diagnostics."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("text", nargs="?", help="Text to speak; omit to read stdin")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--output",
        type=Path,
        help="Destination mono 24 kHz PCM16 WAV; replaces an existing file",
    )
    destination.add_argument(
        "--play",
        action="store_true",
        help="Play through the default system audio device and wait until finished",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Prepared non-Chinese ONNX or OpenVINO XML model",
    )
    parser.add_argument(
        "--voices", type=Path, default=DEFAULT_VOICES, help="Non-Chinese voice archive"
    )
    parser.add_argument(
        "--voice", default="af_heart", help="Voice for non-Han utterances"
    )
    parser.add_argument(
        "--language",
        default="en-us",
        help="Non-Chinese phonemizer language; Han-containing utterances route to Mandarin",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed multiplier in [0.5, 2.0]"
    )
    parser.add_argument(
        "--threads",
        type=int,
        choices=(1, 2),
        default=2,
        help="CPU inference worker/core budget",
    )
    parser.add_argument(
        "--chinese-model-dir",
        type=Path,
        default=DEFAULT_CHINESE_MODEL,
        help="Prepared Chinese v1.1 artifacts; loaded lazily",
    )
    parser.add_argument(
        "--chinese-voice",
        default="zf_001",
        help="Voice for Han-containing and mixed-language utterances",
    )
    parser.add_argument(
        "--chinese-python",
        type=Path,
        default=DEFAULT_SPEECH_PYTHON,
        help="Prepared Python interpreter for the cached Chinese G2P worker",
    )
    parser.add_argument(
        "--no-fast-activations",
        action="store_true",
        help="Use native CPU activations instead of the optional AVX2 kernel",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(".cache/hoast/diagnostics/tts.log"),
        help="Durable timing and diagnostic log",
    )
    args = parser.parse_args()
    configure_logging(log_file=args.log_file, level="WARNING")
    try:
        configure_cpu_budget(args.threads)
        text = args.text if args.text is not None else sys.stdin.read()
        config = TTSConfig(
            args.model,
            args.voices,
            args.threads,
            args.voice,
            args.language,
            args.speed,
            activation_kernel=None if args.no_fast_activations else DEFAULT_KERNEL,
            chinese_model_dir=args.chinese_model_dir,
            chinese_voice=args.chinese_voice,
            chinese_python=args.chinese_python,
        )
        engine = TTS(config)
        try:
            if args.play:
                engine.play(text)
            else:
                engine.write_wav(text, args.output)
        finally:
            engine.close()
        sys.stdout.write(
            "Finished speaking.\n" if args.play else "Saved speech audio.\n"
        )
    except Exception:
        logger.exception(
            "tts.cli status=failed output=%s play=%s threads=%s",
            args.output,
            args.play,
            args.threads,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
