"""Serialized assistant turns with current music context and bounded system speech."""

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from .agent import LocalAgent
from .lfm import LFM2
from .logging import get_logger
from .memory import log_memory_usage
from .music import MusicAssistantError, MusicClient
from .prompts import build_system_prompt
from .session import Session
from .stt2 import STT
from .tts2 import TTS
from .voice import VoiceTurn

logger = get_logger(__name__)


def _startup_audio(
    samples: NDArray[np.float32], sample_rate: int
) -> NDArray[np.float32]:
    """Level startup speech only, then prepend 200 ms of output-settling silence.

    Target whole-clip RMS 0.07 (about -23 dBFS), capped at 4x gain and 0.8 peak
    before resampling, leaving headroom for interpolation. This is scalar gain,
    not clipping/compression; silence remains silence and input is never mutated.
    Full-array statistics are limited to this one short startup announcement.

    Args:
        samples:
            Nonempty mono float32 normalized speech returned by the synthesizer.

        sample_rate:
            Positive integer native waveform rate in Hz.

    """
    if samples.dtype != np.float32 or samples.ndim != 1 or not samples.size:
        raise ValueError("Startup speech must be nonempty mono float32 PCM")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("Startup sample rate must be a positive integer")
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if not np.isfinite(peak) or not np.isfinite(rms):
        raise ValueError("Startup speech must be finite")
    gain = min(4.0, 0.07 / rms, 0.8 / peak) if peak > 0 else 1.0
    logger.info(
        "warmup.announcement status=leveled source_rms=%.6f source_peak=%.6f "
        "gain=%.4f output_peak=%.6f preroll_seconds=0.2",
        rms,
        peak,
        gain,
        peak * gain,
    )
    return np.concatenate(
        (np.zeros(round(sample_rate * 0.2), dtype=np.float32), samples * gain)
    )


def bounded_speech(text: str, limit: int = 400) -> str:
    """Bound spoken text at a sentence or word boundary before synthesis.

    Prefer the last complete sentence within the bound; otherwise retain a whole
    word prefix and an ellipsis. Unbroken text is cut by Unicode character count.
    This is a text-size bound rather than an audio-duration guarantee.

    Args:
        text:
            Fully consumed, application-rendered agent output.

        limit:
            Maximum spoken Unicode characters, including a truncation ellipsis.

    """
    if type(limit) is not int or limit < 2:
        raise ValueError("Speech limit must be an integer of at least two characters")
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    prefix = normalized[:limit]
    ends = list(re.finditer(r"[.!?。！？](?=\s|$|[^\x00-\x7f])", prefix))
    if ends:
        result = prefix[: ends[-1].end()]
    else:
        prefix = normalized[: limit - 1]
        if " " in prefix:
            prefix = prefix.rsplit(" ", 1)[0]
        result = prefix.rstrip() + "…"
    logger.info(
        "speech.truncate original_characters=%d spoken_characters=%d",
        len(normalized),
        len(result),
    )
    logger.debug("speech.truncate full_text=%r", normalized)
    return result


def console_text(text: str) -> None:
    """Write a single application text event to stdout and flush for interactive use.

    Args:
        text:
            Transcript or grounded assistant answer, not raw tool diagnostics.

    """
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


@dataclass(slots=True)
class Assistant:
    """Own grounded agent state and refresh the system prompt before each request."""

    agent: LocalAgent
    """Loaded model session with configured weather, music, and light tools."""

    music: MusicClient | None = None
    """Optional live music status source; absence is explicitly stated in the prompt."""

    base_prompt: str = field(init=False)
    """Stable routing prompt without the per-request external-data snapshot."""

    def __post_init__(self) -> None:
        """Remember the model's original routing instructions."""
        self.base_prompt = self.agent.session.model.config.system_prompt

    def refresh_music(self) -> None:
        """Include only on/off music state or explicit unavailability in the system prompt.

        Project the provider result onto status/state keys so media labels cannot
        enter the background prompt, even if a provider supplies extra metadata.
        """
        context = {"status": "not_configured"} if self.music is None else None
        if self.music is not None:
            try:
                context = self.music.prompt_context()
            except MusicAssistantError:
                logger.warning("music.prompt_context status=unavailable", exc_info=True)
                context = {"status": "unavailable"}
        model = self.agent.session.model
        assert context is not None
        prompt = build_system_prompt(context, self.base_prompt)
        if isinstance(model, LFM2):
            model.config = replace(model.config, system_prompt=prompt)
        else:
            model.config = replace(model.config, system_prompt=prompt)

    def new_session(self) -> None:
        """Create fresh conversation state while reusing loaded model and tools."""
        self.agent.session = Session(self.agent.session.model)

    def reply(self, text: str, cancelled: Callable[[], bool] | None = None) -> str:
        """Refresh music context and exhaust the agent stream before returning its text.

        Args:
            text:
                Nonempty user request or recognized follow-up.

            cancelled:
                Thread-safe predicate suppressing undispatched work after disconnect.

        """
        if cancelled is not None and cancelled():
            return ""
        self.refresh_music()
        return "".join(self.agent.stream(text, cancelled=cancelled))


@dataclass(slots=True)
class VoiceApplication:
    """Serialize recognition, grounded agent work, and blocking system playback.

    Run synchronous methods on the application's single inference worker. Each
    accepted voice command uses fresh conversation state regardless of capture
    timestamps, then discards its history on completion, failure or cancellation.
    Loaded models and tools are reused. Playback already started is drained on
    cancellation, while later tool dispatch and playback are suppressed.
    """

    assistant: Assistant
    """Grounded text agent with live music context."""

    stt: STT
    """Reusable prepared speech recognizer."""

    tts: TTS
    """Reusable bilingual synthesizer and system-output playback engine."""

    output: Callable[[str], None] = console_text
    """Text sink for recognized input and full grounded output."""

    memory_diagnostics: bool = False
    """Whether to sample memory counters at voice stage boundaries."""

    def close(self) -> None:
        """Drain playback and release both speech engines on the inference worker.

        Recognition resources are released even if playback cleanup fails. Native
        engines make repeated cleanup safe; the caller retains ownership of LLM.
        """
        try:
            self.tts.close()
        finally:
            self.stt.close()

    def warmup(self) -> None:
        """Warm models without tool effects, then drain leveled initialization speech."""
        clips = []
        for language, text in (
            ("en", "The assistant is ready."),
            ("zh", "你好，我已经准备好了。"),
        ):
            logger.info("warmup.tts language=%s status=starting", language)
            clips.append(self.tts.synthesize(text))
            logger.info("warmup.tts language=%s status=ready", language)
        for samples, rate in clips:
            transcript = self.stt.transcribe_samples(samples, rate)
            logger.debug("warmup.stt transcript=%r sample_rate=%d", transcript, rate)
        logger.info("warmup.stt status=ready")
        self.assistant.refresh_music()
        result = self.assistant.agent.session.model.generate(
            "What is the weather today?"
        )
        logger.debug("warmup.llm generation=%r", result)
        self.assistant.new_session()
        logger.info("warmup.llm status=ready tools_dispatched=0")
        samples, rate = self.tts.synthesize("系统初始化完毕")
        self.tts.play_samples(_startup_audio(samples, rate), rate, blocking=True)
        logger.info("warmup.announcement status=completed")

    def transcribe(self, pcm: bytes) -> str:
        """Recognize PCM, sampling optional memory counters even on failure.

        Args:
            pcm:
                Nonempty mono signed little-endian int16 PCM at 16 kHz.

        """
        if self.memory_diagnostics:
            log_memory_usage("voice.transcribe.before")
        try:
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            return self.stt.transcribe_samples(samples, 16000)
        finally:
            if self.memory_diagnostics:
                log_memory_usage("voice.transcribe.after")

    def respond(self, turn: VoiceTurn) -> None:
        """Run one independent command, drain speech, and discard its history.

        Full grounded text goes to the text sink. Expected top-level turn failures
        are logged with tracebacks and produce a brief spoken failure. A playback
        error propagates to the native pipeline handler. Every accepted turn gets
        fresh state, cleaned up even on empty output or sink/playback errors;
        a turn already cancelled at entry does not run or change session state.
        When enabled, counter observations bracket agent work and follow reset.

        Args:
            turn:
                Transcript and disconnect/abort cancellation flag; capture timestamps
                are metadata and do not control conversation state.

        """
        if turn.cancelled.is_set():
            return
        self.assistant.new_session()
        logger.info("agent.session status=new mode=voice")
        try:
            self.output("You: " + turn.text)
            if self.memory_diagnostics:
                log_memory_usage("voice.agent.before")
            try:
                answer = self.assistant.reply(turn.text, turn.cancelled.is_set)
            except Exception:
                logger.exception("agent.turn status=failed")
                self.assistant.new_session()
                answer = "I couldn't complete that request. Please try again."
            finally:
                if self.memory_diagnostics:
                    log_memory_usage("voice.agent.after")
            if turn.cancelled.is_set() or not answer.strip():
                return
            self.output("Assistant: " + answer)
            spoken = bounded_speech(answer)
            if not turn.cancelled.is_set():
                try:
                    self.tts.play(spoken, blocking=True, cancelled=turn.cancelled)
                finally:
                    if self.memory_diagnostics:
                        log_memory_usage("voice.tts.after")
        finally:
            self.assistant.new_session()
            if self.memory_diagnostics:
                log_memory_usage("voice.turn.after")
