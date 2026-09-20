"""Serialized assistant turns with current music context and bounded system speech."""

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from .agent import LocalAgent
from .lfm import LFM2
from .logging import get_logger
from .music import MusicAssistantError, MusicClient
from .prompts import build_system_prompt
from .session import Session
from .stt import STT
from .tts import TTS
from .voice import VoiceTurn

logger = get_logger(__name__)


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

    Run synchronous methods on the application's single inference worker. A new
    conversation starts when first PCM arrival is over 30 seconds after the end
    of the previous nonempty capture. Recognition/inference/playback time is not
    substituted for those capture timestamps. Playback already started is drained
    on cancellation, while later tool dispatch and playback are suppressed.
    """

    assistant: Assistant
    """Grounded text agent with live music context."""

    stt: STT
    """Reusable prepared speech recognizer."""

    tts: TTS
    """Reusable bilingual synthesizer and system-output playback engine."""

    output: Callable[[str], None] = console_text
    """Text sink for recognized input and full grounded output."""

    previous_audio_end: float | None = field(default=None, init=False)
    """End of the preceding accepted nonempty capture, in monotonic seconds."""

    def warmup(self) -> None:
        """Warm all models without tool effects, then announce initialization."""
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
        self.tts.play("系统初始化完毕", blocking=True)
        logger.info("warmup.announcement status=completed")

    def transcribe(self, pcm: bytes) -> str:
        """Recognize captured PCM on the single inference worker.

        Args:
            pcm:
                Nonempty mono signed little-endian int16 PCM at 16 kHz.

        """
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return self.stt.transcribe_samples(samples, 16000)

    def respond(self, turn: VoiceTurn) -> None:
        """Run one uncancelled command and drain bounded speech before returning.

        Full grounded text goes to the text sink. Expected top-level turn failures
        are logged with tracebacks, reset conversation state, and produce a brief
        spoken failure. A playback error propagates to the native pipeline handler.

        Args:
            turn:
                Transcript, capture timestamps, and disconnect/abort cancellation flag.

        """
        if turn.cancelled.is_set():
            return
        if (
            self.previous_audio_end is None
            or turn.started_at - self.previous_audio_end > 30
        ):
            self.assistant.new_session()
            logger.info(
                "agent.session status=new audio_gap=%s",
                None
                if self.previous_audio_end is None
                else turn.started_at - self.previous_audio_end,
            )
        self.previous_audio_end = turn.ended_at
        self.output("You: " + turn.text)
        try:
            answer = self.assistant.reply(turn.text, turn.cancelled.is_set)
        except Exception:
            logger.exception("agent.turn status=failed")
            self.assistant.new_session()
            answer = "I couldn't complete that request. Please try again."
        if turn.cancelled.is_set():
            self.assistant.new_session()
            return
        if not answer.strip():
            return
        self.output("Assistant: " + answer)
        spoken = bounded_speech(answer)
        if not turn.cancelled.is_set():
            try:
                self.tts.play(spoken, blocking=True)
            except Exception:
                self.assistant.new_session()
                raise
        if turn.cancelled.is_set():
            self.assistant.new_session()
