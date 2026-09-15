"""Lazy, cached access to the official Chinese Kokoro phonemizer in Python 3.12."""

import json
import math
import os
import select
import subprocess
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from .logging import get_logger

logger = get_logger(__name__)
DEFAULT_SPEECH_PYTHON = Path(".cache/hoast/speech-env/.venv/bin/python")


def _release(process: subprocess.Popen[bytes], log: BinaryIO) -> None:
    """Close the worker and its streams without leaving a background process.

    Args:
        process:
            Owned Chinese phonemizer process.

        log:
            Owned stderr log stream.

    """
    if process.stdin is not None:
        process.stdin.close()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if process.stdout is not None:
        process.stdout.close()
    log.close()


@dataclass(slots=True, weakref_slot=True)
class ChineseG2P:
    """Own one lazily started official Misaki v1.1 frontend and reuse it across calls."""

    python: Path = DEFAULT_SPEECH_PYTHON
    """Prepared Python 3.12 interpreter containing the pinned Chinese dependencies."""

    english_language: str = "en-us"
    """English phonemizer language for short Latin-script insertions."""

    timeout: float = 30.0
    """Maximum seconds per startup read, request write, or response read."""

    log_file: Path = Path(".cache/hoast/diagnostics/chinese-g2p.log")
    """Durable worker diagnostic log."""

    _process: subprocess.Popen[bytes] | None = field(
        default=None, init=False, repr=False
    )
    """Owned worker process, absent until first use."""

    _finalizer: weakref.finalize | None = field(default=None, init=False, repr=False)
    """Resource cleanup without a reference cycle to this owner."""

    _buffer: bytearray = field(default_factory=bytearray, init=False, repr=False)
    """Partial JSON-line response bytes."""

    _sequence: int = field(default=0, init=False)
    """Monotonic request identifier for matching responses."""

    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    """Serializes startup and protocol requests."""

    def __post_init__(self) -> None:
        """Reject unusable deadlines and empty insertion-language labels."""
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Chinese phonemizer timeout must be finite and positive")
        if not self.english_language.strip():
            raise ValueError("English insertion language must be nonempty")

    def _write(self, data: bytes) -> None:
        """Write a bounded request completely, enforcing a deadline under backpressure.

        Args:
            data:
                UTF-8 JSON line, at most one MiB including its newline.

        """
        assert self._process is not None and self._process.stdin is not None
        descriptor = self._process.stdin.fileno()
        deadline = time.monotonic() + self.timeout
        remaining_data = memoryview(data)
        while remaining_data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [descriptor], [], remaining)[1]:
                raise TimeoutError("Chinese phonemizer request timed out")
            try:
                written = os.write(descriptor, remaining_data)
            except BlockingIOError:
                continue
            if written == 0:
                raise BrokenPipeError("Chinese phonemizer request pipe closed")
            remaining_data = remaining_data[written:]

    def _read(self) -> dict[str, Any]:
        """Read one bounded response with an actual deadline, including partial writes."""
        assert self._process is not None and self._process.stdout is not None
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([self._process.stdout], [], [], remaining)[0]
            ):
                raise TimeoutError("Chinese phonemizer timed out")
            data = os.read(self._process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError("Chinese phonemizer exited before replying")
            self._buffer.extend(data)
            if len(self._buffer) > 1024 * 1024:
                raise ValueError("Chinese phonemizer response exceeds the size limit")
        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = bytearray(remainder)
        response = json.loads(line)
        if not isinstance(response, dict):
            raise TypeError("Chinese phonemizer returned a non-object response")
        return response

    def _start(self) -> None:
        """Start the prepared worker and validate its protocol handshake."""
        if self._process is not None:
            return
        if not self.python.is_file():
            raise FileNotFoundError(
                "Prepare Chinese TTS first: tools.prepare_tts --chinese"
            )
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_file.open("ab")
        environment = os.environ.copy()
        environment.update(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
        )
        root = str(Path(__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = root + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH")
            else ""
        )
        try:
            process = subprocess.Popen(
                [
                    str(self.python.absolute()),
                    "-m",
                    "hoast.chinese_g2p_worker",
                    "--english-language",
                    self.english_language,
                    "--log-file",
                    str(self.log_file.resolve()),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                bufsize=0,
                env=environment,
            )
        except BaseException:
            log.close()
            raise
        self._process = process
        self._finalizer = weakref.finalize(self, _release, process, log)
        assert process.stdin is not None
        os.set_blocking(process.stdin.fileno(), False)
        response = self._read()
        if response != {"ready": True, "protocol": 1}:
            raise RuntimeError("Chinese phonemizer protocol mismatch")
        logger.info("tts.chinese_g2p status=ready pid=%d", process.pid)

    def phonemize(self, text: str) -> str:
        """Return official v1.1 phonemes, including English insertions.

        Args:
            text:
                Nonempty utterance; its encoded protocol request must fit in one MiB.

        """
        if not text.strip():
            raise ValueError("Chinese phonemizer text must be nonempty")
        with self._lock:
            self._sequence += 1
            data = (
                json.dumps(
                    {"id": self._sequence, "text": text}, ensure_ascii=False
                ).encode("utf-8")
                + b"\n"
            )
            if len(data) > 1024 * 1024:
                raise ValueError("Chinese phonemizer request exceeds the size limit")
            try:
                self._start()
                assert self._process is not None and self._process.stdin is not None
                self._write(data)
                response = self._read()
                if response["id"] != self._sequence:
                    raise RuntimeError(
                        "Chinese phonemizer returned a mismatched request ID"
                    )
                if "error" in response:
                    raise ValueError(
                        f"Chinese phonemization failed: {response['error']}"
                    )
                phonemes = response["phonemes"]
                if not isinstance(phonemes, str) or not phonemes.strip():
                    raise ValueError("Chinese text produced no speakable phonemes")
                return phonemes
            except BaseException:
                logger.exception(
                    "tts.chinese_g2p status=failed characters=%d", len(text)
                )
                self.close()
                raise

    def close(self) -> None:
        """Release the worker; a later request may start it again."""
        with self._lock:
            if self._finalizer is not None and self._finalizer.alive:
                self._finalizer()
            self._process = None
            self._finalizer = None
            self._buffer.clear()
