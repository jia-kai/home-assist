"""JSON-lines worker for the official Chinese frontend's supported Python runtime."""

import argparse
import contextlib
import importlib
import json
import os
import sys
from pathlib import Path

from hoast.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Initialize official G2P once, then process bounded protocol requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english-language", default="en-us")
    parser.add_argument("--log-file", type=Path, required=True)
    args = parser.parse_args()
    configure_logging(log_file=args.log_file, level="WARNING")
    try:
        os.sched_setaffinity(0, [max(os.sched_getaffinity(0))])
        # Optional language packages stay lazy and run only in the prepared environment.
        with contextlib.redirect_stdout(sys.stderr):
            loader = importlib.import_module("espeakng_loader")
            wrapper = importlib.import_module(
                "phonemizer.backend.espeak.wrapper"
            ).EspeakWrapper
            wrapper.set_library(loader.get_library_path())
            wrapper.set_data_path(loader.get_data_path())
            backend = importlib.import_module("phonemizer.backend").EspeakBackend(
                args.english_language,
                preserve_punctuation=True,
                with_stress=True,
                words_mismatch="ignore",
            )

            def english(text: str) -> str:
                """Return English phonemes for an insertion in a Chinese sentence.

                Args:
                    text:
                        English span supplied by the official Chinese frontend.

                """
                return backend.phonemize([text], strip=True)[0]

            phonemizer = importlib.import_module("misaki.zh").ZHG2P(
                version="1.1", en_callable=english
            )
        sys.stdout.write(json.dumps({"ready": True, "protocol": 1}) + "\n")
        sys.stdout.flush()
        while line := sys.stdin.buffer.readline(1024 * 1024 + 1):
            if len(line) > 1024 * 1024 or not line.endswith(b"\n"):
                raise ValueError(
                    "Chinese phonemizer request exceeds the size limit or is incomplete"
                )
            request = json.loads(line)
            identifier = request["id"]
            text = request["text"]
            if not isinstance(identifier, int) or not isinstance(text, str):
                raise TypeError("Invalid Chinese phonemizer request")
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    phonemes, _ = phonemizer(text)
                response = {"id": identifier, "phonemes": phonemes}
            except Exception as error:
                logger.exception(
                    "Chinese G2P failed request=%d characters=%d", identifier, len(text)
                )
                response = {"id": identifier, "error": str(error)}
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    except Exception:
        logger.exception("Chinese phonemizer worker failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
