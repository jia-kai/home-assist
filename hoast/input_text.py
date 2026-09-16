"""Canonical user language preserving basic punctuation and word apostrophes."""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

REPAIR_FEEDBACK_KEY = "_hoast_repair_feedback"
NORMALIZATION_VERSION = 2
BASIC_PUNCTUATION = ",，.。!?！？;；:：'"
SENTENCE_PUNCTUATION = BASIC_PUNCTUATION.replace("'", "")
_UNSUPPORTED_NUMBER = re.compile(
    r"[+−‐‑‒–—-]\s*\d|\d[.,，]\d|(?<!\w)[.,]\d|\d\s*[/⁄∕]\s*\d"
)
_APOSTROPHE = re.compile(r"(?<=\w)[’‘ʼ](?=\w)|(?<=[sS])[’ʼ](?!\w)")


def canonicalize_text(text: str) -> str:
    """Preserve basic punctuation while removing title delimiters and decorative symbols.

    Compatibility forms normalize without changing case or Chinese sentence-mark
    widths. Apostrophes, dotted initials and basic sentence punctuation are kept;
    curly word apostrophes become ASCII. Other punctuation/symbols become separators.
    Signed, fractional, and grouped numeric literals are unsupported and rejected
    before punctuation removal, rather than rewritten into different values. Tool
    schemas enforce integer value ranges. The result is idempotent; punctuation-only
    input becomes an empty string. This is not an arithmetic or date parser.

    Args:
        text:
            Spoken or typed user language, not serialized tools, prompts, or JSON.

    """
    text = unicodedata.normalize(
        "NFC",
        "".join(
            character
            if character in BASIC_PUNCTUATION
            else unicodedata.normalize("NFKC", character)
            for character in text
        ),
    )
    if _UNSUPPORTED_NUMBER.search(text):
        raise ValueError("Command numeric values must be unsigned integers")
    text = _APOSTROPHE.sub("'", text)
    text = "".join(
        character
        if character in BASIC_PUNCTUATION or unicodedata.category(character)[0] in "LNM"
        else " "
        for character in text
    )
    text = " ".join(text.split())
    text = re.sub(r"\s+([,，.。!?！？;；:：])", r"\1", text)
    if _UNSUPPORTED_NUMBER.search(text):
        raise ValueError("Command numeric values must be unsigned integers")
    return text if any(unicodedata.category(c)[0] in "LNM" for c in text) else ""


def content_signature(text: str) -> str:
    """Compare content independent of formatting without changing model input.

    Args:
        text:
            User language or an aligned script representation for offline audits.

    """
    return "".join(c for c in text if unicodedata.category(c)[0] in "LNM").casefold()


def canonical_user_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy history and canonicalize external user content without altering tool data.

    Session repair feedback uses a private message metadata flag; the flag is
    consumed here and never sent through the chat template. Text supplied by users
    cannot set that metadata. Internal diagnostics retain their structured syntax.
    Empty canonical user messages and non-string user content are rejected.

    Args:
        messages:
            Native message mappings containing user language and structured history.

    """
    result: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        internal = copied.pop(REPAIR_FEEDBACK_KEY, False)
        if type(internal) is not bool:
            raise TypeError("Internal feedback metadata must be boolean")
        if copied["role"] == "user" and not internal:
            if not isinstance(copied["content"], str):
                raise TypeError("User message content must be text")
            copied["content"] = canonicalize_text(copied["content"])
            if not copied["content"]:
                raise ValueError("User request must contain words or numbers")
        result.append(copied)
    return result
