"""Canonical user language and separation from structured model history."""

import unicodedata
from copy import deepcopy

import pytest

from hoast.input_text import (
    BASIC_PUNCTUATION,
    REPAIR_FEEDBACK_KEY,
    canonical_user_messages,
    canonicalize_text,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("播放《晴天》，谢谢！", "播放 晴天，谢谢！"),
        ('play "Don\'t Stop Me Now" by G.E.M.', "play Don't Stop Me Now by G.E.M."),
        ("Adele’s recording, please。", "Adele's recording, please。"),
        ("Paris, France，明天？", "Paris, France，明天？"),
        ("音量设为35%", "音量设为35"),
        ("０", "0"),
        ("100%", "100"),
        ("cafe\u0301\tA/B\n", "café A B"),
        ("‘O’Connor’", "O'Connor"),
        ("...《》🎵", ""),
    ],
)
def test_canonical_form(text: str, expected: str) -> None:
    """Preserve numeric/word content and make normalization idempotent.

    Args:
        text:
            Unicode speech or keyboard input containing formatting marks.

        expected:
            Explicit expected canonical representation with basic punctuation retained.

    """
    result = canonicalize_text(text)
    assert result == expected
    assert canonicalize_text(result) == result
    assert all(
        char == " "
        or char in BASIC_PUNCTUATION
        or unicodedata.category(char)[0] in "LNM"
        for char in result
    )


def test_only_user_language_is_normalized() -> None:
    """Keep protocol data and internal repair syntax intact without mutating callers."""
    messages = [
        {"role": "system", "content": "Use tools: [a(), b()]."},
        {"role": "user", "content": "Play 《晴天》!"},
        {
            "role": "assistant",
            "content": "Okay.",
            "tool_calls": [
                {
                    "function": {
                        "name": "play_music",
                        "arguments": {"title": "Don't Stop"},
                    }
                }
            ],
        },
        {"role": "tool", "content": {"status": "ok", "value": "[x, y]"}},
        {
            "role": "user",
            "content": "Repair: [call(x='a,b')].",
            REPAIR_FEEDBACK_KEY: True,
        },
    ]
    original = deepcopy(messages)
    result = canonical_user_messages(messages)
    assert result[1]["content"] == "Play 晴天!"
    assert result[0] == messages[0] and result[2:4] == messages[2:4]
    assert result[4] == {"role": "user", "content": "Repair: [call(x='a,b')]."}
    assert messages == original


def test_empty_or_structured_user_content_is_rejected() -> None:
    """Reject punctuation-only or non-text requests at the model boundary."""
    with pytest.raises(ValueError, match="words or numbers"):
        canonical_user_messages([{"role": "user", "content": "!?"}])
    with pytest.raises(TypeError, match="must be text"):
        canonical_user_messages([{"role": "user", "content": {"text": "Hello"}}])


@pytest.mark.parametrize(
    "text",
    [
        "volume -5",
        "volume 5.2",
        "volume +5",
        "volume 1,000",
        "louder-5",
        "louder –5",
        "volume ½",
        "volume 1/2",
        "音量3 ，5",
    ],
)
def test_unsupported_numeric_forms_are_not_rewritten(text: str) -> None:
    """Reject unsupported numeric syntax before punctuation could turn it into a value.

    Args:
        text:
            A command outside the supported unsigned-integer numeric syntax.

    """
    with pytest.raises(ValueError, match="unsigned integers"):
        canonicalize_text(text)
