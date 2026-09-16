"""Non-spoken title delimiters and explicit Kokoro phoneme compatibility mappings."""

import re
from collections.abc import Callable

_INITIALISM = re.compile(r"(?<!\w)(?:[A-Za-z]\.){2,}(?:[A-Za-z](?!\w))?")
_TITLE_MARKS = str.maketrans("", "", '"“”《》「」『』«»')
_HAN_POSSESSIVE = re.compile(r"(?<=[\u3400-\u9fff])['’]s\b", re.IGNORECASE)


def prepare_speech_text(text: str) -> str:
    """Remove silent title delimiters and spell dotted initialisms as letter sequences.

    Apostrophes and sentence punctuation retain their pronunciation/prosody roles.
    For example, quoted G.E.M.'s becomes G E M's, avoiding an isolated apostrophe
    in mixed-language G2P. This preparation does not change stored user transcripts.

    Args:
        text:
            Text for synthesis, possibly containing book-title marks or initialisms.

    """
    return _INITIALISM.sub(
        lambda match: " ".join(character for character in match[0] if character != "."),
        text.translate(_TITLE_MARKS),
    )


def english_phonemes(phonemes: str) -> str:
    """Decompose rhotic vowels into equivalent supported Kokoro phoneme sequences.

    This is an explicit phonetic mapping, not removal of unknown phonemes. Other
    unsupported symbols remain visible for the caller's vocabulary validation.

    Args:
        phonemes:
            English eSpeak phonemes used inside Mandarin/code-switched synthesis.

    """
    return phonemes.replace("ɚ", "əɹ").replace("ɝ", "ɜɹ")


def mixed_phonemes(text: str, phonemize: Callable[[str], str]) -> str:
    """Render English possessives on Mandarin names without an orphan apostrophe token.

    Mandarin syllables end in voiced sounds, so an English possessive uses /z/.
    The official frontend handles each surrounding text span; no spoken word is
    discarded or translated. Text without this construction uses the frontend
    unchanged.

    Args:
        text:
            Prepared Mandarin/code-switched synthesis text.

        phonemize:
            Official frontend callback returning phonemes for one text span.

    """
    if _HAN_POSSESSIVE.search(text) is None:
        return phonemize(text)
    return " z ".join(
        phonemize(part) if part.strip() else "" for part in _HAN_POSSESSIVE.split(text)
    ).strip()
