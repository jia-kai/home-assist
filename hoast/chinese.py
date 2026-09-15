"""Han detection and lossless phoneme boundaries for Chinese speech."""

import unicodedata


def contains_han(text: str) -> bool:
    """Detect Han ideographs, including compatibility and supplementary characters.

    Args:
        text:
            Original user text, not the Chinese model's phonetic symbols.

    """
    return any(
        character == "〇"
        or unicodedata.name(character, "").startswith(
            ("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-")
        )
        for character in text
    )


def phoneme_batches(phonemes: str, limit: int = 120) -> list[str]:
    """Prefer Chinese word/punctuation boundaries and preserve complete tone syllables.

    Args:
        phonemes:
            Official v1.1 phoneme string, including slash boundaries and tone digits.

        limit:
            Preferred payload character limit, between 8 and 450.

    """
    if not 8 <= limit <= 450:
        raise ValueError("Chinese phoneme limit must be between 8 and 450")
    remaining = phonemes.strip()
    parts: list[str] = []
    while len(remaining) > limit:
        boundary = max(remaining.rfind(mark, 0, limit + 1) for mark in " /.,!?;:") + 1
        if boundary <= 1:
            boundary = max(remaining.rfind(tone, 0, limit) for tone in "12345") + 1
        if boundary <= 0:
            raise ValueError("Chinese phonemes cannot be split at a syllable boundary")
        parts.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts
