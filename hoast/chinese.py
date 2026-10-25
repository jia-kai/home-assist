"""Han detection for Chinese speech routing."""

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
