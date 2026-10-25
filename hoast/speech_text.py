"""Non-spoken title delimiters and dotted initialisms for speech synthesis."""

import re

_INITIALISM = re.compile(r"(?<!\w)(?:[A-Za-z]\.){2,}(?:[A-Za-z](?!\w))?")
_TITLE_MARKS = str.maketrans("", "", '"“”《》「」『』«»')


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
