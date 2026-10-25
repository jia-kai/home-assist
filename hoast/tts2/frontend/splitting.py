"""Dependency-minimal Melo sentence splitting (MyShell.ai, MIT)."""

import re
from typing import Literal


def _numeric_separator(text: str, index: int) -> bool:
    """Identify decimal/grouping punctuation that must remain inside a number.

    Args:
        text:
            Raw text being split, before numeric pronunciation normalization.

        index:
            Valid character index of the punctuation being considered.

    """
    return (
        text[index] in ".,"
        and 0 < index < len(text) - 1
        and text[index + 1].isdigit()
        and (
            text[index - 1].isdigit()
            or (text[index] == "." and text[index - 1] in "$£¥")
        )
    )


def _latin_chunks(
    text: str, desired_length: int = 256, max_length: int = 512
) -> list[str]:
    """Chunk Latin text at sentence/word boundaries, preserving numeric separators.

    Args:
        text:
            Input text after Latin punctuation substitutions.

        desired_length:
            Preferred minimum character count for sentence-boundary commits.

        max_length:
            Character count triggering a boundary search.

    """
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'["]', '"', text)
    text = re.sub(
        r"[,.?!]",
        lambda match: (
            match[0] if _numeric_separator(text, match.start()) else match[0] + " "
        ),
        text,
    )
    text = re.sub(r"\s+", " ", text)
    result: list[str] = []
    in_quote = False
    current = ""
    split_positions: list[int] = []
    position = -1
    end_position = len(text) - 1

    def seek(delta: int) -> str:
        """Move the cursor, updating quote state and current chunk.

        Args:
            delta:
                Signed number of source characters to move.

        """
        nonlocal position, in_quote, current
        for _ in range(abs(delta)):
            if delta < 0:
                position -= 1
                current = current[:-1]
            else:
                position += 1
                current += text[position]
            if text[position] == '"':
                in_quote = not in_quote
        return text[position]

    def peek(delta: int) -> str:
        """Read without advancing, retaining upstream final-character semantics.

        Args:
            delta:
                Signed lookahead offset relative to the cursor.

        """
        index = position + delta
        return text[index] if 0 <= index < end_position else ""

    def commit() -> None:
        """Commit accumulated characters and reset boundary candidates."""
        nonlocal current, split_positions
        result.append(current)
        current = ""
        split_positions = []

    while position < end_position:
        char = seek(1)
        if len(current) >= max_length:
            if split_positions and len(current) > desired_length / 2:
                seek(-(position - split_positions[-1]))
            else:
                while (
                    (char not in "!?.\n " or _numeric_separator(text, position))
                    and position > 0
                    and len(current) > desired_length
                ):
                    char = seek(-1)
            commit()
        elif not in_quote and (char in "!?\n" or (char in ".," and peek(1) in "\n ")):
            while (
                position < len(text) - 1
                and len(current) < max_length
                and peek(1) in "!?."
            ):
                char = seek(1)
            split_positions.append(position)
            if len(current) >= desired_length:
                commit()
        elif in_quote and peek(1) == '"' and peek(2) in "\n ":
            seek(2)
            split_positions.append(position)
    result.append(current)
    return [
        chunk.strip()
        for chunk in result
        if chunk.strip() and not re.match(r"^[\s.,;:!?]*$", chunk)
    ]


def split_text(text: str, language: Literal["EN", "ZH"]) -> list[str]:
    """Split with EN/ZH length policies without separating numeric punctuation.

    Decimal points and grouping commas stay attached to their digits so later
    pronunciation normalization sees complete numbers and currency amounts.

    Args:
        text:
            Raw utterance; normalization for pronunciation happens separately.

        language:
            English Latin splitting or Chinese/mixed punctuation splitting.

    """
    text = re.sub("[。！？；]", ".", text)
    text = re.sub("[，]", ",", text)
    if language == "EN":
        text = re.sub("[“”]", '"', text)
        text = re.sub("[‘’]", "'", text)
        text = re.sub(r'[<>\(\)\[\]"«»]+', "", text)
        return _latin_chunks(text)
    text = re.sub(r"[\n\t ]+", " ", text)
    text = re.sub(
        r"[,.!?;]",
        lambda match: (
            match[0] if _numeric_separator(text, match.start()) else match[0] + " $#!"
        ),
        text,
    )
    sentences = [sentence.strip() for sentence in text.split("$#!")]
    if not sentences[-1]:
        sentences.pop()
    chunks: list[str] = []
    pending: list[str] = []
    count = 0
    for index, sentence in enumerate(sentences):
        pending.append(sentence)
        count += len(sentence)
        if count > 10 or index == len(sentences) - 1:
            chunks.append(" ".join(pending))
            count = 0
            pending = []
    result: list[str] = []
    for chunk in chunks:
        if result and len(result[-1]) <= 2:
            result[-1] += " " + chunk
        else:
            result.append(chunk)
    if len(result) >= 2 and len(result[-1]) <= 2:
        result[-2] += " " + result.pop()
    return result
