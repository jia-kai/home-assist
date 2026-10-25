"""Native offline Melo EN/ZH/mixed frontend with checkpoint-specific symbols.

Only text and tokenizer dependencies are used; no speech-model or Transformers
runtime is imported. See ASSETS.md for the complete artifact contract.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from tokenizers import Encoding, Tokenizer

from .chinese import ChineseG2P
from .english import PHONEMES, EnglishG2P, distribute_phone
from .normalization import PUNCTUATION, normalize
from .splitting import split_text

IntArray = NDArray[np.int64]
Language = Literal["EN", "ZH"]
EN_PHONE_SYMBOLS: frozenset[str] = (
    frozenset(
        ("V" if phone.rstrip("012").lower() == "v" else phone.rstrip("012").lower())
        for phone in PHONEMES[4:]
    )
    | frozenset(PUNCTUATION)
    | {"UNK"}
)


@dataclass(slots=True)
class TextInputs:
    """Token/phone inputs after blank insertion, ready for BERT and Melo graphs."""

    phones: IntArray
    """Checkpoint-specific phone IDs, int64 [1, P], including blank ID zero."""

    tones: IntArray
    """Tone IDs, int64 [1, P]; blank zero, EN shifted by seven, ZH unshifted."""

    languages: IntArray
    """Language IDs, int64 [1, P]; EN two, mixed-ZH three, blank zero."""

    input_ids: IntArray
    """BERT tokenizer IDs, int64 [1, S], including CLS and SEP."""

    word2ph: IntArray
    """BERT repeat counts, int64 [S], after doubling and first-count increment."""

    normalized_text: str
    """Exact normalized utterance passed to the artifact tokenizer."""


class Frontend:
    """Prepare offline English or Chinese/English mixed Melo model inputs."""

    language: Language
    """Selected checkpoint frontend mode."""

    tokenizer: Tokenizer
    """Fast tokenizer loaded from an explicit tokenizer.json artifact."""

    symbol_ids: dict[str, int]
    """Exact checkpoint symbol ordering, with '_' required at index zero."""

    english: EnglishG2P
    """Dictionary, tagger and GRU resources, optionally shared explicitly by caller."""

    chinese: ChineseG2P | None
    """Chinese resources, loaded only for a ZH frontend."""

    def __init__(
        self,
        artifact_dir: Path,
        language: Language,
        *,
        english: EnglishG2P | None = None,
    ) -> None:
        """Load checkpoint-specific tokenizer, symbol alphabet and offline assets.

        An explicit English resource object is reused without reading English
        assets from artifact_dir. Tokenizer, checkpoint symbols and Chinese
        resources remain independent. No global resource cache is introduced.

        Args:
            artifact_dir:
                Directory containing config.json, tokenizer.json and ASSETS.md resources.

            language:
                EN for bert-base-uncased; ZH for multilingual-uncased mixed text.

            english:
                Optional shared English resources. The caller must first verify
                identical checksums for cmudict.rep and every g2p/ asset in both
                manifests, and serialize use of the shared mutable OOV cache.
                None loads a separate resource object from artifact_dir.

        """
        if language not in ("EN", "ZH"):
            raise ValueError(f"Unsupported Melo frontend language: {language!r}")
        self.language = language
        config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
        symbols = config["symbols"]
        if (
            not isinstance(symbols, list)
            or not symbols
            or not all(isinstance(symbol, str) for symbol in symbols)
        ):
            raise ValueError("config.json symbols must be a nonempty string list")
        if len(set(symbols)) != len(symbols) or symbols[0] != "_":
            raise ValueError(
                "Checkpoint symbols must be unique with '_' at blank ID zero"
            )
        self.symbol_ids = {symbol: index for index, symbol in enumerate(symbols)}
        if "UNK" not in self.symbol_ids:
            raise ValueError(
                "Checkpoint symbol alphabet requires an explicit UNK symbol"
            )
        self.tokenizer = Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.english = EnglishG2P(artifact_dir) if english is None else english
        self.chinese = ChineseG2P(artifact_dir) if language == "ZH" else None

    def split(self, text: str) -> list[str]:
        """Split an utterance with dependency-minimal upstream Melo semantics.

        Args:
            text:
                Raw utterance to split before per-chunk normalization.

        """
        return split_text(text, self.language)

    def _english_groups(
        self,
        text: str,
        encoding: Encoding,
        indexes: list[int],
    ) -> tuple[list[str], list[int], list[int]]:
        """Pronounce tokenizer words and distribute phones over their pieces.

        Args:
            text:
                Full normalized source utterance.

            encoding:
                Full BERT encoding used by the eventual feature graph.

            indexes:
                Ordered content token indexes in one English span.

        """
        groups: list[list[int]] = []
        for index in indexes:
            if encoding.tokens[index].startswith("##") and groups:
                groups[-1].append(index)
            else:
                groups.append([index])
        phones: list[str] = []
        tones: list[int] = []
        counts: list[int] = []
        for group in groups:
            pieces = [encoding.tokens[index] for index in group]
            if "[UNK]" in pieces:
                start = encoding.offsets[group[0]][0]
                end = encoding.offsets[group[-1]][1]
                word = text[start:end]
            else:
                word = "".join(piece.removeprefix("##") for piece in pieces)
            symbols, stresses = self.english.pronounce(word)
            symbols = [
                symbol if symbol in EN_PHONE_SYMBOLS else "UNK" for symbol in symbols
            ]
            phones.extend(symbols)
            tones.extend(tone + 7 for tone in stresses)
            counts.extend(distribute_phone(len(symbols), len(group)))
        return phones, tones, counts

    def _mixed(
        self,
        text: str,
        encoding: Encoding,
        content: list[int],
    ) -> tuple[list[str], list[int], list[int]]:
        """Pronounce Han runs with sandhi and English groups with shifted tones.

        Full-tokenizer offsets, rather than a second independently tokenized
        string, associate Han character phones and English wordpieces with BERT.

        Args:
            text:
                Mixed normalized utterance.

            encoding:
                Full multilingual BERT encoding.

            content:
                Ordered non-special token indexes.

        """
        chinese = self.chinese
        if chinese is None:
            raise RuntimeError("Chinese frontend resources are not loaded")
        character_phones: dict[int, tuple[list[str], int]] = {}
        for match in re.finditer(r"[\u4e00-\u9fa5]+", text):
            syllables = chinese.pronounce(match[0])
            for offset, syllable in enumerate(syllables, start=match.start()):
                character_phones[offset] = syllable
        phones: list[str] = []
        tones: list[int] = []
        counts: list[int] = []
        position = 0
        while position < len(content):
            index = content[position]
            start, end = encoding.offsets[index]
            if re.fullmatch(r"[a-zA-Z]+", text[start:end]):
                english_indexes = [index]
                position += 1
                while position < len(content):
                    next_index = content[position]
                    next_start, next_end = encoding.offsets[next_index]
                    if not re.fullmatch(r"[a-zA-Z]+", text[next_start:next_end]):
                        break
                    english_indexes.append(next_index)
                    position += 1
                english_phones, english_tones, english_counts = self._english_groups(
                    text, encoding, english_indexes
                )
                phones.extend(english_phones)
                tones.extend(english_tones)
                counts.extend(english_counts)
                continue
            token_count = 0
            for offset in range(start, end):
                char = text[offset]
                if char.isspace():
                    continue
                if offset in character_phones:
                    symbols, tone = character_phones[offset]
                else:
                    symbols, tone = ([char] if char in PUNCTUATION else ["UNK"]), 0
                phones.extend(symbols)
                tones.extend([tone] * len(symbols))
                token_count += len(symbols)
            counts.append(token_count)
            position += 1
        return phones, tones, counts

    def prepare(self, text: str) -> TextInputs:
        """Normalize, pronounce, align BERT wordpieces and insert zero blanks.

        Missing resources, unsupported checkpoint phones and inconsistent
        tokenizer geometry raise exceptions rather than dropping input tokens.

        Args:
            text:
                Nonempty utterance in English or basic Han/ASCII mixed text.

        """
        normalized = normalize(text, self.language)
        if not normalized.strip():
            raise ValueError("Text is empty after Melo normalization")
        encoding = self.tokenizer.encode(normalized, add_special_tokens=True)
        mask = encoding.special_tokens_mask
        if len(mask) < 3 or mask[0] != 1 or mask[-1] != 1 or sum(mask) != 2:
            raise ValueError(
                "Tokenizer must emit exactly one CLS and one SEP around content"
            )
        if encoding.tokens[0] != "[CLS]" or encoding.tokens[-1] != "[SEP]":
            raise ValueError("Frontend requires BERT CLS/SEP special tokens")
        content = list(range(1, len(mask) - 1))
        if self.language == "EN":
            phones, tones, counts = self._english_groups(normalized, encoding, content)
        else:
            phones, tones, counts = self._mixed(normalized, encoding, content)
        padding_tone = 7 if self.language == "EN" else 0
        phones = ["_"] + phones + ["_"]
        tones = [padding_tone] + tones + [padding_tone]
        counts = [1] + counts + [1]
        if (
            len(counts) != len(encoding.ids)
            or sum(counts) != len(phones)
            or len(tones) != len(phones)
        ):
            raise ValueError("Melo token/phone alignment invariant failed")
        try:
            ids = [self.symbol_ids[symbol] for symbol in phones]
        except KeyError as error:
            raise ValueError(
                f"Phone {error.args[0]!r} is absent from checkpoint symbols"
            ) from error
        length = 2 * len(ids) + 1
        phone_array = np.zeros((1, length), dtype=np.int64)
        tone_array = np.zeros((1, length), dtype=np.int64)
        language_array = np.zeros((1, length), dtype=np.int64)
        phone_array[0, 1::2] = ids
        tone_array[0, 1::2] = tones
        language_array[0, 1::2] = 2 if self.language == "EN" else 3
        repeats = np.asarray(counts, dtype=np.int64) * 2
        repeats[0] += 1
        if int(repeats.sum()) != length:
            raise ValueError("Blank-inserted repeat counts do not match phone length")
        return TextInputs(
            phones=phone_array,
            tones=tone_array,
            languages=language_array,
            input_ids=np.asarray([encoding.ids], dtype=np.int64),
            word2ph=repeats,
            normalized_text=normalized,
        )


__all__ = ["Frontend", "TextInputs"]
