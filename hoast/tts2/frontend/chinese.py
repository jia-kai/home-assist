"""Mandarin initials/finals and tone sandhi adapted from Melo's Chinese G2P."""

from pathlib import Path
from typing import Any

from .normalization import PUNCTUATION
from .tone_sandhi import ToneSandhi


class ChineseG2P:
    """Pronounce Han characters using pypinyin, Jieba and strict OpenCPOP phones."""

    symbols: dict[str, list[str]]
    """Strict tone-free pinyin-to-phone mapping loaded from artifacts."""

    sandhi: ToneSandhi
    """Paddle/Melo tone-sandhi modifier."""

    posseg: Any
    """Jieba POS segmentation module."""

    lazy_pinyin: Any
    """Pypinyin conversion callable."""

    style: Any
    """Pypinyin Style enum."""

    def __init__(self, artifact_dir: Path) -> None:
        """Load explicit pronunciation mappings and Chinese-only tooling.

        Args:
            artifact_dir:
                Artifact root containing opencpop-strict.txt.

        """
        self.symbols = {}
        for line in (
            (artifact_dir / "opencpop-strict.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        ):
            if not line.strip():
                continue
            pinyin, phones = line.split("\t")
            self.symbols[pinyin] = phones.split()
        if not self.symbols:
            raise ValueError("Empty OpenCPOP pronunciation mapping")
        # Optional Chinese language modules remain lazy for English inference.
        from jieba import posseg
        from pypinyin import Style, lazy_pinyin

        self.posseg = posseg
        self.lazy_pinyin = lazy_pinyin
        self.style = Style
        self.sandhi = ToneSandhi()

    def _syllable(self, initial: str, final: str) -> tuple[list[str], int]:
        """Convert pypinyin output to strict OpenCPOP phones with one tone.

        Args:
            initial:
                Pinyin consonant initial or punctuation.

            final:
                Pinyin final with a trailing tone digit, or punctuation.

        """
        if initial == final:
            return ([initial] if initial in PUNCTUATION else ["UNK"]), 0
        if not final or final[-1] not in "12345":
            raise ValueError(f"Unsupported Chinese pronunciation: {initial + final!r}")
        vowel = final[:-1]
        pinyin = initial + vowel
        if initial:
            pinyin = initial + {"uei": "ui", "iou": "iu", "uen": "un"}.get(vowel, vowel)
        else:
            replacements = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
            if pinyin in replacements:
                pinyin = replacements[pinyin]
            elif pinyin:
                pinyin = {"v": "yu", "e": "e", "i": "y", "u": "w"}.get(
                    pinyin[0], pinyin[0]
                ) + pinyin[1:]
        if pinyin not in self.symbols:
            raise ValueError(f"Missing OpenCPOP pronunciation: {pinyin!r}")
        return self.symbols[pinyin], int(final[-1])

    def pronounce(self, text: str) -> list[tuple[list[str], int]]:
        """Return phone groups and tones for each source character in a Han run.

        Args:
            text:
                Contiguous nonempty basic-Han text, without English/whitespace.

        """
        segments = [(pair.word, pair.flag) for pair in self.posseg.lcut(text)]
        segments = self.sandhi.pre_merge_for_modify(segments)
        result: list[tuple[list[str], int]] = []
        for word, pos in segments:
            initials = self.lazy_pinyin(
                word, neutral_tone_with_five=True, style=self.style.INITIALS
            )
            finals = self.lazy_pinyin(
                word, neutral_tone_with_five=True, style=self.style.FINALS_TONE3
            )
            if len(initials) != len(word) or len(finals) != len(word):
                raise ValueError(f"Pypinyin character alignment failed for {word!r}")
            finals = self.sandhi.modified_tone(word, pos, finals)
            result.extend(
                self._syllable(initial, final)
                for initial, final in zip(initials, finals, strict=True)
            )
        if len(result) != len(text):
            raise ValueError("Chinese phone groups do not cover every source character")
        return result
