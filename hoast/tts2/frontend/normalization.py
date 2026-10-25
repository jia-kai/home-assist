"""Melo English and mixed-Chinese normalization without model dependencies.

The rules are adapted from MyShell.ai MeloTTS (MIT); see NOTICE.md.
"""

import re
from functools import lru_cache
from typing import Any, Literal

ABBREVIATIONS: dict[str, str] = {
    "mrs": "misess",
    "mr": "mister",
    "dr": "doctor",
    "st": "saint",
    "co": "company",
    "jr": "junior",
    "maj": "major",
    "gen": "general",
    "drs": "doctors",
    "rev": "reverend",
    "lt": "lieutenant",
    "hon": "honorable",
    "sgt": "sergeant",
    "capt": "captain",
    "esq": "esquire",
    "ltd": "limited",
    "col": "colonel",
    "ft": "fort",
}
PUNCTUATION: str = "!?…,.'-"
REPLACEMENTS: dict[str, str] = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    **dict.fromkeys("“”‘’（）()《》【】[]「」", "'"),
    **dict.fromkeys("—～~", "-"),
}
TIME_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?P<hour>0?[0-9]|1[0-9]|2[0-3]):(?P<minute>[0-5][0-9])"
    r"(?:[ \t]*(?P<meridiem>[ap](?:\.m\.?|m))(?![a-zA-Z])|(?=$|[^a-zA-Z0-9_]))",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _inflector() -> Any:
    """Construct the optional text-only inflection engine once."""
    # Keep language-specific optional dependencies out of package import.
    import inflect

    return inflect.engine()


def _number(match: re.Match[str]) -> str:
    """Expand a number using Melo's year and cardinal conventions.

    Args:
        match:
            Integer match, including an optional minus sign.

    """
    number = int(match.group())
    engine = _inflector()
    if 1000 < number < 3000:
        if number == 2000:
            return "two thousand"
        if 2000 < number < 2010:
            return "two thousand " + engine.number_to_words(number % 100)
        if number % 100 == 0:
            return engine.number_to_words(number // 100) + " hundred"
        return engine.number_to_words(
            number,
            andword="",
            zero="oh",
            group=2,
        ).replace(", ", " ")
    return engine.number_to_words(number, andword="")


def _currency(match: re.Match[str]) -> str:
    """Expand Melo-supported currency amounts before cardinal conversion.

    Args:
        match:
            Currency symbol and decimal amount groups.

    """
    units = {
        "$": {0.01: "cent", 0.02: "cents", 1: "dollar", 2: "dollars"},
        "£": {0.01: "penny", 0.02: "pence", 1: "pound sterling", 2: "pounds sterling"},
        "¥": {0.02: "sen", 2: "yen"},
    }[match[1]]
    value = match[2]
    parts = value.replace(",", "").split(".")
    if len(parts) > 2:
        return f"{value} {units[2]}"
    integer = int(parts[0]) if parts[0] else 0
    fraction = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    words = []
    if integer > 0:
        words.append(f"{integer} {units.get(integer, units[2])}")
    if fraction > 0:
        words.append(f"{fraction} {units.get(fraction / 100, units[0.02])}")
    return " ".join(words) if words else f"zero {units[2]}"


def _time(match: re.Match[str]) -> str:
    """Expand clock times without consuming whitespace before unrelated words.

    Midnight is twelve AM; explicit dotted/undotted meridiems are respected.
    These content-preserving rules intentionally correct upstream cleaner bugs.

    Args:
        match:
            Clock-time expression with hour, minute and optional meridiem.

    """
    hour = int(match["hour"])
    past_noon = hour >= 12
    if hour > 12:
        hour -= 12
    elif hour == 0:
        hour = 12
    result = [_inflector().number_to_words(hour)]
    minute = int(match["minute"])
    if minute:
        if minute < 10:
            result.append("oh")
        result.append(_inflector().number_to_words(minute))
    meridiem = match["meridiem"]
    if meridiem is None:
        result.append("p m" if past_noon else "a m")
    else:
        result.extend(meridiem.replace(".", ""))
    return " ".join(result)


def normalize(text: str, language: Literal["EN", "ZH"]) -> str:
    """Apply upstream number, abbreviation and restricted punctuation rules.

    Grouping commas are removed before number expansion. Chinese mode expands
    every decimal numeral to Chinese, including numerals embedded in English.
    Unsupported Chinese-mode characters are removed by Melo's documented
    cleaner; an entirely empty result is rejected by prepare.

    Args:
        text:
            Input utterance in the selected language.

        language:
            EN for English or ZH for Han/ASCII-English mixed text.

    """
    text = re.sub(r"([0-9][0-9,]+[0-9])", lambda m: m[1].replace(",", ""), text)
    if language == "EN":
        text = re.sub(TIME_PATTERN, _time, text.lower())
        text = re.sub(r"(£|\$|¥)([0-9]+(?:\.[0-9]+)?|\.[0-9]+)", _currency, text)
        text = re.sub(r"([0-9]+\.[0-9]+)", lambda m: m[1].replace(".", " point "), text)
        text = re.sub(
            r"[0-9]+(st|nd|rd|th)", lambda m: _inflector().number_to_words(m[0]), text
        )
        text = re.sub(r"-?[0-9]+", _number, text)
        for abbreviation, expansion in ABBREVIATIONS.items():
            text = re.sub(rf"\b{abbreviation}\.", expansion, text, flags=re.IGNORECASE)
        return text
    # cn2an is needed only for Chinese normalization, never at package import.
    import cn2an

    for number in re.findall(r"\d+(?:\.?\d+)?", text):
        text = text.replace(number, cn2an.an2cn(number), 1)
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = "|".join(re.escape(item) for item in REPLACEMENTS)
    text = re.sub(pattern, lambda m: REPLACEMENTS[m[0]], text)
    text = re.sub(r"[^\u4e00-\u9fa5_a-zA-Z\s!?…,.\'-]+", "", text)
    return re.sub(r"\s+", " ", text)
