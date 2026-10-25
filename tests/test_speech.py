"""Model-independent speech text and Han routing regressions."""

from hoast.chinese import contains_han
from hoast.speech_text import prepare_speech_text


def test_spoken_titles_and_initialisms() -> None:
    """Remove silent delimiters without changing apostrophes or sentence punctuation."""
    assert (
        prepare_speech_text('"G.E.M."\'s recording of 《晴天》.')
        == "G E M's recording of 晴天."
    )
    assert prepare_speech_text("Don't stop!") == "Don't stop!"


def test_han_routing() -> None:
    """Recognize extended Han while excluding other scripts and phonetic symbols."""
    for text in ("你好", "Hello. 你好。", "〇", "\U00020000", "\uf900"):
        assert contains_han(text)
    for text in ("Hello!", "123", "ㄋㄧ3", "かな", "한글", ""):
        assert not contains_han(text)
