"""Small offline fixtures for the repository-owned native Melo frontend."""

import builtins
import io
import json
import pickle
import shutil
import socket
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from hoast.tts2.frontend import Frontend
from hoast.tts2.frontend.english import (
    PHONEMES,
    EnglishG2P,
    distribute_phone,
    read_dictionary,
    refine_phone,
)
from hoast.tts2.frontend.normalization import normalize
from hoast.tts2.frontend.splitting import split_text
from tools.prepare_tts2_frontend import PrimitiveUnpickler, populate_frontend_assets


@pytest.fixture
def assets(tmp_path: Path) -> Path:
    """Create tokenizer/dictionary/tagger/GRU fixtures without machine-local data.

    Args:
        tmp_path:
            Pytest-managed temporary artifact directory.

    """
    vocabulary = [
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "[MASK]",
        "hello",
        "un",
        "##bel",
        "##ievable",
        "read",
        "fallback",
        "florp",
        "!",
        ",",
        ".",
        "'",
        "-",
        "你",
        "好",
        "不",
        "对",
        "一",
        "个",
        "三",
        "点",
        "五",
        "_",
    ]
    tokenizer = Tokenizer(
        models.WordPiece(
            {word: index for index, word in enumerate(vocabulary)}, unk_token="[UNK]"
        )
    )
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    symbols = [
        "_",
        "UNK",
        "V",
        "!",
        "?",
        "…",
        ",",
        ".",
        "'",
        "-",
        "n",
        "i",
        "h",
        "ao",
        "b",
        "u",
        "d",
        "ui",
        "y",
        "g",
        "e",
        "s",
        "an",
        "ian",
        "w",
    ]
    for phone in PHONEMES[4:]:
        symbol, _ = refine_phone(phone)
        if symbol not in symbols:
            symbols.append(symbol)
    (tmp_path / "config.json").write_text(json.dumps({"symbols": symbols}))
    (tmp_path / "cmudict.rep").write_text(
        "## minimal synthetic CMU fixture\nHELLO  HH AH0 - L OW1\n"
        "UNBELIEVABLE  AH2 N - B IH0 - L IY1 - V AH0 - B AH0 L\n"
    )
    (tmp_path / "opencpop-strict.txt").write_text(
        "ni\tn i\nhao\th ao\nbu\tb u\ndui\td ui\nyi\ty i\nge\tg e\n"
        "san\ts an\ndian\td ian\nwu\tw u\n"
    )
    g2p = tmp_path / "g2p"
    tagger = g2p / "tagger"
    tagger.mkdir(parents=True)
    (g2p / "cmudict.json").write_text(
        json.dumps({"fallback": [["F", "AO1", "L", "B", "AE2", "K"]]})
    )
    (g2p / "homographs.en").write_text("# fixture\nREAD|R IY1 D|R EH1 D|V\n")
    (tagger / "weights.json").write_text(json.dumps({"bias": {"NN": 1.0}}))
    (tagger / "tagdict.json").write_text(json.dumps({"read": "VB"}))
    (tagger / "classes.json").write_text(json.dumps(["NN", "VB"]))
    shapes = {
        "enc_emb": (29, 64),
        "enc_w_ih": (384, 64),
        "enc_w_hh": (384, 128),
        "enc_b_ih": (384,),
        "enc_b_hh": (384,),
        "dec_emb": (74, 64),
        "dec_w_ih": (384, 64),
        "dec_w_hh": (384, 128),
        "dec_b_ih": (384,),
        "dec_b_hh": (384,),
        "fc_w": (74, 128),
        "fc_b": (74,),
    }
    variables = {
        name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()
    }
    variables["fc_b"][PHONEMES.index("AH1")] = 1
    np.savez(g2p / "g2p.npz", allow_pickle=False, **variables)
    return tmp_path


@pytest.fixture
def frontend(assets: Path) -> Frontend:
    """Load the English fixture when its optional text dependencies are available.

    Args:
        assets:
            Complete synthetic artifact directory.

    """
    pytest.importorskip("nltk")
    return Frontend(assets, "EN")


def test_dictionary_and_distribution(assets: Path) -> None:
    """Verify CMU syllable flattening and upstream earliest-piece balancing.

    Args:
        assets:
            Synthetic CMU artifact location.

    """
    assert read_dictionary(assets / "cmudict.rep")["HELLO"] == ["HH", "AH0", "L", "OW1"]
    assert distribute_phone(8, 3) == [3, 3, 2]
    assert distribute_phone(1, 3) == [1, 0, 0]
    assert refine_phone("V") == ("V", 0)
    assert refine_phone("AH2") == ("ah", 3)
    with pytest.raises(ValueError):
        distribute_phone(1, 0)


@pytest.mark.parametrize(
    ("hidden_size", "encoder_width", "decoder_width"),
    [(128, 64, 64), (256, 256, 256), (16, 8, 12)],
)
def test_numpy_gru_with_synthetic_weights(
    hidden_size: int, encoder_width: int, decoder_width: int
) -> None:
    """Check GRU gating and the greedy bound without any pretrained resources.

    Args:
        hidden_size:
            Common encoder/decoder hidden width, including the distributed width.

        encoder_width:
            Synthetic encoder embedding width.

        decoder_width:
            Synthetic decoder embedding width, which can differ from the encoder.

    """
    g2p = EnglishG2P.__new__(EnglishG2P)
    g2p.prediction_cache = OrderedDict()
    shapes = {
        "enc_emb": (29, encoder_width),
        "enc_w_ih": (3 * hidden_size, encoder_width),
        "enc_w_hh": (3 * hidden_size, hidden_size),
        "enc_b_ih": (3 * hidden_size,),
        "enc_b_hh": (3 * hidden_size,),
        "dec_emb": (74, decoder_width),
        "dec_w_ih": (3 * hidden_size, decoder_width),
        "dec_w_hh": (3 * hidden_size, hidden_size),
        "dec_b_ih": (3 * hidden_size,),
        "dec_b_hh": (3 * hidden_size,),
        "fc_w": (74, hidden_size),
        "fc_b": (74,),
    }
    g2p.variables = {
        name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()
    }
    g2p.variables["fc_b"][PHONEMES.index("AH1")] = 1
    g2p._validate_variables()
    hidden = g2p._cell(
        np.zeros((1, encoder_width), np.float32),
        np.ones((1, hidden_size), np.float32),
        "enc",
    )
    np.testing.assert_array_equal(hidden, np.full((1, hidden_size), 0.5, np.float32))
    assert g2p.predict("unknown") == ["AH1"] * 20
    g2p.prediction_cache.clear()
    g2p.variables["fc_b"][:] = 0
    g2p.variables["fc_b"][3] = 1
    with pytest.raises(ValueError, match="no phones"):
        g2p.predict("empty")
    g2p.variables["dec_w_hh"] = np.zeros((3 * hidden_size, hidden_size + 1), np.float32)
    with pytest.raises(ValueError, match="Invalid G2P parameter dec_w_hh"):
        g2p._validate_variables()


def test_english_phone_ids_blanks_and_wordpieces(frontend: Frontend) -> None:
    """Check checkpoint symbol mapping, EN offsets and post-blank repeat counts.

    Args:
        frontend:
            Offline English fixture frontend.

    """
    result = frontend.prepare("Hello unbelievable!")
    assert result.normalized_text == "hello unbelievable!"
    assert result.input_ids.shape == (1, 7)
    assert result.word2ph.tolist() == [3, 8, 8, 8, 6, 2, 2]
    for array in (
        result.phones,
        result.tones,
        result.languages,
        result.input_ids,
        result.word2ph,
    ):
        assert array.dtype == np.int64
    assert result.phones.shape == result.tones.shape == result.languages.shape
    assert result.phones.shape[1] == result.word2ph.sum()
    assert result.phones[0, 3] == frontend.symbol_ids["hh"]
    assert result.tones[0, 1:10:2].tolist() == [7, 7, 8, 7, 9]
    assert np.all(result.phones[0, ::2] == 0)
    assert np.all(result.tones[0, ::2] == 0)
    assert np.all(result.languages[0, ::2] == 0)
    assert np.all(result.languages[0, 1::2] == 2)


def test_explicit_english_reuse_keeps_checkpoint_state_separate(
    frontend: Frontend, assets: Path, tmp_path: Path
) -> None:
    """Reuse explicit English resources without reading a second English asset set.

    Args:
        frontend:
            Loaded English fixture supplying the shared resource object.

        assets:
            Source fixtures for an independently tokenized mixed frontend.

        tmp_path:
            Temporary root for a second checkpoint with no CMU or g2p files.

    """
    for name in ("jieba", "pypinyin", "cn2an"):
        pytest.importorskip(name)
    second = tmp_path / "second-checkpoint"
    second.mkdir()
    config = json.loads((assets / "config.json").read_text())
    config["symbols"] = ["_"] + list(reversed(config["symbols"][1:]))
    (second / "config.json").write_text(json.dumps(config))
    tokenizer = json.loads((assets / "tokenizer.json").read_text())
    vocabulary = tokenizer["model"]["vocab"]
    vocabulary["hello"], vocabulary["fallback"] = (
        vocabulary["fallback"],
        vocabulary["hello"],
    )
    (second / "tokenizer.json").write_text(json.dumps(tokenizer))
    shutil.copyfile(assets / "opencpop-strict.txt", second / "opencpop-strict.txt")
    assert not (second / "cmudict.rep").exists()
    assert not (second / "g2p").exists()
    reused = Frontend(second, "ZH", english=frontend.english)
    assert reused.english is frontend.english
    assert reused.tokenizer is not frontend.tokenizer
    assert reused.symbol_ids is not frontend.symbol_ids
    original = frontend.prepare("Hello")
    mixed = reused.prepare("Hello")
    assert original.input_ids.tolist() != mixed.input_ids.tolist()
    assert original.phones[0, 3] == frontend.symbol_ids["hh"]
    assert mixed.phones[0, 3] == reused.symbol_ids["hh"]
    assert original.phones[0, 3] != mixed.phones[0, 3]
    assert original.word2ph.tolist() == mixed.word2ph.tolist()
    assert np.all(original.languages[0, 1::2] == 2)
    assert np.all(mixed.languages[0, 1::2] == 3)


def test_homographs_cmu_precedence_and_oov(frontend: Frontend) -> None:
    """Exercise explicit local POS, fallback CMU and NumPy OOV decoding.

    Args:
        frontend:
            Offline English fixture frontend.

    """
    assert frontend.english.pronounce("read") == (["r", "iy", "d"], [0, 2, 0])
    frontend.english.tagger.tagdict["read"] = "NN"
    assert frontend.english.pronounce("read") == (["r", "eh", "d"], [0, 2, 0])
    frontend.english.dictionary["READ"] = ["R", "IY0", "D"]
    assert frontend.english.pronounce("read") == (["r", "iy", "d"], [0, 1, 0])
    assert frontend.english.pronounce("fallback")[0] == ["f", "ao", "l", "b", "ae", "k"]
    assert frontend.english.pronounce("florp") == (["ah"] * 20, [2] * 20)
    assert frontend.prepare("florp").word2ph.tolist() == [3, 40, 2]


def test_unknown_token_uses_source_spelling(frontend: Frontend) -> None:
    """An unknown BERT token must still receive actual OOV pronunciation.

    Args:
        frontend:
            Offline English fixture frontend.

    """
    result = frontend.prepare("quux")
    assert result.input_ids.tolist() == [[1, 0, 2]]
    assert result.word2ph.tolist() == [3, 40, 2]
    unsupported = frontend.prepare("☃")
    assert unsupported.phones[0, 3] == frontend.symbol_ids["UNK"]
    assert unsupported.word2ph.tolist() == [3, 2, 2]


def test_mixed_alignment_tones_and_numbers(assets: Path) -> None:
    """Preserve mixed language ID three with Chinese and shifted English tones.

    Args:
        assets:
            Complete synthetic mixed-language artifacts.

    """
    for name in ("nltk", "jieba", "pypinyin", "cn2an"):
        pytest.importorskip(name)
    frontend = Frontend(assets, "ZH")
    result = frontend.prepare("你好Hello，3.5！")
    assert result.normalized_text == "你好Hello,三点五!"
    assert len(result.word2ph) == result.input_ids.shape[1]
    assert result.word2ph.sum() == result.phones.shape[1]
    assert result.word2ph.tolist() == [3, 4, 4, 8, 2, 4, 4, 4, 2, 2]
    assert result.tones[0, 1:21:2].tolist() == [0, 2, 2, 3, 3, 7, 8, 7, 9, 0]
    assert np.all(result.languages[0, 1::2] == 3)
    assert np.all(result.languages[0, ::2] == 0)
    assert frontend.chinese is not None
    assert [tone for _, tone in frontend.chinese.pronounce("不对")] == [2, 4]
    assert [tone for _, tone in frontend.chinese.pronounce("一个")] == [2, 5]


def test_normalization_and_split() -> None:
    """Verify non-audition-specific number/time/abbreviation and split rules."""
    pytest.importorskip("inflect")
    assert normalize("Dr. Smith at 13:05 paid $1.25 in 2005.", "EN") == (
        "doctor smith at one oh five p m paid one dollar twenty-five cents in two thousand five."
    )
    assert (
        normalize("21st 1900 2012 -4 1.5", "EN")
        == "twenty-first nineteen hundred twenty twelve minus four one point five"
    )


@pytest.mark.parametrize(
    ("language", "text", "numeric", "expected"),
    [
        (
            "EN",
            "The price is $2.50.",
            "$2.50",
            "the price is two dollars fifty cents.",
        ),
        (
            "EN",
            "The price is $.50.",
            "$.50",
            "the price is fifty cents.",
        ),
        (
            "EN",
            "It is 23.5 degrees.",
            "23.5",
            "it is twenty-three point five degrees.",
        ),
        (
            "EN",
            "The price is $1,234.50.",
            "$1,234.50",
            "the price is twelve thirty-four dollars fifty cents.",
        ),
        (
            "EN",
            "The price is $1,234.50, today.",
            "$1,234.50",
            "the price is twelve thirty-four dollars fifty cents, today.",
        ),
        (
            "EN",
            "There are 12,345 items.",
            "12,345",
            "there are twelve thousand, three hundred forty-five items.",
        ),
        (
            "EN",
            "It is -4.5 degrees.",
            "-4.5",
            "it is minus four point five degrees.",
        ),
        ("ZH", "温度是23.5度。", "23.5", "温度是二十三点五度."),
        ("ZH", "Hello 3.5！", "3.5", "Hello 三点五."),
        ("ZH", "温度是-3.5度。", "-3.5", "温度是-三点五度."),
        ("ZH", "余额1,234.50元。", "1,234.50", "余额一千二百三十四点五零元."),
    ],
)
def test_numeric_split_then_normalize(
    language: Literal["EN", "ZH"], text: str, numeric: str, expected: str
) -> None:
    """Keep numeric punctuation intact until language-specific expansion.

    Args:
        language:
            English or mixed-Chinese splitting and normalization mode.

        text:
            Short utterance containing a decimal or grouped numeral.

        numeric:
            Exact source numeral, including its sign and currency symbol.

        expected:
            Expanded words with sentence punctuation preserved; English uses
            the normalizer's year-style reading for numbers from 1001 to 2999.

    """
    pytest.importorskip("inflect" if language == "EN" else "cn2an")
    chunks = split_text(text, language)
    assert len(chunks) == 1
    assert numeric in chunks[0]
    assert normalize(chunks[0], language) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$2.50.", "two dollars fifty cents."),
        ("$2.50,", "two dollars fifty cents,"),
        ("$1,234.50.", "twelve thirty-four dollars fifty cents."),
        ("$1,234.50,", "twelve thirty-four dollars fifty cents,"),
        ("$.50.", "fifty cents."),
        ("$.50,", "fifty cents,"),
    ],
)
def test_currency_normalization_preserves_sentence_punctuation(
    text: str, expected: str
) -> None:
    """Keep sentence-final dots and commas outside a currency amount match.

    Args:
        text:
            Currency amount followed immediately by sentence punctuation.

        expected:
            Complete monetary expansion retaining the trailing punctuation.

    """
    pytest.importorskip("inflect")
    assert normalize(text, "EN") == expected


@pytest.mark.parametrize("prefix_repeats", [83, 84, 85])
def test_english_numeric_split_near_hard_length_limit(prefix_repeats: int) -> None:
    """Keep a grouped currency decimal whole near the 512-character fallback.

    Args:
        prefix_repeats:
            Number of six-character words placing the amount around the limit.

    """
    pytest.importorskip("inflect")
    numeric = "$1,234.50"
    text = "hello " * prefix_repeats + numeric + " today. " + "hello " * 40
    chunks = split_text(text, "EN")
    assert len(chunks) > 1
    assert all(len(chunk) <= 512 for chunk in chunks)
    assert sum(chunk.count(numeric) for chunk in chunks) == 1
    normalized = [normalize(chunk, "EN") for chunk in chunks]
    assert (
        sum(
            chunk.count("twelve thirty-four dollars fifty cents")
            for chunk in normalized
        )
        == 1
    )


@pytest.mark.parametrize(
    ("text", "numeric", "spoken"),
    [
        (
            (
                "今天室外温度是23.5度，请注意及时增加衣服。"
                "明天室外温度是23.5度，请继续关注天气变化。"
            ),
            "23.5",
            "二十三点五",
        ),
        (
            (
                "Hello，今天测量结果是3.5，请认真记录测量结果。"
                "下一次测量仍然是3.5，请保持设备运行稳定。"
            ),
            "3.5",
            "三点五",
        ),
    ],
)
def test_long_chinese_numeric_split_then_normalize(
    text: str, numeric: str, spoken: str
) -> None:
    """Keep repeated decimals whole across multiple Chinese or mixed chunks.

    Args:
        text:
            Multi-sentence utterance exceeding the Chinese chunk threshold.

        numeric:
            Repeated decimal expression that must stay inside individual chunks.

        spoken:
            Chinese expansion expected once for each source expression.

    """
    pytest.importorskip("cn2an")
    chunks = split_text(text, "ZH")
    assert len(chunks) > 1
    assert sum(chunk.count(numeric) for chunk in chunks) == text.count(numeric)
    normalized = [normalize(chunk, "ZH") for chunk in chunks]
    assert sum(chunk.count(spoken) for chunk in normalized) == text.count(numeric)
    assert not any(character.isdigit() for chunk in normalized for character in chunk)


@pytest.mark.parametrize(
    ("text", "numeric", "expected"),
    [
        (
            "The price is $2.50.",
            "$2.50",
            "the price is two dollars fifty cents.",
        ),
        (
            "It is -23.5 degrees.",
            "-23.5",
            "it is minus twenty-three point five degrees.",
        ),
    ],
)
def test_english_numeric_split_then_prepare(
    frontend: Frontend, text: str, numeric: str, expected: str
) -> None:
    """Exercise numeric expansion and phone alignment through the offline frontend.

    Args:
        frontend:
            English frontend backed by synthetic tokenizer and G2P fixtures.

        text:
            Short numeric utterance passed through the frontend splitter.

        numeric:
            Exact numeric expression required in the raw split output.

        expected:
            Normalized words and punctuation supplied to tokenization.

    """
    chunks = frontend.split(text)
    assert len(chunks) == 1
    assert numeric in chunks[0]
    result = frontend.prepare(chunks[0])
    assert result.normalized_text == expected
    assert len(result.word2ph) == result.input_ids.shape[1]
    assert result.word2ph.sum() == result.phones.shape[1]


def test_mixed_numeric_split_then_prepare(assets: Path) -> None:
    """Expand a mixed decimal after splitting using existing tiny Chinese assets.

    Args:
        assets:
            Synthetic mixed-language tokenizer, dictionary and G2P artifacts.

    """
    for name in ("nltk", "jieba", "pypinyin", "cn2an"):
        pytest.importorskip(name)
    frontend = Frontend(assets, "ZH")
    chunks = frontend.split("你好Hello，3.5！")
    assert len(chunks) == 1
    assert "3.5" in chunks[0]
    result = frontend.prepare(chunks[0])
    assert result.normalized_text == "你好Hello, 三点五."
    assert len(result.word2ph) == result.input_ids.shape[1]
    assert result.word2ph.sum() == result.phones.shape[1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("13:05 paid", "one oh five p m paid"),
        ("13:05   paid", "one oh five p m   paid"),
        ("13:05\tpaid", "one oh five p m\tpaid"),
        ("13:05", "one oh five p m"),
        ("00:00 tomorrow", "twelve a m tomorrow"),
        ("00:05!", "twelve oh five a m!"),
        ("12:00 today", "twelve p m today"),
        ("23:59 tonight", "eleven fifty-nine p m tonight"),
        ("9:30am next", "nine thirty a m next"),
        ("9:30 AM next", "nine thirty a m next"),
        ("9:30 p.m. next", "nine thirty p m next"),
        ("9:30 a.m next", "nine thirty a m next"),
        ("09:00 pm", "nine p m"),
        ("09:00 p.m.!", "nine p m!"),
    ],
)
def test_clock_times_preserve_content(text: str, expected: str) -> None:
    """Correct time-word boundaries and midnight while respecting meridiems.

    Args:
        text:
            Clock time with trailing content or an explicit meridiem spelling.

        expected:
            Exact content-preserving normalized English output.

    """
    assert normalize(text, "EN") == expected


@pytest.mark.parametrize("legacy_tokenizer", [False, True])
def test_frontend_asset_conversion(
    assets: Path, tmp_path: Path, legacy_tokenizer: bool
) -> None:
    """Convert primitive tagger and CMU fixtures offline with complete inventory.

    Args:
        assets:
            Synthetic tokenizer/config and G2P source data.

        tmp_path:
            Temporary root for explicitly separated source and output directories.

        legacy_tokenizer:
            Omit the optional WordPiece type discriminator as in official BERT JSON.

    """
    source = tmp_path / "source"
    text = source / "melo/text"
    text.mkdir(parents=True)
    shutil.copyfile(assets / "cmudict.rep", text / "cmudict.rep")
    (source / "LICENSE").write_text("Synthetic test license\n")
    g2p = tmp_path / "g2p-input"
    g2p.mkdir()
    shutil.copyfile(assets / "g2p/g2p.npz", g2p / "checkpoint20.npz")
    shutil.copyfile(assets / "g2p/homographs.en", g2p / "homographs.en")
    nltk = tmp_path / "nltk-input"
    cmu = nltk / "corpora/cmudict"
    cmu.mkdir(parents=True)
    (cmu / "cmudict").write_text("A 1 AH0\nA 2 EY1\n")
    (cmu / "README").write_text("Synthetic CMU license\n")
    tagger = nltk / "taggers/averaged_perceptron_tagger"
    tagger.mkdir(parents=True)
    values = ({"bias": {"NN": 1.0}}, {"read": "VB"}, {"NN", "VB"})
    (tagger / "averaged_perceptron_tagger.pickle").write_bytes(
        pickle.dumps(values, protocol=2)
    )
    destination = tmp_path / "prepared"
    destination.mkdir()
    for filename in ("config.json", "tokenizer.json"):
        shutil.copyfile(assets / filename, destination / filename)
    if legacy_tokenizer:
        tokenizer_path = destination / "tokenizer.json"
        tokenizer_data = json.loads(tokenizer_path.read_text())
        del tokenizer_data["model"]["type"]
        tokenizer_data["decoder"] = {
            "type": "WordPiece",
            "prefix": "##",
            "cleanup": True,
        }
        tokenizer_path.write_text(json.dumps(tokenizer_data))
    emitted = populate_frontend_assets(
        source, destination, "EN", g2p_resource_dir=g2p, nltk_resource_dir=nltk
    )
    actual = {
        path.relative_to(destination)
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert actual == set(emitted) | {Path("config.json"), Path("tokenizer.json")}
    assert json.loads((destination / "g2p/cmudict.json").read_text()) == {
        "a": [["AH0"], ["EY1"]]
    }
    assert json.loads((destination / "g2p/tagger/classes.json").read_text()) == [
        "NN",
        "VB",
    ]
    provenance = json.loads((destination / "frontend-assets.json").read_text())
    assert set(provenance["files"]) == {
        path.as_posix() for path in emitted if path != Path("frontend-assets.json")
    }
    assert Frontend(destination, "EN").prepare("hello").word2ph.tolist() == [3, 8, 2]
    with pytest.raises(FileExistsError, match="already exists"):
        populate_frontend_assets(
            source, destination, "EN", g2p_resource_dir=g2p, nltk_resource_dir=nltk
        )


def test_restricted_tagger_reader() -> None:
    """Allow primitive reference tuples but reject non-built-in globals."""
    values = ({"bias": {"NN": 1.0}}, {"word": "NN"}, {"NN"})
    assert (
        PrimitiveUnpickler(io.BytesIO(pickle.dumps(values, protocol=2))).load()
        == values
    )
    with pytest.raises(pickle.UnpicklingError, match="Forbidden"):
        PrimitiveUnpickler(io.BytesIO(pickle.dumps(Path))).load()


def test_prediction_cache_is_instance_owned(frontend: Frontend, assets: Path) -> None:
    """Prevent cached OOV values from retaining other frontend state or mutation.

    Args:
        frontend:
            English fixture whose prediction cache is populated and modified.

        assets:
            Synthetic artifact directory for a second independent frontend.

    """
    first = frontend.english.predict("florp")
    first.clear()
    assert frontend.english.predict("florp") == ["AH1"] * 20
    other = Frontend(assets, "EN")
    assert not other.english.prediction_cache
    assert len(frontend.english.prediction_cache) == 1


def test_split_semantics() -> None:
    """Check short/empty/long EN and ZH chunking without optional dependencies."""
    assert split_text("Hello. Bye!", "EN") == ["Hello. Bye!"]
    assert split_text("你好！这是第二个句子。好的。", "ZH") == [
        "你好. 这是第二个句子.",
        "好的.",
    ]
    assert split_text("", "EN") == []
    assert split_text("", "ZH") == []
    chunks = split_text("hello " * 200, "EN")
    assert len(chunks) > 1
    assert all(len(chunk) <= 512 for chunk in chunks)


def test_missing_assets_and_invalid_symbols(assets: Path) -> None:
    """Reject missing dictionaries and symbol alphabets instead of defaulting.

    Args:
        assets:
            Synthetic artifact directory whose files may be removed or modified.

    """
    (assets / "cmudict.rep").unlink()
    with pytest.raises(FileNotFoundError, match="cmudict.rep"):
        Frontend(assets, "EN")
    (assets / "config.json").write_text(json.dumps({"symbols": ["UNK", "_"]}))
    with pytest.raises(ValueError, match="blank ID zero"):
        Frontend(assets, "EN")


def test_missing_fallback_assets(assets: Path) -> None:
    """Require explicit fallback resources even for dictionary-only utterances.

    Args:
        assets:
            Synthetic artifacts from which one fallback resource is removed.

    """
    (assets / "g2p" / "homographs.en").unlink()
    with pytest.raises(FileNotFoundError, match="homographs.en"):
        Frontend(assets, "EN")


def test_empty_input_and_missing_checkpoint_phone(frontend: Frontend) -> None:
    """Reject empty utterances and incompatible alphabets, keeping exact geometry.

    Args:
        frontend:
            Offline English fixture frontend.

    """
    with pytest.raises(ValueError, match="empty"):
        frontend.prepare("  ")
    del frontend.symbol_ids["hh"]
    with pytest.raises(ValueError, match="absent from checkpoint"):
        frontend.prepare("Hello")


def test_no_downloads_or_unrelated_runtime_imports(
    assets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Construct and prepare offline while forbidding network and speech imports.

    Args:
        assets:
            Complete synthetic frontend artifacts.

        monkeypatch:
            Scoped patch manager guarding network and unwanted modules.

    """
    nltk = pytest.importorskip("nltk")
    original_import = builtins.__import__

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        """Fail immediately on attempted downloader/network use.

        Args:
            args:
                Positional call arguments.

            kwargs:
                Keyword call arguments.

        """
        raise AssertionError("Offline frontend attempted network access")

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Reject model wrappers and unrelated language runtimes.

        Args:
            name:
                Imported module name.

            args:
                Other positional import arguments.

            kwargs:
                Keyword import arguments.

        """
        if name.split(".")[0] in {
            "melo",
            "funasr",
            "g2p_en",
            "transformers",
            "torch",
            "subprocess",
            "pyopenjtalk",
        }:
            raise AssertionError(f"Forbidden runtime import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(nltk, "download", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert Frontend(assets, "EN").prepare("hello florp").word2ph.sum() > 0
