# NumPy G2P adapted from Kyubyong Park and Jongseok Kim (Apache License 2.0).
# Modified for hoast: explicit assets, offline loading, validation and typing.
"""Offline dictionary and NumPy GRU English G2P adapted from Melo/g2p_en.

No NLTK resource discovery, download, pickle loading, or g2p_en import occurs.
See NOTICE.md for source attribution and licenses.
"""

import json
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
PHONEMES: tuple[str, ...] = (
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "AA0",
    "AA1",
    "AA2",
    "AE0",
    "AE1",
    "AE2",
    "AH0",
    "AH1",
    "AH2",
    "AO0",
    "AO1",
    "AO2",
    "AW0",
    "AW1",
    "AW2",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH0",
    "EH1",
    "EH2",
    "ER0",
    "ER1",
    "ER2",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH0",
    "IH1",
    "IH2",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW0",
    "OW1",
    "OW2",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
)


def read_dictionary(path: Path) -> dict[str, list[str]]:
    """Read Melo's syllabified CMU dictionary without writing a pickle cache.

    Args:
        path:
            UTF-8 cmudict.rep file with comment lines and two-space delimiters.

    """
    result: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("##"):
            continue
        word, pronunciation = re.split(r"\s{2,}", line.strip(), maxsplit=1)
        result[word] = pronunciation.replace(" - ", " ").split()
    if not result:
        raise ValueError(f"Empty pronunciation dictionary: {path}")
    return result


def distribute_phone(n_phone: int, n_word: int) -> list[int]:
    """Distribute phones evenly over wordpieces, favoring earlier pieces.

    Args:
        n_phone:
            Number of phones in the complete word.

        n_word:
            Positive number of tokenizer pieces in the complete word.

    """
    if n_phone < 0 or n_word <= 0:
        raise ValueError("Phone count must be nonnegative and wordpiece count positive")
    quotient, remainder = divmod(n_phone, n_word)
    return [quotient + int(index < remainder) for index in range(n_word)]


def refine_phone(phone: str) -> tuple[str, int]:
    """Convert ARPAbet vowel stress to Melo tones and its lowercase alphabet.

    Args:
        phone:
            ARPAbet symbol or punctuation from a dictionary/G2P output.

    """
    if (phone in PHONEMES[4:] and phone != "UW") or phone in {"ER", "IH"}:
        tone = int(phone[-1]) + 1 if phone[-1].isdigit() else 0
        symbol = phone[:-1].lower() if phone[-1].isdigit() else phone.lower()
        return ("V" if symbol == "v" else symbol), tone
    return phone, 0


class EnglishG2P:
    """CMU-first pronunciation with explicitly loaded offline fallback assets."""

    dictionary: dict[str, list[str]]
    """Melo CMU dictionary, keyed by uppercase spelling."""

    cmu: dict[str, list[list[str]]]
    """NLTK CMU fallback dictionary, keyed by lowercase spelling."""

    homographs: dict[str, tuple[list[str], list[str], str]]
    """Fallback POS-conditioned pronunciation pairs and POS prefix."""

    tagger: Any
    """NLTK PerceptronTagger populated only from explicit JSON files."""

    word_tokenize: Any
    """NLTK TweetTokenizer tokenization callable, requiring no corpus assets."""

    variables: dict[str, FloatArray]
    """Float32 encoder, decoder and classifier arrays from g2p.npz."""

    prediction_cache: OrderedDict[str, tuple[str, ...]]
    """Instance-owned LRU cache of at most 4096 immutable OOV pronunciations."""

    hidden_size: int
    """Validated encoder/decoder GRU hidden width H, inferred from artifact arrays."""

    def __init__(self, artifact_dir: Path) -> None:
        """Load all English resources, failing on missing or malformed assets.

        Args:
            artifact_dir:
                Frontend artifact root containing cmudict.rep and g2p/ assets.

        """
        self.dictionary = read_dictionary(artifact_dir / "cmudict.rep")
        self.prediction_cache = OrderedDict()
        root = artifact_dir / "g2p"
        self.cmu = json.loads((root / "cmudict.json").read_text(encoding="utf-8"))
        self.homographs = {}
        for line in (root / "homographs.en").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            word, first, second, pos = line.strip().split("|")
            self.homographs[word.lower()] = (first.split(), second.split(), pos)
        with np.load(root / "g2p.npz", allow_pickle=False) as archive:
            self.variables = {
                key: np.asarray(archive[key], dtype=np.float32) for key in archive.files
            }
        self._validate_variables()
        # Optional language tooling is lazy; never import g2p_en's downloader.
        from nltk.tag.perceptron import PerceptronTagger
        from nltk.tokenize import TweetTokenizer

        self.tagger = PerceptronTagger(load=False)
        tagger_root = root / "tagger"
        self.tagger.model.weights = json.loads(
            (tagger_root / "weights.json").read_text()
        )
        self.tagger.tagdict = json.loads((tagger_root / "tagdict.json").read_text())
        self.tagger.classes = set(
            json.loads((tagger_root / "classes.json").read_text())
        )
        if not self.tagger.classes:
            raise ValueError("The explicit English POS tagger has no classes")
        self.tagger.model.classes = self.tagger.classes
        self.word_tokenize = TweetTokenizer().tokenize

    def _validate_variables(self) -> None:
        """Validate vocabulary sizes and GRU dimension relationships at the boundary.

        Embedding/hidden widths are inferred because upstream source comments
        describe smaller dimensions than the distributed checkpoint arrays.
        """
        encoder_embedding = self.variables["enc_emb"]
        decoder_embedding = self.variables["dec_emb"]
        encoder_recurrent = self.variables["enc_w_hh"]
        if any(
            array.ndim != 2
            for array in (encoder_embedding, decoder_embedding, encoder_recurrent)
        ):
            raise ValueError("G2P embeddings and recurrent weights must be matrices")
        self.hidden_size = encoder_recurrent.shape[1]
        encoder_width = encoder_embedding.shape[1]
        decoder_width = decoder_embedding.shape[1]
        if min(self.hidden_size, encoder_width, decoder_width) <= 0:
            raise ValueError("G2P embedding/hidden dimensions must be positive")
        gates = 3 * self.hidden_size
        shapes = {
            "enc_emb": (29, encoder_width),
            "enc_w_ih": (gates, encoder_width),
            "enc_w_hh": (gates, self.hidden_size),
            "enc_b_ih": (gates,),
            "enc_b_hh": (gates,),
            "dec_emb": (74, decoder_width),
            "dec_w_ih": (gates, decoder_width),
            "dec_w_hh": (gates, self.hidden_size),
            "dec_b_ih": (gates,),
            "dec_b_hh": (gates,),
            "fc_w": (74, self.hidden_size),
            "fc_b": (74,),
        }
        for name, shape in shapes.items():
            if name not in self.variables or self.variables[name].shape != shape:
                raise ValueError(f"Invalid G2P parameter {name}: expected {shape}")
            if not np.isfinite(self.variables[name]).all():
                raise ValueError(f"Nonfinite G2P parameter {name}")

    def _cell(self, x: FloatArray, hidden: FloatArray, prefix: str) -> FloatArray:
        """Evaluate the upstream reset/update/new GRU cell in NumPy float32.

        Args:
            x:
                Input embedding with shape [1, E], where E is the selected width.

            hidden:
                Hidden state with shape [1, H], inferred from artifact parameters.

            prefix:
                Encoder 'enc' or decoder 'dec' parameter prefix.

        """
        variables = self.variables
        ih = x @ variables[f"{prefix}_w_ih"].T + variables[f"{prefix}_b_ih"]
        hh = hidden @ variables[f"{prefix}_w_hh"].T + variables[f"{prefix}_b_hh"]
        boundary = 2 * hidden.shape[1]
        with np.errstate(over="ignore"):
            gates = np.float32(1) / (
                np.float32(1) + np.exp(-(ih[:, :boundary] + hh[:, :boundary]))
            )
        reset, update = np.split(gates, 2, axis=-1)
        candidate = np.tanh(ih[:, boundary:] + reset * hh[:, boundary:])
        return np.asarray(
            (np.float32(1) - update) * candidate + update * hidden, dtype=np.float32
        )

    def predict(self, word: str) -> list[str]:
        """Predict OOV phones with the twenty-step GRU decoder and bounded LRU.

        Cached values do not retain other frontend instances and are returned
        as fresh lists so callers cannot mutate subsequent pronunciations.

        Args:
            word:
                Lowercase, accent-stripped English spelling.

        """
        cached = self.prediction_cache.get(word)
        if cached is not None:
            self.prediction_cache.move_to_end(word)
            return list(cached)
        graphemes = [
            ord(char) - ord("a") + 3 if "a" <= char <= "z" else 1 for char in word
        ] + [2]
        hidden = np.zeros((1, self.hidden_size), dtype=np.float32)
        for index in graphemes:
            hidden = self._cell(self.variables["enc_emb"][[index]], hidden, "enc")
        prediction = 2
        result = []
        for _ in range(20):
            hidden = self._cell(self.variables["dec_emb"][[prediction]], hidden, "dec")
            prediction = int(
                (hidden @ self.variables["fc_w"].T + self.variables["fc_b"]).argmax()
            )
            if prediction == 3:
                break
            result.append(PHONEMES[prediction])
        if not result:
            raise ValueError(f"G2P fallback produced no phones for {word!r}")
        if len(self.prediction_cache) >= 4096:
            self.prediction_cache.popitem(last=False)
        self.prediction_cache[word] = tuple(result)
        return result

    def pronounce(self, word: str) -> tuple[list[str], list[int]]:
        """Apply Melo CMU precedence and g2p_en's per-word fallback semantics.

        Homographs in cmudict.rep retain that dictionary's pronunciation. Only
        fallback words use POS selection, with standalone-word tagging as in
        upstream english.g2p rather than sentence-context retagging.

        Args:
            word:
                Complete tokenizer word, or source spelling for an UNK token.

        """
        if word.upper() in self.dictionary:
            pronunciation = self.dictionary[word.upper()]
        else:
            text = "".join(
                char
                for char in unicodedata.normalize("NFD", word)
                if unicodedata.category(char) != "Mn"
            ).lower()
            text = re.sub("[^ a-z'.,?!\\-]", "", text)
            text = text.replace("i.e.", "that is").replace("e.g.", "for example")
            words = self.word_tokenize(text)
            pronunciation = []
            for token, pos in self.tagger.tag(words):
                if re.search("[a-z]", token) is None:
                    pronunciation.append(token)
                elif token in self.homographs:
                    first, second, prefix = self.homographs[token]
                    pronunciation.extend(first if pos.startswith(prefix) else second)
                elif token in self.cmu:
                    pronunciation.extend(self.cmu[token][0])
                else:
                    pronunciation.extend(self.predict(token))
            if not pronunciation:
                # An explicit unknown symbol preserves the token/phone geometry.
                pronunciation = ["UNK"]
        refined = [refine_phone(phone) for phone in pronunciation]
        return [item[0] for item in refined], [item[1] for item in refined]
