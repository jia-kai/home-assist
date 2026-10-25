"""Populate native Melo frontend assets from explicit offline local resources.

This Python 3.14 preparation helper uses only the standard library. It never
imports Melo, g2p_en, NLTK, speech models, or an alternate interpreter and never
downloads anything. The graph preparer must first copy checkpoint config.json
and the matching BERT tokenizer.json into its staging directory, then invoke
populate_frontend_assets before publishing and recursively inventorying files.

CLI: python -m tools.prepare_tts2_frontend --help

All paths are mandatory. The default downloaded input resources are located at:
    source: models/speech2-inputs/MeloTTS
    g2p-resource-dir: models/speech2-inputs/g2p_en
    nltk-resource-dir: models/speech2-inputs/nltk
Choose a staging destination under models with config/tokenizer already copied.

The Python version in a resource pathname is immaterial: files are treated as
static data, never imported or executed. A primitive-only restricted unpickler
converts the reference NLTK perceptron tuple to JSON. Native JSON taggers are
also accepted. Runtime uses JSON only, not this preparation-only reader.
"""

import argparse
import hashlib
import io
import json
import logging
import math
import pickle
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PrimitiveUnpickler(pickle.Unpickler):
    """Read NLTK's dictionary/dictionary/set tuple without arbitrary globals."""

    def find_class(self, module: str, name: str) -> Any:
        """Allow only the set constructor needed by Python-2-format tagger data.

        Args:
            module:
                Pickle GLOBAL module spelling, never imported by this method.

            name:
                Pickle GLOBAL symbol spelling.

        """
        if module in {"__builtin__", "builtins"} and name == "set":
            return set
        raise pickle.UnpicklingError(f"Forbidden tagger pickle global: {module}.{name}")


def _sha256(path: Path) -> str:
    """Hash a resource with bounded memory, following explicit local symlinks.

    Args:
        path:
            Existing source or emitted artifact file.

    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cmu_dictionary(path: Path) -> dict[str, list[list[str]]]:
    """Parse NLTK's word/counter/phones corpus preserving pronunciation order.

    Args:
        path:
            Explicit unpacked NLTK corpora/cmudict/cmudict text resource.

    """
    dictionary: dict[str, list[list[str]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.startswith(";;;"):
                continue
            fields = line.split()
            if len(fields) < 3 or not fields[1].isdigit():
                raise ValueError(f"Invalid CMU corpus record at {path}:{line_number}")
            dictionary.setdefault(fields[0].lower(), []).append(fields[2:])
    if not dictionary:
        raise ValueError(f"Empty NLTK CMU corpus: {path}")
    return dictionary


def _tagger_values(
    resource_dir: Path,
) -> tuple[dict[str, Any], dict[str, str], list[str], list[Path]]:
    """Load explicit unpacked reference pickle or current NLTK JSON tagger data.

    The reference pickle is preferred when present because choosing a newly
    trained JSON model can change homograph decisions. No NLTK APIs or corpus
    lookup are used. ZIP-only resources must first be unpacked by the caller.

    Args:
        resource_dir:
            Explicit NLTK resource root containing taggers/.

    """
    legacy = (
        resource_dir
        / "taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle"
    )
    if legacy.is_file():
        value = PrimitiveUnpickler(io.BytesIO(legacy.read_bytes())).load()
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError(
                "Reference tagger must contain a weights/tagdict/classes tuple"
            )
        weights, tagdict, classes = value
        inputs = [legacy]
    else:
        root = resource_dir / "taggers/averaged_perceptron_tagger_eng"
        inputs = [
            root / f"averaged_perceptron_tagger_eng.{part}.json"
            for part in ("weights", "tagdict", "classes")
        ]
        weights, tagdict, classes = [
            json.loads(path.read_text(encoding="utf-8")) for path in inputs
        ]
    if not isinstance(weights, dict) or not isinstance(tagdict, dict):
        raise TypeError("Perceptron weights and tag dictionary must be JSON objects")
    if (
        not isinstance(classes, (set, list, tuple))
        or not classes
        or not all(isinstance(tag, str) for tag in classes)
    ):
        raise ValueError("Perceptron classes must be a nonempty string collection")
    class_set = set(classes)
    for feature, values in weights.items():
        if not isinstance(feature, str) or not isinstance(values, dict):
            raise TypeError("Invalid perceptron feature weights")
        for tag, weight in values.items():
            if (
                tag not in class_set
                or isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
            ):
                raise ValueError(f"Invalid perceptron weight for feature {feature!r}")
    if not all(
        isinstance(word, str) and isinstance(tag, str) and tag in class_set
        for word, tag in tagdict.items()
    ):
        raise ValueError("Invalid perceptron tag dictionary entry")
    return weights, tagdict, sorted(class_set), inputs


def _validate_checkpoint(destination: Path) -> None:
    """Require a prepared official alphabet and fast-tokenizer artifact.

    Args:
        destination:
            Staging directory containing graph-preparer-owned config/tokenizer files.

    """
    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    symbols = config["symbols"]
    if (
        not isinstance(symbols, list)
        or not symbols
        or not all(isinstance(symbol, str) for symbol in symbols)
    ):
        raise ValueError("Checkpoint symbols must be a nonempty string list")
    if symbols[0] != "_" or len(set(symbols)) != len(symbols) or "UNK" not in symbols:
        raise ValueError(
            "Checkpoint symbols require unique values, '_' at ID zero and explicit 'UNK'"
        )
    tokenizer = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
    model = tokenizer["model"]
    # Official pinned BERT JSON predates the optional model type discriminator.
    wordpiece = model.get("type") == "WordPiece" or (
        "type" not in model
        and tokenizer["decoder"]["type"] == "WordPiece"
        and model["continuing_subword_prefix"] == "##"
        and model["unk_token"] == "[UNK]"
        and isinstance(model["vocab"], dict)
        and "merges" not in model
    )
    if not wordpiece:
        raise ValueError("Melo EN/ZH preparation requires a BERT WordPiece tokenizer")


def populate_frontend_assets(
    source: Path,
    destination: Path,
    language: str,
    *,
    g2p_resource_dir: Path,
    nltk_resource_dir: Path,
) -> list[Path]:
    """Copy and convert explicit local frontend data for native Python 3.14.

    Returns sorted paths relative to destination, including every emitted data,
    license, notice, documentation and frontend provenance file. Graph preparers
    must inventory recursively (rglob('*')), not only destination.iterdir(), so
    g2p/tagger resources receive checksums. Config/tokenizer are caller-owned and
    must already exist; they are neither generated nor modified by this helper.
    Input data is read completely before creating output files. Existing emitted
    paths are rejected. Use a disposable staging directory for atomic publication.

    Args:
        source:
            Offline MeloTTS checkout root containing melo/text and LICENSE.

        destination:
            Graph preparation staging root with config.json and tokenizer.json.

        language:
            Exact EN or ZH frontend mode; ZH includes English fallback resources.

        g2p_resource_dir:
            Explicit data directory containing checkpoint20.npz and homographs.en.
            An installed package directory is accepted as data, never imported.

        nltk_resource_dir:
            Explicit unpacked NLTK corpus/tagger resource root, containing corpora/
            and taggers/. No global NLTK resource paths are searched.

    """
    if language not in {"EN", "ZH"}:
        raise ValueError(f"Unsupported frontend language: {language!r}")
    _validate_checkpoint(destination)
    cmu_source = nltk_resource_dir / "corpora/cmudict/cmudict"
    dictionary = _cmu_dictionary(cmu_source)
    weights, tagdict, classes, tagger_inputs = _tagger_values(nltk_resource_dir)
    frontend = Path(__file__).resolve().parents[1] / "hoast/tts2/frontend"
    copies = {
        Path("cmudict.rep"): source / "melo/text/cmudict.rep",
        Path("g2p/g2p.npz"): g2p_resource_dir / "checkpoint20.npz",
        Path("g2p/homographs.en"): g2p_resource_dir / "homographs.en",
        Path("g2p/CMU-README.txt"): nltk_resource_dir / "corpora/cmudict/README",
        Path("frontend/MeloTTS-LICENSE.txt"): source / "LICENSE",
        Path("frontend/NOTICE.md"): frontend / "NOTICE.md",
        Path("frontend/LICENSE-APACHE-2.0.txt"): frontend / "LICENSE-APACHE-2.0.txt",
        Path("frontend/ASSETS.md"): frontend / "ASSETS.md",
    }
    if language == "ZH":
        copies[Path("opencpop-strict.txt")] = source / "melo/text/opencpop-strict.txt"
    generated = {
        Path("g2p/cmudict.json"): dictionary,
        Path("g2p/tagger/weights.json"): weights,
        Path("g2p/tagger/tagdict.json"): tagdict,
        Path("g2p/tagger/classes.json"): classes,
    }
    emitted = sorted([*copies, *generated, Path("frontend-assets.json")])
    for relative in emitted:
        if (destination / relative).exists():
            raise FileExistsError(
                f"Frontend output already exists: {destination / relative}"
            )
    source_hashes = {str(relative): _sha256(path) for relative, path in copies.items()}
    source_hashes["nltk/cmudict"] = _sha256(cmu_source)
    for index, path in enumerate(tagger_inputs):
        source_hashes[f"nltk/tagger-input-{index}/{path.name}"] = _sha256(path)
    for relative, path in copies.items():
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
    for relative, value in generated.items():
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
    provenance = {
        "version": 1,
        "language": language,
        "preparation": "native Python standard library; explicit offline resources; restricted primitive-only tagger pickle conversion",
        "cmu_words": len(dictionary),
        "tagger_classes": len(classes),
        "source_sha256": source_hashes,
        "files": {
            str(relative): {
                "sha256": _sha256(destination / relative),
                "bytes": (destination / relative).stat().st_size,
            }
            for relative in emitted
            if relative != Path("frontend-assets.json")
        },
    }
    (destination / "frontend-assets.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "tts2.frontend.prepare status=ok language=%s files=%d cmu_words=%d",
        language,
        len(emitted),
        len(dictionary),
    )
    return emitted


def main() -> None:
    """Parse an explicit offline CLI and log success or complete failure details."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Offline MeloTTS checkout; never imported",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="Staging artifact root with config.json and tokenizer.json already copied",
    )
    parser.add_argument("--language", choices=("EN", "ZH"), required=True)
    parser.add_argument(
        "--g2p-resource-dir",
        type=Path,
        required=True,
        help="Explicit directory with checkpoint20.npz and homographs.en; never imported",
    )
    parser.add_argument(
        "--nltk-resource-dir",
        type=Path,
        required=True,
        help="Explicit unpacked NLTK corpora/ and taggers/ root; no downloads or global lookup",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        populate_frontend_assets(
            args.source,
            args.destination,
            args.language,
            g2p_resource_dir=args.g2p_resource_dir,
            nltk_resource_dir=args.nltk_resource_dir,
        )
    except Exception:
        logger.exception("tts2.frontend.prepare status=error")
        raise


if __name__ == "__main__":
    main()
