"""Fetch or validate pinned native speech inputs without importing upstream code.

Offline is the default; --download explicitly permits network access. Only EN/ZH
Melo/BERT, official SenseVoice, static English G2P/NLTK data and optionally the
community UINT8 source are handled. No package is installed, model is exported,
or alternate interpreter is executed. Existing files are verified, never replaced.
The resulting inventory can be passed to the native preparation tools.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)
SENSEVOICE_REPOSITORY = "FunAudioLLM/SenseVoiceSmall"
SENSEVOICE_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
MELO_REVISION = "209145371cff8fc3bd60d7be902ea69cbdb7965a"
NLTK_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
SENSEVOICE_FILES = {
    "model.pt": "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea",
    "config.yaml": "f71e239ba36705564b5bf2d2ffd07eece07b8e3f2bbf6d2c99d8df856339ac19",
    "am.mvn": "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5",
    "chn_jpn_yue_eng_ko_spectok.bpe.model": "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8",
    "README.md": "6add90487ea3d685b6604ad0932b565fc5cf7e9daaf8aebc674de49da4560958",
}
LICENSE_INPUTS = {
    "MODEL_LICENSE": (
        (
            "https://raw.githubusercontent.com/modelscope/FunASR/"
            "66d7a4c264a5993a2a63ed00c1f402c296ee521a/MODEL_LICENSE"
        ),
        "7dba975a2069691db4992b0592d70828b330d2f8a30a71450f4e152a554e84f8",
        5306,
    ),
    "FunASR-LICENSE": (
        (
            "https://raw.githubusercontent.com/modelscope/FunASR/"
            "66d7a4c264a5993a2a63ed00c1f402c296ee521a/LICENSE"
        ),
        "f382b62dcc61fc566b9215b1e3b604b539bfd96ed52ae1eeff5090f6dc176057",
        1061,
    ),
    "SenseVoice-LICENSE": (
        (
            "https://raw.githubusercontent.com/FunAudioLLM/SenseVoice/"
            "a80e676461b24419cf1130a33d4dd2f04053e5cc/LICENSE"
        ),
        "221c6df10b0931a5629adad671ea48fb7747e034c414b6d2bfa275bc3dd4ea17",
        71,
    ),
}
LICENSE_NOTES = [
    (
        "Official model card references FunASR main/MODEL_LICENSE, a mutable URL; "
        "this is its observed pinned text, not a claim of historical license applicability."
    ),
    (
        "SenseVoice source revision is the commit referenced in community export-onnx.py; "
        "its LICENSE is itself a URL reference."
    ),
    (
        "No separate tokenizer/CMVN/config license declaration was found in the official "
        "snapshot. No resource license has been invented."
    ),
]
# These inventories match the audited benchmark snapshots, not floating HF refs.
HF_ASSETS = {
    "myshell-ai/MeloTTS-English": (
        "bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef",
        {
            "checkpoint.pth": "acd278040eaf9536908e2b965273df5a731c44d8f0da66cc5fed7972772ed23c",
            "config.json": "039116c927c70eaa4458d315ea83aaaa99e1fca1c621b50c8ca56b4a5700eb77",
        },
    ),
    "myshell-ai/MeloTTS-Chinese": (
        "af5d207a364ea4208c6f589c89f57f88414bdd16",
        {
            "checkpoint.pth": "a74e9eadffff065c75eb6dfa040efa72cad23e72cfea70d39190bc174fb97093",
            "config.json": "d58b5acdab89ad2bbd65325affab309ae3cb964834b02f9a60587474e81c8bb9",
        },
    ),
    "bert-base-uncased": (
        "86b5e0934494bd15c9632b12f734a8a67f723594",
        {
            "config.json": "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20",
            "pytorch_model.bin": "097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a",
            "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
        },
    ),
    "bert-base-multilingual-uncased": (
        "7cbf9a625e29989f6b9c6c2fa68234c304f7e38f",
        {
            "config.json": "fba5d4b0a351a43f6ccb7a6587301fd9f6876ca36aae62af762af67c8f18db1c",
            "pytorch_model.bin": "2fec0e2a13cde5fa386fa00ba3e1bfea14b5d8fd8760f37f051799812a320e8d",
            "tokenizer.json": "41a9e8ccfd02a7bf5bb8426ed22eec3a5bd81d85d7c10ef4a5cbdbfa21faa21b",
        },
    ),
}
G2P_FILES = {
    "checkpoint20.npz": "b8af35e4596d8dd5836dfd3fe9b2ba4f97b9c311efe8879544cbcfcbd566d8c6",
    "homographs.en": "0403edcc6aefe8945f8987be398279fd98bf6b767e54b741331d3bc99a307071",
}
NLTK_FILES = {
    "corpora/cmudict/cmudict": "cad209c39eb87677d64e93d97f8eed10b7e6f9bdd42de8e7ca8efc8e17d62e8a",
    "corpora/cmudict/README": "b0556be7a2b12bea6a667277b75864f802dd4854480657ce298c62e6897e766b",
    "taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle": "25a5a19c7ced7b2bac3831da5bc0afcc2c34e5dd01cd4f361bb799949a696238",
}
G2P_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/d7/d9/"
    "b77dc634a7a0c0c97716ba97dd0a28cbfa6267c96f359c4f27ed71cbd284/"
    "g2p_en-2.1.0-py3-none-any.whl"
)
G2P_WHEEL_SHA256 = "2a7aabf1fc7f270fcc3349881407988c9245173c2413debbe5432f4a4f31319f"
NLTK_ARCHIVES = {
    "corpora/cmudict.zip": "d07cca47fd72ad32ea9d8ad1219f85301eeaf4568f8b6b73747506a71fb5afd6",
    "taggers/averaged_perceptron_tagger.zip": "e1f13cf2532daadfd6f3bc481a49859f0b8ea6432ccdcd83e6a49a5f19008de9",
}
COMMUNITY_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
COMMUNITY_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    + COMMUNITY_NAME
    + ".tar.bz2"
)
COMMUNITY_ARCHIVE_SHA256 = (
    "7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e"
)
COMMUNITY_FILES = {
    "model.int8.onnx": "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51",
    "tokens.txt": "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc",
    "LICENSE": "221c6df10b0931a5629adad671ea48fb7747e034c414b6d2bfa275bc3dd4ea17",
    "README.md": "763991a00edaea534ab36bf1b7cf89e61e911666dcfabbba71f91f9f7c593a63",
    "export-onnx.py": "c97f6a33f9d7135efd4d55b3e24e288c47d925f3b4f04b8b3418c2821c0a89ce",
}


def sha256(path: Path) -> str:
    """Hash a local input with bounded-memory reads.

    Args:
        path:
            Existing file; explicit cached symlinks are followed.

    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(directory: Path, files: dict[str, str]) -> None:
    """Require every pinned input; a corrupt cache is never repaired silently.

    Args:
        directory:
            Explicit source/data directory.

        files:
            Relative filenames mapped to their pinned SHA256 values.

    """
    for name, expected in files.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned speech input: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Speech input SHA256 mismatch: {path}: {actual}")


def publish_stream(stream: IO[bytes], destination: Path, expected: str) -> None:
    """Verify staged bytes and publish without replacing any existing path.

    Args:
        stream:
            Binary source stream; consumed using bounded-memory reads.

        destination:
            New file, or an existing byte-identical pinned file to reuse.

        expected:
            Pinned SHA256 of the complete stream.

    """
    if destination.exists() or destination.is_symlink():
        verify_files(destination.parent, {destination.name: expected})
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as staged:
        temporary = Path(staged.name)
        try:
            shutil.copyfileobj(stream, staged, length=1024 * 1024)
            staged.flush()
            if sha256(temporary) != expected:
                raise ValueError(f"Fetched speech input SHA256 mismatch: {destination}")
            # Hard-link publication is atomic and cannot overwrite a racing writer.
            os.link(temporary, destination)
        finally:
            temporary.unlink()


def fetch_file(destination: Path, url: str, expected: str, download: bool) -> None:
    """Reuse verified bytes or explicitly download one fully hash-pinned input.

    Args:
        destination:
            Cache file; corrupt and dangling existing paths are rejected.

        url:
            Revision-specific HTTPS source, never a floating model reference.

        expected:
            Pinned file SHA256.

        download:
            Permit network access; false produces an actionable prerequisite error.

    """
    if destination.exists() or destination.is_symlink():
        verify_files(destination.parent, {destination.name: expected})
        return
    if not download:
        raise FileNotFoundError(
            f"Offline speech input missing: {destination}. Supply explicit local "
            "source/resource paths or run tools.fetch_speech2 --download first."
        )
    logger.info("speech2.fetch destination=%s", destination)
    with urllib.request.urlopen(url, timeout=120) as stream:
        publish_stream(stream, destination, expected)


def extract_data(
    archive: Path, destination: Path, members: dict[str, tuple[str, str]]
) -> None:
    """Extract only named, hash-pinned regular data members; never execute code.

    Args:
        archive:
            Already verified ZIP/wheel or tar.bz2 file.

        destination:
            Data directory; existing output bytes must match the pins.

        members:
            Exact archive member names mapped to relative output name and SHA256.
            Unknown archive members are ignored, including Python source code.

    """
    for relative, _ in members.values():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe data destination: {relative}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zipped:
            for name, (relative, expected) in members.items():
                info = zipped.getinfo(name)
                if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Data archive member is not regular: {name}")
                with zipped.open(info) as stream:
                    publish_stream(stream, destination / relative, expected)
    else:
        with tarfile.open(archive, "r:bz2") as tar:
            for name, (relative, expected) in members.items():
                info = tar.getmember(name)
                if not info.isfile():
                    raise ValueError(f"Data archive member is not regular: {name}")
                stream = tar.extractfile(info)
                if stream is None:
                    raise ValueError(f"Unreadable data archive member: {name}")
                with stream:
                    publish_stream(stream, destination / relative, expected)


def select_directory(explicit: Path | None, cached: Path, target: Path) -> Path:
    """Prefer an explicit input, then benchmark cache, then the pinned input root.

    Args:
        explicit:
            User path; missing inputs here must not select another source.

        cached:
            Existing benchmark directory to reuse without copying model weights.

        target:
            Permanent pinned input destination when no benchmark directory exists.

    """
    return explicit if explicit is not None else cached if cached.exists() else target


def prepare_source(source: Path, download: bool) -> None:
    """Require an exact clean Melo git checkout, fetching only its pinned commit.

    Args:
        source:
            Existing git checkout, or new destination to publish atomically.

        download:
            Allow creating a checkout; offline never invokes a network git command.

    """
    if not source.exists():
        if not download or source.is_symlink():
            raise FileNotFoundError(f"Missing pinned Melo git checkout: {source}")
        source.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=source.parent) as temporary:
            checkout = Path(temporary) / "MeloTTS"
            subprocess.run(
                ["git", "init", str(checkout)], check=True, capture_output=True
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "fetch",
                    "--depth",
                    "1",
                    "https://github.com/myshell-ai/MeloTTS.git",
                    MELO_REVISION,
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"],
                check=True,
                capture_output=True,
            )
            checkout.rename(source)
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != MELO_REVISION:
        raise ValueError(f"Melo source revision mismatch: {source}: {revision}")
    subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--"],
        check=True,
        capture_output=True,
    )


def write_inventory(path: Path, inventory: dict[str, Any]) -> None:
    """Publish JSON exclusively, or reuse an existing identical inventory.

    Args:
        path:
            Output inventory path; an incompatible existing file is rejected.

        inventory:
            Fully validated input provenance with absolute local data paths.

    """
    if path.exists() or path.is_symlink():
        if json.loads(path.read_text(encoding="utf-8")) != inventory:
            raise FileExistsError(f"Refusing to replace speech inventory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(inventory, stream, indent=2)
        stream.write("\n")


def prepare_licenses(directory: Path, download: bool) -> None:
    """Verify pinned legal documents and local provenance without importing source.

    Existing provenance must inventory the exact pinned URLs, bytes and hashes.
    Missing provenance is generated from those verified pins, without inventing a
    retrieval date or claiming historical applicability of the model terms.

    Args:
        directory:
            Selected local legal-document directory; existing files are preserved.

        download:
            Permit fetching missing pinned texts; false is local-only validation.

    """
    records: dict[str, Any] = {}
    for name, (url, expected, size) in LICENSE_INPUTS.items():
        fetch_file(directory / name, url, expected, download)
        if (directory / name).stat().st_size != size:
            raise ValueError(f"Pinned license size mismatch: {directory / name}")
        records[name] = {"url": url, "bytes": size, "sha256": expected}
    provenance_path = directory / "provenance.json"
    if provenance_path.exists() or provenance_path.is_symlink():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict) or provenance.get("files") != records:
            raise ValueError(f"Pinned license provenance mismatch: {provenance_path}")
    else:
        write_inventory(
            provenance_path,
            {
                "pins_observed_utc": "2026-10-05",
                "verification": "Exact pinned legal text SHA256 and byte sizes; local data only",
                "files": records,
                "notes": LICENSE_NOTES,
            },
        )


def collect_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch/validate selected inputs and publish a preparer-compatible inventory.

    Args:
        args:
            Parsed CLI paths, explicit network permission and community selection.

    """
    root: Path = args.root.resolve()
    bench: Path = args.bench_cache.resolve()
    download: bool = args.download
    stt = select_directory(
        args.stt_source, bench / "models/sensevoice", root / "sensevoice"
    ).resolve()
    for name, digest in SENSEVOICE_FILES.items():
        fetch_file(
            stt / name,
            f"https://huggingface.co/{SENSEVOICE_REPOSITORY}/resolve/{SENSEVOICE_REVISION}/{name}",
            digest,
            download and args.stt_source is None,
        )
    cached_licenses = stt / "licenses"
    if not cached_licenses.exists():
        cached_licenses = bench / "licenses/sensevoice"
    licenses = select_directory(
        args.license_dir, cached_licenses, stt / "licenses"
    ).resolve()
    prepare_licenses(licenses, download and args.license_dir is None)
    source = select_directory(
        args.melo_source, bench / "source/MeloTTS", root / "MeloTTS"
    ).resolve()
    prepare_source(source, download and args.melo_source is None)
    supplied_manifest = args.assets_manifest
    cache_manifest = bench / "results/melo/assets.json"
    if supplied_manifest is None and cache_manifest.is_file():
        supplied_manifest = cache_manifest
    supplied = (
        json.loads(supplied_manifest.read_text(encoding="utf-8"))
        if supplied_manifest is not None
        else {}
    )
    assets: dict[str, Any] = {}
    for repository, (revision, files) in HF_ASSETS.items():
        if supplied_manifest is not None:
            entry = supplied[repository]
            if entry["revision"] != revision:
                raise ValueError(f"Melo/BERT inventory revision mismatch: {repository}")
            snapshot = Path(entry["snapshot"]).resolve()
            verify_files(snapshot, files)
        else:
            cached = (
                bench
                / "models/melo-hf/hub"
                / ("models--" + repository.replace("/", "--"))
                / "snapshots"
                / revision
            )
            snapshot = select_directory(
                None, cached, root / "hf" / repository / revision
            )
            for name, digest in files.items():
                fetch_file(
                    snapshot / name,
                    f"https://huggingface.co/{repository}/resolve/{revision}/{name}",
                    digest,
                    download,
                )
        assets[repository] = {
            "revision": revision,
            "snapshot": str(snapshot),
            "files": [
                {
                    "name": name,
                    "bytes": (snapshot / name).stat().st_size,
                    "sha256": digest,
                }
                for name, digest in files.items()
            ],
        }
    g2p = select_directory(
        args.g2p_resource_dir,
        bench / "envs/melo/lib/python3.10/site-packages/g2p_en",
        root / "g2p_en",
    ).resolve()
    if args.g2p_resource_dir is None and not all(
        (g2p / name).is_file() for name in G2P_FILES
    ):
        wheel = root / "archives/g2p_en-2.1.0-py3-none-any.whl"
        fetch_file(wheel, G2P_WHEEL_URL, G2P_WHEEL_SHA256, download)
        extract_data(
            wheel,
            g2p,
            {"g2p_en/" + name: (name, digest) for name, digest in G2P_FILES.items()},
        )
    verify_files(g2p, G2P_FILES)
    nltk = select_directory(
        args.nltk_resource_dir, bench / "models/melo-nltk", root / "nltk"
    ).resolve()
    if args.nltk_resource_dir is None:
        for relative, digest in NLTK_ARCHIVES.items():
            category = relative.split("/")[0]
            required = {
                name: value
                for name, value in NLTK_FILES.items()
                if name.startswith(category + "/")
            }
            if all((nltk / name).is_file() for name in required):
                verify_files(nltk, required)
                continue
            cached_archive = nltk / relative
            archive = (
                cached_archive
                if cached_archive.is_file()
                else root / "archives/nltk" / relative
            )
            fetch_file(
                archive,
                f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_REVISION}/packages/{relative}",
                digest,
                download,
            )
            extract_data(
                archive,
                nltk,
                {
                    name.removeprefix(category + "/"): (name, value)
                    for name, value in required.items()
                },
            )
    verify_files(nltk, NLTK_FILES)
    community: Path | None = None
    if args.with_community:
        community = select_directory(
            args.community_source,
            bench / "models/sensevoice-community" / COMMUNITY_NAME,
            root / "community",
        ).resolve()
        if args.community_source is None and not all(
            (community / name).is_file() for name in COMMUNITY_FILES
        ):
            cached_archive = (
                bench / "models/sensevoice-community/sherpa-int8-20240717.tar.bz2"
            )
            archive = (
                cached_archive
                if cached_archive.is_file()
                else root / "archives/community.tar.bz2"
            )
            fetch_file(archive, COMMUNITY_URL, COMMUNITY_ARCHIVE_SHA256, download)
            extract_data(
                archive,
                community,
                {
                    COMMUNITY_NAME + "/" + name: (name, digest)
                    for name, digest in COMMUNITY_FILES.items()
                },
            )
        verify_files(community, COMMUNITY_FILES)
    asset_inventory = args.output.resolve().with_suffix(".melo-assets.json")
    write_inventory(asset_inventory, assets)
    result = {
        "version": 1,
        "stt_source": str(stt),
        "melo_source": str(source),
        "assets_manifest": str(asset_inventory),
        "g2p_resource_dir": str(g2p),
        "nltk_resource_dir": str(nltk),
        "community_source": str(community) if community is not None else None,
        "license_dir": str(licenses),
        "pins": {
            "sensevoice": SENSEVOICE_REVISION,
            "melo": MELO_REVISION,
            "nltk": NLTK_REVISION,
            "g2p_en": "2.1.0",
        },
    }
    write_inventory(args.output, result)
    return result


def main() -> None:
    """Parse explicit input policy and retain complete failures in a unique log."""
    parser = argparse.ArgumentParser(description=__doc__)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument(
        "--offline",
        action="store_true",
        help="No speech input network access (default)",
    )
    policy.add_argument(
        "--download", action="store_true", help="Fetch only the pinned missing inputs"
    )
    parser.add_argument("--root", type=Path, default=Path("models/speech2-inputs"))
    parser.add_argument("--bench-cache", type=Path, default=Path(".cache/speech-bench"))
    parser.add_argument(
        "--output", type=Path, default=Path("models/speech2-inputs/inputs.json")
    )
    for option in (
        "stt-source",
        "melo-source",
        "assets-manifest",
        "g2p-resource-dir",
        "nltk-resource-dir",
        "community-source",
        "license-dir",
    ):
        parser.add_argument("--" + option, type=Path)
    parser.add_argument("--with-community", action="store_true")
    args = parser.parse_args()
    diagnostics = Path(".cache/hoast/diagnostics/fetch-speech2")
    diagnostics.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="run-", suffix=".log", dir=diagnostics, delete=False
    ) as stream:
        log_path = Path(stream.name).resolve()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logger.info(
        "speech2.fetch diagnostics=%s policy=%s inputs=%s",
        log_path,
        "download" if args.download else "offline",
        vars(args),
    )
    try:
        if sys.version_info[:2] != (3, 14):
            raise RuntimeError("Speech input preparation requires native Python 3.14")
        if args.community_source is not None and not args.with_community:
            raise ValueError("--community-source requires --with-community")
        collect_inputs(args)
    except Exception as error:
        logger.exception("speech2.fetch status=error diagnostics=%s", log_path)
        if isinstance(error, subprocess.CalledProcessError):
            logger.error("git stdout=%r stderr=%r", error.stdout, error.stderr)
        raise
    logger.info("speech2.fetch status=ok inventory=%s", args.output)


if __name__ == "__main__":
    main()
