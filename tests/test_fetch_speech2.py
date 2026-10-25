"""Offline fixture checks for pinned speech input acquisition and data extraction."""

import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools import fetch_speech2 as fetch


def digest(data: bytes) -> str:
    """Return the fixture's expected SHA256.

    Args:
        data:
            Tiny fixture content, not a model or external corpus.

    """
    return hashlib.sha256(data).hexdigest()


def test_offline_missing_never_connects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing offline inputs fail without invoking HTTP.

    Args:
        tmp_path:
            Empty fixture workspace.

        monkeypatch:
            Test-scoped network replacement.

    """

    def forbidden(*args: object, **kwargs: object) -> None:
        """Reject any attempted HTTP call.

        Args:
            args:
                Unexpected positional network arguments.

            kwargs:
                Unexpected keyword network arguments.

        """
        pytest.fail("Offline fetching attempted HTTP")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", forbidden)
    with pytest.raises(FileNotFoundError, match="--download"):
        fetch.fetch_file(
            tmp_path / "missing", "https://invalid.test/pinned", digest(b"x"), False
        )


def test_cache_mismatch_never_replaced(tmp_path: Path) -> None:
    """A corrupt cache remains untouched, even with network permission.

    Args:
        tmp_path:
            Tiny local cache fixture.

    """
    path = tmp_path / "cached"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetch.fetch_file(
            path, "https://invalid.test/pinned", digest(b"replacement"), True
        )
    assert path.read_bytes() == b"original"
    fetch.fetch_file(path, "https://invalid.test/pinned", digest(b"original"), False)


def test_bad_stream_leaves_no_artifact(tmp_path: Path) -> None:
    """Checksum failure removes staging bytes without publishing a partial file.

    Args:
        tmp_path:
            New artifact destination.

    """
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetch.publish_stream(io.BytesIO(b"bad"), tmp_path / "artifact", digest(b"good"))
    assert list(tmp_path.iterdir()) == []


def test_wheel_extracts_only_pinned_data(tmp_path: Path) -> None:
    """Wheel extraction ignores Python code and archive traversal entries.

    Args:
        tmp_path:
            Local fake wheel and resource root.

    """
    archive = tmp_path / "fake.whl"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("g2p_en/checkpoint20.npz", b"weights")
        zipped.writestr("g2p_en/g2p.py", "raise RuntimeError('must not execute')")
        zipped.writestr("../../escape", b"unselected")
    data = tmp_path / "data"
    members = {"g2p_en/checkpoint20.npz": ("checkpoint20.npz", digest(b"weights"))}
    fetch.extract_data(archive, data, members)
    fetch.extract_data(archive, data, members)
    assert list(data.iterdir()) == [data / "checkpoint20.npz"]
    assert (data / "checkpoint20.npz").read_bytes() == b"weights"
    with pytest.raises(ValueError, match="Unsafe data destination"):
        fetch.extract_data(
            archive,
            data,
            {"g2p_en/checkpoint20.npz": ("../escape", digest(b"weights"))},
        )


def test_tar_links_are_rejected(tmp_path: Path) -> None:
    """Whitelisted names still cannot introduce a tar symlink.

    Args:
        tmp_path:
            Tiny fake community archive.

    """
    archive = tmp_path / "community.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        info = tarfile.TarInfo("model")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(ValueError, match="not regular"):
        fetch.extract_data(
            archive, tmp_path / "data", {"model": ("model", digest(b"x"))}
        )
    assert not (tmp_path / "data").exists()


def test_inventory_refuses_replacement(tmp_path: Path) -> None:
    """An identical inventory is reusable, but different provenance is rejected.

    Args:
        tmp_path:
            Fixture inventory destination.

    """
    path = tmp_path / "inputs.json"
    fetch.write_inventory(path, {"version": 1})
    fetch.write_inventory(path, {"version": 1})
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        fetch.write_inventory(path, {"version": 2})
    assert json.loads(path.read_text()) == {"version": 1}


def test_local_inventory_is_preparer_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Minimal local fixtures exercise inventory publication without any model data.

    Args:
        tmp_path:
            Tiny fake source/resource snapshots.

        monkeypatch:
            Replace audited file pins and git validation with isolated fixtures.

    """
    expected = digest(b"fixture")
    pins = {"file": expected}
    monkeypatch.setattr(fetch, "SENSEVOICE_FILES", pins)
    monkeypatch.setattr(fetch, "G2P_FILES", pins)
    monkeypatch.setattr(fetch, "NLTK_FILES", pins)
    monkeypatch.setattr(fetch, "COMMUNITY_FILES", pins)
    monkeypatch.setattr(fetch, "HF_ASSETS", {"fixture/repo": ("a" * 40, pins)})
    monkeypatch.setattr(
        fetch,
        "LICENSE_INPUTS",
        {"MODEL_LICENSE": ("https://invalid.test/pin", expected, 7)},
    )
    monkeypatch.setattr(fetch, "prepare_source", lambda source, download: None)
    for name in ("stt", "melo", "g2p", "nltk", "community", "hf", "licenses"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "file").write_bytes(b"fixture")
    (tmp_path / "licenses/MODEL_LICENSE").write_bytes(b"fixture")
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {"fixture/repo": {"revision": "a" * 40, "snapshot": str(tmp_path / "hf")}}
        )
    )
    args = argparse.Namespace(
        root=tmp_path / "cache",
        bench_cache=tmp_path / "bench",
        download=False,
        stt_source=tmp_path / "stt",
        melo_source=tmp_path / "melo",
        assets_manifest=manifest,
        g2p_resource_dir=tmp_path / "g2p",
        nltk_resource_dir=tmp_path / "nltk",
        community_source=tmp_path / "community",
        with_community=True,
        output=tmp_path / "inputs.json",
        license_dir=tmp_path / "licenses",
    )
    result = fetch.collect_inputs(args)
    inventory = json.loads(Path(result["assets_manifest"]).read_text())
    assert inventory["fixture/repo"]["files"] == [
        {"name": "file", "bytes": 7, "sha256": expected}
    ]
    assert result["g2p_resource_dir"] == str(tmp_path / "g2p")
    assert result["community_source"] == str(tmp_path / "community")
    assert result["license_dir"] == str(tmp_path / "licenses")
    assert json.loads(args.output.read_text()) == result


def test_legal_provenance_is_verified_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local legal texts create verifiable records and reject altered provenance.

    Args:
        tmp_path:
            Tiny offline legal-document fixture directory.

        monkeypatch:
            Replace upstream pins with in-memory test-document digests.

    """
    names = ("MODEL_LICENSE", "FunASR-LICENSE", "SenseVoice-LICENSE")
    pins = {}
    for name in names:
        data = name.encode()
        (tmp_path / name).write_bytes(data)
        pins[name] = ("https://invalid.test/" + name, digest(data), len(data))
    monkeypatch.setattr(fetch, "LICENSE_INPUTS", pins)
    fetch.prepare_licenses(tmp_path, False)
    fetch.prepare_licenses(tmp_path, False)
    provenance = tmp_path / "provenance.json"
    value = json.loads(provenance.read_text())
    assert value["files"]["MODEL_LICENSE"]["sha256"] == digest(b"MODEL_LICENSE")
    value["files"]["MODEL_LICENSE"]["url"] = "https://invalid.test/mutable"
    provenance.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="provenance mismatch"):
        fetch.prepare_licenses(tmp_path, False)
    assert json.loads(provenance.read_text()) == value


def test_legal_fetch_uses_verified_pinned_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mocked HTTP supplies legal text, with pins checked before publication.

    Args:
        tmp_path:
            Empty fixture legal-document destination.

        monkeypatch:
            Replace upstream pins and HTTP with tiny in-memory fixtures.

    """
    data = b"fixture legal text"
    url = "https://invalid.test/full-commit/LICENSE"
    monkeypatch.setattr(
        fetch, "LICENSE_INPUTS", {"MODEL_LICENSE": (url, digest(data), len(data))}
    )
    monkeypatch.setattr(
        fetch.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(data)
    )
    fetch.prepare_licenses(tmp_path, True)
    value = json.loads((tmp_path / "provenance.json").read_text())
    assert (tmp_path / "MODEL_LICENSE").read_bytes() == data
    assert value["files"]["MODEL_LICENSE"] == {
        "url": url,
        "bytes": len(data),
        "sha256": digest(data),
    }


def test_missing_legal_text_is_an_offline_prerequisite(tmp_path: Path) -> None:
    """No network is attempted to fill absent offline legal inputs.

    Args:
        tmp_path:
            Empty fixture legal-directory destination.

    """
    with pytest.raises(FileNotFoundError, match="Offline speech input missing"):
        fetch.prepare_licenses(tmp_path, False)
    assert not (tmp_path / "provenance.json").exists()


def test_pins_are_full_hashes() -> None:
    """All downloadable files and snapshots have immutable complete pins."""
    for revision in (
        fetch.SENSEVOICE_REVISION,
        fetch.MELO_REVISION,
        fetch.NLTK_REVISION,
    ):
        assert len(revision) == 40
        assert int(revision, 16) >= 0
    for files in (
        fetch.SENSEVOICE_FILES,
        fetch.G2P_FILES,
        fetch.NLTK_FILES,
        fetch.COMMUNITY_FILES,
        fetch.NLTK_ARCHIVES,
    ):
        for expected in files.values():
            assert len(expected) == 64
            assert int(expected, 16) >= 0
    for revision, files in fetch.HF_ASSETS.values():
        assert len(revision) == 40
        for expected in files.values():
            assert len(expected) == 64
            assert int(expected, 16) >= 0
    for url, expected, size in fetch.LICENSE_INPUTS.values():
        assert url.startswith("https://raw.githubusercontent.com/")
        assert len(expected) == 64
        assert int(expected, 16) >= 0
        assert size > 0
