"""Offline, synthetic-byte fixtures for pinned VAD acquisition and provenance."""

import hashlib
import io
import json
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import URLError

import pytest

from hoast.endpoint import DEFAULT_VAD_MODEL
from tools import prepare_vad


@pytest.fixture
def asset(monkeypatch: pytest.MonkeyPatch) -> bytes:
    """Use tiny fixture bytes with their digest in place of a machine-local ONNX file.

    Args:
        monkeypatch:
            Temporarily replace only the pinned digest; preserve URL/provenance logic.

    """
    content = b"synthetic Silero fixture, not a parseable ONNX model"
    monkeypatch.setattr(
        prepare_vad, "MODEL_SHA256", hashlib.sha256(content).hexdigest()
    )
    return content


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Forbid actual network access in every acquisition test unless a fixture overrides it.

    Args:
        monkeypatch:
            Restore the urllib request function after each test.

    """
    request = MagicMock(side_effect=AssertionError("Tests must not access the network"))
    monkeypatch.setattr(prepare_vad.urllib.request, "urlopen", request)
    return request


def test_trusted_pin_and_runtime_default() -> None:
    """Keep the verified immutable source and bootstrap/runtime destination consistent."""
    assert prepare_vad.MODEL_SHA256 == (
        "4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2"
    )
    assert prepare_vad.MODEL_URL == (
        "https://raw.githubusercontent.com/SYSTRAN/faster-whisper/"
        "65882eee9f5cdbeeb2d877f1131d48cf241b327d/"
        "faster_whisper/assets/silero_vad_v6.onnx"
    )
    assert prepare_vad.DEFAULT_MODEL == DEFAULT_VAD_MODEL


@pytest.mark.parametrize("offline", [False, True])
def test_reuse_validated_output(
    tmp_path: Path, asset: bytes, network: MagicMock, offline: bool
) -> None:
    """Reuse a hash-valid local asset and materialize licensing without contacting upstream.

    Args:
        tmp_path:
            Isolated output directory.

        asset:
            Synthetic bytes whose digest replaces the pinned model digest.

        network:
            Mock proving that no request is made.

        offline:
            Reuse must be network-free in either preparation mode.

    """
    model = tmp_path / "vad.onnx"
    model.write_bytes(asset)
    assert prepare_vad.prepare_model(model, offline=offline) == model
    assert model.read_bytes() == asset
    network.assert_not_called()
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["sha256"] == hashlib.sha256(asset).hexdigest()
    assert provenance["size_bytes"] == len(asset)
    assert provenance["source_url"] == prepare_vad.MODEL_URL
    assert provenance["source_revision"] == prepare_vad.REVISION
    assert provenance["silero_license_url"] == prepare_vad.SILERO_LICENSE_URL
    assert provenance["interface"]["recurrent_shape"] == [1, 1, 128]
    assert (tmp_path / "LICENSE.silero").read_text() == prepare_vad.SILERO_LICENSE
    assert "Copyright (c) 2020-present Silero Team" in prepare_vad.SILERO_LICENSE
    assert "THE SOFTWARE IS PROVIDED" in prepare_vad.SILERO_LICENSE
    assert (
        tmp_path / "LICENSE.faster-whisper"
    ).read_text() == prepare_vad.SYSTRAN_LICENSE


@pytest.mark.parametrize("offline", [False, True])
def test_reject_corrupted_output(
    tmp_path: Path, asset: bytes, network: MagicMock, offline: bool
) -> None:
    """Fail on changed existing bytes without replacing them, even when online is allowed.

    Args:
        tmp_path:
            Isolated output directory.

        asset:
            Synthetic expected model bytes.

        network:
            Mock proving a mismatch is not repaired by a download.

        offline:
            Both online and offline calls must reject corrupted output.

    """
    model = tmp_path / "vad.onnx"
    model.write_bytes(asset + b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare_vad.prepare_model(model, offline=offline)
    assert model.read_bytes() == asset + b"corrupt"
    assert not (tmp_path / "provenance.json").exists()
    network.assert_not_called()


def test_copy_local_source_offline(
    tmp_path: Path, asset: bytes, network: MagicMock
) -> None:
    """Validate a local source and publish its bytes plus license/provenance offline.

    Args:
        tmp_path:
            Isolated source and output directory.

        asset:
            Synthetic model bytes.

        network:
            Mock proving that local-source preparation never requests a download.

    """
    source = tmp_path / "source.onnx"
    source.write_bytes(asset)
    model = tmp_path / "output" / "vad.onnx"
    assert prepare_vad.prepare_model(model, source=source, offline=True) == model
    assert source.read_bytes() == model.read_bytes() == asset
    assert (model.parent / "LICENSE.silero").is_file()
    assert sorted(path.name for path in model.parent.iterdir()) == [
        "LICENSE.faster-whisper",
        "LICENSE.silero",
        "provenance.json",
        "vad.onnx",
    ]
    network.assert_not_called()


def test_reject_corrupted_source(
    tmp_path: Path, asset: bytes, network: MagicMock
) -> None:
    """Fail before creating output when a supplied source has a mismatched digest.

    Args:
        tmp_path:
            Isolated source/output directory.

        asset:
            Synthetic expected model bytes.

        network:
            Mock proving invalid sources cannot trigger a fallback download.

    """
    source = tmp_path / "source.onnx"
    source.write_bytes(asset + b"corrupt")
    model = tmp_path / "output" / "vad.onnx"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare_vad.prepare_model(model, source=source)
    assert not model.parent.exists()
    network.assert_not_called()


def test_missing_asset_offline(tmp_path: Path, network: MagicMock) -> None:
    """Report an absent offline asset without creating partial files or using the network.

    Args:
        tmp_path:
            Isolated output directory.

        network:
            Mock proving offline failure never requests a download.

    """
    model = tmp_path / "output" / "vad.onnx"
    with pytest.raises(FileNotFoundError, match="unavailable offline"):
        prepare_vad.prepare_model(model, offline=True)
    assert not model.parent.exists()
    network.assert_not_called()


def test_download_pinned_asset(
    tmp_path: Path, asset: bytes, network: MagicMock
) -> None:
    """Use the immutable trusted URL, validate downloaded fixture bytes, then publish them.

    Args:
        tmp_path:
            Isolated output directory.

        asset:
            Synthetic bytes returned by the mocked upstream response.

        network:
            urllib mock configured with a local in-memory response.

    """
    network.side_effect = None
    network.return_value = io.BytesIO(asset)
    model = tmp_path / "vad.onnx"
    prepare_vad.prepare_model(model)
    assert model.read_bytes() == asset
    network.assert_called_once_with(prepare_vad.MODEL_URL, timeout=60)


@pytest.mark.parametrize("failure", ["digest", "network"])
def test_download_failures_do_not_publish(
    tmp_path: Path, asset: bytes, network: MagicMock, failure: str
) -> None:
    """Keep failed acquisition loud and leave neither model nor staging/provenance files.

    Args:
        tmp_path:
            Isolated output directory.

        asset:
            Synthetic expected model bytes.

        network:
            urllib mock producing corruption or an explicit network exception.

        failure:
            The external acquisition failure to simulate.

    """
    model = tmp_path / "vad.onnx"
    if failure == "digest":
        network.side_effect = None
        network.return_value = io.BytesIO(asset + b"corrupt")
        expected, message = ValueError, "SHA256 mismatch"
    else:
        network.side_effect = URLError("fixture failure")
        expected, message = URLError, "fixture failure"
    with pytest.raises(expected, match=message):
        prepare_vad.prepare_model(model)
    assert not model.exists()
    assert list(tmp_path.iterdir()) == []
