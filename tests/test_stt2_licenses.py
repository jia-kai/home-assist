"""Offline legal-document bundling tests; no model loading, exports or inference."""

import json
from pathlib import Path
from typing import Any

import pytest

from hoast.stt2.artifacts import file_record
from tools.stt2_provenance import bundle_provenance, validate_license_inputs


def _documents(root: Path) -> Path:
    """Create local verbatim legal/provenance fixtures with reference-only notices.

    Args:
        root:
            Empty fixture directory to create.

    """
    root.mkdir()
    (root / "README.md").write_text("license: other\nlicense_name: model-license\n")
    licenses = root / "licenses"
    licenses.mkdir()
    records: dict[str, Any] = {}
    for name in ("MODEL_LICENSE", "FunASR-LICENSE", "SenseVoice-LICENSE"):
        content = (
            b"Ref to https://example.invalid/license\n"
            if name == "SenseVoice-LICENSE"
            else b"Verbatim upstream fixture text\n"
        )
        (licenses / name).write_bytes(content)
        records[name] = {
            **file_record(licenses / name),
            "url": f"https://example.invalid/pinned/{name}",
        }
    (licenses / "provenance.json").write_text(json.dumps({"files": records}))
    return root


def test_license_bundle_and_inventory(tmp_path: Path) -> None:
    """Preserve legal bytes, hash every output and avoid inventing resource terms.

    Args:
        tmp_path:
            Isolated official/community/staging fixture directories.

    """
    official = _documents(tmp_path / "official")
    staging = tmp_path / "staging"
    staging.mkdir()
    community = tmp_path / "community"
    community.mkdir()
    for name in ("LICENSE", "README.md", "export-onnx.py"):
        (community / name).write_bytes(b"Ref to upstream; not a license grant\n")
    result = bundle_provenance(official, staging, community=community)
    for name, record in result["files"].items():
        assert file_record(staging / name) == record
    assert (staging / "licenses/community-LICENSE").read_bytes() == (
        community / "LICENSE"
    ).read_bytes()
    assert "No separate tokenizer/CMVN/config" in result["resource_license"]
    assert "not a standalone license grant" in result["community_license"]
    assert "custom" in result["model_license"]


def test_license_inputs_missing_or_changed(tmp_path: Path) -> None:
    """Reject missing full terms or changed license bytes before heavy preparation.

    Args:
        tmp_path:
            Isolated model-card and legal-document fixture directories.

    """
    root = tmp_path / "missing"
    root.mkdir()
    (root / "README.md").write_text("license: other\n")
    with pytest.raises(FileNotFoundError, match="license/provenance"):
        validate_license_inputs(root)
    official = _documents(tmp_path / "official")
    (official / "licenses/MODEL_LICENSE").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest/size mismatch"):
        validate_license_inputs(official)
