"""Offline bundling of verbatim STT model/source license and provenance documents."""

import json
import shutil
from pathlib import Path
from typing import Any

from hoast.stt2.artifacts import file_record

LICENSE_NAMES = (
    "MODEL_LICENSE",
    "FunASR-LICENSE",
    "SenseVoice-LICENSE",
    "provenance.json",
)


def validate_license_inputs(official: Path, license_dir: Path | None = None) -> Path:
    """Require local license texts and their provenance before heavy preparation.

    Args:
        official:
            Official snapshot containing its unmodified README.md model card.

        license_dir:
            Offline directory containing LICENSE_NAMES; None selects
            official/licenses. Preparers never fetch license documents.

    """
    directory = official / "licenses" if license_dir is None else license_dir
    for path in (official / "README.md", *(directory / name for name in LICENSE_NAMES)):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing STT2 license/provenance document: {path}; supply --license-dir"
            )
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise TypeError("License provenance must be a JSON mapping")
    records = provenance.get("files")
    if not isinstance(records, dict):
        raise TypeError("License provenance must inventory upstream files")
    for name in LICENSE_NAMES[:-1]:
        record = records.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("url"), str):
            raise TypeError(f"Missing license provenance URL: {name}")
        actual = file_record(directory / name)
        if any(record.get(key) != value for key, value in actual.items()):
            raise ValueError(f"License provenance digest/size mismatch: {name}")
    return directory


def bundle_provenance(
    official: Path,
    staging: Path,
    license_dir: Path | None = None,
    community: Path | None = None,
) -> dict[str, Any]:
    """Copy verbatim legal/provenance files and inventory them separately from weights.

    No license is inferred for tokenizer/CMVN/config. The official model card and
    custom MODEL_LICENSE are distinct from source-code terms. A community LICENSE
    URL reference is preserved as a reference, never presented as a license grant.

    Args:
        official:
            Official resource snapshot containing README.md.

        staging:
            New unpublished artifact directory; creates its licenses subdirectory.

        license_dir:
            Verified local upstream legal texts and fetch provenance.

        community:
            Optional extracted community archive with LICENSE/README.md/export-onnx.py.

    """
    directory = validate_license_inputs(official, license_dir)
    target = staging / "licenses"
    target.mkdir()
    sources = [(directory / name, name) for name in LICENSE_NAMES]
    sources.append((official / "README.md", "official-model-card.md"))
    sources.append(
        (
            Path(__file__).resolve().parents[1] / "hoast/stt2/LICENSES.md",
            "native-source-LICENSES.md",
        )
    )
    if community is not None:
        for name in ("LICENSE", "README.md", "export-onnx.py"):
            sources.append((community / name, f"community-{name}"))
    for source, name in sources:
        shutil.copyfile(source, target / name)
    return {
        "model_name": "SenseVoiceSmall",
        "model_license": "custom; see official-model-card.md and MODEL_LICENSE",
        "resource_license": "No separate tokenizer/CMVN/config license declaration was found; consult upstream before redistribution",
        "community_license": "community-LICENSE is a URL reference, not a standalone license grant"
        if community is not None
        else None,
        "files": {
            f"licenses/{name}": file_record(target / name) for _, name in sources
        },
        "upstream": json.loads(
            (directory / "provenance.json").read_text(encoding="utf-8")
        ),
    }
