"""Offline model publication and disposable-cache separation regressions."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import huggingface_hub
import pytest

from hoast.lfm import LFMConfig
from hoast.llm import LLMConfig
from tools import fetch_speech2, prepare_llm, prepare_tts2


@pytest.mark.parametrize("custom_root", [False, True])
def test_factories_select_permanent_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom_root: bool
) -> None:
    """Select models independently of cache contents, preserving factory names.

    Args:
        tmp_path:
            Isolated model and disposable cache directories.

        monkeypatch:
            Sets the working directory for default relative model roots.

        custom_root:
            Whether to supply an explicit permanent root instead of models.

    """
    monkeypatch.chdir(tmp_path)
    root = Path("selected-models" if custom_root else "models")
    lfm = root / "lfm/model.json"
    gemma = root / "functiongemma/int8.json"
    lfm.parent.mkdir(parents=True)
    gemma.parent.mkdir(parents=True)
    lfm.write_text(json.dumps({"model_id": prepare_llm.LFM_ID, "path": "lfm-export"}))
    gemma.write_text(json.dumps({"path": "gemma-export"}))
    cache = Path("disposable")
    cache.mkdir()
    (cache / "int8.json").write_text("invalid cache selection")
    if custom_root:
        lfm_config = LFMConfig.from_cache(cache, model_root=root)
        gemma_config = LLMConfig.from_cache(cache, model_root=root)
    else:
        lfm_config = LFMConfig.from_cache(cache)
        gemma_config = LLMConfig.from_cache(cache)
    assert lfm_config.model_path == Path("lfm-export")
    assert gemma_config.model_path == Path("gemma-export")
    assert lfm_config.cache_dir == gemma_config.cache_dir == cache
    assert lfm_config.threads == 1 and lfm_config.device == "GPU"
    shutil.rmtree(cache)
    assert LFMConfig.from_cache(cache, model_root=root) == lfm_config
    assert LLMConfig.from_cache(cache, model_root=root) == gemma_config


@pytest.mark.parametrize("model", ["lfm", "functiongemma"])
def test_download_publishes_cache_independent_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: prepare_llm.Model
) -> None:
    """Dereference HF snapshot symlinks before publishing permanent selections.

    Args:
        tmp_path:
            Local tiny snapshot fixture; no network or model weights are required.

        monkeypatch:
            Replaces Hugging Face API and snapshot acquisition.

        model:
            Public LFM export or FunctionGemma source checkpoint family.

    """
    cache = tmp_path / "cache"
    snapshot = cache / "hub/snapshot"
    snapshot.mkdir(parents=True)
    blob = cache / "blob"
    blob.write_bytes(b"fixture weights")
    (snapshot / "weights.bin").symlink_to(blob)
    (snapshot / "config.json").write_text("{}")
    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(sha="immutable-sha")
    monkeypatch.setattr(huggingface_hub, "HfApi", MagicMock(return_value=api))
    acquisition = MagicMock(return_value=str(snapshot))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", acquisition)
    root = tmp_path / "models"
    path = prepare_llm.download(cache, model, None, root)
    category = "checkpoints" if model == "functiongemma" else "exports"
    assert path == root / model / category / "immutable-sha"
    assert acquisition.call_args.kwargs["cache_dir"] == cache / "hub"
    assert not (path / "weights.bin").is_symlink()
    manifest = json.loads((root / model / "model.json").read_text())
    assert Path(manifest["path"]) == path
    shutil.rmtree(cache)
    assert (path / "weights.bin").read_bytes() == b"fixture weights"
    assert prepare_llm.publish_snapshot(path, path) == path
    if model == "lfm":
        assert LFMConfig.from_cache(cache, model_root=root).model_path == path
    else:
        assert prepare_llm.checkpoint(root)[0] == path


def test_export_keeps_selection_and_ir_outside_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persist FunctionGemma IR and precision selection but cache compilation only.

    Args:
        tmp_path:
            Small local checkpoint and separate artifact/cache roots.

        monkeypatch:
            Replaces native export, dependency version discovery and compilation.

    """
    root = tmp_path / "models"
    source = root / "functiongemma/checkpoints/source"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    prepare_llm.write_json(
        root / "functiongemma/model.json",
        {"model_id": prepare_llm.FUNCTIONGEMMA_ID, "path": str(source)},
    )
    cache = tmp_path / "cache"
    monkeypatch.setattr(prepare_llm, "versions", lambda: {"fixture": "1"})
    export = MagicMock()
    compile_model = MagicMock()
    monkeypatch.setattr(prepare_llm.subprocess, "run", export)
    monkeypatch.setattr(prepare_llm, "compile_model", compile_model)
    output = prepare_llm.export_model(cache, "int8", 2, "CPU", root)
    assert output.parent == root / "functiongemma/exports"
    assert (output / "hoast-manifest.json").is_file()
    assert not (cache / "int8.json").exists()
    compile_model.assert_called_once_with(cache, "functiongemma", output, 2, "CPU")
    export.assert_called_once()
    shutil.rmtree(cache)
    assert LLMConfig.from_cache(cache, model_root=root).model_path == output


def test_fetch_default_inventory_is_permanent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep pinned speech inputs and their inventories in the permanent model root.

    Args:
        tmp_path:
            Empty working directory for diagnostic creation.

        monkeypatch:
            Replaces the fetch boundary without touching network or source data.

    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["fetch_speech2"])
    collect = MagicMock()
    monkeypatch.setattr(fetch_speech2, "collect_inputs", collect)
    fetch_speech2.main()
    args = collect.call_args.args[0]
    assert args.root == Path("models/speech2-inputs")
    assert args.output == Path("models/speech2-inputs/inputs.json")


def test_melo_preparation_defaults_are_permanent() -> None:
    """Use pinned source and frontend resources that survive disposable cache removal."""
    root = Path("models/speech2-inputs")
    assert prepare_tts2.DEFAULT_SOURCE == root / "MeloTTS"
    assert prepare_tts2.DEFAULT_ASSETS == root / "inputs.melo-assets.json"
    assert prepare_tts2.DEFAULT_G2P_RESOURCE_DIR == root / "g2p_en"
    assert prepare_tts2.DEFAULT_NLTK_RESOURCE_DIR == root / "nltk"
