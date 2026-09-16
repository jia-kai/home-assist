"""Load exported fast tokenizers offline through the application's LFM runtime."""

import json
from pathlib import Path
from typing import Any

import openvino_genai
import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from hoast.lfm import LFM2, LFMConfig
from hoast.llm import ToolRegistry


@pytest.mark.parametrize(
    ("tokenizer_class", "backend_name", "valid"),
    [
        ("PreTrainedTokenizerFast", "tokenizers", True),
        ("PreTrainedTokenizerFast", None, True),
        ("PreTrainedTokenizerFast", "unsupported", False),
        ("UnsupportedTokenizer", "tokenizers", False),
    ],
)
def test_exported_tokenizer_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_class: str,
    backend_name: str | None,
    valid: bool,
) -> None:
    """Preserve saved vocabulary/template and reject unknown metadata before compilation.

    Args:
        tmp_path:
            Temporary export directory containing an embedded tokenizer fixture.

        monkeypatch:
            Replace model compilation to avoid external checkpoints and devices.

        tokenizer_class:
            Declared tokenizer implementation in the exported configuration.

        backend_name:
            Optional backend metadata; generic fast tokenizers may omit this field.

        valid:
            Whether the declared class/backend pair should load successfully.

    """
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    tokenizer.save_pretrained(tmp_path)
    metadata_path = tmp_path / "tokenizer_config.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["tokenizer_class"] = tokenizer_class
    if backend_name is not None:
        metadata["backend"] = backend_name
    metadata_path.write_text(json.dumps(metadata))
    (tmp_path / "openvino_model.xml").touch()
    serialized = (tmp_path / "tokenizer.json").read_bytes()
    attempts: list[str] = []

    def pipeline(path: str, device: str, **properties: Any) -> object:
        """Capture compilation without loading a graph or accessing a device.

        Args:
            path:
                Absolute local model directory.

            device:
                Explicit OpenVINO device requested by the application.

            properties:
                Runtime compilation and cache settings.

        """
        assert Path(path) == tmp_path.resolve()
        assert properties["INFERENCE_NUM_THREADS"] == 2
        attempts.append(device)
        return object()

    monkeypatch.setattr(openvino_genai, "LLMPipeline", pipeline)
    model = LFM2(
        LFMConfig(tmp_path, cache_dir=tmp_path, device="CPU", threads=2),
        ToolRegistry([]),
    )
    if valid:
        with model:
            assert model._tokenizer.get_vocab() == tokenizer.get_vocab()
            assert model._tokenizer.chat_template == tokenizer.chat_template
            assert model._tokenizer.encode("hello", add_special_tokens=False) == [1]
        assert attempts == ["CPU"]
    else:
        with pytest.raises(ValueError, match="Unexpected LFM tokenizer backend"):
            model.load()
        assert not attempts
    assert model._model is None and model._tokenizer is None
    assert (tmp_path / "tokenizer.json").read_bytes() == serialized
