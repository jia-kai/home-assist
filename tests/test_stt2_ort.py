"""Mocked native ORT/importer contracts; no models, export or device execution."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from hoast import stt2
from hoast.stt2.artifacts import (
    ARCHITECTURE_SPEC,
    ARTIFACT_NAMES,
    FORMAT,
    FRONTEND_SPEC,
    ONNX_ARTIFACT_NAMES,
    file_record,
    validate_artifacts,
)
from hoast.stt2.ort_engine import ORTEngine
from tools import benchmark_speech2, prepare_stt2_community


def _artifacts(directory: Path, backend: str = "onnxruntime") -> Path:
    """Write fake hashed artifacts with the selected backend's exact file set.

    Args:
        directory:
            Isolated fixture output path.

        backend:
            onnxruntime or openvino manifest backend.

    """
    directory.mkdir()
    names = ONNX_ARTIFACT_NAMES if backend == "onnxruntime" else ARTIFACT_NAMES
    for name in names:
        (directory / name).write_bytes(b"fixture")
    manifest = {
        "format": FORMAT,
        "frontend": FRONTEND_SPEC,
        "architecture": ARCHITECTURE_SPEC,
        "backend": backend,
        "files": {name: file_record(directory / name) for name in names},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture
def mock_ort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Replace native session/options creation with inspectable pure Python mocks.

    Args:
        monkeypatch:
            Scoped session factory and current-directory replacement helper.

        tmp_path:
            Isolated diagnostic output directory.

    """
    monkeypatch.chdir(tmp_path)
    options = SimpleNamespace(add_session_config_entry=Mock())
    session = Mock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    session.get_inputs.return_value = [
        SimpleNamespace(name="speech", type="tensor(float)", shape=[1, "T", 560]),
        SimpleNamespace(name="language", type="tensor(int64)", shape=[1]),
        SimpleNamespace(name="textnorm", type="tensor(int64)", shape=[1]),
    ]
    session.get_outputs.return_value = [
        SimpleNamespace(name="frame_ids", type="tensor(int64)", shape=[1, "T4"])
    ]
    session.run.return_value = [np.arange(6, dtype=np.int64)[None, :]]
    profile = tmp_path / "trace.json"
    profile.write_text(
        json.dumps(
            [
                {"cat": "Session", "name": "model_run", "dur": 9999},
                {
                    "cat": "Node",
                    "name": "quant_kernel_time",
                    "dur": 1250,
                    "args": {
                        "op_name": "MatMulInteger",
                        "provider": "CPUExecutionProvider",
                    },
                },
            ]
        )
    )
    session.end_profiling.return_value = str(profile)
    factory = Mock(return_value=session)
    monkeypatch.setattr(
        "hoast.stt2.ort_engine.ort.SessionOptions", Mock(return_value=options)
    )
    monkeypatch.setattr("hoast.stt2.ort_engine.ort.InferenceSession", factory)
    return SimpleNamespace(
        session=session, options=options, factory=factory, profile=profile
    )


@pytest.mark.parametrize("threads", [1, 2])
def test_ort_cpu_policy_and_owned_ids(mock_ort: SimpleNamespace, threads: int) -> None:
    """Use CPU exclusively, disable spinning and copy frame IDs without profiling.

    Args:
        mock_ort:
            Pure Python session/options fixture.

        threads:
            One or two allowed intra-operation workers.

    """
    engine = ORTEngine(Path("model.onnx"), threads)
    options = mock_ort.options
    assert options.intra_op_num_threads == threads
    assert options.inter_op_num_threads == 1
    assert options.execution_mode.name == "ORT_SEQUENTIAL"
    assert options.enable_profiling is False
    assert options.add_session_config_entry.call_args_list[0].args == (
        "session.intra_op.allow_spinning",
        "0",
    )
    assert options.add_session_config_entry.call_args_list[1].args == (
        "session.inter_op.allow_spinning",
        "0",
    )
    assert mock_ort.factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    mock_ort.session.disable_fallback.assert_called_once()
    features = np.zeros((2, 560), dtype=np.float32)
    result = engine.infer(features, 4, 15)
    feeds = mock_ort.session.run.call_args.args[1]
    assert feeds["speech"].shape == (1, 2, 560)
    assert feeds["speech"].flags.c_contiguous
    assert feeds["language"].dtype == np.int64
    assert feeds["textnorm"].tolist() == [15]
    mock_ort.session.run.return_value[0][:] = 99
    assert result.tolist() == list(range(6))
    assert engine.get_profile() == []
    engine.close()
    engine.close()
    mock_ort.session.end_profiling.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        engine.infer(features, 4, 15)


def test_ort_profile_finalized_on_close(mock_ort: SimpleNamespace) -> None:
    """Retain a durable native trace and copied JSON-safe records after close.

    Args:
        mock_ort:
            Pure Python session with a two-event fixture trace.

    """
    engine = ORTEngine(Path("model.onnx"), 1, profiling=True)
    assert engine.get_profile() == []
    assert engine.profile_path is None
    assert "diagnostics" in mock_ort.options.profile_file_prefix
    engine.close()
    engine.close()
    mock_ort.session.end_profiling.assert_called_once()
    assert engine.profile_path == mock_ort.profile
    assert engine.profile_path is not None
    assert engine.profile_path.is_file()
    records = engine.get_profile()
    assert len(records) == 1
    assert records[0]["node_type"] == "MatMulInteger"
    assert records[0]["real_seconds"] == 0.00125
    assert records[0]["cpu_seconds"] == 0.0
    json.dumps(records)
    records[0]["node_name"] = "changed"
    assert engine.get_profile()[0]["node_name"] == "quant_kernel_time"


def test_ort_recognizer_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mock_ort: SimpleNamespace
) -> None:
    """Budget before session loading and preserve finalized recognizer diagnostics.

    Args:
        monkeypatch:
            Scoped tokenizer/CMVN and process-budget replacements.

        tmp_path:
            Isolated hashed ONNX fixture artifact directory.

        mock_ort:
            Native session replacement; no device workload is run.

    """
    directory = _artifacts(tmp_path / "artifacts")
    budget = Mock(return_value=(0,))
    monkeypatch.setattr(stt2, "configure_cpu_budget", budget)
    monkeypatch.setattr(
        stt2.CMVN,
        "load",
        Mock(
            return_value=stt2.CMVN(
                np.zeros(560, dtype=np.float32), np.ones(560, dtype=np.float32)
            )
        ),
    )
    monkeypatch.setattr(
        stt2,
        "make_tokenizer",
        Mock(return_value=SimpleNamespace(decode=lambda ids: "<|en|>fixture")),
    )
    mock_ort.session.run.return_value = [np.arange(5, dtype=np.int64)[None, :]]
    recognizer = stt2.STT(
        stt2.STTConfig(directory, threads=1, backend="ort", profiling=True)
    )
    budget.assert_called_once_with(1)
    assert mock_ort.factory.call_args.args == (str(directory / "model.onnx"),)
    assert (
        recognizer.transcribe_samples(np.zeros(800, dtype=np.float32), 16000)
        == "fixture"
    )
    assert recognizer.last_profile == []
    assert recognizer.profile_path is None
    recognizer.close()
    recognizer.close()
    assert recognizer.profile_path == mock_ort.profile
    assert recognizer.last_profile[0]["node_type"] == "MatMulInteger"
    assert recognizer.tokenizer is None
    assert recognizer.frontend is None


@pytest.mark.parametrize("fault", ["provider", "speech", "control", "output"])
def test_ort_rejects_contract_mismatch(mock_ort: SimpleNamespace, fault: str) -> None:
    """Reject invalid graph/provider contracts and release failed construction.

    Args:
        mock_ort:
            Replaceable graph metadata fixture.

        fault:
            Provider, feature, control or output metadata mismatch to inject.

    """
    if fault == "provider":
        mock_ort.session.get_providers.return_value = ["CUDAExecutionProvider"]
    elif fault == "speech":
        mock_ort.session.get_inputs.return_value[0].shape = [1, 2, 560]
    elif fault == "control":
        mock_ort.session.get_inputs.return_value[1].type = "tensor(int32)"
    else:
        mock_ort.session.get_outputs.return_value[0].type = "tensor(float)"
    with pytest.raises(ValueError):
        ORTEngine(Path("model.onnx"), 1, profiling=True)
    mock_ort.session.end_profiling.assert_called_once()


def test_onnx_manifest_and_runtime_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Validate ONNX file integrity and reject backend mismatch before budgeting.

    Args:
        monkeypatch:
            Scoped budget and backend factory replacements.

        tmp_path:
            Isolated fake artifact directories.

    """
    directory = _artifacts(tmp_path / "onnx")
    assert validate_artifacts(directory)["backend"] == "onnxruntime"
    assert ARTIFACT_NAMES == (
        "model.xml",
        "model.bin",
        "am.mvn",
        "tokenizer.model",
        "config.yaml",
    )
    budget = Mock(side_effect=AssertionError("budget changed"))
    monkeypatch.setattr(stt2, "configure_cpu_budget", budget)
    with pytest.raises(ValueError, match="backend"):
        stt2.STT(stt2.STTConfig(directory, backend="cpu"))
    budget.assert_not_called()
    factory = Mock()
    monkeypatch.setattr("hoast.stt2.ort_engine.ORTEngine", factory)
    stt2.make_engine(directory / "model.onnx", 2, "ort", False)
    factory.assert_called_once_with(directory / "model.onnx", 2, False)
    (directory / "model.onnx").write_bytes(b"tamper!")
    with pytest.raises(ValueError, match="SHA256"):
        validate_artifacts(directory)


def test_community_onnx_preparation_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preserve quantized nodes and source pin without OpenVINO conversion/export.

    Args:
        monkeypatch:
            Scoped fake ONNX/SentencePiece modules and source-hash replacement.

        tmp_path:
            Isolated source and output fixture directories.

    """
    community = tmp_path / "community"
    official = tmp_path / "official"
    community.mkdir()
    official.mkdir()
    (official / "README.md").write_text("license: other\nlicense_name: model-license\n")
    licenses = official / "licenses"
    licenses.mkdir()
    records: dict[str, Any] = {}
    for name in ("MODEL_LICENSE", "FunASR-LICENSE", "SenseVoice-LICENSE"):
        (licenses / name).write_bytes(b"fixture legal document")
        records[name] = {
            **file_record(licenses / name),
            "url": f"https://example.invalid/{name}",
        }
    (licenses / "provenance.json").write_text(json.dumps({"files": records}))
    for name in ("LICENSE", "README.md", "export-onnx.py"):
        (community / name).write_bytes(b"fixture community attribution")
    (community / "model.int8.onnx").write_bytes(b"source")
    (community / "tokens.txt").write_text("blank 0\nword 1\n")
    shifts = " ".join(["0"] * 560)
    scales = " ".join(["1"] * 560)
    (official / "am.mvn").write_text(
        f"<AddShift> 560 560\n<LearnRateCoef> 0 [ {shifts} ]\n"
        f"<Rescale> 560 560\n<LearnRateCoef> 0 [ {scales} ]\n"
    )
    for name in ("config.yaml", "chn_jpn_yue_eng_ko_spectok.bpe.model"):
        (official / name).write_bytes(b"fixture")
    metadata = {
        "model_type": "sense_voice_ctc",
        "vocab_size": "25055",
        "lfr_window_size": "7",
        "lfr_window_shift": "6",
        "normalize_samples": "0",
        "neg_mean": ",".join(["0"] * 560),
        "inv_stddev": ",".join(["1"] * 560),
    }
    nodes = [
        SimpleNamespace(op_type="DynamicQuantizeLinear", input=["x"], output=["q"]),
        SimpleNamespace(
            op_type="MatMulInteger",
            input=["q", "language", "text_norm"],
            output=["logits"],
        ),
    ]
    model = SimpleNamespace(
        metadata_props=[SimpleNamespace(key=k, value=v) for k, v in metadata.items()],
        graph=SimpleNamespace(
            node=list(nodes),
            input=[
                SimpleNamespace(name=n)
                for n in ("x", "x_length", "language", "text_norm")
            ],
            output=[SimpleNamespace(name="logits")],
        ),
    )
    helper = SimpleNamespace(
        make_node=Mock(
            side_effect=lambda op, inputs, outputs, **kwargs: SimpleNamespace(
                op_type=op, input=inputs, output=outputs
            )
        ),
        make_tensor=Mock(),
        make_tensor_value_info=Mock(
            side_effect=lambda name, dtype, shape: SimpleNamespace(
                name=name, dtype=dtype, shape=shape
            )
        ),
    )

    def save_fixture(graph: Any, destination: Path) -> None:
        """Write fixture bytes instead of exporting a model.

        Args:
            graph:
                Rewritten in-memory fake graph, retained for assertions.

            destination:
                Importer's staged model.onnx path.

        """
        destination.write_bytes(b"mock transformed graph")

    fake_onnx = SimpleNamespace(
        load=Mock(return_value=model),
        helper=helper,
        TensorProto=SimpleNamespace(INT32=6, INT64=7, FLOAT=1),
        checker=SimpleNamespace(check_model=Mock()),
        save=Mock(side_effect=save_fixture),
    )
    tokenizer = SimpleNamespace(
        vocab_size=lambda: 2, id_to_piece=lambda i: ["blank", "word"][i]
    )
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(
        sys.modules,
        "sentencepiece",
        SimpleNamespace(SentencePieceProcessor=Mock(return_value=tokenizer)),
    )
    monkeypatch.setitem(sys.modules, "openvino", None)
    monkeypatch.setattr(prepare_stt2_community, "configure_cpu_budget", Mock())
    pinned_sha = prepare_stt2_community.COMMUNITY_SHA256
    assert (
        pinned_sha == "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"
    )
    source_hash = Mock(return_value=pinned_sha)
    monkeypatch.setattr(prepare_stt2_community, "sha256_file", source_hash)
    output = tmp_path / "output"
    prepare_stt2_community.prepare(community, official, output, 1, "onnxruntime")
    assert validate_artifacts(output)["backend"] == "onnxruntime"
    assert {path.name for path in output.iterdir()} == {
        *ONNX_ARTIFACT_NAMES,
        "manifest.json",
        "licenses",
    }
    license_inventory = validate_artifacts(output)["licenses"]["files"]
    assert "licenses/community-LICENSE" in license_inventory
    assert (
        output / "licenses/community-LICENSE"
    ).read_bytes() == b"fixture community attribution"
    assert (output / "licenses/MODEL_LICENSE").read_bytes() == b"fixture legal document"
    assert model.graph.node[6:8] == nodes
    assert nodes[0].input == ["speech"]
    assert nodes[1].input == ["q", "_language32", "_textnorm32"]
    assert [port.name for port in model.graph.input] == [
        "speech",
        "language",
        "textnorm",
    ]
    assert model.graph.output[0].name == "frame_ids"
    assert model.graph.node[-1].op_type == "ArgMax"
    with pytest.raises(FileExistsError):
        prepare_stt2_community.prepare(community, official, output, 1, "onnxruntime")
    source_hash.return_value = "0" * 64
    with pytest.raises(ValueError, match="audited release"):
        prepare_stt2_community.prepare(
            community, official, tmp_path / "rejected", 1, "onnxruntime"
        )
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("model", ["sensevoice", "whisper", "kokoro", "melo"])
def test_benchmark_ort_only_sensevoice(model: str) -> None:
    """Allow explicit ORT only for SenseVoice, without creating any model worker.

    Args:
        model:
            Benchmark model family to validate.

    """
    args = argparse.Namespace(model=model, backend="ort", profile=False, model_dir=None)
    if model == "sensevoice":
        benchmark_speech2.validate_options(args)
    else:
        with pytest.raises(ValueError, match="Unsupported model/backend"):
            benchmark_speech2.validate_options(args)


def test_benchmark_forwards_ort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Forward ORT settings to native STT without creating a benchmark workload.

    Args:
        monkeypatch:
            Scoped recognizer factory replacement.

        tmp_path:
            Explicit candidate model-directory fixture path.

    """
    factory = Mock()
    monkeypatch.setattr(stt2, "STT", factory)
    args = argparse.Namespace(
        model="sensevoice",
        backend="ort",
        profile=False,
        model_dir=tmp_path,
        cache=tmp_path,
        threads=1,
    )
    benchmark_speech2.load_engine(args)
    config = factory.call_args.args[0]
    assert config.backend == "ort"
    assert config.model_path == tmp_path
    assert config.threads == 1
    assert config.profiling is False
