"""Offline, model-independent tests for the LFM CPU parity harness."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import openvino as ov
import pytest

from tools.lfm_gpu_reference import (
    capture,
    compare,
    digest,
    logit_summary,
    parse_chunks,
)


def test_chunks() -> None:
    """Keep chunk boundaries and reject empty, signed, or out-of-range IDs."""
    assert parse_chunks("1, 2;3") == [[1, 2], [3]]
    for text in ("", "1;", "1,,2", "-1", "+1", "65536", "١"):
        with pytest.raises(ValueError):
            parse_chunks(text)


def test_comparison_and_logits() -> None:
    """Enforce explicit asymmetric tolerance, shape, finite values, and stable ties."""
    reference = np.array([[1.0, 2.0]])
    assert compare(reference, reference + 0.001, atol=0.0011, rtol=0)["passed"]
    result = compare(reference, np.array([[1.0, 2.1]]), atol=0, rtol=0)
    assert not result["passed"] and result["max_index"] == [0, 1]
    assert json.loads(json.dumps(result))["max_index"] == [0, 1]
    for a, b, atol in (
        (reference, np.array([1.0, 2.0]), 0),
        (reference, reference, float("nan")),
        (reference, np.array([[1.0, float("nan")]]), 0),
    ):
        with pytest.raises(ValueError):
            compare(a, b, atol=atol, rtol=0)
    assert logit_summary(np.array([[[1.0, 3.0, 3.0]]])) == {
        "top_token": 1,
        "margin": 0.0,
    }
    with pytest.raises(ValueError):
        logit_summary(np.array([1.0]))


def test_capture_state_and_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture all 16 layers and copied states while preserving chunk position.

    Args:
        tmp_path:
            Isolated synthetic IR placeholder directory.

        monkeypatch:
            Replaces OpenVINO graph and request with deterministic in-memory doubles.
    """
    xml = tmp_path / "model.xml"
    xml.write_bytes(b"fixture")
    xml.with_suffix(".bin").write_bytes(b"weights")
    calls: list[Any] = []

    class Node:
        """Provide the named layer outputs expected by the capture routine."""

        def __init__(self, index: int) -> None:
            """Save a layer identifier.

            Args:
                index:
                    Layer number.
            """
            self.index: int = index

        def get_friendly_name(self) -> str:
            """Return the exact fixture layer node name."""
            return f"__module.model.model.layers.{self.index}/aten::add/Add_1"

        def get_type_name(self) -> str:
            """Return the expected operation type."""
            return "Add"

        def output(self, index: int) -> int:
            """Return the layer identifier for the only output.

            Args:
                index:
                    Output port number.
            """
            assert index == 0
            return self.index

    class Model:
        """Small mutable output list for graph instrumentation."""

        def get_ordered_ops(self) -> list[Node]:
            """Return all layer nodes."""
            return [Node(i) for i in range(16)]

        def add_outputs(self, index: int) -> None:
            """Record an added layer output.

            Args:
                index:
                    Layer number.
            """
            calls.append(("output", index))

    class Request:
        """Simulate one mutable stateful inference request."""

        def __init__(self) -> None:
            """Initialize a single mutable state tensor."""
            self.state: np.ndarray[Any, Any] = np.array([0])

        def reset_state(self) -> None:
            """Record reset of request state."""
            calls.append("reset")
            self.state[:] = 0

        def infer(self, inputs: dict[str, np.ndarray[Any, Any]]) -> None:
            """Record input shapes and advance the state.

            Args:
                inputs:
                    Current token chunk and accumulated attention mask.
            """
            calls.append((inputs["input_ids"].tolist(), inputs["attention_mask"].shape))
            self.state += inputs["input_ids"].shape[1]

        def get_tensor(self, name: str) -> Any:
            """Return deterministic logits for the requested name.

            Args:
                name:
                    Output name, which must be logits.
            """
            assert name == "logits"
            return SimpleNamespace(data=np.array([[[0.0, 2.0, 1.0]]]))

        def get_output_tensor(self, index: int) -> Any:
            """Return an activation identifying output index and current state.

            Args:
                index:
                    Model output index.
            """
            return SimpleNamespace(data=np.array([index, self.state[0]]))

        def query_state(self) -> list[Any]:
            """Return a mutable named state object."""
            return [
                SimpleNamespace(name="cache", state=SimpleNamespace(data=self.state))
            ]

    class Core:
        """Provide a CPU-only graph and request."""

        def read_model(self, xml_path: str, bin_path: str) -> Model:
            """Read the fixture graph.

            Args:
                xml_path:
                    XML path.

                bin_path:
                    BIN path.
            """
            assert xml_path == str(xml) and bin_path == str(xml.with_suffix(".bin"))
            return Model()

        def compile_model(
            self, model: Model, device: str, options: dict[str, str]
        ) -> Any:
            """Compile the fixture graph only for the selected CPU precision.

            Args:
                model:
                    Instrumented fixture graph.

                device:
                    Required CPU device.

                options:
                    Required CPU inference precision hint.
            """
            assert isinstance(model, Model) and device == "CPU"
            assert options == {"INFERENCE_PRECISION_HINT": "f32"}
            return SimpleNamespace(create_infer_request=Request)

    monkeypatch.setattr(ov, "Core", Core)
    with pytest.raises(ValueError, match="digest mismatch"):
        capture(xml, "bad", digest(xml.with_suffix(".bin")), [[1]])
    arrays, metadata = capture(
        xml, digest(xml), digest(xml.with_suffix(".bin")), [[1, 2], [3]]
    )
    assert "reset" in calls
    assert ([[1, 2]], (1, 2)) in calls and ([[3]], (1, 3)) in calls
    assert arrays["step0/state/cache"].tolist() == [2]
    assert arrays["step1/state/cache"].tolist() == [3]
    assert arrays["step1/layer15"].tolist() == [16, 3]
    assert metadata["chunks"] == [[1, 2], [3]]
