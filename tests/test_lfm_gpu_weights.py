"""Isolated OpenVINO IR weight-index fixtures, independent of model downloads."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest

from hoast.lfm_gpu.weights import read_weights


def fixture_ir(tmp_path: Path) -> tuple[Path, Path]:
    """Serialize a tiny real OpenVINO asymmetric U8 decompression graph.

    Args:
        tmp_path:
            Test-owned directory for the generated XML and BIN files.

    """
    weight = ops.constant(np.array([[0, 128, 255], [4, 5, 6]], dtype=np.uint8))
    weight.set_friendly_name("model.weight")
    zero = ops.constant(np.array([[127], [5]], dtype=np.uint8))
    zero.set_friendly_name("model.weight/zero_point")
    scale = ops.constant(np.array([[0.125], [0.3]], dtype=np.float16))
    scale.set_friendly_name("model.weight/scale")
    diff = ops.subtract(ops.convert(weight, "f16"), ops.convert(zero, "f16"))
    diff.set_friendly_name("weight/subtract")
    out = ops.multiply(diff, scale)
    out.set_friendly_name("weight/multiply")
    model = ov.Model([out], [], "fixture")  # pyright: ignore[reportGeneralTypeIssues]
    xml, binary = tmp_path / "model.xml", tmp_path / "model.bin"
    ov.serialize(model, xml, binary)
    return xml, binary


def test_extract_preserves_bytes_names_shapes_and_f16_arithmetic(
    tmp_path: Path,
) -> None:
    """Extract mapped U8 values and compare exact F16 operations with OpenVINO.

    Args:
        tmp_path:
            Temporary directory for generated model artifacts.

    """
    xml, binary = fixture_ir(tmp_path)
    index = read_weights(xml, binary)
    assert set(index.weights) == {"model.weight"}
    item = index.weights["model.weight"]
    assert item.subtract_name == "weight/subtract"
    assert item.multiply_name == "weight/multiply"
    assert item.weight.shape == (2, 3)
    assert item.zero_point.shape == item.scale.shape == (2, 1)
    w, z, s = (
        constant.map(binary) for constant in (item.weight, item.zero_point, item.scale)
    )
    assert all(not array.flags.writeable for array in (w, z, s))
    np.testing.assert_array_equal(w, [[0, 128, 255], [4, 5, 6]])
    actual = (w.astype(np.float16) - z.astype(np.float16)) * s
    result = ov.Core().compile_model(xml, "CPU")([])[0]
    np.testing.assert_array_equal(actual, result)


def test_invalid_graph_rejected(tmp_path: Path) -> None:
    """Reject a changed arithmetic operation rather than losing a byte constant.

    Args:
        tmp_path:
            Temporary directory for generated model artifacts.

    """
    xml, binary = fixture_ir(tmp_path)
    tree = ET.parse(xml)
    sub = tree.find("./layers/layer[@name='weight/subtract']")
    assert sub is not None
    sub.set("type", "Add")
    tree.write(xml)
    with pytest.raises(ValueError, match="Subtract"):
        read_weights(xml, binary)


def test_invalid_range_and_scale_rejected(tmp_path: Path) -> None:
    """Reject truncated BIN data and unsupported per-output scale metadata.

    Args:
        tmp_path:
            Temporary directory for generated model artifacts.

    """
    xml, binary = fixture_ir(tmp_path)
    tree = ET.parse(xml)
    scale = tree.find("./layers/layer[@name='model.weight/scale']/data")
    assert scale is not None
    scale.set("shape", "2")
    tree.write(xml)
    with pytest.raises(ValueError, match="quantization shapes"):
        read_weights(xml, binary)
    xml, binary = fixture_ir(tmp_path)
    binary.write_bytes(b"\0")
    with pytest.raises(ValueError, match="byte range"):
        read_weights(xml, binary)


def test_trusted_digests_detect_wrong_same_size_binary(tmp_path: Path) -> None:
    """Verify artifact identity independently of structural range checks.

    Args:
        tmp_path:
            Temporary directory for generated model artifacts.

    """
    xml, binary = fixture_ir(tmp_path)
    xml_hash = hashlib.sha256(xml.read_bytes()).hexdigest()
    bin_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    read_weights(
        xml, binary, expected_xml_sha256=xml_hash, expected_bin_sha256=bin_hash
    )
    with pytest.raises(ValueError, match="Both XML and BIN"):
        read_weights(xml, binary, expected_bin_sha256=bin_hash)
    replacement = bytearray(binary.read_bytes())
    replacement[0] ^= 1
    binary.write_bytes(replacement)
    # Size and graph remain valid, but this is not the trusted artifact.
    read_weights(xml, binary)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_weights(
            xml, binary, expected_xml_sha256=xml_hash, expected_bin_sha256=bin_hash
        )


@pytest.mark.parametrize("mutation", ["signed", "port_shape", "version", "broadcast"])
def test_unsupported_quantization_is_rejected(tmp_path: Path, mutation: str) -> None:
    """Reject unsupported storage or operation contracts at indexing time.

    Args:
        tmp_path:
            Temporary directory for generated model artifacts.

        mutation:
            Graph metadata field to corrupt.

    """
    xml, binary = fixture_ir(tmp_path)
    tree = ET.parse(xml)
    weight = tree.find("./layers/layer[@name='model.weight']")
    subtract = tree.find("./layers/layer[@name='weight/subtract']")
    assert weight is not None and subtract is not None
    if mutation == "signed":
        data = weight.find("data")
        assert data is not None
        data.set("element_type", "i8")
    elif mutation == "port_shape":
        dim = weight.find("output/port/dim")
        assert dim is not None
        dim.text = "3"
    elif mutation == "version":
        subtract.set("version", "opset99")
    else:
        data = subtract.find("data")
        assert data is not None
        data.set("auto_broadcast", "none")
    tree.write(xml)
    with pytest.raises(ValueError):
        read_weights(xml, binary)
