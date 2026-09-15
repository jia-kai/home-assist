"""Optional verified AVX2 CPU activation fusion for Kokoro OpenVINO models."""

import hashlib
import json
from pathlib import Path

import openvino as ov
from numpy._core._multiarray_umath import __cpu_features__
from openvino.utils.node_factory import NodeFactory

from .logging import get_logger

logger = get_logger(__name__)
DEFAULT_KERNEL = Path(".cache/hoast/tts/cpu-kernels/libhoast_speech.so")
KERNEL_SOURCE = Path(__file__).with_name("kernels") / "snake.cpp"


def _same(left: ov.Output, right: ov.Output) -> bool:
    """Compare exact graph-output identity.

    Args:
        left:
            First graph output.

        right:
            Second graph output.

    """
    return (
        left.get_node().get_name() == right.get_node().get_name()
        and left.get_index() == right.get_index()
    )


def snake_inputs(node: ov.Node) -> list[ov.Output] | None:
    """Match x + scale * sin(alpha*x)^2 without changing its coefficients.

    Args:
        node:
            Candidate Add node in a Kokoro graph.

    """
    if node.get_type_name() != "Add":
        return None
    for index in (0, 1):
        data = node.input_value(index)
        scaled = node.input_value(1 - index).get_node()
        if scaled.get_type_name() != "Multiply":
            continue
        for square_index in (0, 1):
            square = scaled.input_value(square_index).get_node()
            if square.get_type_name() != "Power":
                continue
            exponent = square.input_value(1).get_node()
            while exponent.get_type_name() == "Convert":
                exponent = exponent.input_value(0).get_node()
            if exponent.get_type_name() != "Constant":
                continue
            value = exponent.get_data()
            if value.size != 1 or value.item() != 2:
                continue
            sine = square.input_value(0).get_node()
            if sine.get_type_name() != "Sin":
                continue
            product = sine.input_value(0).get_node()
            if product.get_type_name() != "Multiply":
                continue
            for data_index in (0, 1):
                if _same(data, product.input_value(data_index)):
                    inputs = [
                        data,
                        product.input_value(1 - data_index),
                        scaled.input_value(1 - square_index),
                    ]
                    if _supported_layout(inputs):
                        return inputs
    return None


def _supported_layout(inputs: list[ov.Output]) -> bool:
    """Restrict fusion to FP32 rank-three data and scalar/channel broadcasting.

    Args:
        inputs:
            Data, alpha and scale graph outputs; unknown layouts retain native ops.

    """
    shape = inputs[0].get_partial_shape()
    if shape.rank.is_dynamic or shape.rank.get_length() != 3:
        return False
    for index, port in enumerate(inputs):
        if port.get_element_type() != ov.Type.f32:
            return False
        if index == 0:
            continue
        coefficient = port.get_partial_shape()
        if not coefficient.is_static or coefficient.rank.get_length() > 3:
            return False
        dimensions = list(coefficient.to_shape())
        padded = [1] * (3 - len(dimensions)) + dimensions
        if padded == [1, 1, 1]:
            continue
        if shape[1].is_dynamic or padded != [1, shape[1].get_length(), 1]:
            return False
    return True


def fuse_cpu_activations(core: ov.Core, model: ov.Model, library: Path | None) -> int:
    """Fuse verified CPU activations in place; return the number of replacements.

    Missing optional kernels retain native operations with a warning. A present
    but mismatched/corrupt library fails rather than loading incompatible code.

    Args:
        core:
            CPU model's OpenVINO core.

        model:
            Mutable FP32-compute model graph.

        library:
            Prepared library and adjacent JSON manifest; None disables fusion.

    """
    if library is None:
        return 0
    if not library.is_file():
        logger.warning(
            "tts.kernels status=unavailable path=%s; using native activations", library
        )
        return 0
    manifest = json.loads(library.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest["openvino"] != ov.__version__:
        raise ValueError("CPU speech kernels need rebuilding for this OpenVINO version")
    if (
        manifest["source_sha256"]
        != hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest()
    ):
        raise ValueError("CPU speech kernels need rebuilding for the current source")
    with library.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != manifest["library_sha256"]:
        raise ValueError("CPU speech kernel library checksum mismatch")
    if not __cpu_features__["AVX2"]:
        raise RuntimeError("Prepared CPU speech kernels require AVX2")
    core.add_extension(library)
    factory = NodeFactory()
    factory.add_extension(library)
    count = 0
    for node in model.get_ops():
        inputs = snake_inputs(node)
        if inputs is not None:
            replacement = factory.create("HoastSnake", [*inputs], {})
            replacement.set_friendly_name(node.get_friendly_name() + "/fused_snake")
            node.output(0).replace(replacement.output(0))
            count += 1
    model.validate_nodes_and_infer_types()
    logger.info("tts.kernels status=ok fused=%d library=%s", count, library)
    return count
