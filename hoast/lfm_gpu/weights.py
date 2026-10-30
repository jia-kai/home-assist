"""Read the asymmetric U8 weight-decompression subset of OpenVINO IR v11.

This is an offline artifact reader, not an inference backend. It indexes XML and
maps individual BIN ranges on demand; it does not expand or copy model weights.
The supported graph is Convert(u8 -> f16), Subtract of a separately converted
U8 zero point, and Multiply by an F16 scale, all with NumPy broadcasting.
Both arithmetic operations occur in F16, including the intermediate rounding.
Unrecognized byte constants are rejected rather than silently omitted.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Constant:
    """A named, contiguous constant in the IR's little-endian BIN file."""

    name: str
    """Exact IR layer name."""

    shape: tuple[int, ...]
    """Logical tensor shape in IR order; an empty tuple is a scalar."""

    element_type: str
    """OpenVINO element type, such as u8 or f16."""

    offset: int
    """Byte offset in the BIN file."""

    size: int
    """Byte length in the BIN file."""

    def map(self, bin_path: Path) -> NDArray:
        """Return a read-only memory-mapped view without loading the entire tensor.

        Args:
            bin_path:
                BIN path belonging to the indexed IR; it must remain unchanged.

        """
        types = {
            "u8": "u1",
            "i8": "i1",
            "f16": "<f2",
            "f32": "<f4",
            "i32": "<i4",
            "i64": "<i8",
            "boolean": "?",
        }
        if self.element_type not in types:
            raise ValueError(f"Unsupported constant type: {self.element_type}")
        dtype = np.dtype(types[self.element_type])
        if self.offset < 0 or self.size != math.prod(self.shape) * dtype.itemsize:
            raise ValueError(f"Invalid constant range or size: {self.name}")
        if self.offset + self.size > bin_path.stat().st_size:
            raise ValueError(f"Constant exceeds BIN file: {self.name}")
        return np.memmap(
            bin_path, dtype=dtype, mode="r", offset=self.offset, shape=self.shape
        )


@dataclass(frozen=True, slots=True)
class QuantizedWeight:
    """One original asymmetric per-output weight, preserving all IR metadata."""

    weight: Constant
    """U8 weight tensor in original IR order, shape (outputs, ...)."""

    zero_point: Constant
    """U8 zero points, shape (outputs, 1, ...), broadcast over remaining axes."""

    scale: Constant
    """F16 scales, same shape as zero points and broadcast over remaining axes."""

    subtract_name: str
    """Exact IR Subtract layer name."""

    multiply_name: str
    """Exact IR Multiply layer name identifying the dequantized tensor."""


@dataclass(frozen=True, slots=True)
class WeightIndex:
    """Validated graph metadata and the BIN path; arrays are mapped only on request."""

    bin_path: Path
    """Associated BIN path, which must not be modified while views are in use."""

    constants: dict[str, Constant]
    """All IR constants by exact layer name, including unquantized model constants."""

    weights: dict[str, QuantizedWeight]
    """Recognized quantized weights keyed by the exact weight layer name."""


def read_weights(
    xml_path: Path,
    bin_path: Path,
    *,
    expected_xml_sha256: str | None = None,
    expected_bin_sha256: str | None = None,
) -> WeightIndex:
    """Index an IR v11 and validate every byte constant as an asymmetric weight.

    Only U8 per-output weights and U8 zero points, converted to F16 and multiplied
    by F16 per-output scales, are supported. Any other I8/U8 graph is rejected.
    Other IR operations are not validated as a complete LFM execution graph.
    The IR does not bind its XML and BIN cryptographically. Without trusted
    digests the reader cannot distinguish a same-sized, incorrect BIN. Both
    artifacts must remain unchanged throughout indexing and subsequent mapping.

    Args:
        xml_path:
            OpenVINO XML graph path.

        bin_path:
            Its associated unmodified BIN weights path.

        expected_xml_sha256:
            Optional trusted lowercase SHA-256 hex digest of the XML. Supply both
            expected digests together to verify exact artifact identity.

        expected_bin_sha256:
            Optional trusted lowercase SHA-256 hex digest of the BIN. Hashing
            streams the entire file and adds startup I/O without loading it all.

    """
    if (expected_xml_sha256 is None) != (expected_bin_sha256 is None):
        raise ValueError("Both XML and BIN digests are required for identity checking")
    if expected_xml_sha256 is not None and expected_bin_sha256 is not None:
        for path, expected in (
            (xml_path, expected_xml_sha256),
            (bin_path, expected_bin_sha256),
        ):
            if len(expected) != 64 or any(
                c not in "0123456789abcdef" for c in expected
            ):
                raise ValueError("Expected SHA-256 must be 64 lowercase hex digits")
            with path.open("rb") as file:
                actual = hashlib.file_digest(file, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Artifact SHA-256 mismatch: {path}")
    root = ET.parse(xml_path).getroot()
    if root.tag != "net" or root.get("version") != "11":
        raise ValueError("Expected an OpenVINO IR version 11 net")
    layer_root, edge_root = root.find("layers"), root.find("edges")
    if layer_root is None or edge_root is None:
        raise ValueError("IR requires layers and edges")
    layers: dict[str, ET.Element] = {}
    names: set[str] = set()
    for layer in layer_root:
        ident, name = layer.get("id"), layer.get("name")
        if ident is None or name is None or ident in layers or name in names:
            raise ValueError("Missing or duplicate IR layer id/name")
        layers[ident] = layer
        names.add(name)
    incoming: dict[str, dict[str, str]] = {}
    source_ports: dict[tuple[str, str], str] = {}
    outgoing: dict[str, list[tuple[str, str]]] = {}
    for edge in edge_root:
        src, dst = edge.get("from-layer"), edge.get("to-layer")
        src_port, dst_port = edge.get("from-port"), edge.get("to-port")
        if (
            src not in layers
            or dst not in layers
            or src_port is None
            or dst_port is None
        ):
            raise ValueError("Invalid IR edge")
        assert src is not None and dst is not None
        if dst_port in incoming.setdefault(dst, {}):
            raise ValueError("Duplicate input edge")
        incoming[dst][dst_port] = src
        source_ports[(dst, dst_port)] = src_port
        outgoing.setdefault(src, []).append((dst, dst_port))
        if (
            layers[src].find(f"output/port[@id='{src_port}']") is None
            or layers[dst].find(f"input/port[@id='{dst_port}']") is None
        ):
            raise ValueError("IR edge references a missing port")
    bin_size = bin_path.stat().st_size
    constants: dict[str, Constant] = {}
    by_id: dict[str, Constant] = {}
    types = {"u8": 1, "i8": 1, "f16": 2, "f32": 4, "i32": 4, "i64": 8, "boolean": 1}
    for ident, layer in layers.items():
        if layer.get("type") != "Const":
            continue
        data = layer.find("data")
        if data is None:
            raise ValueError("Constant missing data")
        try:
            element_type = data.attrib["element_type"]
            raw_shape = data.attrib["shape"].strip()
            shape = (
                tuple(int(d.strip()) for d in raw_shape.split(",")) if raw_shape else ()
            )
            offset, size = int(data.attrib["offset"]), int(data.attrib["size"])
            name = layer.attrib["name"]
            if element_type not in types or any(d <= 0 for d in shape):
                raise ValueError("Unsupported constant type or shape")
            if (
                offset < 0
                or size != math.prod(shape) * types[element_type]
                or offset + size > bin_size
            ):
                raise ValueError("Invalid constant byte range")
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Invalid constant {layer.get('name')}: {error}"
            ) from error
        constant = Constant(name, shape, element_type, offset, size)
        constants[name] = constant
        by_id[ident] = constant

    def only_child(ident: str, kind: str, port: str) -> str:
        """Find a unique expected consumer at its specified input port.

        Args:
            ident:
                Producer layer id.

            kind:
                Required consumer operation type.

            port:
                Required consumer input port id.

        """
        children = outgoing.get(ident, [])
        if len(children) != 1 or children[0][1] != port:
            raise ValueError(f"Unsupported consumers of {layers[ident].get('name')}")
        child = children[0][0]
        if layers[child].get("type") != kind:
            raise ValueError(f"Expected {kind} after {layers[ident].get('name')}")
        return child

    def converted_byte(ident: str) -> str:
        """Validate a single-input U8-to-F16 Convert and return its consumer id.

        Args:
            ident:
                Byte constant layer id.

        """
        convert = only_child(ident, "Convert", "0")
        data = layers[convert].find("data")
        if (
            data is None
            or data.get("destination_type") != "f16"
            or incoming.get(convert) != {"0": ident}
        ):
            raise ValueError("Expected U8-to-F16 Convert")
        return convert

    weights: dict[str, QuantizedWeight] = {}
    used: set[str] = set()
    byte_ids = {i for i, c in by_id.items() if c.element_type in ("u8", "i8")}
    for ident in sorted(byte_ids, key=int):
        if ident in used:
            continue
        if by_id[ident].element_type != "u8":
            raise ValueError("Signed INT8 constants are not supported")
        cv = converted_byte(ident)
        consumers = outgoing.get(cv, [])
        if len(consumers) != 1 or layers[consumers[0][0]].get("type") != "Subtract":
            raise ValueError("Expected asymmetric Subtract")
        sub, port = consumers[0]
        # Discover from either input so traversal order cannot affect validation.
        if port not in ("0", "1") or set(incoming.get(sub, {})) != {"0", "1"}:
            raise ValueError("Unsupported Subtract inputs")
        wcv, zcv = incoming[sub]["0"], incoming[sub]["1"]

        def source(convert: str) -> str:
            """Get the unique source byte constant of a Convert.

            Args:
                convert:
                    Convert layer id.

            """
            inputs = incoming.get(convert, {})
            if set(inputs) != {"0"} or inputs["0"] not in byte_ids:
                raise ValueError("Expected byte constant Convert input")
            result = inputs["0"]
            if by_id[result].element_type != "u8" or converted_byte(result) != convert:
                raise ValueError("Expected U8 weight and zero point")
            return result

        wid, zid = source(wcv), source(zcv)
        if wid == zid or wid in used or zid in used:
            raise ValueError("Reused quantized byte constant")
        if (
            only_child(wcv, "Subtract", "0") != sub
            or only_child(zcv, "Subtract", "1") != sub
        ):
            raise ValueError("Invalid Subtract graph")
        mul = only_child(sub, "Multiply", "0")
        for op in (sub, mul):
            data = layers[op].find("data")
            if data is None or data.get("auto_broadcast") != "numpy":
                raise ValueError("Expected NumPy broadcasting")
        args = incoming.get(mul, {})
        if set(args) != {"0", "1"} or args["0"] != sub or args["1"] not in by_id:
            raise ValueError("Expected constant scale at Multiply input 1")
        weight, zero, scale = by_id[wid], by_id[zid], by_id[args["1"]]
        if (
            len(weight.shape) < 2
            or zero.shape != (weight.shape[0],) + (1,) * (len(weight.shape) - 1)
            or scale.shape != zero.shape
            or scale.element_type != "f16"
        ):
            raise ValueError(
                f"Unsupported per-output quantization shapes: {weight.name}"
            )
        for node in (wid, zid, args["1"], wcv, zcv, sub, mul):
            if layers[node].get("version") != "opset1":
                raise ValueError(
                    f"Unsupported operation version: {layers[node].get('name')}"
                )

        def port_shape(
            node: str, direction: str, port: str, shape: tuple[int, ...], precision: str
        ) -> None:
            """Check the declared static shape and type of one operation port.

            Args:
                node:
                    Layer identifier.

                direction:
                    Input or output port collection.

                port:
                    Port identifier.

                shape:
                    Required logical tensor shape.

                precision:
                    Required IR precision spelling.

            """
            element = layers[node].find(f"{direction}/port[@id='{port}']")
            if element is None or element.get("precision") != precision:
                raise ValueError(
                    f"Invalid port precision for {layers[node].get('name')}"
                )
            try:
                actual = tuple(int(dim.text or "") for dim in element.findall("dim"))
            except ValueError as error:
                raise ValueError("Invalid port shape") from error
            if actual != shape:
                raise ValueError(f"Invalid port shape for {layers[node].get('name')}")

        for node, shape, precision in (
            (wid, weight.shape, "U8"),
            (zid, zero.shape, "U8"),
            (args["1"], scale.shape, "FP16"),
        ):
            port_shape(node, "output", "0", shape, precision)
        for node, shape in ((wcv, weight.shape), (zcv, zero.shape)):
            port_shape(node, "input", "0", shape, "U8")
            port_shape(node, "output", "1", shape, "FP16")
        for node in (sub, mul):
            port_shape(node, "input", "0", weight.shape, "FP16")
            port_shape(node, "input", "1", zero.shape, "FP16")
            port_shape(node, "output", "2", weight.shape, "FP16")
        for node, port, expected in (
            (wcv, "0", "0"),
            (zcv, "0", "0"),
            (sub, "0", "1"),
            (sub, "1", "1"),
            (mul, "0", "2"),
            (mul, "1", "0"),
        ):
            if source_ports.get((node, port)) != expected:
                raise ValueError(f"Invalid source port for {layers[node].get('name')}")
        weights[weight.name] = QuantizedWeight(
            weight, zero, scale, layers[sub].attrib["name"], layers[mul].attrib["name"]
        )
        used.update((wid, zid))
    if not weights:
        raise ValueError("No supported asymmetric quantized weights")
    return WeightIndex(bin_path, constants, weights)
