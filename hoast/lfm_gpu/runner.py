"""Experimental batch-one OpenCL LFM runner, separate from the live service.

Two exact IR/BIN pairs are supported: the local deployed ``openvino-int8``
export and the published ``lfm/checkpoint`` export. The graph and weights
fingerprints are paired and unsupported revisions fail closed. The local
fingerprints identify inspected artifacts, not independently authenticated
upstream provenance. Both graphs use the same LFM2.5-350M execution dimensions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .weights import Constant, read_weights

LOGGER = logging.getLogger(__name__)
PREPARED_PREFILL_MIN_TOKENS = 128
PREPARED_WEIGHT_BYTES = 4608 * 1024 * 2
PERSISTENT_KN_BYTES = 548 * 1024 * 1024
# Pair each supported graph fingerprint with its own weights fingerprint.
ARTIFACT_DIGESTS = {
    "ef6e654baeadb7ef7b804779cbe9b68170e64b4985a7ad7dafc8e7436f64306b": "406c1b9ebc07ec3ee5e043850a093934dad655dbb81d3f01e99e8fe3c1ec7c1e",
    "437536541509f83062d535922381da0dd9e4a70f92c0d86fee9c706895fde819": "f0d31d1c89bac2d2a3419edf9081fcd598d896f933e70428c8f69cfdeeb435c5",
}


def _digest(path: Path) -> str:
    """Return an artifact's SHA-256 using bounded reads.

    Args:
        path:
            Artifact path.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_digest(path: Path, expected: str) -> None:
    """Reject an artifact whose SHA-256 does not match the supported model.

    Args:
        path:
            File to hash using bounded-size reads.

        expected:
            Expected lowercase hexadecimal SHA-256 digest.
    """
    if _digest(path) != expected:
        raise ValueError(f"Unexpected model artifact digest: {path.name}")


def _verify_artifacts(xml: Path, bin_path: Path) -> None:
    """Reject unsupported IRs and graph/weight pairs from different exports.

    Args:
        xml:
            OpenVINO XML graph path.

        bin_path:
            Associated OpenVINO weights path.
    """
    expected_bin = ARTIFACT_DIGESTS.get(_digest(xml))
    if expected_bin is None:
        raise ValueError("Unsupported OpenVINO model graph digest")
    _verify_digest(bin_path, expected_bin)


def _validate_norm(name: str, constant: Constant, width: int) -> None:
    """Reject an IR norm tensor with a mismatched shape, type or byte size.

    Args:
        name:
            IR name of the normalization operation.

        constant:
            Metadata of its associated weight constant.

        width:
            Required number of FP32 features, either 64 or 1024.
    """
    expected_shape = (1, 1, 1024) if width == 1024 else (1, 1, 1, 64)
    if (
        constant.element_type != "f32"
        or constant.shape != expected_shape
        or constant.size != width * 4
    ):
        raise ValueError(f"Invalid normalization weight: {name}")


def _validate_memory_limits(
    device: Any,
    workspace_sizes: dict[str, int],
    model_sizes: list[int],
    persistent_sizes: dict[str, int],
) -> None:
    """Reject aggregate or individual allocations beyond reported device limits.

    Driver availability and shared system memory can be lower than reported
    limits, so actual allocation can still fail.

    Args:
        device:
            OpenCL device exposing global and maximum-allocation byte limits.

        workspace_sizes:
            Named workspace allocations in bytes.

        model_sizes:
            Individual quantized model and norm allocations in bytes.

        persistent_sizes:
            Individual retained K-major projection allocations in bytes.
    """
    sizes = [*workspace_sizes.values(), *model_sizes, *persistent_sizes.values()]
    if any(size <= 0 or size > device.max_mem_alloc_size for size in sizes):
        raise ValueError("A model or workspace allocation exceeds device limit")
    if sum(sizes) > device.global_mem_size:
        raise ValueError("Model and workspace exceed device global memory")


def _validate_device_workgroup(device: Any, name: str, shape: tuple[int, ...]) -> None:
    """Reject a fixed local shape unsupported by device-wide limits.

    Args:
        device:
            OpenCL device exposing total and per-axis work-item limits.

        name:
            Kernel name for the diagnostic.

        shape:
            Required local workgroup dimensions.
    """
    dimensions = device.max_work_item_sizes
    if (
        len(dimensions) < len(shape)
        or any(required > available for required, available in zip(shape, dimensions))
        or math.prod(shape) > device.max_work_group_size
    ):
        raise RuntimeError(f"Device workgroup limits do not support {name}: {shape}")


def _persistent_projection_sizes() -> dict[str, int]:
    """Return fixed IR projection names and FP16 K-major byte sizes."""
    sizes: dict[str, int] = {}
    for layer in range(16):
        prefix = f"self.model.model.layers.{layer}."
        for name, shape in (
            ("feed_forward.w1.weight", (4608, 1024)),
            ("feed_forward.w3.weight", (4608, 1024)),
            ("feed_forward.w2.weight", (1024, 4608)),
        ):
            sizes[prefix + name] = shape[0] * shape[1] * 2
        if layer in (2, 5, 8, 10, 12, 14):
            for part, output in (("q", 1024), ("k", 512), ("v", 512), ("out", 1024)):
                sizes[prefix + f"self_attn.{part}_proj.weight"] = output * 1024 * 2
        else:
            sizes[prefix + "conv.in_proj.weight"] = 3072 * 1024 * 2
            sizes[prefix + "conv.out_proj.weight"] = 1024 * 1024 * 2
    return sizes


@dataclass(frozen=True, slots=True)
class WorkspaceSpec:
    """Fixed dimensions for a single-sequence LFM2.5-350M workspace."""

    capacity: int
    """Maximum number of cached tokens, including prefill and decode."""

    chunk_size: int
    """Maximum number of tokens in an activation workspace chunk."""

    prepared_prefill: bool = False
    """Enable the 9 MiB prepared-weight scratch path for chunks of at least 128 tokens."""

    prefix_cache: bool = False
    """Preallocate prefix snapshots with capacity equal to the sequence capacity."""

    prepared_kn: bool = False
    """Use K-major blocked GEMM for prepared prefill; requires prepared_prefill."""

    vec4_matvec: bool = False
    """Use the vectorized U8 matvec for single-row projections."""

    tiled_attention: bool = False
    """Use tiled causal GQA score computation for the fixed model dimensions."""

    persistent_kn: bool = False
    """Retain K-major layer projections across chunks and sequence resets."""

    persistent_kn_budget_bytes: int = 0
    """Explicit upper bound in bytes for all retained K-major projections."""

    prefix_slots: int = 1
    """Number of independent snapshot slots when prefix caching is enabled."""

    def __post_init__(self) -> None:
        """Reject invalid dimensions before allocating device memory."""
        for name, value in (
            ("capacity", self.capacity),
            ("chunk_size", self.chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.chunk_size > self.capacity:
            raise ValueError("chunk_size must not exceed capacity")
        if self.capacity > 128000:
            raise ValueError("capacity exceeds the checkpoint's maximum position")
        if type(self.prepared_prefill) is not bool:
            raise TypeError("prepared_prefill must be a boolean")
        if type(self.prefix_cache) is not bool:
            raise TypeError("prefix_cache must be a boolean")
        if type(self.prefix_slots) is not int or self.prefix_slots not in (1, 2):
            raise ValueError("prefix_slots must be 1 or 2")
        if self.prefix_slots != 1 and not self.prefix_cache:
            raise ValueError("Multiple prefix slots require prefix caching")
        if type(self.prepared_kn) is not bool:
            raise TypeError("prepared_kn must be a boolean")
        if self.prepared_kn and not self.prepared_prefill:
            raise ValueError("prepared_kn requires prepared_prefill")
        if type(self.vec4_matvec) is not bool:
            raise TypeError("vec4_matvec must be a boolean")
        if type(self.tiled_attention) is not bool:
            raise TypeError("tiled_attention must be a boolean")
        if type(self.persistent_kn) is not bool:
            raise TypeError("persistent_kn must be a boolean")
        if type(self.persistent_kn_budget_bytes) is not int:
            raise TypeError("persistent_kn_budget_bytes must be an integer")
        if self.persistent_kn:
            if not self.prepared_kn:
                raise ValueError("persistent_kn requires prepared_kn")
            if self.persistent_kn_budget_bytes < PERSISTENT_KN_BYTES:
                raise ValueError("persistent_kn budget is below required model bytes")
        elif self.persistent_kn_budget_bytes != 0:
            raise ValueError("persistent_kn budget requires persistent_kn")

    @property
    def buffer_sizes(self) -> dict[str, int]:
        """Return per-buffer byte sizes, including persistent projection storage."""
        f32 = np.dtype(np.float32).itemsize
        chunk = self.chunk_size
        sizes = {
            "hidden": chunk * 1024 * f32,
            "norm": chunk * 1024 * f32,
            "projection": chunk * 3072 * f32,
            "q": chunk * 1024 * f32,
            "k": chunk * 512 * f32,
            "v": chunk * 512 * f32,
            "rotated_q": chunk * 1024 * f32,
            "rotated_k": chunk * 512 * f32,
            "operator": chunk * 1024 * f32,
            "bx": chunk * 1024 * f32,
            "gate_conv": chunk * 1024 * f32,
            "last": 1024 * f32,
            "tokens": chunk * 4,
            "scores": chunk * 16 * self.capacity * f32,
            "gate": chunk * 4608 * f32,
            "up": chunk * 4608 * f32,
            "logits": 65536 * f32,
        }
        if self.prepared_prefill and not self.persistent_kn:
            sizes["prepared_weights"] = PREPARED_WEIGHT_BYTES
        if self.persistent_kn:
            sizes.update(
                {
                    f"prepared_kn:{name}": size
                    for name, size in _persistent_projection_sizes().items()
                }
            )
        for layer in range(16):
            if layer in (2, 5, 8, 10, 12, 14):
                sizes[f"key_{layer}"] = self.capacity * 512 * f32
                sizes[f"value_{layer}"] = self.capacity * 512 * f32
                if self.prefix_cache:
                    sizes[f"prefix_key_{layer}"] = self.capacity * 512 * f32
                    sizes[f"prefix_value_{layer}"] = self.capacity * 512 * f32
                    if self.prefix_slots == 2:
                        sizes[f"prefix1_key_{layer}"] = self.capacity * 512 * f32
                        sizes[f"prefix1_value_{layer}"] = self.capacity * 512 * f32
            else:
                sizes[f"conv_{layer}"] = 1024 * 3 * f32
                if self.prefix_cache:
                    sizes[f"prefix_conv_{layer}"] = 1024 * 3 * f32
                    if self.prefix_slots == 2:
                        sizes[f"prefix1_conv_{layer}"] = 1024 * 3 * f32
        return sizes


def validate_config(path: Path) -> None:
    """Validate the architectural assumptions made by the fixed workspace.

    Args:
        path:
            Path to the checkpoint's JSON model configuration.

    Raises:
        ValueError: If the architecture differs from the supported checkpoint.
    """
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise TypeError("Checkpoint configuration must be a JSON object")
    expected: dict[str, object] = {
        "model_type": "lfm2",
        "hidden_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "vocab_size": 65536,
        "conv_L_cache": 3,
        "conv_bias": False,
        "intermediate_size": 6656,
        "block_auto_adjust_ff_dim": True,
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 256,
        "norm_eps": 1e-5,
        "max_position_embeddings": 128000,
        "layer_types": [
            "full_attention" if i in (2, 5, 8, 10, 12, 14) else "conv"
            for i in range(16)
        ],
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    }
    for key, value in expected.items():
        if (
            key not in config
            or type(config[key]) is not type(value)
            or config[key] != value
        ):
            raise ValueError(f"Unsupported checkpoint configuration: {key}")
    tied = [
        config[key] for key in ("tie_embedding", "tie_word_embeddings") if key in config
    ]
    if not tied or any(value is not True for value in tied):
        raise ValueError("Unsupported checkpoint configuration: tied embeddings")


class Runner:
    """Experimental batch-one runner for the published LFM2.5-350M IR.

    Input tokens are unpadded, logits are returned only for the final token,
    and one runner may serve only one sequence at a time. Runner calls are
    serialized internally; callers must serialize any other use of its queue.
    Arithmetic is not bit-identical to OpenVINO.
    When explicitly enabled by the workspace specification, prefill chunks of
    at least 128 tokens prepare each projection in a shared 9 MiB FP16 scratch
    buffer; smaller chunks use direct U8 projections. Decode always uses matvec.
    When prepared_kn is enabled, prepared weights use K-major layout and a
    blocked GEMM; otherwise the original prepared-weight path is used.
    When vec4_matvec is enabled, single-row projections use the vectorized
    matvec, whose reduction order may change output rounding.
    When tiled_attention is enabled, attention scores use the tiled fixed-shape
    kernel; the existing attention application consumes the same workspace.
    Persistent K-major mode preallocates layer projection buffers and prepares
    their contents lazily, within the explicitly supplied 548 MiB minimum budget.
    Optional prefix snapshots survive reset and are never applied unless
    the complete input begins with the exact stored token sequence.
    """

    cl: Any
    """PyOpenCL module used by the runner."""

    workspace: Workspace
    """Fixed-capacity single-sequence state and activations."""

    kernels: dict[str, Any]
    """Compiled OpenCL kernels keyed by function name."""

    weights: dict[str, tuple[Any, Any, Any, tuple[int, ...]]]
    """Device U8 weights, zero points, FP32 scales, and logical shapes."""

    conv: dict[int, Any]
    """Dequantized depthwise-convolution weights by layer index."""

    norms: dict[str, Any]
    """Device normalization weights keyed by IR operation name."""

    valid: bool
    """Whether the current sequence is usable after the last operation."""

    _lock: Any
    """Serializes inference and reset calls on this runner."""

    _token_host: np.ndarray
    """Preallocated int32 token input, shape [maximum chunk length]."""

    _logits_host: np.ndarray
    """Preallocated float32 transfer buffer, shape [vocabulary size]."""

    _prefix_tokens: tuple[int, ...] | None
    """Immutable cached token IDs, or None when no snapshot is valid."""

    _prefix_logits: np.ndarray
    """Private FP32 last-prefix logits, shape [vocabulary size]."""

    _extra_prefix_tokens: tuple[int, ...] | None
    """Token IDs for the second snapshot, when allocated."""

    _extra_prefix_logits: np.ndarray
    """Second snapshot's FP32 final logits, shape [vocabulary size]."""

    _prepared_weights: dict[str, Any]
    """Preallocated device FP16 K-major buffers keyed by projection name."""

    _prepared_ready: set[str]
    """Projection names with preparation enqueued successfully on the in-order queue."""

    _persistent_sizes: dict[str, int]
    """Validated byte sizes of persistent projections, keyed by IR weight name."""

    def __init__(self, queue: Any, checkpoint: Path, spec: WorkspaceSpec) -> None:
        """Upload the checkpoint to a caller-owned in-order GPU queue.

        Args:
            queue:
                Caller-owned PyOpenCL in-order command queue.

            checkpoint:
                Directory containing config.json and the IR pair.

            spec:
                Maximum sequence and chunk sizes.
        """
        import pyopencl as cl  # GPU runtime is an optional heavy dependency.

        validate_config(checkpoint / "config.json")
        xml = checkpoint / "openvino_model.xml"
        bin_path = checkpoint / "openvino_model.bin"
        _verify_artifacts(xml, bin_path)
        if "cl_khr_fp16" not in queue.device.extensions.split():
            raise RuntimeError("IR-rounded dequantization requires cl_khr_fp16")
        if not queue.device.half_fp_config & cl.device_fp_config.DENORM:
            raise RuntimeError(
                "IR-rounded dequantization requires FP16 denormal support"
            )
        if spec.prepared_kn and (
            queue.device.max_work_group_size < 256
            or len(queue.device.max_work_item_sizes) < 2
            or min(queue.device.max_work_item_sizes[:2]) < 16
        ):
            raise RuntimeError("K-major preparation requires 16x16 workgroups")
        optional_shapes = []
        if spec.vec4_matvec:
            optional_shapes.append(("lfm_matvec_u8_f16_dequant_vec4", (128, 1)))
        if spec.tiled_attention:
            optional_shapes.append(("lfm_gqa_scores_tiled_16_8_64", (64, 1, 1)))
        for name, shape in optional_shapes:
            _validate_device_workgroup(queue.device, name, shape)
        self.cl: Any = cl
        self._lock = threading.Lock()
        self._token_host = np.empty(spec.chunk_size, dtype=np.int32)
        self._logits_host = np.empty(65536, dtype=np.float32)
        self._prefix_logits = np.empty(65536, dtype=np.float32)
        self._prefix_tokens = None
        self._extra_prefix_tokens = None
        self._extra_prefix_logits = np.empty(65536, dtype=np.float32)
        source = "\n".join(
            (Path(__file__).parent / n).read_text()
            for n in ("lfm_gpu_kernels.cl", "lfm_gpu_ops.cl", "lfm_gpu_runner.cl")
        )
        program = cl.Program(queue.context, source).build(options=["-cl-std=CL1.2"])
        names = (
            "lfm_matvec_u8_f16_dequant",
            "lfm_matmul_u8_f16_dequant",
            "lfm_prepare_u8_f16",
            "lfm_matmul_prepared_f16",
            "lfm_rmsnorm_parallel",
            "lfm_rope_qk",
            "lfm_kv_append",
            "lfm_gqa_scores",
            "lfm_gqa_apply",
            "lfm_causal_conv3",
            "lfm_embedding",
            "lfm_conv_split",
            "lfm_add",
            "lfm_swiglu",
            "lfm_last",
        )
        self.kernels: dict[str, Any] = {n: cl.Kernel(program, n) for n in names}
        if spec.prepared_kn:
            optional_shapes.extend(
                (
                    ("lfm_prepare_u8_f16_kn", (16, 16)),
                    ("lfm_matmul_prepared_kn_f16", (8, 8)),
                )
            )
        for name, shape in optional_shapes:
            kernel = cl.Kernel(program, name)
            if kernel.get_work_group_info(
                cl.kernel_work_group_info.WORK_GROUP_SIZE, queue.device
            ) < math.prod(shape):
                raise RuntimeError(f"Insufficient workgroup size for {name}")
            self.kernels[name] = kernel
        index = read_weights(xml, bin_path)
        self._prepared_weights = {}
        self._prepared_ready = set()
        self._persistent_sizes = {}
        if spec.persistent_kn:
            self._persistent_sizes = {
                name: weight.weight.size * 2
                for name, weight in index.weights.items()
                if "/aten::_convolution/Reshape" not in name
                and name != "self.model.model.embed_tokens.weight"
            }
            if (
                self._persistent_sizes != _persistent_projection_sizes()
                or sum(self._persistent_sizes.values()) != PERSISTENT_KN_BYTES
                or any(
                    size <= 0 or size > PREPARED_WEIGHT_BYTES
                    for size in self._persistent_sizes.values()
                )
            ):
                raise ValueError("Unexpected persistent projection inventory")
        model_sizes = [1024 * 4] * 33 + [64 * 4] * 12
        for name, weight in index.weights.items():
            if "/aten::_convolution/Reshape" in name:
                model_sizes.append(1024 * 3 * 4)
            else:
                model_sizes.extend(
                    (
                        weight.weight.size,
                        weight.zero_point.size,
                        weight.weight.shape[0] * 4,
                    )
                )
        _validate_memory_limits(queue.device, spec.buffer_sizes, model_sizes, {})
        self.workspace = Workspace(queue, spec)
        self._prepared_weights = {
            name: self.workspace.buffers[f"prepared_kn:{name}"]
            for name in self._persistent_sizes
        }
        expected_weights: dict[str, tuple[int, ...]] = {
            "self.model.model.embed_tokens.weight": (65536, 1024),
        }
        attention_layers = (2, 5, 8, 10, 12, 14)
        expected_conv: set[int] = set()
        expected_norms = {
            "__module.model.model.embedding_norm/aten::mul/Multiply_1": 1024
        }
        for layer in range(16):
            prefix = f"self.model.model.layers.{layer}."
            norm_prefix = f"__module.model.model.layers.{layer}."
            for norm in ("operator_norm", "ffn_norm"):
                expected_norms[norm_prefix + norm + "/aten::mul/Multiply_1"] = 1024
            for name, shape in (
                ("w1", (4608, 1024)),
                ("w3", (4608, 1024)),
                ("w2", (1024, 4608)),
            ):
                expected_weights[prefix + "feed_forward." + name + ".weight"] = shape
            if layer in attention_layers:
                for part, output in (
                    ("q", 1024),
                    ("k", 512),
                    ("v", 512),
                    ("out", 1024),
                ):
                    expected_weights[prefix + f"self_attn.{part}_proj.weight"] = (
                        output,
                        1024,
                    )
                for part in ("q", "k"):
                    expected_norms[
                        norm_prefix + f"self_attn.{part}_layernorm/aten::mul/Multiply_1"
                    ] = 64
            else:
                expected_conv.add(layer)
                expected_weights[prefix + "conv.in_proj.weight"] = (3072, 1024)
                expected_weights[prefix + "conv.out_proj.weight"] = (1024, 1024)

        def upload(a: np.ndarray) -> Any:
            """Copy an immutable model array to the GPU.

            Args:
                a:
                    Contiguous model tensor.
            """
            return cl.Buffer(
                queue.context,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=np.ascontiguousarray(a),
            )

        self.weights: dict[str, tuple[Any, Any, Any, tuple[int, ...]]] = {}
        self.conv: dict[int, Any] = {}
        for name, weight in index.weights.items():
            q, z, s = (
                v.map(index.bin_path)
                for v in (weight.weight, weight.zero_point, weight.scale)
            )
            if "/aten::_convolution/Reshape" in name:
                layer = int(name.split("layers.")[1].split(".")[0])
                if (
                    layer not in expected_conv
                    or layer in self.conv
                    or weight.weight.shape != (1024, 1, 1, 3)
                ):
                    raise ValueError(f"Unexpected convolution weight: {name}")
                self.conv[layer] = upload(
                    ((q.astype(np.float16) - z.astype(np.float16)) * s)
                    .astype(np.float32)
                    .reshape(1024, 3)
                )
            else:
                if expected_weights.get(name) != weight.weight.shape:
                    raise ValueError(f"Unexpected projection weight: {name}")
                self.weights[name] = (
                    upload(q),
                    upload(z),
                    upload(s.astype(np.float32)),
                    weight.weight.shape,
                )
        root = ET.parse(xml).getroot()
        ls, es = root.find("layers"), root.find("edges")
        if ls is None or es is None:
            raise ValueError("Missing IR layers or edges")
        nodes = {n.attrib["id"]: n for n in ls}
        self.norms: dict[str, Any] = {}
        for edge in es:
            src, dst = nodes[edge.attrib["from-layer"]], nodes[edge.attrib["to-layer"]]
            name, cn = dst.attrib["name"], src.attrib["name"]
            if (
                src.get("type") == "Const"
                and name in expected_norms
                and cn in index.constants
            ):
                c = index.constants[cn]
                expected_width = expected_norms[name]
                if name in self.norms:
                    raise ValueError(f"Duplicate normalization weight: {name}")
                _validate_norm(name, c, expected_width)
                self.norms[name] = upload(c.map(index.bin_path))
        if (
            set(self.weights) != set(expected_weights)
            or set(self.conv) != expected_conv
            or set(self.norms) != set(expected_norms)
        ):
            raise ValueError(
                f"Unexpected model inventory: {len(self.weights)}, {len(self.conv)}, {len(self.norms)}"
            )
        self.valid: bool = True

    def reset(self) -> None:
        """Reset live state, invalidating saved prefix if device reset fails."""
        with self._lock:
            self.valid = False
            try:
                self.workspace.reset()
            except BaseException:
                self._prefix_tokens = None
                self._extra_prefix_tokens = None
                if hasattr(self, "_prepared_ready"):
                    self._prepared_ready.clear()
                LOGGER.exception("LFM GPU reset failed; prefix snapshot invalidated")
                raise
            self.valid = True

    def _run(
        self,
        name: str,
        global_size: tuple[int, ...],
        local_size: tuple[int, ...] | None,
        *args: Any,
    ) -> None:
        """Dispatch a kernel to the in-order queue.

        Args:
            name:
                Kernel name.

            global_size:
                Global work size.

            local_size:
                Workgroup size or None.

            *args:
                Kernel arguments.
        """
        self.kernels[name](self.workspace.queue, global_size, local_size, *args)

    def _project(
        self, x: Any, y: Any, name: str, count: int, inputs: int, outputs: int
    ) -> None:
        """Apply a quantized linear projection.

        Args:
            x:
                Input FP32 activation buffer with shape [count, inputs].

            y:
                Output FP32 activation buffer with shape [count, outputs].

            name:
                IR weight name.

            count:
                Number of token rows.

            inputs:
                Input width.

            outputs:
                Output width.
        """
        q, z, s, shape = self.weights[name]
        if shape != (outputs, inputs):
            raise ValueError(f"Unexpected projection shape {name}: {shape}")
        if count == 1:
            self._run(
                "lfm_matvec_u8_f16_dequant_vec4"
                if self.workspace.spec.vec4_matvec
                else "lfm_matvec_u8_f16_dequant",
                (128, (outputs + 7) // 8),
                (128, 1),
                x,
                q,
                z,
                s,
                y,
                np.int32(inputs),
                np.int32(outputs),
            )
        elif (
            self.workspace.spec.prepared_prefill
            and count >= PREPARED_PREFILL_MIN_TOKENS
        ):
            if inputs * outputs * 2 > PREPARED_WEIGHT_BYTES:
                raise ValueError(f"Projection exceeds prepared-weight scratch: {name}")
            kn = self.workspace.spec.prepared_kn
            persistent = self.workspace.spec.persistent_kn
            if persistent:
                size = self._persistent_sizes.get(name)
                if size != inputs * outputs * 2:
                    raise ValueError(f"Unexpected persistent projection: {name}")
                scratch = self._prepared_weights[name]
                prepare = name not in self._prepared_ready
            else:
                scratch = self.workspace.buffers["prepared_weights"]
                prepare = True
            if prepare:
                self._run(
                    "lfm_prepare_u8_f16_kn" if kn else "lfm_prepare_u8_f16",
                    (16 * ((inputs + 15) // 16), 16 * ((outputs + 15) // 16))
                    if kn
                    else (64 * ((inputs * outputs + 63) // 64),),
                    (16, 16) if kn else (64,),
                    q,
                    z,
                    s,
                    scratch,
                    np.int32(inputs),
                    np.int32(outputs),
                )
                if persistent:
                    self._prepared_ready.add(name)
            # The in-order queue orders preparation and readers, including
            # reuse of the shared scratch buffer or persistent buffers.
            self._run(
                "lfm_matmul_prepared_kn_f16" if kn else "lfm_matmul_prepared_f16",
                (8 * ((outputs + 31) // 32), 8 * ((count + 31) // 32))
                if kn
                else (64, (outputs + 7) // 8, (count + 7) // 8),
                (8, 8) if kn else (64, 1, 1),
                x,
                scratch,
                y,
                np.int32(count),
                np.int32(inputs),
                np.int32(outputs),
            )
        else:
            self._run(
                "lfm_matmul_u8_f16_dequant",
                (128, (outputs + 7) // 8, (count + 7) // 8),
                (128, 1, 1),
                x,
                q,
                z,
                s,
                y,
                np.int32(count),
                np.int32(inputs),
                np.int32(outputs),
            )

    def _norm(self, x: Any, y: Any, name: str, rows: int, width: int) -> None:
        """Apply learned RMSNorm to contiguous rows.

        Args:
            x:
                Input FP32 buffer with shape [rows, width].

            y:
                Output FP32 buffer with shape [rows, width]; may alias input.

            name:
                IR norm multiplication name.

            rows:
                Row count.

            width:
                Row width.
        """
        self._run(
            "lfm_rmsnorm_parallel",
            (64, rows),
            (64, 1),
            x,
            self.norms[name],
            y,
            np.int32(rows),
            np.int32(width),
            np.float32(1e-5),
        )

    def _chunk(self, tokens: list[int], *, final: bool) -> np.ndarray | None:
        """Append one chunk, returning last-position logits only for the final chunk.

        Args:
            tokens:
                Valid vocabulary IDs fitting the workspace.

            final:
                Whether to project and return the chunk's final-token logits.
        """
        w, b = self.workspace, self.workspace.buffers
        count, start, cap = len(tokens), w.position, w.spec.capacity

        def ints(*v: int) -> tuple[np.int32, ...]:
            """Convert integer kernel arguments to OpenCL int32.

            Args:
                *v:
                    Integer arguments.
            """
            return tuple(np.int32(i) for i in v)

        self._token_host[:count] = tokens
        self.cl.enqueue_copy(
            w.queue, b["tokens"], self._token_host[:count], is_blocking=True
        )
        q, z, s, shape = self.weights["self.model.model.embed_tokens.weight"]
        if shape != (65536, 1024):
            raise ValueError("Unexpected embedding shape")
        self._run(
            "lfm_embedding",
            (count * 1024,),
            None,
            b["tokens"],
            q,
            z,
            s,
            b["hidden"],
            *ints(count, 1024),
        )
        for layer in range(16):
            p, n = (
                f"self.model.model.layers.{layer}.",
                f"__module.model.model.layers.{layer}.",
            )
            self._norm(
                b["hidden"],
                b["norm"],
                n + "operator_norm/aten::mul/Multiply_1",
                count,
                1024,
            )
            if layer not in (2, 5, 8, 10, 12, 14):
                self._project(
                    b["norm"],
                    b["projection"],
                    p + "conv.in_proj.weight",
                    count,
                    1024,
                    3072,
                )
                self._run(
                    "lfm_conv_split",
                    (count * 1024,),
                    None,
                    b["projection"],
                    b["bx"],
                    b["gate_conv"],
                    *ints(count, 1024),
                )
                self._run(
                    "lfm_causal_conv3",
                    (1024,),
                    None,
                    b["bx"],
                    b["gate_conv"],
                    self.conv[layer],
                    b[f"conv_{layer}"],
                    b["operator"],
                    *ints(count, 1024),
                )
                self._project(
                    b["operator"],
                    b["projection"],
                    p + "conv.out_proj.weight",
                    count,
                    1024,
                    1024,
                )
            else:
                for part, width in (("q", 1024), ("k", 512), ("v", 512)):
                    self._project(
                        b["norm"],
                        b[part],
                        p + f"self_attn.{part}_proj.weight",
                        count,
                        1024,
                        width,
                    )
                self._norm(
                    b["q"],
                    b["q"],
                    n + "self_attn.q_layernorm/aten::mul/Multiply_1",
                    count * 16,
                    64,
                )
                self._norm(
                    b["k"],
                    b["k"],
                    n + "self_attn.k_layernorm/aten::mul/Multiply_1",
                    count * 8,
                    64,
                )
                self._run(
                    "lfm_rope_qk",
                    (count, 16, 32),
                    None,
                    b["q"],
                    b["k"],
                    b["rotated_q"],
                    b["rotated_k"],
                    *ints(count, 16, 8, 64, start),
                    np.float32(1e6),
                )
                self._run(
                    "lfm_kv_append",
                    (count, 8, 64),
                    None,
                    b["rotated_k"],
                    b["v"],
                    b[f"key_{layer}"],
                    b[f"value_{layer}"],
                    *ints(count, 8, 64, start, cap),
                )
                if w.spec.tiled_attention:
                    self._run(
                        "lfm_gqa_scores_tiled_16_8_64",
                        (64 * ((start + count + 7) // 8), (count + 3) // 4, 8),
                        (64, 1, 1),
                        b["rotated_q"],
                        b[f"key_{layer}"],
                        b["scores"],
                        *ints(count, start, cap),
                    )
                else:
                    self._run(
                        "lfm_gqa_scores",
                        (start + count, count * 16),
                        None,
                        b["rotated_q"],
                        b[f"key_{layer}"],
                        b["scores"],
                        *ints(count, 16, 8, 64, start, cap),
                    )
                self._run(
                    "lfm_gqa_apply",
                    (64, count * 16),
                    (64, 1),
                    b["scores"],
                    b[f"value_{layer}"],
                    b["operator"],
                    *ints(count, 16, 8, 64, start, cap),
                )
                self._project(
                    b["operator"],
                    b["projection"],
                    p + "self_attn.out_proj.weight",
                    count,
                    1024,
                    1024,
                )
            self._run(
                "lfm_add",
                (count * 1024,),
                None,
                b["hidden"],
                b["projection"],
                *ints(count * 1024),
            )
            self._norm(
                b["hidden"],
                b["norm"],
                n + "ffn_norm/aten::mul/Multiply_1",
                count,
                1024,
            )
            self._project(
                b["norm"], b["gate"], p + "feed_forward.w1.weight", count, 1024, 4608
            )
            self._project(
                b["norm"], b["up"], p + "feed_forward.w3.weight", count, 1024, 4608
            )
            self._run(
                "lfm_swiglu",
                (count * 4608,),
                None,
                b["gate"],
                b["up"],
                *ints(count * 4608),
            )
            self._project(
                b["gate"],
                b["projection"],
                p + "feed_forward.w2.weight",
                count,
                4608,
                1024,
            )
            self._run(
                "lfm_add",
                (count * 1024,),
                None,
                b["hidden"],
                b["projection"],
                *ints(count * 1024),
            )
        if not final:
            w.position += count
            return None
        self._norm(
            b["hidden"],
            b["norm"],
            "__module.model.model.embedding_norm/aten::mul/Multiply_1",
            count,
            1024,
        )
        self._run("lfm_last", (1024,), None, b["norm"], b["last"], *ints(count, 1024))
        self._project(
            b["last"],
            b["logits"],
            "self.model.model.embed_tokens.weight",
            1,
            1024,
            65536,
        )
        self.cl.enqueue_copy(w.queue, self._logits_host, b["logits"], is_blocking=True)
        w.position += count
        return self._logits_host.copy()

    def prefill(self, token_ids: list[int]) -> np.ndarray:
        """Run a prompt from empty state and return last-position logits.

        Args:
            token_ids:
                Nonempty prompt token IDs.
        """
        with self._lock:
            if self.workspace.position != 0 or not self.valid:
                raise RuntimeError("Reset is required before prefill")
            if not isinstance(token_ids, list):
                raise TypeError("Expected a list of integer vocabulary IDs")
            return self._execute(token_ids.copy())

    def _copy_prefix_state(self, length: int, *, restore: bool, slot: int = 0) -> None:
        """Copy prefix KV and convolution state between live and snapshot buffers.

        Args:
            length:
                Number of KV token positions to copy.

            restore:
                If true, restore the snapshot; otherwise capture live state.

            slot:
                Snapshot index, zero or one.
        """
        buffers = self.workspace.buffers
        for layer in range(16):
            if layer in (2, 5, 8, 10, 12, 14):
                entries = (
                    (f"key_{layer}", length * 512 * 4),
                    (f"value_{layer}", length * 512 * 4),
                )
            else:
                entries = ((f"conv_{layer}", 1024 * 3 * 4),)
            for name, size in entries:
                label = "prefix" if slot == 0 else "prefix1"
                live, snapshot = buffers[name], buffers[f"{label}_{name}"]
                dest, src = (live, snapshot) if restore else (snapshot, live)
                self.cl.enqueue_copy(self.workspace.queue, dest, src, byte_count=size)
        self.workspace.queue.finish()

    def cache_prefix(self, token_ids: list[int], *, slot: int = 0) -> np.ndarray:
        """Prefill and snapshot a nonempty prefix from an empty sequence.

        Capture replaces only the selected slot. Reset preserves completed
        snapshots; device failure invalidates all and requires reset of live state.

        Args:
            token_ids:
                Nonempty prefix IDs, bounded by the configured capacity.

            slot:
                Allocated snapshot index.

        Returns:
            Independent FP32 logits for the last prefix token, shape [65536].
        """
        with self._lock:
            if not self.workspace.spec.prefix_cache:
                raise RuntimeError("Prefix caching was not enabled at construction")
            if type(slot) is not int or not 0 <= slot < self.workspace.spec.prefix_slots:
                raise ValueError("Invalid prefix slot")
            if self.workspace.position != 0 or not self.valid:
                raise RuntimeError("Reset is required before caching a prefix")
            if not isinstance(token_ids, list):
                raise TypeError("Expected a list of integer vocabulary IDs")
            tokens = token_ids.copy()
            if not tokens or any(
                type(t) is not int or t < 0 or t >= 65536 for t in tokens
            ):
                raise ValueError("Expected nonempty integer vocabulary IDs")
            if len(tokens) > self.workspace.spec.capacity:
                raise ValueError("Sequence exceeds workspace capacity")
            if slot == 0:
                self._prefix_tokens = None
            else:
                self._extra_prefix_tokens = None
            try:
                logits = self._execute(tokens)
                if slot == 0:
                    self._copy_prefix_state(len(tokens), restore=False)
                    self._prefix_logits[:] = logits
                    self._prefix_tokens = tuple(tokens)
                else:
                    self._copy_prefix_state(len(tokens), restore=False, slot=slot)
                    self._extra_prefix_logits[:] = logits
                    self._extra_prefix_tokens = tuple(tokens)
                return logits
            except BaseException:
                self.valid = False
                self._prefix_tokens = None
                self._extra_prefix_tokens = None
                if hasattr(self, "_prepared_ready"):
                    self._prepared_ready.clear()
                LOGGER.exception("LFM GPU prefix capture failed; reset is required")
                raise

    def prefill_cached(self, token_ids: list[int], *, slot: int = 0) -> np.ndarray:
        """Restore a matching snapshot and prefill any remaining token suffix.

        A mismatch raises without modifying live or snapshot state. Reset is
        required before each independent sequence. Exact-prefix inputs return
        an independent copy of the stored prefix logits.

        Args:
            token_ids:
                Full nonempty prompt, starting with the exact cached prefix.

            slot:
                Allocated snapshot index.

        Returns:
            Independent FP32 final-token logits, shape [65536].
        """
        with self._lock:
            if not self.workspace.spec.prefix_cache:
                raise RuntimeError("Prefix caching was not enabled at construction")
            if type(slot) is not int or not 0 <= slot < self.workspace.spec.prefix_slots:
                raise ValueError("Invalid prefix slot")
            if self.workspace.position != 0 or not self.valid:
                raise RuntimeError("Reset is required before cached prefill")
            if not isinstance(token_ids, list):
                raise TypeError("Expected a list of integer vocabulary IDs")
            tokens = token_ids.copy()
            if not tokens or any(
                type(t) is not int or t < 0 or t >= 65536 for t in tokens
            ):
                raise ValueError("Expected nonempty integer vocabulary IDs")
            if len(tokens) > self.workspace.spec.capacity:
                raise ValueError("Sequence exceeds workspace capacity")
            prefix = self._prefix_tokens if slot == 0 else self._extra_prefix_tokens
            if prefix is None:
                raise RuntimeError("No valid prefix snapshot")
            if len(tokens) < len(prefix) or tuple(tokens[: len(prefix)]) != prefix:
                raise ValueError("Prompt does not match cached prefix")
            try:
                if slot == 0:
                    self._copy_prefix_state(len(prefix), restore=True)
                else:
                    self._copy_prefix_state(len(prefix), restore=True, slot=slot)
                self.workspace.position = len(prefix)
                suffix = tokens[len(prefix) :]
                logits = self._prefix_logits if slot == 0 else self._extra_prefix_logits
                return self._execute(suffix) if suffix else logits.copy()
            except BaseException:
                self.valid = False
                self._prefix_tokens = None
                self._extra_prefix_tokens = None
                if hasattr(self, "_prepared_ready"):
                    self._prepared_ready.clear()
                LOGGER.exception("LFM GPU cached prefill failed; reset is required")
                raise

    def invalidate_prefix(self) -> None:
        """Invalidate all saved prefixes without changing the live sequence."""
        with self._lock:
            self._prefix_tokens = None
            self._extra_prefix_tokens = None

    def decode(self, token_id: int) -> np.ndarray:
        """Append one token and return its logits.

        Args:
            token_id:
                Integer vocabulary ID.
        """
        with self._lock:
            if self.workspace.position == 0 or not self.valid:
                raise RuntimeError("Prefill is required before decode")
            return self._execute([token_id])

    def _execute(self, tokens: list[int]) -> np.ndarray:
        """Validate tokens and append them in bounded chunks.

        Args:
            tokens:
                Nonempty list of vocabulary IDs.
        """
        if not tokens or any(type(t) is not int or t < 0 or t >= 65536 for t in tokens):
            raise ValueError("Expected nonempty integer vocabulary IDs")
        if len(tokens) > self.workspace.spec.capacity - self.workspace.position:
            raise ValueError("Sequence exceeds workspace capacity")
        try:
            result: np.ndarray | None = None
            for start in range(0, len(tokens), self.workspace.spec.chunk_size):
                end = start + self.workspace.spec.chunk_size
                result = self._chunk(tokens[start:end], final=end >= len(tokens))
            assert result is not None
            return result
        except BaseException:
            self.valid = False
            self._prefix_tokens = None
            self._extra_prefix_tokens = None
            if hasattr(self, "_prepared_ready"):
                self._prepared_ready.clear()
            LOGGER.exception("LFM GPU execution failed; reset is required")
            raise


class Workspace:
    """Preallocated buffers for one sequence on a caller-owned in-order queue."""

    spec: WorkspaceSpec
    """Immutable workspace capacity and maximum chunk length."""

    queue: Any
    """Caller-owned in-order OpenCL queue."""

    buffers: dict[str, Any]
    """Named preallocated read-write OpenCL buffers."""

    position: int
    """Number of tokens committed in the current sequence."""

    _cl: Any
    """PyOpenCL module used to allocate and clear buffers."""

    def __init__(self, queue: Any, spec: WorkspaceSpec) -> None:
        """Allocate all workspace buffers and initialize convolution state.

        Args:
            queue:
                Caller-owned in-order PyOpenCL queue.

            spec:
                Validated maximum sequence and activation chunk sizes.
        """
        import pyopencl as cl  # Optional native dependency is needed only for GPU use.

        if queue.properties & cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE:
            raise ValueError("Workspace requires an in-order queue")
        sizes = spec.buffer_sizes
        if any(size > queue.device.max_mem_alloc_size for size in sizes.values()):
            raise ValueError("A workspace buffer exceeds the device allocation limit")
        if sum(sizes.values()) > queue.device.global_mem_size:
            raise ValueError("Workspace exceeds device global memory")
        self.spec = spec
        self.queue = queue
        self._cl = cl
        self.buffers = {
            name: cl.Buffer(queue.context, cl.mem_flags.READ_WRITE, size)
            for name, size in sizes.items()
        }
        self.position = 0
        self.reset()

    def reset(self) -> None:
        """Clear all convolution histories and reset the logical token position."""
        for name, buffer in self.buffers.items():
            if name.startswith("conv_"):
                self._cl.enqueue_fill_buffer(
                    self.queue, buffer, np.float32(0), 0, 1024 * 3 * 4
                )
        self.queue.finish()
        self.position = 0
