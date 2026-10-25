"""Native OpenVINO SANM inference graph built directly from checkpoint tensors.

Inference equations are adapted from the MIT-licensed SenseVoice implementation
by Alibaba/FunASR; attribution and license text are in LICENSES.md. This module
has no FunASR dependency. Torch is needed only by the separate preparation tool
for weights-only checkpoint loading, never by the STT runtime.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import openvino as ov
from numpy.typing import NDArray
from openvino import opset13 as ops

FloatArray = NDArray[np.float32]


@dataclass(slots=True, frozen=True)
class Architecture:
    """SANM dimensions; defaults describe the audited SenseVoiceSmall checkpoint.

    Args:
        input_size:
            Even feature/embedding width, 560 for LFR7 stacked 80-bin FBANK.

        hidden_size:
            Encoder output/residual width, divisible by heads; reference is 512.

        heads:
            Positive number of equal-width attention heads; reference is four.

        feedforward_size:
            Positive ReLU intermediate width; reference is 2048.

        encoder_blocks:
            Main block count including the input-width-changing first block;
            reference is 50. Smaller values are useful for independent fixtures.

        tp_blocks:
            Nonnegative post-encoder block count after after_norm; reference is 20.

        kernel_size:
            Odd symmetric depthwise temporal convolution width; reference is 11.

        vocab_size:
            CTC vocabulary size including blank and metadata; reference is 25055.

        embedding_count:
            Prefix embedding rows, at least 16 to include both ITN controls.

    """

    input_size: int = 560
    """LFR feature/embedding width before the first SANM block."""

    hidden_size: int = 512
    """Attention, residual and encoder output width."""

    heads: int = 4
    """Equal-width attention heads."""

    feedforward_size: int = 2048
    """ReLU feed-forward intermediate width."""

    encoder_blocks: int = 50
    """Main encoder count, including the feature-width-changing first block."""

    tp_blocks: int = 20
    """Post-normalization encoder count."""

    kernel_size: int = 11
    """Odd symmetric depthwise temporal convolution width."""

    vocab_size: int = 25055
    """CTC output vocabulary including blank zero and metadata pieces."""

    embedding_count: int = 16
    """Prefix embedding rows; indices 14 and 15 are the ITN controls."""

    def __post_init__(self) -> None:
        """Reject dimensions incompatible with sinusoidal encoding and attention."""
        if (
            self.input_size < 4
            or self.input_size % 2
            or self.hidden_size < 1
            or self.heads < 1
            or self.hidden_size % self.heads
            or self.feedforward_size < 1
            or self.encoder_blocks < 1
            or self.tp_blocks < 0
            or self.kernel_size < 1
            or self.kernel_size % 2 == 0
            or self.vocab_size < 1
            or self.embedding_count < 16
        ):
            raise ValueError("Invalid SANM architecture dimensions")


def block_prefixes(architecture: Architecture) -> list[tuple[str, int]]:
    """Return checkpoint block prefixes and their input widths in execution order.

    Args:
        architecture:
            Main and post-encoder counts and dimensions.

    """
    return (
        [("encoder.encoders0.0", architecture.input_size)]
        + [
            (f"encoder.encoders.{index}", architecture.hidden_size)
            for index in range(architecture.encoder_blocks - 1)
        ]
        + [
            (f"encoder.tp_encoders.{index}", architecture.hidden_size)
            for index in range(architecture.tp_blocks)
        ]
    )


def expected_shapes(architecture: Architecture) -> dict[str, tuple[int, ...]]:
    """Describe every inference tensor using audited checkpoint key names.

    Args:
        architecture:
            Dimensions used to validate all tensors before graph construction.

    """
    hidden = architecture.hidden_size
    ff = architecture.feedforward_size
    shapes: dict[str, tuple[int, ...]] = {}
    for prefix, width in block_prefixes(architecture):
        local = {
            "self_attn.linear_out.weight": (hidden, hidden),
            "self_attn.linear_out.bias": (hidden,),
            "self_attn.linear_q_k_v.weight": (3 * hidden, width),
            "self_attn.linear_q_k_v.bias": (3 * hidden,),
            "self_attn.fsmn_block.weight": (hidden, 1, architecture.kernel_size),
            "feed_forward.w_1.weight": (ff, hidden),
            "feed_forward.w_1.bias": (ff,),
            "feed_forward.w_2.weight": (hidden, ff),
            "feed_forward.w_2.bias": (hidden,),
            "norm1.weight": (width,),
            "norm1.bias": (width,),
            "norm2.weight": (hidden,),
            "norm2.bias": (hidden,),
        }
        shapes.update({f"{prefix}.{key}": shape for key, shape in local.items()})
    for prefix in ("encoder.after_norm", "encoder.tp_norm"):
        shapes[f"{prefix}.weight"] = (hidden,)
        shapes[f"{prefix}.bias"] = (hidden,)
    shapes["ctc.ctc_lo.weight"] = (architecture.vocab_size, hidden)
    shapes["ctc.ctc_lo.bias"] = (architecture.vocab_size,)
    shapes["embed.weight"] = (architecture.embedding_count, architecture.input_size)
    return shapes


@dataclass(slots=True)
class _Builder:
    """One graph construction context; constants own copied checkpoint storage."""

    weights: Mapping[str, FloatArray]
    """Validated float32 arrays keyed by checkpoint module paths."""

    architecture: Architecture
    """SANM dimensions and layer counts."""

    def constant(self, key: str) -> ov.Node:
        """Create a named graph constant from one audited checkpoint tensor.

        Args:
            key:
                Exact validated checkpoint key.

        """
        return ops.constant(self.weights[key], name=key)

    def linear(self, x: ov.Node, prefix: str) -> ov.Node:
        """Apply a PyTorch-layout affine matrix on the last tensor axis.

        Args:
            x:
                Float32 graph tensor shaped (1, frames, input width).

            prefix:
                Checkpoint module prefix containing weight and bias.

        """
        return ops.add(
            ops.matmul(x, self.constant(f"{prefix}.weight"), False, True),
            self.constant(f"{prefix}.bias"),
            name=prefix,
        )

    def norm(self, x: ov.Node, prefix: str) -> ov.Node:
        """Apply float32 LayerNorm with biased variance and epsilon 1e-5.

        Args:
            x:
                Graph tensor shaped (1, frames, width).

            prefix:
                Checkpoint affine normalization module path.

        """
        normalized = ops.mvn(
            x, ops.constant(np.array([-1], dtype=np.int64)), True, 1e-5, "inside_sqrt"
        )
        return ops.add(
            ops.multiply(normalized, self.constant(f"{prefix}.weight")),
            self.constant(f"{prefix}.bias"),
            name=prefix,
        )

    def attention(self, x: ov.Node, prefix: str) -> ov.Node:
        """Combine full attention with depthwise FSMN memory including value skip.

        Args:
            x:
                Normalized float32 input shaped (1, frames, input width).

            prefix:
                Checkpoint self_attn module prefix.

        """
        architecture = self.architecture
        hidden = architecture.hidden_size
        heads = architecture.heads
        head_width = hidden // heads
        packed = self.linear(x, f"{prefix}.linear_q_k_v")
        pieces = ops.split(packed, np.int64(-1), 3)
        transformed: list[ov.Node] = []
        for index in range(3):
            transformed.append(
                ops.transpose(
                    ops.reshape(
                        pieces.output(index), [1, -1, heads, head_width], False
                    ),
                    [0, 2, 1, 3],
                )
            )
        query, key, value = transformed
        scores = ops.matmul(
            ops.multiply(query, np.float32(head_width**-0.5)), key, False, True
        )
        context = ops.matmul(ops.softmax(scores, -1), value, False, False)
        context = ops.reshape(
            ops.transpose(context, [0, 2, 1, 3]), [1, -1, hidden], False
        )
        projected = self.linear(context, f"{prefix}.linear_out")
        values = ops.reshape(pieces.output(2), [1, -1, hidden], False)
        filters = ops.reshape(
            self.constant(f"{prefix}.fsmn_block.weight"),
            [hidden, 1, 1, architecture.kernel_size],
            False,
        )
        padding = architecture.kernel_size // 2
        memory = ops.group_convolution(
            ops.transpose(values, [0, 2, 1]),
            filters,
            [1],
            [padding],
            [padding],
            [1],
        )
        return ops.add(
            projected,
            ops.add(ops.transpose(memory, [0, 2, 1]), values),
            name=prefix,
        )

    def block(self, x: ov.Node, prefix: str, width: int) -> ov.Node:
        """Apply pre-norm attention and residual ReLU feed-forward equations.

        Args:
            x:
                Float32 graph tensor shaped (1, frames, width).

            prefix:
                Exact checkpoint SANM block module path.

            width:
                Block input width; the first 560-to-512 block has no attention skip.

        """
        attended = self.attention(
            self.norm(x, f"{prefix}.norm1"), f"{prefix}.self_attn"
        )
        if width == self.architecture.hidden_size:
            attended = ops.add(x, attended)
        feedforward = self.linear(
            ops.relu(
                self.linear(
                    self.norm(attended, f"{prefix}.norm2"), f"{prefix}.feed_forward.w_1"
                )
            ),
            f"{prefix}.feed_forward.w_2",
        )
        return ops.add(attended, feedforward, name=prefix)


def build_graph(
    weights: Mapping[str, FloatArray],
    architecture: Architecture | None = None,
    output_logits: bool = False,
) -> ov.Model:
    """Build batch-one dynamic-frame float32 inference without framework export.

    All frames are valid, so no padding mask or length input is needed. Outputs
    are frame_ids shaped (1, frames+4), including metadata. Optional ctc_logits
    shaped (1, frames+4, vocab) supports numerical qualification only. Argmax is
    applied directly to logits; log-softmax cannot alter its choice.

    Args:
        weights:
            Native float32 checkpoint tensors; missing/unexpected keys, shapes
            and dtypes raise ValueError before any graph nodes are allocated.

        architecture:
            None selects SenseVoiceSmall; reduced dimensions support tiny tests.

        output_logits:
            Include full logits for independent numerical audits, increasing
            output memory and copy cost. Runtime preparation defaults to false.

    """
    architecture = Architecture() if architecture is None else architecture
    expected = expected_shapes(architecture)
    missing = sorted(expected.keys() - weights.keys())
    unexpected = sorted(weights.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}"
        )
    for key, shape in expected.items():
        array = weights[key]
        if array.dtype != np.float32 or array.shape != shape:
            raise ValueError(
                f"Checkpoint tensor {key} must be float32 with shape {shape}"
            )
    builder = _Builder(weights, architecture)
    speech = ops.parameter([1, -1, architecture.input_size], np.float32, name="speech")
    language = ops.parameter([1], np.int64, name="language")
    textnorm = ops.parameter([1], np.int64, name="textnorm")
    for parameter in (speech, language, textnorm):
        parameter.output(0).set_names({parameter.get_friendly_name()})
    prefix_ids = ops.concat(
        [language, ops.constant(np.array([1, 2], dtype=np.int64)), textnorm], 0
    )
    prefix = ops.unsqueeze(
        ops.gather(builder.constant("embed.weight"), prefix_ids, np.int64(0)), [0]
    )
    x = ops.multiply(
        ops.concat([prefix, speech], 1), np.float32(architecture.hidden_size**0.5)
    )
    frames = ops.gather(ops.shape_of(x, "i64"), np.int64(1), np.int64(0))
    positions = ops.range(np.int64(1), ops.add(frames, np.int64(1)), np.int64(1), "i64")
    depth = architecture.input_size
    inverse_timescales = np.exp(
        np.arange(depth // 2, dtype=np.float32)
        * np.float32(-np.log(np.float32(10000)) / (depth / 2 - 1))
    )
    scaled_time = ops.multiply(
        ops.reshape(ops.convert(positions, "f32"), [1, -1, 1], False),
        inverse_timescales.reshape(1, 1, -1),
    )
    x = ops.add(
        x,
        ops.concat([ops.sin(scaled_time), ops.cos(scaled_time)], 2),
        name="positioned",
    )
    for index, (prefix, width) in enumerate(block_prefixes(architecture)):
        if index == architecture.encoder_blocks:
            x = builder.norm(x, "encoder.after_norm")
        x = builder.block(x, prefix, width)
    if architecture.tp_blocks == 0:
        x = builder.norm(x, "encoder.after_norm")
    x = builder.norm(x, "encoder.tp_norm")
    logits = builder.linear(x, "ctc.ctc_lo")
    logits.output(0).set_names({"ctc_logits"})
    top = ops.topk(logits, np.int64(1), -1, "max", "value", "i64", True)
    ids = ops.squeeze(top.output(1), [-1])
    ids.output(0).set_names({"frame_ids"})
    outputs: list[ov.Output] = [ids.output(0)]
    if output_logits:
        outputs.append(logits.output(0))
    # OpenVINO's public constructor stub names _pyopenvino.Model as self,
    # conflicting with the public _ov_api.Model wrapper used here.
    return ov.Model(outputs, [speech, language, textnorm], "SenseVoiceSmall")  # pyright: ignore[reportGeneralTypeIssues]
