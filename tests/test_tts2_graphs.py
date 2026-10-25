"""Offline numerical tests for Melo's dense spline and dynamic relative attention.

Reference equations follow the MIT-licensed pinned MeloTTS neural helpers; the
copyright/permission notice is preserved in hoast.tts2.graphs. Fixtures are small
and embedded: tests do not load any external checkpoint, source checkout or data.
"""

import ast
import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional

from hoast.tts2.graphs import RelativeAttention, inverse_rational_spline
from tools.prepare_tts2 import (
    PCM16_LSB,
    PRECISION_TOLERANCES,
    QUIET_WAVEFORM_RMS,
    STAGE_CONTRACTS,
    WAVEFORM_SNR_FLOORS,
    NeuralImports,
    compare_arrays,
    expand_prior,
    inventory_files,
    save_pipeline_evidence,
    select_stage_precision,
    validate_config,
    waveform_snr_gate,
)


def scalar_inverse_reference(
    value: Tensor,
    widths_logits: Tensor,
    heights_logits: Tensor,
    derivatives_logits: Tensor,
) -> Tensor:
    """Evaluate the original single-value inverse equation including linear tails.

    Args:
        value:
            Scalar FP32 input value, spline domain [-5,5].

        widths_logits:
            FP32 width logits [10].

        heights_logits:
            FP32 height logits [10].

        derivatives_logits:
            FP32 interior derivative logits [9].

    """
    if value < -5 or value > 5:
        return value
    widths = 0.001 + 0.99 * functional.softmax(widths_logits, -1)
    heights = 0.001 + 0.99 * functional.softmax(heights_logits, -1)
    cw = functional.pad(widths.cumsum(-1), (1, 0)) * 10 - 5
    ch = functional.pad(heights.cumsum(-1), (1, 0)) * 10 - 5
    cw[0], cw[-1] = -5, 5
    ch[0], ch[-1] = -5, 5
    widths = cw[1:] - cw[:-1]
    heights = ch[1:] - ch[:-1]
    padded = functional.pad(derivatives_logits, (1, 1))
    padded[0] = padded[-1] = math.log(math.expm1(0.999))
    derivatives = 0.001 + functional.softplus(padded)
    # Upstream adds 1e-6 only to the last edge for inclusive tail endpoints.
    searched = ch.clone()
    searched[-1] += 1e-6
    index = int((value >= searched).sum().item()) - 1
    delta = heights[index] / widths[index]
    shifted = value - ch[index]
    curve = derivatives[index] + derivatives[index + 1] - 2 * delta
    a = shifted * curve + heights[index] * (delta - derivatives[index])
    b = heights[index] * derivatives[index] - shifted * curve
    c = -delta * shifted
    root = 2 * c / (-b - (b.square() - 4 * a * c).sqrt())
    return root * widths[index] + cw[index]


@pytest.mark.parametrize("all_tails", [False, True])
def test_dense_spline_parity_and_linear_tails(all_tails: bool) -> None:
    """Compare dense inverse to scalar upstream equations, including exact bounds.

    Args:
        all_tails:
            Exercise an entirely out-of-domain tensor as well as mixed inputs.

    """
    generator = torch.Generator().manual_seed(19)
    values = torch.tensor([-20, -5.001, -5, -4, -0.5, 0, 0.5, 4, 5, 5.001, 20])
    if all_tails:
        values = torch.tensor([-20, -6, 6, 20], dtype=torch.float32)
    values = values.reshape(1, 1, -1).float()
    widths = torch.randn(*values.shape, 10, generator=generator)
    heights = torch.randn(*values.shape, 10, generator=generator)
    derivatives = torch.randn(*values.shape, 9, generator=generator)
    actual = inverse_rational_spline(values, widths, heights, derivatives)
    expected = torch.stack(
        [
            scalar_inverse_reference(value, width, height, derivative)
            for value, width, height, derivative in zip(
                values.flatten(),
                widths.reshape(-1, 10),
                heights.reshape(-1, 10),
                derivatives.reshape(-1, 9),
                strict=True,
            )
        ]
    ).reshape(values.shape)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    outside = values.abs() > 5
    assert torch.equal(actual[outside], values[outside])
    assert torch.isfinite(actual).all()


class AttentionFixture(nn.Module):
    """Small trained-like relative attention parameters with upstream field names."""

    window_size: int = 4
    """Relative embedding window, not an attention sparsity constraint."""

    heads_share: bool = True
    """Relative embeddings broadcast across heads."""

    proximal_bias: bool = False
    """No proximal attention bias in the pinned model."""

    block_length: None = None
    """No local attention block masking."""

    n_heads: int = 2
    """Number of query/key/value heads."""

    k_channels: int = 4
    """Channels per head."""

    conv_q: nn.Conv1d
    """Query projection, eight input and output channels."""

    conv_k: nn.Conv1d
    """Key projection, eight input and output channels."""

    conv_v: nn.Conv1d
    """Value projection, eight input and output channels."""

    conv_o: nn.Conv1d
    """Output projection, eight input and output channels."""

    emb_rel_k: nn.Parameter
    """Shared relative key embeddings [1,9,4]."""

    emb_rel_v: nn.Parameter
    """Shared relative value embeddings [1,9,4]."""

    def __init__(self) -> None:
        """Create embedded deterministic-size neural fixtures without model assets."""
        super().__init__()
        self.conv_q = nn.Conv1d(8, 8, 1)
        self.conv_k = nn.Conv1d(8, 8, 1)
        self.conv_v = nn.Conv1d(8, 8, 1)
        self.conv_o = nn.Conv1d(8, 8, 1)
        self.emb_rel_k = nn.Parameter(torch.randn(1, 9, 4))
        self.emb_rel_v = nn.Parameter(torch.randn(1, 9, 4))


def padded_relative_reference(layer: AttentionFixture, hidden: Tensor) -> Tensor:
    """Evaluate upstream pad/slice and relative/absolute reshape attention.

    Args:
        layer:
            Small fixture with the upstream projection/embedding field names.

        hidden:
            FP32 batch-one hidden states [1,8,L].

    """
    length = hidden.shape[-1]
    query = layer.conv_q(hidden).reshape(1, 2, 4, length).transpose(2, 3)
    key = layer.conv_k(hidden).reshape(1, 2, 4, length).transpose(2, 3)
    value = layer.conv_v(hidden).reshape(1, 2, 4, length).transpose(2, 3)
    pad = max(length - 5, 0)
    start = max(5 - length, 0)
    relative_k = functional.pad(layer.emb_rel_k, (0, 0, pad, pad))[
        :, start : start + 2 * length - 1
    ]
    relative_v = functional.pad(layer.emb_rel_v, (0, 0, pad, pad))[
        :, start : start + 2 * length - 1
    ]
    relative_logits = (query / 2) @ relative_k.unsqueeze(0).transpose(-2, -1)
    # Exact upstream relative-to-absolute transform (zero column, flatten/pad).
    padded = functional.pad(relative_logits, (0, 1)).reshape(1, 2, length * 2 * length)
    padded = functional.pad(padded, (0, length - 1))
    absolute_logits = padded.reshape(1, 2, length + 1, 2 * length - 1)[
        :, :, :length, length - 1 :
    ]
    probabilities = functional.softmax(
        (query / 2) @ key.transpose(-2, -1) + absolute_logits, -1
    )
    # Exact upstream absolute-to-relative transform used by relative values.
    relative_weights = functional.pad(probabilities, (0, length - 1)).reshape(1, 2, -1)
    relative_weights = functional.pad(relative_weights, (length, 0)).reshape(
        1, 2, length, 2 * length
    )[:, :, :, 1:]
    output = probabilities @ value + relative_weights @ relative_v.unsqueeze(0)
    return layer.conv_o(output.transpose(2, 3).reshape(1, 8, length))


@pytest.mark.parametrize("length", [1, 4, 5, 6, 11])
def test_relative_attention_window_boundary_parity(length: int) -> None:
    """Verify relative attention below, at and above its pad/slice boundary.

    Args:
        length:
            Dynamic self-attention sequence length.

    """
    torch.manual_seed(23)
    source = AttentionFixture().eval()
    graph = RelativeAttention(source).eval()
    hidden = torch.randn(1, 8, length)
    with torch.inference_mode():
        actual = graph(hidden, hidden, torch.ones(1, 1, length, length))
        expected = padded_relative_reference(source, hidden)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_relative_attention_trace_preserves_dynamic_lengths() -> None:
    """Trace at five positions and compare shorter/longer upstream reference outputs."""
    torch.manual_seed(29)
    source = AttentionFixture().eval()
    graph = RelativeAttention(source).eval()
    example = torch.randn(1, 8, 5)
    traced = torch.jit.trace(graph, (example, example, torch.ones(1, 1, 5, 5)))
    assert isinstance(traced, torch.jit.ScriptModule)
    for length in (1, 4, 5, 6, 11):
        hidden = torch.randn(1, 8, length)
        with torch.inference_mode():
            actual = traced(hidden, hidden, torch.ones(1, 1, length, length))
            expected = padded_relative_reference(source, hidden)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_alignment_import_is_removed_and_neural_imports_rewritten() -> None:
    """Inspect AST adaptation without executing upstream code or loading alignment."""
    tree = ast.parse(
        "from melo import commons\nimport melo.monotonic_align as monotonic_align\n"
    )
    adapter = NeuralImports("_fixture_neural")
    adapted = adapter.visit(tree)
    assert adapter.removed_alignment == 1
    assert ast.unparse(adapted) == "from _fixture_neural import commons"
    with pytest.raises(ValueError, match="Unexpected"):
        adapter.visit(ast.parse("from melo import utils"))


def test_stage_contract_noise_and_dynamic_dtypes() -> None:
    """Keep the exported fixed input/output names and dynamic axis contracts stable."""
    encoder = STAGE_CONTRACTS["encoder"]
    assert list(encoder["outputs"]) == [
        "mean",
        "logscale",
        "log_dp",
        "log_sdp",
        "speaker_condition",
    ]
    assert encoder["inputs"]["duration_noise"] == {
        "dtype": "float32",
        "shape": [1, 2, "P"],
    }
    assert STAGE_CONTRACTS["decoder"]["outputs"]["audio"]["shape"] == [1, 1, "512*F"]
    with pytest.raises(KeyError):
        validate_config({"data": {}, "model": {}})


def test_zero_bert_projection_fold_keeps_bias() -> None:
    """Verify the folded 1024-channel zero-feature projection retains trained bias."""
    projection = nn.Conv1d(1024, 192, 1)
    assert projection.bias is not None
    with torch.no_grad():
        projection.bias.copy_(torch.linspace(-0.25, 0.25, 192))
    for length in (1, 4, 5, 9):
        with torch.inference_mode():
            original = projection(torch.zeros(1, 1024, length)).transpose(1, 2)
            folded = projection.bias.reshape(1, 1, -1).expand(1, length, 192)
        torch.testing.assert_close(original, folded, atol=0, rtol=0)
        assert not torch.equal(original, torch.zeros_like(original))


def test_repeat_prior_expansion_matches_dense_alignment() -> None:
    """Compare repeat/gather duration expansion with an independently built alignment."""
    mean = torch.arange(192 * 4, dtype=torch.float32).reshape(1, 192, 4)
    logscale = mean * 0.001
    log_duration = torch.tensor(
        [math.log(1.25), -1000, math.log(2.25), math.log(0.75)]
    ).reshape(1, 1, 4)
    counts, expanded_mean, expanded_logscale = expand_prior(
        mean,
        logscale,
        log_duration,
        log_duration,
    )
    assert counts.flatten().tolist() == [2, 0, 3, 1]
    alignment = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    torch.testing.assert_close(
        expanded_mean, (alignment @ mean.transpose(1, 2)).transpose(1, 2)
    )
    torch.testing.assert_close(
        expanded_logscale, (alignment @ logscale.transpose(1, 2)).transpose(1, 2)
    )


def test_precision_comparison_requires_finite_matching_geometry() -> None:
    """Check explicit precision budgets, SNR evidence and malformed output rejection."""
    expected = torch.tensor([1.0, -1.0]).numpy()
    exact = compare_arrays(expected, expected, *PRECISION_TOLERANCES["FP32"])
    assert exact["exact"] and exact["snr_db"] is None
    perturbed = compare_arrays(
        expected + 0.001, expected, *PRECISION_TOLERANCES["FP16"]
    )
    assert 59 < perturbed["snr_db"] < 61
    with pytest.raises(ValueError, match="geometry"):
        compare_arrays(expected[:1], expected, 0.1, 0.1)
    with pytest.raises(ValueError, match="nonfinite"):
        compare_arrays(torch.tensor([float("nan"), 1.0]).numpy(), expected, 0.1, 0.1)


def test_recursive_inventory_covers_nested_frontend_assets(tmp_path: Path) -> None:
    """Inventory nested static data with unambiguous relative POSIX keys and hashes.

    Args:
        tmp_path:
            Isolated pytest temporary directory; no cached resources are needed.

    """
    nested = tmp_path / "g2p" / "tagger"
    nested.mkdir(parents=True)
    (tmp_path / "weights.json").write_text("root", encoding="utf-8")
    (nested / "weights.json").write_text("nested", encoding="utf-8")
    (nested / "empty-directory").mkdir()
    inventory = inventory_files(tmp_path)
    assert set(inventory) == {"weights.json", "g2p/tagger/weights.json"}
    assert inventory["weights.json"]["bytes"] == 4
    assert inventory["g2p/tagger/weights.json"]["bytes"] == 6
    assert (
        inventory["weights.json"]["sha256"]
        != inventory["g2p/tagger/weights.json"]["sha256"]
    )


def test_stage_precision_selection_preserves_sensitive_fp32_defaults() -> None:
    """Check mixed FP16 defaults, explicit overrides and unchanged INT8 selection."""
    assert {
        stage: select_stage_precision(stage, "FP16") for stage in STAGE_CONTRACTS
    } == {"bert": "FP16", "encoder": "FP32", "flow": "FP32", "decoder": "FP16"}
    for stage in STAGE_CONTRACTS:
        assert select_stage_precision(stage, "FP32") == "FP32"
        assert select_stage_precision(stage, "INT8") == (
            "INT8" if stage == "bert" else "FP32"
        )
        assert select_stage_precision(stage, "FP16", fp16_stages=("decoder",)) == (
            "FP16" if stage == "decoder" else "FP32"
        )
        assert (
            select_stage_precision(stage, "FP16", fp16_stages=tuple(STAGE_CONTRACTS))
            == "FP16"
        )
    with pytest.raises(ValueError, match="selection"):
        select_stage_precision("bert", "FP16", fp16_stages=())
    with pytest.raises(ValueError, match="selection"):
        select_stage_precision("bert", "FP16", fp16_stages=("unknown",))


@pytest.mark.parametrize("precision", ["FP32", "FP16", "INT8"])
def test_waveform_snr_quiet_exception_boundaries(precision: str) -> None:
    """Keep strict RMS, inclusive PCM16 error and precision-specific SNR boundaries.

    Args:
        precision:
            Root storage policy retaining its original SNR floor.

    """
    floor = WAVEFORM_SNR_FLOORS[precision]
    below_rms = math.nextafter(QUIET_WAVEFORM_RMS, 0)
    gate = waveform_snr_gate(below_rms, PCM16_LSB, floor - 1, precision)
    assert gate["quiet_exception_used"]
    for rms, error in (
        (QUIET_WAVEFORM_RMS, PCM16_LSB),
        (math.nextafter(QUIET_WAVEFORM_RMS, math.inf), PCM16_LSB),
        (below_rms, math.nextafter(PCM16_LSB, math.inf)),
    ):
        with pytest.raises(ValueError, match="SNR"):
            waveform_snr_gate(rms, error, floor - 1, precision)
    assert not waveform_snr_gate(0.5, 0.01, floor, precision)["quiet_exception_used"]
    with pytest.raises(ValueError, match="SNR"):
        waveform_snr_gate(0.5, 0.01, math.nextafter(floor, 0), precision)
    assert not waveform_snr_gate(0.5, 0, None, precision)["quiet_exception_used"]


def test_pipeline_evidence_survives_staging_cleanup(tmp_path: Path) -> None:
    """Preserve waveform arrays and rejection metrics outside deleted staging.

    Args:
        tmp_path:
            Isolated pytest temporary root, with no model or external resources.

    """
    staging = tmp_path / "staging"
    durable = tmp_path / "diagnostics" / "run"
    staging.mkdir()
    durable.mkdir(parents=True)
    arrays = {"reference_audio": torch.zeros(1, 1, 4).numpy()}
    metadata = {"status": "rejected", "waveform": {"signal_rms": 0.0}}
    evidence = save_pipeline_evidence(staging, durable, 4, arrays, metadata)
    assert (staging / evidence).read_bytes() == (durable / evidence).read_bytes()
    shutil.rmtree(staging)
    assert (durable / evidence).is_file()
    assert json.loads((durable / "qualification-4.json").read_text()) == metadata
