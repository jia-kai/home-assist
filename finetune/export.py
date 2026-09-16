"""Export merged LFM2 weights with a shape-safe convolution cache to INT8 OpenVINO."""

import argparse
import logging
import shutil
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import nncf
import openvino as ov
import torch
from optimum.exporters.openvino import main_export, model_patcher
from transformers.models.lfm2.modeling_lfm2 import apply_mask_to_padding_states

logger = logging.getLogger(__name__)


def short_conv_forward(
    self: Any,
    x: torch.Tensor,
    past_key_values: Any = None,
    cache_position: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute causal convolution using concatenated history for any positive length.

    Prefill starts with zeroed cache; decoding retains the final kernel-width input
    states. Concatenation avoids advanced-index scatter shapes dependent on the
    trace-time sequence length. Returns hidden states with the same shape as x.

    Args:
        self:
            LFM2 short-convolution module with projections and depthwise kernel.

        x:
            Floating hidden states shaped (batch, tokens, hidden_channels).

        past_key_values:
            Optimum cache wrapper; convolution states have shape
            (batch, hidden_channels, kernel_width), in the same dtype as x.

        cache_position:
            Framework-compatible position argument; history concatenation supplies
            the causal ordering without indexing by absolute positions.

        attention_mask:
            Optional (batch, tokens) padding mask using the native model semantics.

    """
    x = apply_mask_to_padding_states(x, attention_mask)
    b, c, values = self.in_proj(x).transpose(-1, -2).chunk(3, dim=-2)
    gated = b * values
    if past_key_values is None:
        conv = self.conv(gated)[..., : x.shape[1]]
    else:
        index = past_key_values.conv_layer_idx_mapping[self.layer_idx]
        history = past_key_values.conv_cache[index]
        extended = torch.cat((history, gated.to(history.dtype)), dim=-1)
        conv = self.conv(extended)[..., self.L_cache : self.L_cache + x.shape[1]]
        past_key_values.conv_cache[index].copy_(extended[..., -self.L_cache :])
    return self.out_proj((c * conv).transpose(-1, -2).contiguous())


def main() -> None:
    """Apply the version-scoped export patch, compress weights, and save diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.output / "export.log"),
            logging.StreamHandler(),
        ],
    )
    original = model_patcher.lfm2_short_conv_forward_patched
    try:
        assert version("optimum-intel") == "1.27.0", (
            "Review patch against this exporter version"
        )
        model_patcher.lfm2_short_conv_forward_patched = short_conv_forward
        with TemporaryDirectory(prefix="export-", dir=args.output) as temporary:
            staging = Path(temporary)
            main_export(
                args.model,
                output=staging,
                task="text-generation-with-past",
                stateful=True,
                convert_tokenizer=True,
            )
            graph = ov.Core().read_model(staging / "openvino_model.xml")
            compressed = nncf.compress_weights(
                graph, mode=nncf.CompressWeightsMode.INT8_ASYM
            )
            ov.save_model(compressed, args.output / "openvino_model.xml")
            for source in staging.iterdir():
                if source.is_file() and source.name not in {
                    "openvino_model.xml",
                    "openvino_model.bin",
                }:
                    shutil.copy2(source, args.output / source.name)
        logger.info("export status=ok format=int8_asym cache=concatenated-history")
    except Exception:
        logger.exception("export status=failed")
        raise
    finally:
        model_patcher.lfm2_short_conv_forward_patched = original


if __name__ == "__main__":
    main()
