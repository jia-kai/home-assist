"""Check shape-safe exported convolution against native prefill and incremental decode."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache, Lfm2ShortConv

from finetune.export import short_conv_forward


@pytest.mark.parametrize("length", [2, 3, 7])
def test_cache_prefill_and_decode(length: int) -> None:
    """Compare outputs and retained history over prefill and three decode steps.

    Args:
        length:
            Initial prompt token count, including prompts shorter than the kernel.

    """
    torch.manual_seed(42)
    config = Lfm2Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        layer_types=["conv", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    module: Any = Lfm2ShortConv(config, layer_idx=0).eval()
    native = Lfm2HybridConvCache(config, max_batch_size=2, dtype=torch.float32)
    patched = SimpleNamespace(
        conv_layer_idx_mapping={0: 0},
        conv_cache=[native.conv_cache[0].clone()],
    )
    with torch.inference_mode():
        for step, count in enumerate([length, 1, 1, 1]):
            x = torch.randn(2, count, 8)
            positions = (
                torch.arange(count) if step == 0 else torch.tensor([length + step - 1])
            )
            expected = module.slow_forward(
                x, past_key_values=native, cache_position=positions
            )
            actual = short_conv_forward(
                module, x, past_key_values=patched, cache_position=positions
            )
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(patched.conv_cache[0], native.conv_cache[0])
