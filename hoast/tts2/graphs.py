"""Preparation-only Torch graphs for pinned MeloTTS inference.

Nothing in the OpenVINO runtime needs to import this module or MeloTTS. Relative
attention stays full quadratic; window four limits *relative embeddings*, not
attention. Batch one has no padding. Torch reference arithmetic/weights remain
FP32; artifact weight compression is an independent preparation option.

Spline and attention adaptations derive from MeloTTS commit
209145371cff8fc3bd60d7be902ea69cbdb7965a (melo/transforms.py, attentions.py,
models.py, modules.py). Copyright (c) 2024 MyShell.ai, MIT license:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils import remove_weight_norm

SOURCE_REVISION = "209145371cff8fc3bd60d7be902ea69cbdb7965a"
DURATION_NOISE_SCALE = 0.8


def inverse_rational_spline(
    inputs: Tensor,
    widths_logits: Tensor,
    heights_logits: Tensor,
    derivatives_logits: Tensor,
    tail_bound: float = 5.0,
) -> Tensor:
    """Invert Melo's rational-quadratic spline using dense where/gather operations.

    Values outside the closed interval are unchanged (linear tails). Clipping
    only the temporary evaluation input prevents invalid dense tail arithmetic.
    End derivatives equal one; all three upstream minimum constants are 1e-3.
    Entirely tail-only tensors are supported without upstream's empty-partition
    reduction failure. The square-root argument is guarded against negative
    floating-point roundoff; valid-domain inverse equations are unchanged.

    Args:
        inputs:
            FP32 values [B, C, P], where P is phone count.

        widths_logits:
            FP32 bin width logits [B, C, P, K].

        heights_logits:
            FP32 bin height logits [B, C, P, K].

        derivatives_logits:
            FP32 interior derivative logits [B, C, P, K-1].

        tail_bound:
            Positive endpoint of the symmetric spline domain.

    """
    bins = widths_logits.shape[-1]
    widths = 0.001 + (1.0 - 0.001 * bins) * functional.softmax(widths_logits, dim=-1)
    heights = 0.001 + (1.0 - 0.001 * bins) * functional.softmax(heights_logits, dim=-1)
    # Set both cumulative endpoints exactly, as in upstream, without mutation.
    cw_inner = widths.cumsum(-1)[..., :-1] * (2 * tail_bound) - tail_bound
    ch_inner = heights.cumsum(-1)[..., :-1] * (2 * tail_bound) - tail_bound
    endpoint = torch.ones_like(widths[..., :1]) * tail_bound
    cw = torch.cat((-endpoint, cw_inner, endpoint), dim=-1)
    ch = torch.cat((-endpoint, ch_inner, endpoint), dim=-1)
    widths = cw[..., 1:] - cw[..., :-1]
    heights = ch[..., 1:] - ch[..., :-1]
    # Upstream pads the logits with inverse-softplus(1-min_derivative).
    boundary_logit = math.log(math.expm1(1.0 - 0.001))
    boundary = torch.ones_like(derivatives_logits[..., :1]) * boundary_logit
    derivatives = 0.001 + functional.softplus(
        torch.cat((boundary, derivatives_logits, boundary), dim=-1)
    )
    evaluated = inputs.clamp(-tail_bound, tail_bound)
    # Count left edges, including exact internal knots; clamp handles +bound.
    index = (evaluated.unsqueeze(-1) >= ch).sum(-1) - 1
    index = index.clamp(0, bins - 1).unsqueeze(-1)
    x0 = cw.gather(-1, index).squeeze(-1)
    y0 = ch.gather(-1, index).squeeze(-1)
    width = widths.gather(-1, index).squeeze(-1)
    height = heights.gather(-1, index).squeeze(-1)
    delta = (heights / widths).gather(-1, index).squeeze(-1)
    d0 = derivatives.gather(-1, index).squeeze(-1)
    d1 = derivatives[..., 1:].gather(-1, index).squeeze(-1)
    shifted = evaluated - y0
    curve = d0 + d1 - 2 * delta
    a = shifted * curve + height * (delta - d0)
    b = height * d0 - shifted * curve
    c = -delta * shifted
    discriminant = b.square() - 4 * a * c
    root = (2 * c) / (-b - discriminant.clamp_min(0).sqrt())
    transformed = root * width + x0
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    return torch.where(inside, transformed, inputs)


class RelativeAttention(nn.Module):
    """Dynamic-length equivalent of Melo's full relative self-attention."""

    original: Any
    """Pinned upstream attention module with the original trained projections."""

    def __init__(self, original: Any) -> None:
        """Reuse loaded attention weights without changing attention semantics.

        Args:
            original:
                Upstream MultiHeadAttention, with shared relative heads/window 4.

        """
        super().__init__()
        if (
            original.window_size != 4
            or not original.heads_share
            or original.proximal_bias
            or original.block_length is not None
        ):
            raise ValueError("Unsupported pinned Melo attention configuration")
        self.original = original

    def forward(
        self, x: Tensor, context: Tensor, attn_mask: Tensor | None = None
    ) -> Tensor:
        """Apply quadratic self-attention with indexed relative scores and values.

        Args:
            x:
                FP32 hidden states [1, H, P] or [1, H, F].

            context:
                Same self-attention hidden states as x.

            attn_mask:
                Optional broadcast attention mask [1, 1, P, P].

        """
        layer = self.original
        batch, channels, length = x.shape
        query = (
            layer.conv_q(x)
            .reshape(batch, layer.n_heads, layer.k_channels, length)
            .transpose(2, 3)
        )
        key = (
            layer.conv_k(context)
            .reshape(batch, layer.n_heads, layer.k_channels, length)
            .transpose(2, 3)
        )
        value = (
            layer.conv_v(context)
            .reshape(batch, layer.n_heads, layer.k_channels, length)
            .transpose(2, 3)
        )
        scaled_query = query / math.sqrt(layer.k_channels)
        scores = torch.matmul(scaled_query, key.transpose(-2, -1))
        positions = torch.arange(length, device=x.device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_index = (distance + 4).clamp(0, 8)
        local_logits = torch.matmul(
            scaled_query, layer.emb_rel_k.unsqueeze(0).transpose(-2, -1)
        )
        gathered = local_logits.gather(
            -1,
            relative_index.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch, layer.n_heads, length, length),
        )
        scores = scores + gathered * (distance.abs() <= 4).to(scores.dtype)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)
        probabilities = functional.softmax(scores, dim=-1)
        output = torch.matmul(probabilities, value)
        # Gather only the nine nonzero relative contributions, not [P,P,D].
        offsets = torch.arange(-4, 5, device=x.device)
        absolute = positions.unsqueeze(1) + offsets.unsqueeze(0)
        valid = (absolute >= 0) & (absolute < length)
        local_probabilities = probabilities.gather(
            -1,
            absolute.clamp(0, length - 1)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch, layer.n_heads, length, 9),
        ) * valid.to(probabilities.dtype)
        output = output + torch.matmul(
            local_probabilities, layer.emb_rel_v.unsqueeze(0)
        )
        output = output.transpose(2, 3).reshape(batch, channels, length)
        return layer.conv_o(output)


class InverseConvFlow(nn.Module):
    """Melo SDP inverse ConvFlow using the dense rational spline."""

    original: Any
    """Loaded upstream ConvFlow with DDSConv conditioning and spline projection."""

    def __init__(self, original: Any) -> None:
        """Retain original trained conditioning weights.

        Args:
            original:
                Upstream ConvFlow with two input channels.

        """
        super().__init__()
        self.original = original

    def forward(self, latent: Tensor, mask: Tensor, condition: Tensor) -> Tensor:
        """Invert the conditioned two-channel spline coupling.

        Args:
            latent:
                FP32 duration latent [1, 2, P].

            mask:
                FP32 all-one phone mask [1, 1, P].

            condition:
                FP32 SDP conditioning [1, H, P].

        """
        layer = self.original
        x0, x1 = latent.split(layer.half_channels, dim=1)
        hidden = layer.pre(x0)
        hidden = layer.convs(hidden, mask, g=condition)
        hidden = layer.proj(hidden) * mask
        batch, channels, length = x0.shape
        parameters = hidden.reshape(batch, channels, -1, length).permute(0, 1, 3, 2)
        bins = layer.num_bins
        scale = math.sqrt(layer.filter_channels)
        x1 = inverse_rational_spline(
            x1,
            parameters[..., :bins] / scale,
            parameters[..., bins : 2 * bins] / scale,
            parameters[..., 2 * bins :],
            layer.tail_bound,
        )
        return torch.cat((x0, x1), dim=1) * mask


class EncoderGraph(nn.Module):
    """Phone encoder and deterministic/explicit-noise stochastic duration heads."""

    enc_p: Any
    """Loaded text encoder; the zero 1024-feature projection retains its bias."""

    emb_g: Any
    """Speaker embedding [speaker vocabulary, 256]."""

    dp: Any
    """Deterministic duration predictor."""

    sdp_pre: Any
    """Stochastic duration hidden-state input projection."""

    sdp_cond: Any
    """Stochastic duration speaker-conditioning projection."""

    sdp_convs: Any
    """Stochastic duration DDSConv conditioning, without training posterior heads."""

    sdp_proj: Any
    """Stochastic duration conditioning output projection."""

    inverse_flows: nn.ModuleList
    """Exactly three reverse spline couplings, interleaved with flips."""

    use_vc: bool
    """Whether the original configuration disables encoder speaker conditioning."""

    def __init__(self, synthesizer: Any) -> None:
        """Select inference branches only from a strictly loaded SynthesizerTrn.

        Args:
            synthesizer:
                Pinned upstream model after strict checkpoint loading.

        """
        super().__init__()
        self.enc_p = synthesizer.enc_p
        self.emb_g = synthesizer.emb_g
        self.dp = synthesizer.dp
        self.sdp_pre = synthesizer.sdp.pre
        self.sdp_cond = synthesizer.sdp.cond
        self.sdp_convs = synthesizer.sdp.convs
        self.sdp_proj = synthesizer.sdp.proj
        self.use_vc = synthesizer.use_vc
        reverse = list(reversed(synthesizer.sdp.flows))
        # This upstream rule deliberately drops the last ConvFlow and Flip.
        reverse = reverse[:-2] + [reverse[-1]]
        self.inverse_flows = nn.ModuleList(
            InverseConvFlow(flow) if type(flow).__name__ == "ConvFlow" else flow
            for flow in reverse
        )
        if sum(isinstance(flow, InverseConvFlow) for flow in self.inverse_flows) != 3:
            raise ValueError("Pinned SDP must use exactly three inverse ConvFlows")

    def forward(
        self,
        phones: Tensor,
        tones: Tensor,
        languages: Tensor,
        phone_features: Tensor,
        speaker_id: Tensor,
        duration_noise: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return mean, logscale, log_dp, log_sdp and speaker_condition.

        The graph scales standard-normal duration_noise by 0.8 exactly once.
        Runtime blends log durations (default SDP ratio 0.2), exponentiates,
        applies length scale, ceils, repeats prior phone indices and samples
        frame latent noise (default scale 0.6); no dense alignment is exported.

        Args:
            phones:
                INT64 phone IDs [1, P], no padding.

            tones:
                INT64 globally offset tone IDs [1, P].

            languages:
                INT64 language IDs [1, P].

            phone_features:
                FP32 repeated BERT features [1, 768, P].

            speaker_id:
                INT64 speaker index [1].

            duration_noise:
                FP32 independent standard-normal samples [1, 2, P].

        """
        encoder = self.enc_p
        speaker = self.emb_g(speaker_id).unsqueeze(-1)
        # bert_proj(zeros) equals its trained bias; never omit this bias.
        embedding = (
            encoder.emb(phones)
            + encoder.tone_emb(tones)
            + encoder.language_emb(languages)
            + encoder.bert_proj.bias.reshape(1, 1, -1)
            + encoder.ja_bert_proj(phone_features).transpose(1, 2)
        ) * math.sqrt(encoder.hidden_channels)
        hidden = embedding.transpose(1, 2)
        mask = torch.ones_like(hidden[:, :1, :])
        hidden = encoder.encoder(hidden, mask, g=None if self.use_vc else speaker)
        mean, logscale = encoder.proj(hidden).split(encoder.out_channels, dim=1)
        log_dp = self.dp(hidden, mask, g=speaker)
        condition = self.sdp_pre(hidden)
        condition = condition + self.sdp_cond(speaker)
        condition = self.sdp_convs(condition, mask)
        condition = self.sdp_proj(condition) * mask
        latent = duration_noise * DURATION_NOISE_SCALE
        for flow in self.inverse_flows:
            if isinstance(flow, InverseConvFlow):
                latent = flow(latent, mask, condition)
            else:
                latent = flow(latent, mask, g=condition, reverse=True)
        return mean, logscale, log_dp, latent[:, :1, :], speaker


class FlowGraph(nn.Module):
    """Inverse four-coupling flow, separate for independent frame-stage profiling."""

    flow: Any
    """Loaded transformer flow, each coupling containing three attention layers."""

    def __init__(self, synthesizer: Any) -> None:
        """Select the inverse flow branch.

        Args:
            synthesizer:
                Strictly loaded upstream SynthesizerTrn.

        """
        super().__init__()
        self.flow = synthesizer.flow

    def forward(self, latent: Tensor, speaker_condition: Tensor) -> Tensor:
        """Return FP32 decoder latent [1, 192, F] with an all-one frame mask.

        Args:
            latent:
                FP32 sampled prior latent [1, 192, F].

            speaker_condition:
                FP32 encoder speaker embedding [1, 256, 1].

        """
        mask = torch.ones_like(latent[:, :1, :])
        return self.flow(latent, mask, g=speaker_condition, reverse=True)


class DecoderGraph(nn.Module):
    """FP32 HiFi-GAN decoder with folded weight normalization."""

    decoder: Any
    """Loaded decoder with upsample rates 8,8,2,2,2 and 90 residual convolutions."""

    def __init__(self, synthesizer: Any) -> None:
        """Select the decoder without posterior/training branches.

        Args:
            synthesizer:
                Strictly loaded upstream SynthesizerTrn.

        """
        super().__init__()
        self.decoder = synthesizer.dec

    def forward(self, latent: Tensor, speaker_condition: Tensor) -> Tensor:
        """Return waveform FP32 [1, 1, 512*F], at native 44100 samples/second.

        Args:
            latent:
                FP32 flow output [1, 192, F].

            speaker_condition:
                FP32 encoder speaker embedding [1, 256, 1].

        """
        return self.decoder(latent, g=speaker_condition)


class BertGraph(nn.Module):
    """BERT embeddings plus first ten layers, exactly upstream hidden_states[-3]."""

    bert: Any
    """BertModel without pooler/MLM head, retaining loaded layers 0 through 9."""

    def __init__(self, bert: Any) -> None:
        """Truncate twelve trained layers only after loading original weights.

        Args:
            bert:
                Locally loaded BertModel with twelve hidden layers, no pooler.

        """
        super().__init__()
        if (
            bert.config.num_hidden_layers != 12
            or bert.config.hidden_size != 768
            or bert.config.max_position_embeddings != 512
        ):
            raise ValueError("Expected twelve-layer 768-channel BERT checkpoint")
        bert.encoder.layer = nn.ModuleList(list(bert.encoder.layer)[:10])
        bert.config.num_hidden_layers = 10
        self.bert = bert

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor
    ) -> Tensor:
        """Return FP32 token features [1, S, 768], with no pooling/MLM projection.

        Args:
            input_ids:
                INT64 WordPiece IDs [1, S], including boundary tokens.

            attention_mask:
                INT64 token-validity mask [1, S].

            token_type_ids:
                INT64 segment IDs [1, S], normally all zero.

        """
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )[0]


def adapt_inference_model(model: nn.Module) -> None:
    """Replace length-specialized attention and fold all weight normalization.

    Strict checkpoint loading must occur before this mutation. Decoder and flow
    wrappers exclude posterior/training branches from the exported graph.

    Args:
        model:
            Loaded FP32 upstream synthesizer, modified in place and set to eval.

    """
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if type(child).__name__ == "MultiHeadAttention":
                setattr(module, name, RelativeAttention(child))
        if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            remove_weight_norm(module)
    model.float().eval()
