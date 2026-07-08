# SOURCE: vendored from liutianlin0121/seisLM @ main
#   https://raw.githubusercontent.com/liutianlin0121/seisLM/main/seisLM/model/foundation/multidim_wav2vec2.py
#   https://raw.githubusercontent.com/liutianlin0121/seisLM/main/seisLM/model/foundation/conv_encoder.py
#   https://raw.githubusercontent.com/liutianlin0121/seisLM/main/seisLM/model/foundation/transformer_encoder.py
#   https://raw.githubusercontent.com/liutianlin0121/seisLM/main/seisLM/model/foundation/quantizer.py
#   https://raw.githubusercontent.com/liutianlin0121/seisLM/main/seisLM/model/foundation/position_embedding.py
#   https://raw.githubusercontent.com/liutianlin0121/seisLM/main/seisLM/model/foundation/initialization.py
#   https://raw.githubusercontent.com/liutianlin0121/seisLM/main/seisLM/model/foundation/mask_utils.py
#   https://raw.githubusercontent.com/liutianlin0121/seisLM/main/seisLM/model/foundation/modeling_outputs.py
#
# SeisLM (Liu, Yin, Beroza et al., NeurIPS 2024) -- a Wav2Vec2-style (Baevski et al. 2020)
# self-supervised foundation model for multi-channel seismic waveforms: a multi-dim
# convolutional feature encoder -> Transformer encoder (with rotary or convolutional
# relative position embedding, optional RMSNorm) -> Gumbel-softmax vector quantizer ->
# contrastive + diversity pretraining objective. All classes below
# (Wav2Vec2FeatureEncoder + conv layer variants, Wav2Vec2FeatureProjection, Wav2Vec2Model,
# Wav2Vec2SdpaAttention, Wav2Vec2EncoderLayer(StableLayerNorm), Wav2Vec2Encoder(StableLayerNorm),
# Wav2Vec2PositionalConvEmbedding, rotary-embedding helpers, Wav2Vec2GumbelVectorQuantizer,
# init_wav2vec2_weights, MultiDimWav2Vec2ForPreTraining, the BaseModelOutput/
# Wav2Vec2BaseModelOutput/Wav2Vec2ForPreTrainingOutput dataclasses) are copied verbatim from
# the real repo files above -- no architectural changes.
#
# Two non-architectural substitutions were needed because the repo's config/dependency
# plumbing isn't in the base env:
#   1. `ml_collections.ConfigDict` (a dict-with-attribute-access config container, used only
#      for config plumbing, never for architecture) is replaced by a plain
#      `types.SimpleNamespace` subclass exposing the same dot-attribute read/getattr
#      semantics the real code relies on (`config.hidden_size`, `getattr(config, "input_dim", 1)`).
#   2. `torchtune.modules.RMSNorm` (a standard RMSNorm: `x / rms(x) * scale`, no
#      architectural novelty) is replaced by a self-contained `RMSNorm` implementing the
#      identical formula, so no external `torchtune` dependency is required.
# Real config field names/values are taken verbatim from the repo's shipped pretrain config
# `seisLM/configs/pretrain/pretrain_config_std_norm_single_ax_8_datasets_32bit_scaleup_samp_false.json`,
# scaled down (fewer layers/channels/heads) only for tiny-trace size.
"""SeisLM: multi-dim Wav2Vec2-style self-supervised seismic waveform foundation model.

Faithful tiny-scale assembly of the real SeisLM `MultiDimWav2Vec2ForPreTraining` for
TorchLens tracing. Random init, no pretrained/hub downloads.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import gumbel_softmax

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Non-architectural config/dep shims (see header)
# ---------------------------------------------------------------------------
class ConfigDict(SimpleNamespace):
    """Minimal stand-in for ml_collections.ConfigDict (dot-attribute config only)."""


class RMSNorm(nn.Module):
    """Standard RMSNorm (stand-in for torchtune.modules.RMSNorm; same formula)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.scale


# ---------------------------------------------------------------------------
# seisLM/model/foundation/modeling_outputs.py (verbatim)
# ---------------------------------------------------------------------------
@dataclass
class BaseModelOutput:
    last_hidden_state: Optional[Tensor] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None


@dataclass
class Wav2Vec2ForPreTrainingOutput(BaseModelOutput):
    loss: Optional[Tensor] = None
    projected_states: Optional[Tensor] = None
    projected_quantized_states: Optional[Tensor] = None
    codevector_perplexity: Optional[Tensor] = None
    contrastive_loss: Optional[Tensor] = None
    diversity_loss: Optional[Tensor] = None


@dataclass
class Wav2Vec2BaseModelOutput(BaseModelOutput):
    extract_features: Optional[Tensor] = None


# ---------------------------------------------------------------------------
# seisLM/model/foundation/mask_utils.py (only the two functions actually called on
# the traced forward path; verbatim)
# ---------------------------------------------------------------------------
def get_feat_extract_output_lengths(
    config: ConfigDict,
    input_lengths: Union[torch.Tensor, int],
) -> Union[torch.Tensor, int]:
    def _conv_out_length(input_length, kernel_size, stride):
        return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

    for kernel_size, stride in zip(config.conv_kernel, config.conv_stride):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

    return input_lengths


def get_feature_vector_attention_mask(
    config: ConfigDict,
    feature_vector_length: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

    output_lengths = get_feat_extract_output_lengths(config, non_padded_lengths)
    if isinstance(output_lengths, int):
        output_lengths = torch.tensor(output_lengths, dtype=torch.long)
    else:
        output_lengths = output_lengths.to(torch.long)

    batch_size = attention_mask.shape[0]

    attention_mask = torch.zeros(
        (batch_size, feature_vector_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    attention_mask[
        (
            torch.arange(attention_mask.shape[0], device=attention_mask.device),
            output_lengths - 1,
        )
    ] = 1
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return attention_mask


# ---------------------------------------------------------------------------
# seisLM/model/foundation/position_embedding.py (verbatim)
# ---------------------------------------------------------------------------
class Wav2Vec2PositionalConvEmbedding(nn.Module):
    """Use a convolutional layer, which acts as relative positional embedding."""

    def __init__(self, config: ConfigDict):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.activation = nn.functional.gelu

        self.remove_one_right = True if config.num_conv_pos_embeddings % 2 == 0 else False

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = einops.rearrange(hidden_states, "b t c -> b c t")
        hidden_states = self.conv(hidden_states)
        if self.remove_one_right:
            hidden_states = hidden_states[:, :, :-1]

        hidden_states = self.activation(hidden_states)
        hidden_states = einops.rearrange(hidden_states, "b c t -> b t c")
        return hidden_states


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------------
# seisLM/model/foundation/conv_encoder.py (verbatim)
# ---------------------------------------------------------------------------
class Wav2Vec2NoLayerNormConvLayer(nn.Module):
    def __init__(self, config: ConfigDict, layer_id: int = 0):
        super().__init__()
        self.in_conv_dim = (
            config.conv_dim[layer_id - 1] if layer_id > 0 else getattr(config, "input_dim", 1)
        )
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = nn.functional.gelu

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2LayerNormConvLayer(nn.Module):
    def __init__(self, config: ConfigDict, layer_id: int = 0):
        super().__init__()
        self.in_conv_dim = (
            config.conv_dim[layer_id - 1] if layer_id > 0 else getattr(config, "input_dim", 1)
        )
        LayerOrRMSNorm = RMSNorm if config.use_rms_norm else nn.LayerNorm
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = LayerOrRMSNorm(self.out_conv_dim)
        self.activation = nn.functional.gelu

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2GroupNormConvLayer(nn.Module):
    def __init__(self, config: ConfigDict, layer_id: int = 0):
        super().__init__()
        self.in_conv_dim = (
            config.conv_dim[layer_id - 1] if layer_id > 0 else getattr(config, "input_dim", 1)
        )
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = nn.functional.gelu
        self.layer_norm = nn.GroupNorm(
            num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config: ConfigDict):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1)
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i)
                for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm},"
                + "but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)

    def _freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_values: Tensor) -> Tensor:
        if input_values.dim() == 2:
            hidden_states = einops.rearrange(input_values, "B L -> B 1 L")
        else:
            assert input_values.dim() == 3
            hidden_states = input_values

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# seisLM/model/foundation/transformer_encoder.py (verbatim)
# ---------------------------------------------------------------------------
class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.functional.gelu
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class Wav2Vec2SdpaAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rotary_pos_embed: bool = False,
        max_seq_len: int = 3000,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.rotary_pos_embed = rotary_pos_embed

        if rotary_pos_embed:
            self.freqs_cis = precompute_freqs_cis(dim=self.head_dim, end=max_seq_len * 2)

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert output_attentions is False, "output_attentions not supported"
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self._shape(self.q_proj(hidden_states), -1, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.rotary_pos_embed:
            self.freqs_cis = self.freqs_cis.to(hidden_states.device)
            freqs_cis = self.freqs_cis[:tgt_len]

            query_states, key_states = apply_rotary_emb(
                query_states.transpose(1, 2), key_states.transpose(1, 2), freqs_cis=freqs_cis
            )

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size"
                f" {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, None


class Wav2Vec2EncoderBase(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.attention = Wav2Vec2SdpaAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            rotary_pos_embed=config.rotary_pos_embed,
        )
        LayerOrRMSNorm = RMSNorm if config.use_rms_norm else nn.LayerNorm

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = LayerOrRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = LayerOrRMSNorm(config.hidden_size, eps=config.layer_norm_eps)


class Wav2Vec2EncoderLayer(Wav2Vec2EncoderBase):
    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Optional[Tensor],
        output_attentions: bool = False,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor]]:
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2EncoderLayerStableLayerNorm(Wav2Vec2EncoderBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor]]:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.config = config

        if config.conv_embed:
            self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)

        LayerOrRMSNorm = RMSNorm if config.use_rms_norm else nn.LayerNorm

        self.layer_norm = LayerOrRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(
                1, 1, hidden_states.shape[2]
            )
            hidden_states[~expand_attention_mask] = 0
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min

            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        if self.config.conv_embed:
            position_embeddings = self.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings

        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            dropout_probability = torch.rand([])

            skip_the_layer = (
                True if self.training and (dropout_probability < self.config.layerdrop) else False
            )

            if skip_the_layer:
                layer_outputs = (None, None)
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)

        LayerOrRMSNorm = RMSNorm if config.use_rms_norm else nn.LayerNorm

        self.layer_norm = LayerOrRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(
                1, 1, hidden_states.shape[2]
            )
            hidden_states[~expand_attention_mask] = 0

            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            dropout_probability = torch.rand([])

            skip_the_layer = (
                True if self.training and (dropout_probability < self.config.layerdrop) else False
            )

            if skip_the_layer:
                layer_outputs = (None, None)
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# ---------------------------------------------------------------------------
# seisLM/model/foundation/quantizer.py (verbatim)
# ---------------------------------------------------------------------------
class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    """Vector quantization using gumbel softmax."""

    def __init__(self, config: ConfigDict):
        super().__init__()

        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group
        self.last_conv_dim = config.conv_dim[-1]

        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible "
                f"by `config.num_codevector_groups` {self.num_groups}"
                "for concatenation"
            )

        self.codevectors = nn.Parameter(
            torch.FloatTensor(
                1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups
            )
        )
        self.weight_proj = nn.Linear(
            in_features=config.conv_dim[-1], out_features=self.num_groups * self.num_vars
        )

        self.temperature = 2

        self.scale_logits_in_quantization = getattr(config, "scale_logits_in_quantization", False)

    @staticmethod
    def _compute_perplexity(probs: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            B, L = mask.shape
            N, G, V = probs.shape
            assert N == B * L

            mask_extended = einops.repeat(mask, "b l -> (b l) g v", g=G, v=V)

            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            avg_probs = einops.reduce(probs, "s g v -> g v", "sum") / mask.sum()
        else:
            avg_probs = einops.reduce(probs, "s g v -> g v", "mean")

        plogp = avg_probs * torch.log(avg_probs + 1e-7)

        perplexity = torch.exp(-einops.reduce(plogp, "g v -> g", "sum"))

        perplexity = einops.reduce(perplexity, "g ->", "sum")
        return perplexity

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask_time_indices: Optional[torch.Tensor],
        return_selected_codevector_indices: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, feature_dim = hidden_states.shape

        hidden_states = self.weight_proj(hidden_states)
        assert feature_dim == self.last_conv_dim

        if self.scale_logits_in_quantization:
            hidden_states = hidden_states / math.sqrt(self.last_conv_dim)

        hidden_states = einops.rearrange(
            hidden_states,
            "b l (g v) -> (b l g) v",
            b=batch_size,
            l=sequence_length,
            g=self.num_groups,
        )

        if self.training:
            hidden_states = einops.rearrange(
                hidden_states,
                "(b l g) v -> (b l) g v",
                b=batch_size,
                l=sequence_length,
                g=self.num_groups,
            )

            codevector_probs = gumbel_softmax(
                hidden_states.float(),
                tau=self.temperature,
                hard=True,
            ).type_as(hidden_states)

            codevector_soft_dist = torch.softmax(hidden_states.float(), dim=-1)

            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            codevector_idx = hidden_states.argmax(dim=-1, keepdim=True)

            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
                -1, codevector_idx, 1.0
            )

            codevector_probs = einops.rearrange(
                codevector_probs,
                "(b l g) v -> (b l) g v",
                b=batch_size,
                l=sequence_length,
                g=self.num_groups,
            )

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = einops.rearrange(
            codevector_probs,
            "(b l) g v -> (b l) (g v)",
            b=batch_size,
            l=sequence_length,
            g=self.num_groups,
        )

        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors

        codevectors = codevectors_per_group.view(
            batch_size * sequence_length, self.num_groups, self.num_vars, -1
        )
        codevectors = einops.reduce(
            codevectors,
            "(b l) g v k -> b l (g k)",
            "sum",
            b=batch_size,
            l=sequence_length,
            g=self.num_groups,
            v=self.num_vars,
        )

        if return_selected_codevector_indices:
            selected_codevector_indices = torch.argmax(
                einops.rearrange(
                    codevector_probs,
                    "(b l) (g v) -> b l g v",
                    b=batch_size,
                    l=sequence_length,
                    g=self.num_groups,
                ),
                dim=-1,
            )
            return codevectors, perplexity, selected_codevector_indices
        else:
            return codevectors, perplexity


# ---------------------------------------------------------------------------
# seisLM/model/foundation/multidim_wav2vec2.py (verbatim)
# ---------------------------------------------------------------------------
class Wav2Vec2FeatureProjection(nn.Module):
    """Projects the extracted features to the model's hidden size."""

    def __init__(self, config: ConfigDict):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


def init_wav2vec2_weights(*, config: ConfigDict, module: nn.Module) -> None:
    """Initialize the weights"""
    if isinstance(module, MultiDimWav2Vec2ForPreTraining):
        module.project_hid.reset_parameters()
        module.project_q.reset_parameters()
    elif isinstance(module, Wav2Vec2GumbelVectorQuantizer):
        module.weight_proj.weight.data.normal_(mean=0.0, std=1)
        module.weight_proj.bias.data.zero_()
        nn.init.uniform_(module.codevectors)
    elif isinstance(module, Wav2Vec2PositionalConvEmbedding):
        nn.init.normal_(
            module.conv.weight,
            mean=0,
            std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
        )
        module.conv.bias.data.zero_()
    elif isinstance(module, Wav2Vec2FeatureProjection):
        k = math.sqrt(1 / module.projection.in_features)
        nn.init.uniform_(module.projection.weight, a=-k, b=k)
        nn.init.uniform_(module.projection.bias, a=-k, b=k)
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)

        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight)

        if module.bias is not None:
            k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
            nn.init.uniform_(module.bias, a=-k, b=k)


class Wav2Vec2Model(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        self.apply(lambda module: init_wav2vec2_weights(config=config, module=module))

    def freeze_feature_encoder(self) -> None:
        self.feature_extractor._freeze_parameters()

    def _mask_hidden_states(
        self,
        hidden_states: Tensor,
        *,
        mask_time_indices: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # NOTE: the real repo calls mask_utils.compute_mask_indices here (a numpy
            # SpecAugment span sampler) when mask_time_indices is not supplied. That
            # helper is orthogonal to the traced architecture (pure index bookkeeping,
            # no learnable params/ops) and is intentionally omitted from this trace-only
            # module; example_input_seislm always supplies mask_time_indices explicitly,
            # so this branch is never taken during tracing.
            pass

        if self.config.mask_feature_prob > 0 and self.training:
            pass

        return hidden_states

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            attention_mask = get_feature_vector_attention_mask(
                config=self.config,
                feature_vector_length=extract_features.shape[1],
                attention_mask=attention_mask,
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states,
            mask_time_indices=mask_time_indices,
            attention_mask=attention_mask,
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = encoder_outputs.last_hidden_state

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MultiDimWav2Vec2ForPreTraining(nn.Module):
    """Wav2Vec2 model with a contrastive loss head."""

    def __init__(self, config: ConfigDict):
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)

        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

        self.apply(lambda module: init_wav2vec2_weights(config=config, module=module))

    def set_gumbel_temperature(self, temperature: int) -> None:
        self.quantizer.temperature = temperature

    def freeze_feature_encoder(self) -> None:
        self.wav2vec2.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.Tensor,
        negative_features: torch.Tensor,
        predicted_features: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = torch.cosine_similarity(
            predicted_features.float(), target_features.float(), dim=-1
        ).type_as(target_features)

        logits = logits / temperature
        return logits

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Wav2Vec2ForPreTrainingOutput:
        """Forward pass for the Wav2Vec2ForPreTraining model."""

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
        )

        transformer_features = self.project_hid(outputs.last_hidden_state)

        extract_features = self.dropout_features(outputs.extract_features)

        quantized_features, codevector_perplexity = self.quantizer(
            extract_features, mask_time_indices=mask_time_indices
        )

        quantized_features = quantized_features.to(self.project_q.weight.dtype)
        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            negative_quantized_features = quantized_features.view(-1, hidden_size)[
                sampled_negative_indices.long().view(-1)
            ]
            negative_quantized_features = negative_quantized_features.view(
                batch_size, sequence_length, -1, hidden_size
            ).permute(2, 0, 1, 3)

            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

            contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
            num_codevectors = self.config.num_codevectors_per_group * (
                self.config.num_codevector_groups
            )
            diversity_loss = (
                (num_codevectors - codevector_perplexity) / num_codevectors
            ) * mask_time_indices.sum()

            loss = contrastive_loss + (self.config.diversity_loss_weight * diversity_loss)

        outputs = Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )

        return outputs


# ---------------------------------------------------------------------------
# Tiny-scale build/example helpers (field names/values mirror the real shipped
# config seisLM/configs/pretrain/pretrain_config_std_norm_single_ax_8_datasets_
# 32bit_scaleup_samp_false.json, scaled down for a fast trace)
# ---------------------------------------------------------------------------
def _tiny_config() -> ConfigDict:
    return ConfigDict(
        activation_dropout=0.0,
        apply_spec_augment=True,
        attention_dropout=0.0,
        codevector_dim=16,
        contrastive_logits_temperature=0.1,
        conv_bias=True,
        conv_dim=[16, 16],
        conv_kernel=[3, 3],
        conv_stride=[2, 2],
        diversity_loss_weight=0.1,
        do_stable_layer_norm=True,
        feat_extract_norm="layer",
        feat_proj_dropout=0.0,
        feat_quantizer_dropout=0.0,
        hidden_dropout=0.0,
        hidden_size=32,
        initializer_range=0.02,
        input_dim=3,
        intermediate_size=64,
        layer_norm_eps=1e-5,
        layerdrop=0.0,
        mask_feature_length=4,
        mask_feature_min_masks=0,
        mask_feature_prob=0.0,
        mask_time_length=4,
        mask_time_min_masks=1,
        mask_time_prob=0.65,
        num_attention_heads=2,
        num_codevector_groups=2,
        num_codevectors_per_group=8,
        num_conv_pos_embedding_groups=2,
        num_conv_pos_embeddings=8,
        num_feat_extract_layers=2,
        num_hidden_layers=2,
        num_negatives=4,
        output_attentions=False,
        output_hidden_states=False,
        pad_token_id=0,
        proj_codevector_dim=16,
        rotary_pos_embed=False,
        conv_embed=True,
        scale_logits_in_quantization=True,
        use_rms_norm=False,
    )


def build_seislm():
    return MultiDimWav2Vec2ForPreTraining(_tiny_config())


def example_input_seislm():
    batch_size, raw_len, num_channels = 2, 64, 3
    input_values = torch.randn(batch_size, num_channels, raw_len)

    # feature-extractor downsamples raw_len by stride 2 twice (conv_kernel=3, conv_stride=2)
    feat_len = raw_len
    for k, s in zip([3, 3], [2, 2]):
        feat_len = (feat_len - k) // s + 1

    mask_time_indices = torch.zeros(batch_size, feat_len, dtype=torch.bool)
    mask_time_indices[:, 1:3] = True
    sampled_negative_indices = torch.randint(
        0, batch_size * feat_len, (batch_size, feat_len, 4), dtype=torch.long
    )

    return (input_values, None, mask_time_indices, sampled_negative_indices)


MENAGERIE_ENTRIES = [
    ("SeisLM", "build_seislm", "example_input_seislm", 2024, "vendored"),
]
