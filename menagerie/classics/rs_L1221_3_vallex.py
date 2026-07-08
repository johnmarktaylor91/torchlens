# SOURCE: vendored from https://github.com/Plachtaa/VALL-E-X @ master
# Files: models/vallex.py, models/transformer.py (upstream-credited as
# https://github.com/lucidrains/... no -- see header there: this repo's own
# GPT-style decoder), modules/transformer.py, modules/activation.py,
# modules/embedding.py, modules/scaling.py (only the classes actually
# reachable from the VALLE forward/continual path are kept; unreferenced
# scaling helpers -- SRLinear/SRConv1d/Whiten/MaxEig/RandomGrad/etc. that
# VALLE never instantiates on its default construction path -- are omitted).
# Only import/name changes made: relative/package imports flattened into
# this single module; architecture and forward logic are the upstream code.
"""VALL-E (Plachtaa/VALL-E-X open reimplementation of Microsoft's VALL-E).

"Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
(https://arxiv.org/abs/2301.02111). VALL-E treats zero-shot TTS as a
conditional codec-language-modeling problem: an autoregressive (AR)
Transformer predicts the first EnCodec RVQ codebook, and a non-autoregressive
(NAR) Transformer with adaptive layer norm predicts the remaining 7
codebooks conditioned on a per-stage embedding.

We trace ``VALLE.continual`` -- the one fully deterministic forward path in
this repo (fixed ``num_quantizers == 8``, plain Python for-loops, no
autoregressive ``while True`` sampling loop) -- via a thin wrapper module,
which exercises the real AR text/audio embeddings, both Transformer stacks
(AR TransformerEncoder + NAR AdaptiveLayerNorm TransformerEncoder), and all
NAR predict/embedding heads.
"""

import copy
import math
import numbers
import random
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Linear, Module
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"

NUM_TEXT_TOKENS = 2048
NUM_AUDIO_TOKENS = 1024  # EnCodec RVQ bins

# ---------------------------------------------------------------------------
# modules/embedding.py
# ---------------------------------------------------------------------------


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_model)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def forward(self, x: torch.Tensor):
        X = self.word_embeddings(x)
        X = self.dropout(X)

        return X


class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.dim_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(output)


# ---------------------------------------------------------------------------
# modules/scaling.py (only the classes reachable from VALLE's default
# construction path; VALLE never sets layer_norm_cls=IdentityNorm so
# ActivationBalancer/BalancedBasicNorm/Whiten/MaxEig/etc. are never
# instantiated -- omitted as dead weight for this staged trace)
# ---------------------------------------------------------------------------


class BasicNorm(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,
        eps: float = 0.25,
        learn_eps: bool = True,
        eps_min: float = -3.0,
        eps_max: float = 3.0,
    ) -> None:
        super(BasicNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        if learn_eps:
            self.eps = nn.Parameter(torch.tensor(eps).log().detach())
        else:
            self.register_buffer("eps", torch.tensor(eps).log().detach())
        self.eps_min = eps_min
        self.eps_max = eps_max

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels
        eps = self.eps
        scales = (torch.mean(x**2, dim=self.channel_dim, keepdim=True) + eps.exp()) ** -0.5
        return x * scales


class ActivationBalancer(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        min_positive: float = 0.05,
        max_positive: float = 0.95,
        max_factor: float = 0.04,
        sign_gain_factor: float = 0.01,
        scale_gain_factor: float = 0.02,
        min_abs: float = 0.2,
        max_abs: float = 100.0,
        min_prob: float = 0.1,
    ):
        super(ActivationBalancer, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.min_prob = min_prob
        self.sign_gain_factor = sign_gain_factor
        self.scale_gain_factor = scale_gain_factor
        self.cpu_count = 0
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: Tensor) -> Tensor:
        # VALLE's default construction path never instantiates this module
        # (layer_norm_cls stays LayerNorm), so this is unreachable on the
        # traced continual() path; upstream's fast-path branch (no_grad /
        # scripting / tracing -> identity) is kept, the ActivationBalancer
        # Function/_compute_* backprop-only machinery is not needed.
        if torch.jit.is_scripting() or not x.requires_grad or torch.jit.is_tracing():
            return x
        self.cpu_count += 1
        if random.random() < 0.01:
            self.cpu_count = max(self.cpu_count, self.count.item())
            self.count.fill_(self.cpu_count)
        return x


class DoubleSwish(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x - 1.0)


def BalancedDoubleSwish(d_model, channel_dim=-1, max_abs=10.0, min_prob=0.25) -> nn.Sequential:
    """ActivationBalancer -> DoubleSwish"""
    balancer = ActivationBalancer(
        d_model, channel_dim=channel_dim, max_abs=max_abs, min_prob=min_prob
    )
    return nn.Sequential(
        balancer,
        DoubleSwish(),
    )


# ---------------------------------------------------------------------------
# modules/activation.py
# ---------------------------------------------------------------------------


def multi_head_attention_forward(
    x,
    ipw,
    ipb,
    opw,
    opb,
    n_head,
    attn_mask,
    past_kv=None,
    use_cache=False,
):
    B, T, C = x.size()

    q, k, v = torch._C._nn.linear(x, ipw, ipb).chunk(3, dim=-1)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    if past_kv is not None:
        past_key = past_kv[0]
        past_value = past_kv[1]
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)

    FULL_T = k.shape[-2]

    if use_cache is True:
        present = (k, v)
    else:
        present = None

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(attn_mask[FULL_T - T : FULL_T, :FULL_T], float("-inf"))
    att = F.softmax(att, dim=-1)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = torch._C._nn.linear(y, opw, opb)
    return (y, present)


class MultiheadAttention(Module):
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        linear1_cls=Linear,
        linear2_cls=Linear,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        if linear1_cls == Linear:
            if not self._qkv_same_embed_dim:
                self.q_proj_weight = Parameter(
                    torch.empty((embed_dim, embed_dim), **factory_kwargs)
                )
                self.k_proj_weight = Parameter(
                    torch.empty((embed_dim, self.kdim), **factory_kwargs)
                )
                self.v_proj_weight = Parameter(
                    torch.empty((embed_dim, self.vdim), **factory_kwargs)
                )
                self.register_parameter("in_proj_weight", None)
            else:
                self.in_proj_weight = Parameter(
                    torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
                )
                self.register_parameter("q_proj_weight", None)
                self.register_parameter("k_proj_weight", None)
                self.register_parameter("v_proj_weight", None)

            if bias:
                self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
            else:
                self.register_parameter("in_proj_bias", None)
            self.out_proj = NonDynamicallyQuantizableLinear(
                embed_dim, embed_dim, bias=bias, **factory_kwargs
            )

            self._reset_parameters()
        else:
            if not self._qkv_same_embed_dim:
                raise NotImplementedError
            else:
                self.in_proj_linear = linear1_cls(
                    embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs
                )
                self.in_proj_weight = self.in_proj_linear.weight

                self.register_parameter("q_proj_weight", None)
                self.register_parameter("k_proj_weight", None)
                self.register_parameter("v_proj_weight", None)

                if bias:
                    self.in_proj_bias = self.in_proj_linear.bias
                else:
                    self.register_parameter("in_proj_bias", None)

            self.out_proj = linear2_cls(embed_dim, embed_dim, bias=bias, **factory_kwargs)

            if self.bias_k is not None:
                xavier_normal_(self.bias_k)
            if self.bias_v is not None:
                xavier_normal_(self.bias_v)

        self.add_zero_attn = add_zero_attn

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def infer(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        past_kv=None,
        use_cache=False,
    ):
        y, kv = multi_head_attention_forward(
            x=x,
            ipw=self.in_proj_weight,
            ipb=self.in_proj_bias,
            opw=self.out_proj.weight,
            opb=self.out_proj.bias,
            n_head=self.num_heads,
            attn_mask=attn_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        return (y, kv)


# ---------------------------------------------------------------------------
# modules/transformer.py
# ---------------------------------------------------------------------------

_shape_t = Union[int, List[int], torch.Size]


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return (
                F.layer_norm(
                    input,
                    self.normalized_shape,
                    self.weight,
                    self.bias,
                    self.eps,
                ),
                embedding,
            )

        assert embedding is None
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            weight, bias = torch.split(
                self.project_layer(embedding),
                split_size_or_sections=self.d_model,
                dim=-1,
            )
            return (weight * self.norm(input) + bias, embedding)

        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias


class TransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        linear1_self_attention_cls: nn.Module = nn.Linear,
        linear2_self_attention_cls: nn.Module = nn.Linear,
        linear1_feedforward_cls: nn.Module = nn.Linear,
        linear2_feedforward_cls: nn.Module = nn.Linear,
        layer_norm_cls: nn.Module = LayerNorm,
        layer_norm_eps: float = 1e-5,
        adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )

        self.linear1 = linear1_feedforward_cls(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            activation = BalancedDoubleSwish(d_model)

        self.activation = activation

        norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
        norm2 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)

        if adaptive_layer_norm:
            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
        else:
            self.norm1 = norm1
            self.norm2 = norm2

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding),
                src_mask,
                src_key_padding_mask,
            )
            x = x + self._ff_block(self.norm2(x, stage_embedding))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask),
                stage_embedding,
            )
            x = self.norm2(x + self._ff_block(x), stage_embedding)

        if is_src_tuple:
            return (x, stage_embedding)
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_layer_states: bool = False,
    ) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# ---------------------------------------------------------------------------
# models/vallex.py
# ---------------------------------------------------------------------------


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)


class VALLF(nn.Module):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        decoder_cls: Union[nn.TransformerDecoder, nn.TransformerEncoder] = nn.TransformerDecoder,
        decoder_layer_cls: Union[Any, TransformerEncoderLayer] = TransformerEncoderLayer,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        prepend_bos: bool = True,
        num_quantizers: int = 8,
    ):
        super().__init__()
        nar_d_model = int(d_model * nar_scale_factor)

        self.ar_text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)  # W_x
        self.nar_text_embedding = TokenEmbedding(nar_d_model, NUM_TEXT_TOKENS)

        self.ar_audio_prepend_bos = prepend_bos
        self.ar_audio_embedding = TokenEmbedding(d_model, NUM_AUDIO_TOKENS + 1 + int(prepend_bos))

        if add_prenet:
            self.ar_text_prenet = nn.Sequential(
                Transpose(),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                Transpose(),
                nn.Linear(d_model, d_model),
            )

            self.ar_audio_prenet = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, d_model),
            )
        else:
            self.ar_text_prenet = nn.Identity()
            self.ar_audio_prenet = nn.Identity()

        self.ar_text_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
            alpha=True,
        )

        self.ar_decoder = decoder_cls(
            decoder_layer_cls(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )
        self.ar_predict_layer = nn.Linear(d_model, NUM_AUDIO_TOKENS + 1, bias=False)

        self.num_heads = nhead
        self.prefix_mode = prefix_mode
        self.num_quantizers = num_quantizers

        assert num_quantizers >= 1
        if num_quantizers > 1:
            self.nar_audio_embeddings = nn.ModuleList(
                [TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS + 1)]
                + [TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS) for i in range(num_quantizers - 1)]
            )  # W_a

            if add_prenet:
                self.nar_text_prenet = nn.Sequential(
                    Transpose(),
                    nn.Conv1d(nar_d_model, nar_d_model, kernel_size=5, padding="same"),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(nar_d_model, nar_d_model, kernel_size=5, padding="same"),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(nar_d_model, nar_d_model, kernel_size=5, padding="same"),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    Transpose(),
                    nn.Linear(nar_d_model, nar_d_model),
                )
                self.nar_audio_prenet = nn.Sequential(
                    nn.Linear(nar_d_model, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, nar_d_model),
                )
            else:
                self.nar_text_prenet = nn.Identity()
                self.nar_audio_prenet = nn.Identity()

            self.nar_text_position = SinePositionalEmbedding(
                nar_d_model,
                dropout=0.0,
                scale=False,
                alpha=False,
            )
            self.nar_audio_position = SinePositionalEmbedding(
                nar_d_model,
                dropout=0.1,
                scale=False,
                alpha=False,
            )

            self.nar_decoder = decoder_cls(
                decoder_layer_cls(
                    nar_d_model,
                    int(nhead * nar_scale_factor),
                    dim_feedforward=nar_d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=norm_first,
                    adaptive_layer_norm=True,
                ),
                num_layers=int(num_layers * nar_scale_factor),
                norm=AdaptiveLayerNorm(nar_d_model, norm=nn.LayerNorm(nar_d_model))
                if norm_first
                else None,
            )
            self.nar_predict_layers = nn.ModuleList(
                [
                    nn.Linear(nar_d_model, NUM_AUDIO_TOKENS, bias=False)
                    for i in range(num_quantizers - 1)
                ]
            )
            self.nar_stage_embeddings = nn.ModuleList(
                [TokenEmbedding(nar_d_model, 1) for i in range(num_quantizers - 1)]
            )

            if share_embedding:
                for j in range(0, num_quantizers - 2):
                    self.nar_predict_layers[j].weight = self.nar_audio_embeddings[j + 2].weight


class VALLE(VALLF):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        **kwargs,
    ):
        super(VALLE, self).__init__(
            d_model,
            nhead,
            num_layers,
            norm_first=norm_first,
            add_prenet=add_prenet,
            decoder_cls=TransformerEncoder,
            decoder_layer_cls=TransformerEncoderLayer,
            prefix_mode=prefix_mode,
            share_embedding=share_embedding,
            nar_scale_factor=nar_scale_factor,
            **kwargs,
        )
        self.language_ID = {
            "en": 0,
            "zh": 1,
            "ja": 2,
        }
        self.ar_language_embedding = TokenEmbedding(d_model, len(self.language_ID))
        self.nar_language_embedding = TokenEmbedding(d_model, len(self.language_ID))

    def continual(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x: A 2-D tensor of shape (1, S).
          x_lens: A 1-D tensor of shape (1,).
          y: A 3-D tensor of shape (1, T, 8) -- all 8 EnCodec RVQ codebooks.
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)
        assert self.num_quantizers == 8

        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()

        prefix_len = min(int(y.shape[1] * 0.5), 3 * 75)

        prompts = y[:, :prefix_len]

        codes = [y[:, prefix_len:, 0]]
        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        y_emb = self.nar_audio_embeddings[0](y[..., 0])

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_position(y_emb)
                y_pos = self.nar_audio_prenet(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder((xy_pos, self.nar_stage_embeddings[i].weight))
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < 6:
                    y_emb[:, :prefix_len] += embedding_layer(prompts[..., i + 1])
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, 8):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](prompts[..., j])

            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder((xy_pos, self.nar_stage_embeddings[i].weight))
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < 6:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == 8
        return torch.stack(codes, dim=-1)


# ---------------------------------------------------------------------------
# menagerie staging entrypoints
# ---------------------------------------------------------------------------


class VALLEContinualWrapper(nn.Module):
    """Thin forward()-shim over VALLE.continual (the deterministic path)."""

    def __init__(self, vallex_model: VALLE):
        super().__init__()
        self.model = vallex_model

    def forward(self, x, x_lens, y):
        return self.model.continual(x, x_lens, y)


def build_vallex():
    # Tiny d_model/nhead/num_layers for a fast trace; architecture unchanged.
    vallex = VALLE(d_model=16, nhead=4, num_layers=2, norm_first=True, add_prenet=False)
    return VALLEContinualWrapper(vallex)


def example_input_vallex():
    torch.manual_seed(0)
    text_len = 4
    audio_len = 12  # must exceed prefix_len = min(0.5*audio_len, 225)
    x = torch.randint(0, NUM_TEXT_TOKENS, (1, text_len))
    x_lens = torch.tensor([text_len])
    y = torch.randint(0, NUM_AUDIO_TOKENS, (1, audio_len, 8))
    return (x, x_lens, y)


MENAGERIE_ENTRIES = [
    ("VALL-E (VALL-E-X)", "build_vallex", "example_input_vallex", 2023, MENAGERIE_ZOO),
]
