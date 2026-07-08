# SOURCE: vendored from espnet/espnet @ master
#
# CTC-Conformer (ASR): ESPnet's Conformer encoder (espnet2/asr/encoder/conformer_encoder.py)
# paired with the CTC module (espnet2/asr/ctc.py) for end-to-end CTC-based conformer ASR.
# This file vendors the REAL espnet2 nn.Module classes (unmodified architecture) plus their
# transitive base-lib-only dependency closure, flattened into one module because espnet is
# not pip-installed in this environment (heavy dependency tree) but the encoder + CTC head
# themselves depend only on torch / numpy / typeguard / logging.
#
# Files vendored verbatim (only import paths flattened to this single module; no
# architectural changes):
#   espnet2/asr/encoder/conformer_encoder.py   (ConformerEncoder)
#   espnet2/asr/encoder/abs_encoder.py         (AbsEncoder)
#   espnet2/asr/ctc.py                         (CTC)
#   espnet2/legacy/nets/pytorch_backend/conformer/convolution.py    (ConvolutionModule)
#   espnet2/legacy/nets/pytorch_backend/conformer/encoder_layer.py  (EncoderLayer)
#   espnet2/legacy/nets/pytorch_backend/conformer/swish.py          (Swish)
#   espnet2/legacy/nets/pytorch_backend/nets_utils.py   (make_pad_mask + helpers, get_activation)
#   espnet2/legacy/nets/pytorch_backend/transformer/attention.py    (MultiHeadedAttention family)
#   espnet2/legacy/nets/pytorch_backend/transformer/embedding.py    (PositionalEncoding family)
#   espnet2/legacy/nets/pytorch_backend/transformer/layer_norm.py   (LayerNorm)
#   espnet2/legacy/nets/pytorch_backend/transformer/multi_layer_conv.py
#   espnet2/legacy/nets/pytorch_backend/transformer/positionwise_feed_forward.py
#   espnet2/legacy/nets/pytorch_backend/transformer/repeat.py
#   espnet2/legacy/nets/pytorch_backend/transformer/subsampling.py  (trimmed to Conv2dSubsampling
#       + TooShortUttError + check_short_utt; other subsampling variants -- Conv1dSubsampling*,
#       Conv2dSubsampling1/2/6/8 -- are unused by the default "conv2d" input_layer config and are
#       omitted; check_short_utt's isinstance dispatch is trimmed to only the Conv2dSubsampling
#       case actually reachable in this staging module)
#   espnet2/asr/frontend/cnn.py                 (dim_1_layer_norm helper only)
#
# Ref: https://github.com/espnet/espnet/blob/master/espnet2/asr/encoder/conformer_encoder.py

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

try:  # pragma: no cover - typeguard is a base-env dep here
    import inspect as _inspect
    import sys as _sys

    from typeguard import typechecked as _typeguard_typechecked

    # typeguard's @typechecked instruments the *source module* via
    # inspect.getsource(sys.modules[fn.__module__]); that lookup fails when this
    # file is loaded ad hoc (importlib.util.spec_from_file_location without
    # registering the module in sys.modules, as the menagerie staging validator
    # does). Fall back to a no-op decorator in that case rather than raise --
    # this only affects whether argument types are runtime-checked, not the
    # architecture itself.
    def typechecked(fn):
        try:
            _inspect.getsource(_sys.modules[fn.__module__])
        except Exception:
            return fn
        return _typeguard_typechecked(fn)

except Exception:  # pragma: no cover - fallback no-op decorator

    def typechecked(fn):
        return fn


# --------------------------------------------------------------------------
# espnet2/asr/frontend/cnn.py (dim_1_layer_norm only)
# --------------------------------------------------------------------------
def dim_1_layer_norm(x, eps=1e-05, gamma=None, beta=None):
    """Functional version of Dim1LayerNorm."""
    B, D, T = x.shape
    mean = torch.mean(x, 1, keepdim=True)
    variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

    x = (x - mean) * torch.rsqrt(variance + eps)

    if gamma is not None:
        x = x * gamma.view(1, -1, 1)
        if beta is not None:
            x = x + beta.view(1, -1, 1)
    return x


# --------------------------------------------------------------------------
# espnet2/legacy/nets/pytorch_backend/transformer/layer_norm.py
# --------------------------------------------------------------------------
class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module."""

    def __init__(self, nout, dim=-1):
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(self.dim, -1)).transpose(self.dim, -1)


# --------------------------------------------------------------------------
# espnet2/legacy/nets/pytorch_backend/conformer/swish.py
# --------------------------------------------------------------------------
class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        return x * torch.sigmoid(x)


# --------------------------------------------------------------------------
# espnet2/legacy/nets/pytorch_backend/conformer/convolution.py
# --------------------------------------------------------------------------
class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        super(ConvolutionModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.activation = activation

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


# --------------------------------------------------------------------------
# espnet2/legacy/nets/pytorch_backend/transformer/positionwise_feed_forward.py
# --------------------------------------------------------------------------
class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer."""

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# --------------------------------------------------------------------------
# espnet2/legacy/nets/pytorch_backend/transformer/multi_layer_conv.py
# --------------------------------------------------------------------------
class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block."""

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans, hidden_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )
        self.w_2 = torch.nn.Conv1d(
            hidden_chans, in_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask.transpose(-1, 1), 0)
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        if mask is not None:
            x = x.masked_fill(~mask.transpose(-1, 1), 0)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class Conv1dLinear(torch.nn.Module):
    """Conv1D + Linear for Transformer block."""

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        super(Conv1dLinear, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans, hidden_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )
        self.w_2 = torch.nn.Linear(hidden_chans, in_chans)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x))


# --------------------------------------------------------------------------
# espnet2/legacy/nets/pytorch_backend/transformer/repeat.py
# --------------------------------------------------------------------------
class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def __init__(self, *args, layer_drop_rate=0.0):
        super(MultiSequential, self).__init__(*args)
        self.layer_drop_rate = layer_drop_rate

    def forward(self, *args):
        _probs = torch.empty(len(self)).uniform_()
        for idx, m in enumerate(self):
            if not self.training or (_probs[idx] >= self.layer_drop_rate):
                args = m(*args)
        return args


def repeat(N, fn, layer_drop_rate=0.0):
    """Repeat module N times."""
    return MultiSequential(*[fn(n) for n in range(N)], layer_drop_rate=layer_drop_rate)


# --------------------------------------------------------------------------
# espnet2/legacy/nets/pytorch_backend/transformer/attention.py
# --------------------------------------------------------------------------
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
except Exception as e:  # pragma: no cover - flash_attn optional upstream too
    print(f"Failed to import Flash Attention, using ESPnet default: {e}")


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer."""

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        qk_norm=False,
        use_flash_attn=False,
        causal=False,
        cross_attn=False,
        use_sdpa=False,
    ):
        super(MultiHeadedAttention, self).__init__()

        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate) if not use_flash_attn else nn.Identity()
        self.dropout_rate = dropout_rate

        self.q_norm = LayerNorm(self.d_k) if qk_norm else nn.Identity()
        self.k_norm = LayerNorm(self.d_k) if qk_norm else nn.Identity()

        self.use_flash_attn = use_flash_attn
        self.causal = causal
        self.cross_attn = cross_attn

        self.use_sdpa = use_sdpa

    def forward_qkv(self, query, key, value, expand_kv=False):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)

        if expand_kv:
            k_shape = key.shape
            k = (
                self.linear_k(key[:1, :, :])
                .expand(n_batch, k_shape[1], k_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
            v_shape = value.shape
            v = (
                self.linear_v(value[:1, :, :])
                .expand(n_batch, v_shape[1], v_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
        else:
            k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
            v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)

        return self.linear_out(x)

    def forward(self, query, key, value, mask, expand_kv=False):
        if getattr(self, "use_sdpa", False):
            q, k, v = self.forward_qkv(query, key, value, expand_kv)
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                mask.unsqueeze(1) if mask is not None else None,
                dropout_p=self.dropout_rate if self.training else 0.0,
            )
            out = out.transpose(1, 2)
            out = out.reshape(out.shape[0], out.shape[1], -1)
            return self.linear_out(out)

        if self.use_flash_attn:
            try:
                key_nonpad_mask = mask[:, -1, :]
                if self.cross_attn:
                    query_nonpad_mask = torch.ones(
                        size=query.shape[:2], dtype=torch.bool, device=query.device
                    )
                else:
                    query_nonpad_mask = key_nonpad_mask

                if key_nonpad_mask.eq(0).any():
                    q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
                        query, query_nonpad_mask
                    )[:4]
                    k, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(key, key_nonpad_mask)[:4]
                    v, _, _, _ = unpad_input(value, key_nonpad_mask)[:4]

                    q = self.linear_q(q).reshape(-1, self.h, self.d_k)
                    k = self.linear_k(k).reshape(-1, self.h, self.d_k)
                    v = self.linear_v(v).reshape(-1, self.h, self.d_k)

                    q = self.q_norm(q)
                    k = self.k_norm(k)

                    out = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        dropout_p=self.dropout_rate if self.training else 0.0,
                        causal=self.causal,
                    )

                    out = out.reshape(out.shape[0], -1)
                    out = self.linear_out(out)

                    out = pad_input(out, indices_q, query.shape[0], query.shape[1])
                    return out

                else:
                    del key_nonpad_mask
                    q, k, v = self.forward_qkv(query, key, value)

                    out = flash_attn_func(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        dropout_p=self.dropout_rate if self.training else 0.0,
                        causal=self.causal,
                    )
                    del q, k, v

                    out = out.reshape(out.shape[0], out.shape[1], -1)
                    out = self.linear_out(out)
                    return out

            except Exception as e:
                logging.warning(f"Flash Attention failed, falling back to default attention: {e}")
                self.use_flash_attn = False

        q, k, v = self.forward_qkv(query, key, value, expand_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version)."""

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation)."""

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)


# --------------------------------------------------------------------------
# espnet2/legacy/nets/pytorch_backend/transformer/embedding.py
# --------------------------------------------------------------------------
def _pre_hook(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding."""

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LegacyRelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module (old version)."""

    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len, reverse=True)

    def forward(self, x):
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation)."""

    def __init__(self, d_model, dropout_rate, max_len=5000):
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[
            :, self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1)
        ]
        return self.dropout(x), self.dropout(pos_emb)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module."""

    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(x)


class ConvolutionalPositionalEmbedding(torch.nn.Module):
    """Convolutional positional embedding. Used in wav2vec2/HuBERT SSL models."""

    def __init__(
        self,
        embed_dim: int,
        dropout: float,
        max_len: int = 5000,
        num_layers: int = 1,
        kernel_size: int = 128,
        groups: int = 16,
        weight_norm: str = "new",
        use_residual: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.weight_norm = weight_norm

        convs = []
        for layer in range(num_layers):
            conv = torch.nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups,
            )
            if weight_norm != "none" and weight_norm is not None:
                std = math.sqrt((4 * (1.0)) / (kernel_size * embed_dim))
                torch.nn.init.normal_(conv.weight, mean=0, std=std)
                torch.nn.init.constant_(conv.bias, 0)
                if weight_norm == "new":
                    conv = torch.nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)
                if weight_norm == "legacy":
                    conv = torch.nn.utils.weight_norm(conv, name="weight", dim=2)
            convs.append(conv)
        self.convs = torch.nn.ModuleList(convs)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0
        self.use_residual = use_residual

    def forward(self, x):
        if self.use_residual:
            residual = x

        x = x.transpose(-2, -1)
        for conv in self.convs:
            x = conv(x)
            if self.num_remove > 0:
                x = x[..., : -self.num_remove]
            x = torch.nn.functional.gelu(x)
            if self.weight_norm is None or self.weight_norm == "none":
                x = dim_1_layer_norm(x)
        x = x.transpose(-2, -1)

        if self.use_residual:
            x = x + residual
        return x


# --------------------------------------------------------------------------
# espnet2/legacy/nets/pytorch_backend/transformer/subsampling.py
# (trimmed to Conv2dSubsampling -- the default input_layer -- plus the
#  TooShortUttError / check_short_utt machinery that guards it)
# --------------------------------------------------------------------------
class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling."""

    def __init__(self, message, actual_size, limit):
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(ins, size):
    """Check if the utterance is too short for subsampling."""
    if isinstance(ins, Conv2dSubsampling) and size < 7:
        return True, 7
    return False, -1


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)."""

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)

    def forward(self, x, x_mask, prefix_embeds=None):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is not None:
            x_mask = x_mask[:, :, :-2:2][:, :, :-2:2]

        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
            if x_mask is not None:
                x_mask = torch.cat(
                    [
                        torch.ones(
                            x_mask.shape[0],
                            1,
                            prefix_embeds.size(1),
                            dtype=x_mask.dtype,
                            device=x_mask.device,
                        ),
                        x_mask,
                    ],
                    dim=-1,
                )

        x = self.pos_enc(x)

        return x, x_mask


# --------------------------------------------------------------------------
# espnet2/legacy/nets/pytorch_backend/nets_utils.py (mask + activation helpers)
# --------------------------------------------------------------------------
def triu_onnx(x):
    """Make TriU for ONNX."""
    arange = torch.arange(x.size(0), device=x.device)
    mask = arange.unsqueeze(-1).expand(-1, x.size(0)) <= arange
    return x * mask


def _make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None, "When maxlen is specified, xs must not be specified."
        assert maxlen >= int(max(lengths)), (
            f"maxlen {maxlen} must be >= max(lengths) {max(lengths)}"
        )

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (
            f"The size of x.size(0) {xs.size(0)} must match the batch size {bs}"
        )

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def _make_pad_mask_traceable(lengths, xs, length_dim, maxlen=None):
    """Simplified implementation of make_pad_mask that supports JIT tracing."""
    if xs is None:
        device = lengths.device
    else:
        device = xs.device

    if xs is not None and len(xs.shape) == 3:
        if length_dim == 1:
            lengths = lengths.unsqueeze(1).expand(*xs.transpose(1, 2).shape[:2])
        else:
            if length_dim not in (-1, 2):
                logging.warning(
                    f"Invalid length_dim {length_dim}."
                    + "We set it to -1, which is the default value."
                )
                length_dim = -1
            lengths = lengths.unsqueeze(1).expand(*xs.shape[:2])

    if maxlen is not None:
        assert xs is None
        assert maxlen >= lengths.max()
    elif xs is not None:
        maxlen = xs.shape[length_dim]
    else:
        maxlen = lengths.max()

    lengths = torch.clamp(lengths, max=maxlen).type(torch.long)

    mask = torch.ones(maxlen + 1, maxlen + 1, dtype=torch.bool, device=device)
    mask = triu_onnx(mask)[1:, :-1]
    mask = mask[lengths - 1][..., :maxlen]

    if xs is not None and len(xs.shape) == 3 and length_dim == 1:
        return mask.transpose(1, 2)
    else:
        return mask


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part."""
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if isinstance(lengths, list):
        logging.warning(
            "Using make_pad_mask with a list of lengths is not tracable. "
            + "If you try to trace this function with type(lengths) == list, "
            + "please change the type of lengths to torch.LongTensor."
        )

    if (
        (xs is None or xs.dim() in (2, 3))
        and length_dim <= 2
        and (not isinstance(lengths, list) and lengths.dim() == 1)
    ):
        return _make_pad_mask_traceable(lengths, xs, length_dim, maxlen)
    else:
        return _make_pad_mask(lengths, xs, length_dim, maxlen)


def trim_by_ctc_posterior(
    h: torch.Tensor,
    ctc_probs: torch.Tensor,
    masks: torch.Tensor,
    pos_emb: torch.Tensor = None,
):
    """Trim the encoder hidden output using CTC posterior."""
    frame_tolerance = 5
    conf_tolerance = 0.95
    blank_id = 0

    assert masks.size(1) == 1
    masks = masks.squeeze(1)
    hlens = masks.sum(dim=1)
    assert h.size()[:2] == ctc_probs.size()[:2]
    assert h.size(0) == hlens.size(0)

    max_values, max_indices = ctc_probs.max(dim=2)
    blank_masks = torch.logical_and(max_values > conf_tolerance, max_indices == blank_id)

    joint_masks = torch.logical_or(blank_masks, ~masks)

    B, T, _ = h.size()
    frame_idx = torch.where(joint_masks, -1, torch.arange(T).unsqueeze(0).repeat(B, 1).to(h.device))
    after_lens = torch.where(
        frame_idx.max(dim=-1)[0] + frame_tolerance + 1 < hlens,
        frame_idx.max(dim=-1)[0] + frame_tolerance + 1,
        hlens,
    )

    h = h[:, : max(after_lens)]
    masks = ~make_pad_mask(after_lens).to(h.device).unsqueeze(1)

    if pos_emb is None:
        pos_emb = None
    elif (hlens.max() * 2 - 1).item() == pos_emb.size(1):
        pos_emb = pos_emb[
            :, pos_emb.size(1) // 2 - h.size(1) + 1 : pos_emb.size(1) // 2 + h.size(1)
        ]
    else:
        pos_emb = pos_emb[:, : h.size(1)]

    return h, masks, pos_emb


def get_activation(act):
    """Return activation function."""
    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
    }
    return activation_funcs[act]()


# --------------------------------------------------------------------------
# espnet2/legacy/nets/pytorch_backend/conformer/encoder_layer.py
# --------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """Encoder layer module."""

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)
        self.norm_mha = LayerNorm(size)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)
            self.norm_final = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x_input, mask, cache=None):
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        skip_layer = False
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


# --------------------------------------------------------------------------
# espnet2/asr/ctc.py
# --------------------------------------------------------------------------
class CTC(torch.nn.Module):
    """CTC module."""

    @typechecked
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: Optional[bool] = None,
        zero_infinity: bool = True,
        brctc_risk_strategy: str = "exp",
        brctc_group_strategy: str = "end",
        brctc_risk_factor: float = 0.0,
    ):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        if ignore_nan_grad is not None:
            zero_infinity = ignore_nan_grad

        if self.ctc_type == "builtin":
            self.ctc_loss = torch.nn.CTCLoss(reduction="none", zero_infinity=zero_infinity)
        elif self.ctc_type == "builtin2":
            self.ignore_nan_grad = True
            logging.warning("builtin2")
            self.ctc_loss = torch.nn.CTCLoss(reduction="none")
        elif self.ctc_type == "gtnctc":
            from espnet2.legacy.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction

            self.ctc_loss = GTNCTCLossFunction.apply
        elif self.ctc_type == "brctc":
            try:
                import k2  # noqa
            except ImportError:
                raise ImportError("You should install K2 to use Bayes Risk CTC")

            from espnet2.asr.bayes_risk_ctc import BayesRiskCTC

            self.ctc_loss = BayesRiskCTC(
                brctc_risk_strategy, brctc_group_strategy, brctc_risk_factor
            )
        else:
            raise ValueError(f'ctc_type must be "builtin" or "gtnctc": {self.ctc_type}')

        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        if self.ctc_type == "builtin" or self.ctc_type == "brctc":
            th_pred = th_pred.log_softmax(2).float()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            if self.ctc_type == "builtin":
                size = th_pred.size(1)
            else:
                size = loss.size(0)

            if self.reduce:
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "builtin2":
            th_pred = th_pred.log_softmax(2).float()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

            if loss.requires_grad and self.ignore_nan_grad:
                ctc_grad = loss.grad_fn(torch.ones_like(loss))
                ctc_grad = ctc_grad.sum([0, 2])
                indices = torch.isfinite(ctc_grad)
                size = indices.long().sum()
                if size == 0:
                    logging.warning(
                        "All samples in this mini-batch got nan grad."
                        " Returning nan value instead of CTC loss"
                    )
                    return loss
                elif size != th_pred.size(1):
                    logging.warning(
                        f"{th_pred.size(1) - size}/{th_pred.size(1)}"
                        " samples got nan grad."
                        " These were ignored for CTC loss."
                    )

                    target_mask = torch.full(
                        [th_target.size(0)], 1, dtype=torch.bool, device=th_target.device
                    )
                    s = 0
                    for ind, le in enumerate(th_olen):
                        if not indices[ind]:
                            target_mask[s : s + le] = 0
                        s += le

                    loss = self.ctc_loss(
                        th_pred[:, indices, :],
                        th_target[target_mask],
                        th_ilen[indices],
                        th_olen[indices],
                    )
            else:
                size = th_pred.size(1)

            if self.reduce:
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate CTC loss."""
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

        if self.ctc_type == "brctc":
            loss = self.loss_fn(ys_hat, ys_pad, hlens, ys_lens).to(
                device=hs_pad.device, dtype=hs_pad.dtype
            )
            return loss

        elif self.ctc_type == "gtnctc":
            ys_true = [y[y != -1] for y in ys_pad]
        else:
            ys_hat = ys_hat.transpose(0, 1)
            ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])  # noqa: E741

        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens).to(
            device=hs_pad.device, dtype=hs_pad.dtype
        )

        return loss

    def softmax(self, hs_pad):
        """softmax of frame activations"""
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations"""
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations"""
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


# --------------------------------------------------------------------------
# espnet2/asr/encoder/abs_encoder.py
# --------------------------------------------------------------------------
class AbsEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


# --------------------------------------------------------------------------
# espnet2/asr/encoder/conformer_encoder.py
# --------------------------------------------------------------------------
class ConformerEncoder(AbsEncoder):
    """Conformer encoder module."""

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        ctc_trim: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        qk_norm: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "conv":
            pos_enc_class = ConvolutionalPositionalEmbedding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning("Using legacy_rel_pos and it will be deprecated in the future.")
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (output_size, linear_units, dropout_rate, activation)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            if use_flash_attn:
                try:
                    use_flash_attn = False  # flash_attn not installed in this env
                    import flash_attn  # noqa
                except Exception:
                    use_flash_attn = False

            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                qk_norm,
                use_flash_attn,
                False,
                False,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads, output_size, attention_dropout_rate)
            logging.warning("Using legacy_rel_selfattn and it will be deprecated in the future.")
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks

        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate[lnum],
            ),
            layer_drop_rate,
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        self.ctc_trim = ctc_trim

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        masks: torch.Tensor = None,
        ctc: CTC = None,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation."""
        if masks is None:
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        else:
            masks = ~masks[:, None, :]

        if isinstance(self.embed, Conv2dSubsampling):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)
                if return_all_hs:
                    if isinstance(xs_pad, tuple):
                        intermediate_outs.append(xs_pad[0])
                    else:
                        intermediate_outs.append(xs_pad)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x = x + self.conditioning_layer(ctc_out)
                            xs_pad = (x, pos_emb)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out)

                    if self.ctc_trim and ctc is not None:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x, masks, pos_emb = trim_by_ctc_posterior(x, ctc_out, masks, pos_emb)
                            xs_pad = (x, pos_emb)
                        else:
                            x, masks, _ = trim_by_ctc_posterior(x, ctc_out, masks)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None


# --------------------------------------------------------------------------
# Staging wrapper: real ConformerEncoder + real CTC head, wired for a single
# fixed-shape example input so torchlens can trace a plain forward() call.
# --------------------------------------------------------------------------
class ConformerCTCASR(torch.nn.Module):
    """Real ESPnet ConformerEncoder + real CTC head wired for one fixed input.

    Mirrors how espnet2's ASRModel composes encoder(xs_pad, ilens) -> CTC(hs_pad, ...)
    but reduced to a plain forward(feats) -> ctc log-probs signature so it traces
    as an ordinary eager nn.Module.
    """

    def __init__(self, input_size=40, output_size=64, vocab_size=32, num_blocks=2):
        super().__init__()
        self.encoder = ConformerEncoder(
            input_size=input_size,
            output_size=output_size,
            attention_heads=4,
            linear_units=128,
            num_blocks=num_blocks,
            input_layer="conv2d",
            macaron_style=True,
            use_cnn_module=True,
            cnn_module_kernel=15,
            use_flash_attn=False,
        )
        self.ctc = CTC(odim=vocab_size, encoder_output_size=output_size)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        ilens = torch.full((feats.size(0),), feats.size(1), dtype=torch.long)
        hs_pad, hlens, _ = self.encoder(feats, ilens)
        return self.ctc.log_softmax(hs_pad)


def build_ctc_conformer():
    return ConformerCTCASR(input_size=40, output_size=64, vocab_size=32, num_blocks=2)


def example_input_ctc_conformer():
    return torch.randn(1, 50, 40)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "CTC-Conformer (ASR)",
        "build_ctc_conformer",
        "example_input_ctc_conformer",
        2020,
        "vendored",
    ),
]
