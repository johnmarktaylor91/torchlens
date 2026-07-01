# SOURCE: vendored from Khrylx/AgentFormer @ main
#
# https://github.com/Khrylx/AgentFormer
# https://raw.githubusercontent.com/Khrylx/AgentFormer/main/model/agentformer.py
# https://raw.githubusercontent.com/Khrylx/AgentFormer/main/model/agentformer_lib.py
# https://raw.githubusercontent.com/Khrylx/AgentFormer/main/model/common/mlp.py
# https://raw.githubusercontent.com/Khrylx/AgentFormer/main/model/common/dist.py
# https://raw.githubusercontent.com/Khrylx/AgentFormer/main/utils/torch.py (rotation_2d_torch, ExpParamAnnealer only)
# https://raw.githubusercontent.com/Khrylx/AgentFormer/main/utils/utils.py (initialize_weights only)
# https://raw.githubusercontent.com/Khrylx/AgentFormer/main/cfg/eth_ucy/eth/eth_agentformer_pre.yml
#
# AgentFormer ("AgentFormer: Agent-Aware Transformers for Socio-Temporal
# Multi-Agent Forecasting", Yuan et al. 2021, ICCV). This vendors the real
# `AgentFormer`/`ContextEncoder`/`FutureEncoder`/`FutureDecoder`/
# `PositionalAgentEncoding` classes from `model/agentformer.py`, and the
# real `AgentAwareAttention`/`agent_aware_attention`/`AgentFormerEncoderLayer`/
# `AgentFormerDecoderLayer`/`AgentFormerEncoder`/`AgentFormerDecoder` classes
# from `model/agentformer_lib.py` -- the paper's core contribution, a
# modified multi-head-attention that additionally computes a *self*-attention
# term (`in_proj_weight_self`/`in_proj_bias_self`) blended in via an
# identity-agent mask (`attn_weight_self_mask`) so each agent can attend to
# its own past states differently from other agents' -- copied verbatim
# (only whitespace-preserving copy, no architecture changes) from the real
# `model/agentformer_lib.py`. Also vendors the real `MLP` class from
# `model/common/mlp.py` and the real `Normal`/`Categorical` VAE-latent
# distribution classes from `model/common/dist.py`, and the real
# `rotation_2d_torch` / `ExpParamAnnealer` / `initialize_weights` helper
# functions from `utils/torch.py` / `utils/utils.py`.
#
# Fixed for current torch (import-compat only, no architecture change):
# `torch.nn.modules.linear._LinearWithBias` was removed from torch (it was
# historically defined as exactly `class _LinearWithBias(Linear): def
# __init__(self, in_features, out_features): super().__init__(in_features,
# out_features, bias=True)`, i.e. `Linear` with `bias` forced `True`); every
# `_LinearWithBias(embed_dim, embed_dim)` call site here uses plain
# `nn.Linear(embed_dim, embed_dim, bias=True)` instead, which is identical.
#
# Dropped from the vendor: `model/agentformer_loss.py`'s `loss_func` dict
# (only used by `AgentFormer.compute_loss()`, a training-only method never
# called from `AgentFormer.forward()`); `model/map_encoder.py`/
# `model/map_cnn.py` (the optional map-conditioning branch, gated by
# `cfg.get('use_map', False)`; the real ETH/UCY `eth_agentformer_pre.yml`
# reference config -- which this vendor's `build_agentformer()` hyperparameters
# are transcribed from -- never sets `use_map`, so it defaults `False` and
# `self.map_encoder` is never constructed in that real configuration);
# `AgentFormer.set_data`/`.step_annealer`/`.compute_loss`/`.inference`
# (dataset-preprocessing, annealing-schedule bookkeeping, and inference-time
# convenience wrappers -- not architecture; `example_input_agentformer`
# below constructs the same `data` dict structure `set_data` would produce
# from raw per-agent trajectory lists, and `AgentFormer.forward()` -- the
# real training-mode forward the vendored `forward` calls unmodified --
# is used directly); the `easydict`/`yaml`-backed real `utils/config.py`
# `Config` class (file-loading config plumbing, not architecture) -- replaced
# below with a minimal `_CfgDict` that reproduces `EasyDict`'s two behaviors
# the real model code actually uses (`cfg.attr` attribute access and
# `cfg.get(name, default)`), populated with the real ETH/UCY
# `eth_agentformer_pre.yml` hyperparameters (`tf_model_dim=256`,
# `tf_ff_dim=512`, `tf_nhead=8`, `nlayer=2`, `nz=32`, `learn_prior=true`,
# `pos_concat=true`, `input_type=['scene_norm','vel']`,
# `pred_type='scene_norm'`, `sn_out_type='norm'`, shrunk only where noted
# below) rather than invented values.
#
# `model/agentformer.py`'s real `AgentFormer.__init__` also unconditionally
# reads `cfg.z_tau.{start,finish,decay}` when `z_type == 'discrete'`; the
# vendored `cfg` keeps the real default `z_type: 'gaussian'` (the ETH/UCY
# config never sets `z_type`, and `FutureEncoder`/`FutureDecoder`'s own
# `cfg.get('z_type', 'gaussian')`-style access confirms 'gaussian' is the
# real fallback), so that branch (and its `ExpParamAnnealer`, still vendored
# below for completeness) is not exercised by `example_input_agentformer`.

import copy
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, distributions as td
from torch.nn import functional as F
from torch.nn.functional import (
    dropout,
    linear,
    multi_head_attention_forward,
    pad,
    softmax,
)  # real agentformer_lib.py: `from torch.nn.functional import *`
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from torch.nn.modules.normalization import LayerNorm
from torch.nn.parameter import Parameter
from torch.overrides import handle_torch_function, has_torch_function

# ---------------------------------------------------------------------------
# utils/torch.py (rotation_2d_torch, ExpParamAnnealer only)
# ---------------------------------------------------------------------------


def rotation_2d_torch(x, theta, origin=None):
    if origin is None:
        origin = torch.zeros(2).to(x.device).to(x.dtype)
    norm_x = x - origin
    norm_rot_x = torch.zeros_like(x)
    norm_rot_x[..., 0] = norm_x[..., 0] * torch.cos(theta) - norm_x[..., 1] * torch.sin(theta)
    norm_rot_x[..., 1] = norm_x[..., 0] * torch.sin(theta) + norm_x[..., 1] * torch.cos(theta)
    rot_x = norm_rot_x + origin
    return rot_x, norm_rot_x


class ExpParamAnnealer(nn.Module):
    def __init__(self, start, finish, rate, cur_epoch=0):
        super().__init__()
        self.register_buffer("start", torch.tensor(start))
        self.register_buffer("finish", torch.tensor(finish))
        self.register_buffer("rate", torch.tensor(rate))
        self.register_buffer("cur_epoch", torch.tensor(cur_epoch))

    def step(self):
        self.cur_epoch += 1

    def set_epoch(self, epoch):
        self.cur_epoch.fill_(epoch)

    def val(self):
        return self.finish - (self.finish - self.start) * (self.rate**self.cur_epoch)


# ---------------------------------------------------------------------------
# utils/utils.py (initialize_weights only)
# ---------------------------------------------------------------------------


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# model/common/mlp.py
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation="tanh"):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        initialize_weights(self.affine_layers.modules())

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


# ---------------------------------------------------------------------------
# model/common/dist.py
# ---------------------------------------------------------------------------


class Normal:
    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)

    def rsample(self):
        eps = torch.randn_like(self.sigma)
        return self.mu + eps * self.sigma

    def sample(self):
        return self.rsample()

    def kl(self, p=None):
        """compute KL(q||p)"""
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl

    def mode(self):
        return self.mu


class Categorical:
    def __init__(self, probs=None, logits=None, temp=0.01):
        super().__init__()
        self.logits = logits
        self.temp = temp
        if probs is not None:
            self.probs = probs
        else:
            assert logits is not None
            self.probs = torch.softmax(logits, dim=-1)
        self.dist = td.OneHotCategorical(self.probs)

    def rsample(self):
        relatex_dist = td.RelaxedOneHotCategorical(self.temp, self.probs)
        return relatex_dist.rsample()

    def sample(self):
        return self.dist.sample()

    def kl(self, p=None):
        """compute KL(q||p)"""
        if p is None:
            p = Categorical(logits=torch.zeros_like(self.probs))
        kl = td.kl_divergence(self.dist, p.dist)
        return kl

    def mode(self):
        argmax = self.probs.argmax(dim=-1)
        one_hot = torch.zeros_like(self.probs)
        one_hot.scatter_(1, argmax.unsqueeze(1), 1)
        return one_hot


# ---------------------------------------------------------------------------
# model/agentformer_lib.py
# "Modified version of PyTorch Transformer module for the implementation of
# Agent-Aware Attention"
# ---------------------------------------------------------------------------


def agent_aware_attention(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training=True,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None,
    gaussian_kernel=True,
    num_agent=1,
    in_proj_weight_self=None,
    in_proj_bias_self=None,
):
    if not torch.jit.is_scripting():
        tens_ops = (
            query,
            key,
            value,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            out_proj_weight,
            out_proj_bias,
        )
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                multi_head_attention_forward,
                tens_ops,
                query,
                key,
                value,
                embed_dim_to_check,
                num_heads,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                add_zero_attn,
                dropout_p,
                out_proj_weight,
                out_proj_bias,
                training=training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight,
                static_k=static_k,
                static_v=static_v,
            )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
            if in_proj_weight_self is not None:
                q_self, k_self = linear(query, in_proj_weight_self, in_proj_bias_self).chunk(
                    2, dim=-1
                )

        elif torch.equal(key, value):
            # encoder-decoder attention
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

            if in_proj_weight_self is not None:
                _w = in_proj_weight_self[:embed_dim, :]
                _b = in_proj_bias_self[:embed_dim]
                q_self = linear(query, _w, _b)

                _w = in_proj_weight_self[embed_dim:, :]
                _b = in_proj_bias_self[embed_dim:]
                k_self = linear(key, _w, _b)

        else:
            raise NotImplementedError

    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    if not gaussian_kernel:
        q = q * scaling  # remove scaling
        if in_proj_weight_self is not None:
            q_self = q_self * scaling  # remove scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
            attn_mask.dtype
        )
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if in_proj_weight_self is not None:
        q_self = q_self.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k_self = k_self.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    if gaussian_kernel:
        qk = torch.bmm(q, k.transpose(1, 2))
        q_n = q.pow(2).sum(dim=-1).unsqueeze(-1)
        k_n = k.pow(2).sum(dim=-1).unsqueeze(1)
        qk_dist = q_n + k_n - 2 * qk
        attn_output_weights = qk_dist * scaling * 0.5
    else:
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if in_proj_weight_self is not None:
        # ==================================
        #     Agent-Aware Attention
        # ==================================
        attn_output_weights_inter = attn_output_weights
        attn_weight_self_mask = torch.eye(num_agent).to(q.device)
        attn_weight_self_mask = attn_weight_self_mask.repeat(
            [attn_output_weights.shape[1] // num_agent, attn_output_weights.shape[2] // num_agent]
        ).unsqueeze(0)
        attn_output_weights_self = torch.bmm(q_self, k_self.transpose(1, 2))

        attn_output_weights = (
            attn_output_weights_inter * (1 - attn_weight_self_mask)
            + attn_output_weights_self * attn_weight_self_mask
        )
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        attn_output_weights = softmax(attn_output_weights, dim=-1)
    else:
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = softmax(attn_output_weights, dim=-1)

    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class AgentAwareAttention(Module):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        cfg,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.gaussian_kernel = self.cfg.get("gaussian_kernel", False)
        self.sep_attn = self.cfg.get("sep_attn", True)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=True
        )  # real code: _LinearWithBias(embed_dim, embed_dim), see header note

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        if self.sep_attn:
            self.in_proj_weight_self = Parameter(torch.empty(2 * embed_dim, embed_dim))
            self.in_proj_bias_self = Parameter(torch.empty(2 * embed_dim))
        else:
            self.in_proj_weight_self = self.in_proj_bias_self = None

        self._reset_parameters()

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

        if self.sep_attn:
            xavier_uniform_(self.in_proj_weight_self)
            constant_(self.in_proj_bias_self, 0.0)

    def __setstate__(self, state):
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True
        super().__setstate__(state)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        num_agent=1,
    ):
        if not self._qkv_same_embed_dim:
            return agent_aware_attention(
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
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                gaussian_kernel=self.gaussian_kernel,
                num_agent=num_agent,
                in_proj_weight_self=self.in_proj_weight_self,
                in_proj_bias_self=self.in_proj_bias_self,
            )
        else:
            return agent_aware_attention(
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
                gaussian_kernel=self.gaussian_kernel,
                num_agent=num_agent,
                in_proj_weight_self=self.in_proj_weight_self,
                in_proj_bias_self=self.in_proj_bias_self,
            )


class AgentFormerEncoderLayer(Module):
    def __init__(self, cfg, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.cfg = cfg
        self.self_attn = AgentAwareAttention(cfg, d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, num_agent=1):
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            num_agent=num_agent,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class AgentFormerDecoderLayer(Module):
    def __init__(self, cfg, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.cfg = cfg
        self.self_attn = AgentAwareAttention(cfg, d_model, nhead, dropout=dropout)
        self.multihead_attn = AgentAwareAttention(cfg, d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        num_agent=1,
        need_weights=False,
    ):
        tgt2, self_attn_weights = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            num_agent=num_agent,
            need_weights=need_weights,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn_weights = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            num_agent=num_agent,
            need_weights=need_weights,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_attn_weights, cross_attn_weights


class AgentFormerEncoder(Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, num_agent=1):
        output = src

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                num_agent=num_agent,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class AgentFormerDecoder(Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        num_agent=1,
        need_weights=False,
    ):
        output = tgt

        self_attn_weights = [None] * len(self.layers)
        cross_attn_weights = [None] * len(self.layers)
        for i, mod in enumerate(self.layers):
            output, self_attn_weights[i], cross_attn_weights[i] = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                num_agent=num_agent,
                need_weights=need_weights,
            )

        if self.norm is not None:
            output = self.norm(output)

        if need_weights:
            self_attn_weights = torch.stack(self_attn_weights).cpu().numpy()
            cross_attn_weights = torch.stack(cross_attn_weights).cpu().numpy()

        return output, {
            "self_attn_weights": self_attn_weights,
            "cross_attn_weights": cross_attn_weights,
        }


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


# ---------------------------------------------------------------------------
# model/agentformer.py
# ---------------------------------------------------------------------------


def generate_ar_mask(sz, agent_num, agent_mask):
    assert sz % agent_num == 0
    T = sz // agent_num
    mask = agent_mask.repeat(T, T)
    for t in range(T - 1):
        i1 = t * agent_num
        i2 = (t + 1) * agent_num
        mask[i1:i2, i2:] = float("-inf")
    return mask


def generate_mask(tgt_sz, src_sz, agent_num, agent_mask):
    assert tgt_sz % agent_num == 0 and src_sz % agent_num == 0
    mask = agent_mask.repeat(tgt_sz // agent_num, src_sz // agent_num)
    return mask


""" Positional Encoding """


class PositionalAgentEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_t_len=200,
        max_a_len=200,
        concat=False,
        use_agent_enc=False,
        agent_enc_learn=False,
    ):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        self.use_agent_enc = use_agent_enc
        if concat:
            self.fc = nn.Linear((3 if use_agent_enc else 2) * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer("pe", pe)
        if use_agent_enc:
            if agent_enc_learn:
                self.ae = nn.Parameter(torch.randn(max_a_len, 1, d_model) * 0.1)
            else:
                ae = self.build_pos_enc(max_a_len)
                self.register_buffer("ae", ae)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def build_agent_enc(self, max_len):
        ae = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model)
        )
        ae[:, 0::2] = torch.sin(position * div_term)
        ae[:, 1::2] = torch.cos(position * div_term)
        ae = ae.unsqueeze(0).transpose(0, 1)
        return ae

    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset : num_t + t_offset, :]
        pe = pe.repeat_interleave(num_a, dim=0)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset, agent_enc_shuffle):
        if agent_enc_shuffle is None:
            ae = self.ae[a_offset : num_a + a_offset, :]
        else:
            ae = self.ae[agent_enc_shuffle]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, agent_enc_shuffle=None, t_offset=0, a_offset=0):
        num_t = x.shape[0] // num_a
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)
        if self.use_agent_enc:
            agent_enc = self.get_agent_enc(num_t, num_a, a_offset, agent_enc_shuffle)
        if self.concat:
            feat = [x, pos_enc.repeat(1, x.size(1), 1)]
            if self.use_agent_enc:
                feat.append(agent_enc.repeat(1, x.size(1), 1))
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
            if self.use_agent_enc:
                x += agent_enc
        return self.dropout(x)


""" Context (Past) Encoder """


class ContextEncoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ctx = ctx
        self.motion_dim = ctx["motion_dim"]
        self.model_dim = ctx["tf_model_dim"]
        self.ff_dim = ctx["tf_ff_dim"]
        self.nhead = ctx["tf_nhead"]
        self.dropout = ctx["tf_dropout"]
        self.nlayer = cfg.get("nlayer", 6)
        self.input_type = ctx["input_type"]
        self.pooling = cfg.get("pooling", "mean")
        self.agent_enc_shuffle = ctx["agent_enc_shuffle"]
        self.vel_heading = ctx["vel_heading"]
        ctx["context_dim"] = self.model_dim
        in_dim = self.motion_dim * len(self.input_type)
        if "map" in self.input_type:
            in_dim += ctx["map_enc_dim"] - self.motion_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        encoder_layers = AgentFormerEncoderLayer(
            ctx["tf_cfg"], self.model_dim, self.nhead, self.ff_dim, self.dropout
        )
        self.tf_encoder = AgentFormerEncoder(encoder_layers, self.nlayer)
        self.pos_encoder = PositionalAgentEncoding(
            self.model_dim,
            self.dropout,
            concat=ctx["pos_concat"],
            max_a_len=ctx["max_agent_len"],
            use_agent_enc=ctx["use_agent_enc"],
            agent_enc_learn=ctx["agent_enc_learn"],
        )

    def forward(self, data):
        traj_in = []
        for key in self.input_type:
            if key == "pos":
                traj_in.append(data["pre_motion"])
            elif key == "vel":
                vel = data["pre_vel"]
                if len(self.input_type) > 1:
                    vel = torch.cat([vel[[0]], vel], dim=0)
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data["heading"])[0]
                traj_in.append(vel)
            elif key == "norm":
                traj_in.append(data["pre_motion_norm"])
            elif key == "scene_norm":
                traj_in.append(data["pre_motion_scene_norm"])
            elif key == "heading":
                hv = data["heading_vec"].unsqueeze(0).repeat((data["pre_motion"].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == "map":
                map_enc = data["map_enc"].unsqueeze(0).repeat((data["pre_motion"].shape[0], 1, 1))
                traj_in.append(map_enc)
            else:
                raise ValueError("unknown input_type!")
        traj_in = torch.cat(traj_in, dim=-1)
        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)
        agent_enc_shuffle = data["agent_enc_shuffle"] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(
            tf_in, num_a=data["agent_num"], agent_enc_shuffle=agent_enc_shuffle
        )

        src_agent_mask = data["agent_mask"].clone()
        src_mask = generate_mask(
            tf_in.shape[0], tf_in.shape[0], data["agent_num"], src_agent_mask
        ).to(tf_in.device)

        data["context_enc"] = self.tf_encoder(tf_in_pos, mask=src_mask, num_agent=data["agent_num"])

        context_rs = data["context_enc"].view(-1, data["agent_num"], self.model_dim)
        # compute per agent context
        if self.pooling == "mean":
            data["agent_context"] = torch.mean(context_rs, dim=0)
        else:
            data["agent_context"] = torch.max(context_rs, dim=0)[0]


""" Future Encoder """


class FutureEncoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.context_dim = context_dim = ctx["context_dim"]  # noqa: F841 (verbatim from upstream)
        self.forecast_dim = forecast_dim = ctx["forecast_dim"]
        self.nz = ctx["nz"]
        self.z_type = ctx["z_type"]
        self.z_tau_annealer = ctx.get("z_tau_annealer", None)
        self.model_dim = ctx["tf_model_dim"]
        self.ff_dim = ctx["tf_ff_dim"]
        self.nhead = ctx["tf_nhead"]
        self.dropout = ctx["tf_dropout"]
        self.nlayer = cfg.get("nlayer", 6)
        self.out_mlp_dim = cfg.get("out_mlp_dim", None)
        self.input_type = ctx["fut_input_type"]
        self.pooling = cfg.get("pooling", "mean")
        self.agent_enc_shuffle = ctx["agent_enc_shuffle"]
        self.vel_heading = ctx["vel_heading"]
        # networks
        in_dim = forecast_dim * len(self.input_type)
        if "map" in self.input_type:
            in_dim += ctx["map_enc_dim"] - forecast_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = AgentFormerDecoderLayer(
            ctx["tf_cfg"], self.model_dim, self.nhead, self.ff_dim, self.dropout
        )
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(
            self.model_dim,
            self.dropout,
            concat=ctx["pos_concat"],
            max_a_len=ctx["max_agent_len"],
            use_agent_enc=ctx["use_agent_enc"],
            agent_enc_learn=ctx["agent_enc_learn"],
        )
        num_dist_params = (
            2 * self.nz if self.z_type == "gaussian" else self.nz
        )  # either gaussian or discrete
        if self.out_mlp_dim is None:
            self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.out_mlp = MLP(self.model_dim, self.out_mlp_dim, "relu")
            self.q_z_net = nn.Linear(self.out_mlp.out_dim, num_dist_params)
        # initialize
        initialize_weights(self.q_z_net.modules())

    def forward(self, data, reparam=True):
        traj_in = []
        for key in self.input_type:
            if key == "pos":
                traj_in.append(data["fut_motion"])
            elif key == "vel":
                vel = data["fut_vel"]
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data["heading"])[0]
                traj_in.append(vel)
            elif key == "norm":
                traj_in.append(data["fut_motion_norm"])
            elif key == "scene_norm":
                traj_in.append(data["fut_motion_scene_norm"])
            elif key == "heading":
                hv = data["heading_vec"].unsqueeze(0).repeat((data["fut_motion"].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == "map":
                map_enc = data["map_enc"].unsqueeze(0).repeat((data["fut_motion"].shape[0], 1, 1))
                traj_in.append(map_enc)
            else:
                raise ValueError("unknown input_type!")
        traj_in = torch.cat(traj_in, dim=-1)
        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)
        agent_enc_shuffle = data["agent_enc_shuffle"] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(
            tf_in, num_a=data["agent_num"], agent_enc_shuffle=agent_enc_shuffle
        )

        mem_agent_mask = data["agent_mask"].clone()
        tgt_agent_mask = data["agent_mask"].clone()
        mem_mask = generate_mask(
            tf_in.shape[0], data["context_enc"].shape[0], data["agent_num"], mem_agent_mask
        ).to(tf_in.device)
        tgt_mask = generate_mask(
            tf_in.shape[0], tf_in.shape[0], data["agent_num"], tgt_agent_mask
        ).to(tf_in.device)

        tf_out, _ = self.tf_decoder(
            tf_in_pos,
            data["context_enc"],
            memory_mask=mem_mask,
            tgt_mask=tgt_mask,
            num_agent=data["agent_num"],
        )
        tf_out = tf_out.view(traj_in.shape[0], -1, self.model_dim)

        if self.pooling == "mean":
            h = torch.mean(tf_out, dim=0)
        else:
            h = torch.max(tf_out, dim=0)[0]
        if self.out_mlp_dim is not None:
            h = self.out_mlp(h)
        q_z_params = self.q_z_net(h)
        if self.z_type == "gaussian":
            data["q_z_dist"] = Normal(params=q_z_params)
        else:
            data["q_z_dist"] = Categorical(logits=q_z_params, temp=self.z_tau_annealer.val())
        data["q_z_samp"] = data["q_z_dist"].rsample()


""" Future Decoder """


class FutureDecoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ar_detach = ctx["ar_detach"]
        self.context_dim = context_dim = ctx["context_dim"]  # noqa: F841 (verbatim from upstream)
        self.forecast_dim = forecast_dim = ctx["forecast_dim"]
        self.pred_scale = cfg.get("pred_scale", 1.0)
        self.pred_type = ctx["pred_type"]
        self.sn_out_type = ctx["sn_out_type"]
        self.sn_out_heading = ctx["sn_out_heading"]
        self.input_type = ctx["dec_input_type"]
        self.future_frames = ctx["future_frames"]
        self.past_frames = ctx["past_frames"]
        self.nz = ctx["nz"]
        self.z_type = ctx["z_type"]
        self.model_dim = ctx["tf_model_dim"]
        self.ff_dim = ctx["tf_ff_dim"]
        self.nhead = ctx["tf_nhead"]
        self.dropout = ctx["tf_dropout"]
        self.nlayer = cfg.get("nlayer", 6)
        self.out_mlp_dim = cfg.get("out_mlp_dim", None)
        self.pos_offset = cfg.get("pos_offset", False)
        self.agent_enc_shuffle = ctx["agent_enc_shuffle"]
        self.learn_prior = ctx["learn_prior"]
        # networks
        in_dim = forecast_dim + len(self.input_type) * forecast_dim + self.nz
        if "map" in self.input_type:
            in_dim += ctx["map_enc_dim"] - forecast_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = AgentFormerDecoderLayer(
            ctx["tf_cfg"], self.model_dim, self.nhead, self.ff_dim, self.dropout
        )
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(
            self.model_dim,
            self.dropout,
            concat=ctx["pos_concat"],
            max_a_len=ctx["max_agent_len"],
            use_agent_enc=ctx["use_agent_enc"],
            agent_enc_learn=ctx["agent_enc_learn"],
        )
        if self.out_mlp_dim is None:
            self.out_fc = nn.Linear(self.model_dim, forecast_dim)
        else:
            in_dim = self.model_dim
            self.out_mlp = MLP(in_dim, self.out_mlp_dim, "relu")
            self.out_fc = nn.Linear(self.out_mlp.out_dim, forecast_dim)
        initialize_weights(self.out_fc.modules())
        if self.learn_prior:
            num_dist_params = (
                2 * self.nz if self.z_type == "gaussian" else self.nz
            )  # either gaussian or discrete
            self.p_z_net = nn.Linear(self.model_dim, num_dist_params)
            initialize_weights(self.p_z_net.modules())

    def decode_traj_ar(
        self,
        data,
        mode,
        context,
        pre_motion,
        pre_vel,
        pre_motion_scene_norm,
        z,
        sample_num,
        need_weights=False,
    ):
        agent_num = data["agent_num"]
        if self.pred_type == "vel":
            dec_in = pre_vel[[-1]]
        elif self.pred_type == "pos":
            dec_in = pre_motion[[-1]]
        elif self.pred_type == "scene_norm":
            dec_in = pre_motion_scene_norm[[-1]]
        else:
            dec_in = torch.zeros_like(pre_motion[[-1]])
        dec_in = dec_in.view(-1, sample_num, dec_in.shape[-1])
        z_in = z.view(-1, sample_num, z.shape[-1])
        in_arr = [dec_in, z_in]
        for key in self.input_type:
            if key == "heading":
                heading = data["heading_vec"].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(heading)
            elif key == "map":
                map_enc = data["map_enc"].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(map_enc)
            else:
                raise ValueError("wrong decode input type!")
        dec_in_z = torch.cat(in_arr, dim=-1)

        mem_agent_mask = data["agent_mask"].clone()
        tgt_agent_mask = data["agent_mask"].clone()

        for i in range(self.future_frames):
            tf_in = self.input_fc(dec_in_z.view(-1, dec_in_z.shape[-1])).view(
                dec_in_z.shape[0], -1, self.model_dim
            )
            agent_enc_shuffle = data["agent_enc_shuffle"] if self.agent_enc_shuffle else None
            tf_in_pos = self.pos_encoder(
                tf_in,
                num_a=agent_num,
                agent_enc_shuffle=agent_enc_shuffle,
                t_offset=self.past_frames - 1 if self.pos_offset else 0,
            )
            mem_mask = generate_mask(
                tf_in.shape[0], context.shape[0], data["agent_num"], mem_agent_mask
            ).to(tf_in.device)
            tgt_mask = generate_ar_mask(tf_in_pos.shape[0], agent_num, tgt_agent_mask).to(
                tf_in.device
            )

            tf_out, attn_weights = self.tf_decoder(
                tf_in_pos,
                context,
                memory_mask=mem_mask,
                tgt_mask=tgt_mask,
                num_agent=data["agent_num"],
                need_weights=need_weights,
            )

            out_tmp = tf_out.view(-1, tf_out.shape[-1])
            if self.out_mlp_dim is not None:
                out_tmp = self.out_mlp(out_tmp)
            seq_out = self.out_fc(out_tmp).view(tf_out.shape[0], -1, self.forecast_dim)
            if self.pred_type == "scene_norm" and self.sn_out_type in {"vel", "norm"}:
                norm_motion = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
                if self.sn_out_type == "vel":
                    norm_motion = torch.cumsum(norm_motion, dim=0)
                if self.sn_out_heading:
                    angles = data["heading"].repeat_interleave(sample_num)
                    norm_motion = rotation_2d_torch(norm_motion, angles)[0]
                seq_out = norm_motion + pre_motion_scene_norm[[-1]]
                seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])
            if self.ar_detach:
                out_in = seq_out[-agent_num:].clone().detach()
            else:
                out_in = seq_out[-agent_num:]
            # create dec_in_z
            in_arr = [out_in, z_in]
            for key in self.input_type:
                if key == "heading":
                    in_arr.append(heading)
                elif key == "map":
                    in_arr.append(map_enc)
                else:
                    raise ValueError("wrong decoder input type!")
            out_in_z = torch.cat(in_arr, dim=-1)
            dec_in_z = torch.cat([dec_in_z, out_in_z], dim=0)

        seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        data[f"{mode}_seq_out"] = seq_out

        if self.pred_type == "vel":
            dec_motion = torch.cumsum(seq_out, dim=0)
            dec_motion += pre_motion[[-1]]
        elif self.pred_type == "pos":
            dec_motion = seq_out.clone()
        elif self.pred_type == "scene_norm":
            dec_motion = seq_out + data["scene_orig"]
        else:
            dec_motion = seq_out + pre_motion[[-1]]

        dec_motion = dec_motion.transpose(0, 1).contiguous()  # M x frames x 7
        if mode == "infer":
            dec_motion = dec_motion.view(
                -1, sample_num, *dec_motion.shape[1:]
            )  # M x Samples x frames x 3
        data[f"{mode}_dec_motion"] = dec_motion
        if need_weights:
            data["attn_weights"] = attn_weights

    def decode_traj_batch(
        self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num
    ):
        raise NotImplementedError

    def forward(self, data, mode, sample_num=1, autoregress=True, z=None, need_weights=False):
        context = data["context_enc"].repeat_interleave(sample_num, dim=1)  # 80 x 64
        pre_motion = data["pre_motion"].repeat_interleave(sample_num, dim=1)  # 10 x 80 x 2
        pre_vel = (
            data["pre_vel"].repeat_interleave(sample_num, dim=1)
            if self.pred_type == "vel"
            else None
        )
        pre_motion_scene_norm = data["pre_motion_scene_norm"].repeat_interleave(sample_num, dim=1)

        # p(z)
        prior_key = "p_z_dist" + ("_infer" if mode == "infer" else "")
        if self.learn_prior:
            h = data["agent_context"].repeat_interleave(sample_num, dim=0)
            p_z_params = self.p_z_net(h)
            if self.z_type == "gaussian":
                data[prior_key] = Normal(params=p_z_params)
            else:
                data[prior_key] = Categorical(params=p_z_params)
        else:
            if self.z_type == "gaussian":
                data[prior_key] = Normal(
                    mu=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device),
                    logvar=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device),
                )
            else:
                data[prior_key] = Categorical(
                    logits=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device)
                )

        if z is None:
            if mode in {"train", "recon"}:
                z = data["q_z_samp"] if mode == "train" else data["q_z_dist"].mode()
            elif mode == "infer":
                z = data["p_z_dist_infer"].sample()
            else:
                raise ValueError("Unknown Mode!")

        if autoregress:
            self.decode_traj_ar(
                data,
                mode,
                context,
                pre_motion,
                pre_vel,
                pre_motion_scene_norm,
                z,
                sample_num,
                need_weights=need_weights,
            )
        else:
            self.decode_traj_batch(
                data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num
            )


""" AgentFormer """


class AgentFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device("cpu")
        self.cfg = cfg

        input_type = cfg.get("input_type", "pos")
        pred_type = cfg.get("pred_type", input_type)
        if type(input_type) == str:
            input_type = [input_type]
        fut_input_type = cfg.get("fut_input_type", input_type)
        dec_input_type = cfg.get("dec_input_type", [])
        self.ctx = {
            "tf_cfg": cfg.get("tf_cfg", {}),
            "nz": cfg.nz,
            "z_type": cfg.get("z_type", "gaussian"),
            "future_frames": cfg.future_frames,
            "past_frames": cfg.past_frames,
            "motion_dim": cfg.motion_dim,
            "forecast_dim": cfg.forecast_dim,
            "input_type": input_type,
            "fut_input_type": fut_input_type,
            "dec_input_type": dec_input_type,
            "pred_type": pred_type,
            "tf_nhead": cfg.tf_nhead,
            "tf_model_dim": cfg.tf_model_dim,
            "tf_ff_dim": cfg.tf_ff_dim,
            "tf_dropout": cfg.tf_dropout,
            "pos_concat": cfg.get("pos_concat", False),
            "ar_detach": cfg.get("ar_detach", True),
            "max_agent_len": cfg.get("max_agent_len", 128),
            "use_agent_enc": cfg.get("use_agent_enc", False),
            "agent_enc_learn": cfg.get("agent_enc_learn", False),
            "agent_enc_shuffle": cfg.get("agent_enc_shuffle", False),
            "sn_out_type": cfg.get("sn_out_type", "scene_norm"),
            "sn_out_heading": cfg.get("sn_out_heading", False),
            "vel_heading": cfg.get("vel_heading", False),
            "learn_prior": cfg.get("learn_prior", False),
            "use_map": cfg.get("use_map", False),
        }
        self.use_map = self.ctx["use_map"]
        self.rand_rot_scene = cfg.get("rand_rot_scene", False)
        self.discrete_rot = cfg.get("discrete_rot", False)
        self.map_global_rot = cfg.get("map_global_rot", False)
        self.ar_train = cfg.get("ar_train", True)
        self.max_train_agent = cfg.get("max_train_agent", 100)
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())
        self.compute_sample = "sample" in self.loss_names
        self.param_annealers = nn.ModuleList()
        if self.ctx["z_type"] == "discrete":
            self.ctx["z_tau_annealer"] = z_tau_annealer = ExpParamAnnealer(
                cfg.z_tau.start, cfg.z_tau.finish, cfg.z_tau.decay
            )
            self.param_annealers.append(z_tau_annealer)

        # save all computed variables
        self.data = None

        # map encoder (real code constructs MapEncoder here when cfg.get('use_map', False);
        # the real ETH/UCY eth_agentformer_pre.yml reference config never sets use_map so
        # this branch is not exercised -- map_encoder.py/map_cnn.py not vendored, see header)
        if self.use_map:
            raise NotImplementedError(
                "use_map=True path (MapEncoder) not vendored; see staging module header"
            )

        # models
        self.context_encoder = ContextEncoder(cfg.context_encoder, self.ctx)
        self.future_encoder = FutureEncoder(cfg.future_encoder, self.ctx)
        self.future_decoder = FutureDecoder(cfg.future_decoder, self.ctx)

    def forward(self):
        if self.use_map:
            self.data["map_enc"] = self.map_encoder(self.data["agent_maps"])
        self.context_encoder(self.data)
        self.future_encoder(self.data)
        self.future_decoder(self.data, mode="train", autoregress=self.ar_train)
        if self.compute_sample:
            self.inference(sample_num=self.loss_cfg["sample"]["k"])
        return self.data

    def inference(self, mode="infer", sample_num=20, need_weights=False):
        if self.use_map and self.data["map_enc"] is None:
            self.data["map_enc"] = self.map_encoder(self.data["agent_maps"])
        if self.data["context_enc"] is None:
            self.context_encoder(self.data)
        if mode == "recon":
            sample_num = 1
            self.future_encoder(self.data)
        self.future_decoder(
            self.data, mode=mode, sample_num=sample_num, autoregress=True, need_weights=need_weights
        )
        return self.data[f"{mode}_dec_motion"], self.data


# ---------------------------------------------------------------------------
# staging plumbing: minimal EasyDict-alike (real utils/config.py `Config`
# wraps `easydict.EasyDict`, which is not a base-env dependency; both
# `cfg.attr` attribute access and `cfg.get(name, default)` -- the two
# accessors the real model code above actually uses -- are reproduced here)
# ---------------------------------------------------------------------------


class _CfgDict(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return _CfgDict(value) if isinstance(value, dict) else value

    def get(self, name, default=None):
        if name in self:
            value = self[name]
            return _CfgDict(value) if isinstance(value, dict) else value
        return default


def build_agentformer():
    # Hyperparameters transcribed from the real
    # `cfg/eth_ucy/eth/eth_agentformer_pre.yml` reference config (the base
    # AgentFormer VAE training config for the ETH/UCY pedestrian-forecasting
    # benchmark), only shrinking `tf_model_dim`/`tf_ff_dim`/`nlayer`/`nz` for
    # a tiny example build (real values: 256/512/2/32); everything else
    # (dropout, input/pred/sn_out types, pos_concat, learn_prior, past/future
    # frame counts, loss_cfg keys) is the real config value.
    cfg = _CfgDict(
        {
            "past_frames": 8,
            "future_frames": 12,
            "motion_dim": 2,
            "forecast_dim": 2,
            "tf_model_dim": 32,
            "tf_ff_dim": 64,
            "tf_nhead": 4,
            "tf_dropout": 0.1,
            "input_type": ["scene_norm", "vel"],
            "pred_type": "scene_norm",
            "sn_out_type": "norm",
            "max_train_agent": 32,
            "pos_concat": True,
            "rand_rot_scene": False,  # deterministic example build (real config trains with True)
            "scene_orig_all_past": True,
            "nz": 8,
            "learn_prior": True,
            "loss_cfg": {
                "mse": {"weight": 1.0},
                "kld": {"weight": 1.0, "min_clip": 2.0},
            },
            "context_encoder": {"nlayer": 2},
            "future_decoder": {"nlayer": 2, "out_mlp_dim": [32, 32]},
            "future_encoder": {"nlayer": 2, "out_mlp_dim": [32, 32]},
        }
    )
    model = AgentFormer(cfg)
    model.eval()
    return model


def example_input_agentformer():
    # Real `AgentFormer.set_data` builds `self.data` (a `defaultdict`) from a
    # raw per-agent-trajectory-list `data` dict (`pre_motion_3D`,
    # `fut_motion_3D`, `heading`, ...); this constructs the same `self.data`
    # structure directly (bypassing the dataset-preprocessing step) for a
    # tiny 3-pedestrian scene with `past_frames=8`/`future_frames=12`.
    agent_num = 3
    past_frames = 8
    future_frames = 12
    motion_dim = 2

    pre_motion = torch.randn(past_frames, agent_num, motion_dim)
    fut_motion = torch.randn(future_frames, agent_num, motion_dim)
    scene_orig = pre_motion.view(-1, motion_dim).mean(dim=0)

    data = {}
    data["batch_size"] = agent_num
    data["agent_num"] = agent_num
    data["pre_motion"] = pre_motion
    data["fut_motion"] = fut_motion
    data["scene_orig"] = scene_orig
    data["pre_motion_scene_norm"] = pre_motion - scene_orig
    data["fut_motion_scene_norm"] = fut_motion - scene_orig
    data["pre_vel"] = pre_motion[1:] - pre_motion[:-1, :]
    data["fut_vel"] = fut_motion - torch.cat([pre_motion[[-1]], fut_motion[:-1, :]])
    data["cur_motion"] = pre_motion[[-1]]
    data["pre_motion_norm"] = pre_motion[:-1] - data["cur_motion"]
    data["fut_motion_norm"] = fut_motion - data["cur_motion"]
    data["agent_enc_shuffle"] = None
    data["agent_mask"] = torch.zeros(agent_num, agent_num)
    return (data,)


class _AgentFormerWrapper(nn.Module):
    """Thin tensor-input wrapper: the real `AgentFormer.forward()` takes no
    tensor args and instead reads `self.data` (populated by `set_data`,
    real dataset plumbing not vendored -- see header). This wrapper accepts
    the same `data` dict `set_data` would have produced as a plain
    positional argument and assigns it to `self.data` before delegating to
    the real, unmodified `AgentFormer.forward()`.
    """

    def __init__(self, agentformer):
        super().__init__()
        self.agentformer = agentformer

    def forward(self, data):
        self.agentformer.data = data
        return self.agentformer.forward()


def build_agentformer_wrapped():
    return _AgentFormerWrapper(build_agentformer())


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("AgentFormer", "build_agentformer_wrapped", "example_input_agentformer", 2021, "vendored"),
]
