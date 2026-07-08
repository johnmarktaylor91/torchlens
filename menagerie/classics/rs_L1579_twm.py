# SOURCE: vendored from https://github.com/jrobine/twm @ main
# (twm/nets.py: get_activation/get_norm_1d/get_norm_2d/init_/_MultilayerModule/
#  MLP/CNN/TransposeCNN/TransformerXLDecoder/TransformerXLDecoderLayer/
#  RelativeMultiheadSelfAttention/PositionalEncoding/PredictionNet, full file;
#  twm/world_model.py: WorldModel/ObservationModel/DynamicsModel, full file;
#  twm/actor_critic.py: ActorCritic, full file; twm/utils.py: the 6
#  model-construction helpers actually imported by the three files above
#  -- update_metrics/combine_metrics/same_batch_shape/
#  same_batch_shape_time_offset/check_no_grad/AdamOptim -- vendored verbatim
#  as `_twm_utils`; the remaining ~20 functions/classes in utils.py are
#  gym/ale_py/wandb Atari-environment training-loop plumbing, not imported by
#  the network-construction files, and are omitted)
#
# TWM / "TWIRL" (Robine, Uelwer & Harmeling, ICLR 2023, "Transformer-based
# World Models Are Happy With 100k Interactions", arXiv:2303.07109). This is
# the paper authors' own official PyTorch repo (jrobine/twm) -- the reference
# implementation, not a community port. Architecture: a CNN
# `ObservationModel` (conv encoder -> categorical/Gumbel-softmax latent `z`
# via `OneHotCategoricalStraightThrough`, transpose-conv decoder) feeding a
# causal `DynamicsModel` built on a TransformerXL decoder with relative
# multi-head self-attention and segment-level recurrence (memory) over
# interleaved latent-state/action/reward/discount tokens -- the paper's
# defining architectural move is replacing the RSSM/GRU dynamics core used by
# prior world models (e.g. DreamerV2/PlaNet) with this TransformerXL, plus a
# standard MLP `ActorCritic` operating on the model's latent state. All
# vendored classes have no dependency beyond torch, so they are copied
# verbatim; only the outer `optimize*`/training-loop methods (which also
# appear in the real classes, unmodified) are exercised only insofar as
# `__init__` constructs their real `AdamOptim` wrappers -- no training step is
# invoked here, only the real forward-pass construction and a forward trace.

import copy
import math
from functools import lru_cache

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions.utils import logits_to_probs

MENAGERIE_ZOO = "vendored-pytorch"


# ---- twm/utils.py (vendored verbatim: only the 6 helpers imported by
#      nets.py / world_model.py / actor_critic.py; renamed module `_twm_utils`
#      to avoid clashing with any top-level `utils` already on sys.path) ----
class _twm_utils:
    @staticmethod
    def update_metrics(metrics, new_metrics, prefix=None):
        def process(key, t):
            if isinstance(t, (int, float)):
                return t
            assert torch.is_tensor(t), key
            assert not t.requires_grad, key
            assert t.ndim == 0 or t.shape == (1,), key
            return t.clone()

        if prefix is None:
            metrics.update({key: process(key, value) for key, value in new_metrics.items()})
        else:
            metrics.update(
                {f"{prefix}{key}": process(key, value) for key, value in new_metrics.items()}
            )
        return metrics

    @staticmethod
    def combine_metrics(metrics, prefix=None):
        result = {}
        if prefix is None:
            for met in metrics:
                _twm_utils.update_metrics(result, met)
        else:
            for met, pre in zip(metrics, prefix):
                _twm_utils.update_metrics(result, met, pre)
        return result

    @staticmethod
    def same_batch_shape(tensors, ndim=2):
        batch_shape = tensors[0].shape[:ndim]
        assert all(t.ndim >= ndim for t in tensors)
        return all(tensors[i].shape[:ndim] == batch_shape for i in range(1, len(tensors)))

    @staticmethod
    def same_batch_shape_time_offset(a, b, offset):
        assert a.ndim >= 2 and b.ndim >= 2
        return a.shape[:2] == (b.shape[0], b.shape[1] + offset)

    @staticmethod
    def check_no_grad(*tensors):
        return all((t is None or not t.requires_grad) for t in tensors)

    class AdamOptim:
        def __init__(
            self, parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, grad_clip=0
        ):
            self.parameters = list(parameters)
            self.grad_clip = grad_clip
            self.optimizer = optim.Adam(
                self.parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
            )

        def step(self, loss):
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
            self.optimizer.step()


utils = _twm_utils
# ---- end vendored twm/utils.py subset ----


# ---- twm/nets.py (vendored verbatim) ----
def get_activation(nonlinearity, param=None):
    if nonlinearity is None or nonlinearity == "none" or nonlinearity == "linear":
        return nn.Identity()
    elif nonlinearity == "relu":
        return nn.ReLU()
    elif nonlinearity == "leaky_relu":
        if param is None:
            param = 1e-2
        return nn.LeakyReLU(negative_slope=param)
    elif nonlinearity == "elu":
        if param is None:
            param = 1.0
        return nn.ELU(alpha=param)
    elif nonlinearity == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")


def get_norm_1d(norm, k):
    if norm is None or norm == "none":
        return nn.Identity()
    elif norm == "batch_norm":
        return nn.BatchNorm1d(k)
    elif norm == "layer_norm":
        return nn.LayerNorm(k)
    else:
        raise ValueError(f"Unsupported norm: {norm}")


def get_norm_2d(norm, c, h=None, w=None):
    if norm == "none":
        return nn.Identity()
    elif norm == "batch_norm":
        return nn.BatchNorm2d(c)
    elif norm == "layer_norm":
        assert h is not None and w is not None
        return nn.LayerNorm([c, h, w])
    else:
        raise ValueError(f"Unsupported norm: {norm}")


def _calculate_gain(nonlinearity, param=None):
    if nonlinearity == "elu":
        nonlinearity = "selu"
        param = 1
    elif nonlinearity == "silu":
        nonlinearity = "relu"
        param = None
    return torch.nn.init.calculate_gain(nonlinearity, param)


def _kaiming_uniform_(tensor, gain):
    # same as torch.nn.init.kaiming_uniform_, but uses gain
    fan = torch.nn.init._calculate_correct_fan(tensor, mode="fan_in")
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    torch.nn.init._no_grad_uniform_(tensor, -bound, bound)


def _get_initializer(name, nonlinearity=None, param=None):
    if nonlinearity is None:
        assert param is None
    if name == "kaiming_uniform":
        if nonlinearity is None:
            # defaults from PyTorch
            nonlinearity = "leaky_relu"
            param = math.sqrt(5)
        return lambda x: _kaiming_uniform_(x, gain=_calculate_gain(nonlinearity, param))
    elif name == "xavier_uniform":
        if nonlinearity is None:
            nonlinearity = "relu"
        return lambda x: torch.nn.init.xavier_uniform_(x, gain=_calculate_gain(nonlinearity, param))
    elif name == "orthogonal":
        if nonlinearity is None:
            nonlinearity = "relu"
        return lambda x: torch.nn.init.orthogonal_(x, gain=_calculate_gain(nonlinearity, param))
    elif name == "zeros":
        return lambda x: torch.nn.init.zeros_(x)
    else:
        raise ValueError(f"Unsupported initializer: {name}")


def init_(mod, weight_initializer=None, bias_initializer=None, nonlinearity=None, param=None):
    weight_initializer = (
        _get_initializer(weight_initializer, nonlinearity, param)
        if weight_initializer is not None
        else lambda x: x
    )
    bias_initializer = (
        _get_initializer(bias_initializer, nonlinearity="linear", param=None)
        if bias_initializer is not None
        else lambda x: x
    )

    def fn(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            weight_initializer(m.weight)
            if m.bias is not None:
                bias_initializer(m.bias)

    return mod.apply(fn)


class _MultilayerModule(nn.Module):
    def __init__(
        self,
        layer_prefix,
        ndim,
        in_dim,
        num_layers,
        nonlinearity,
        param,
        norm,
        dropout_p,
        pre_activation,
        post_activation,
        weight_initializer,
        bias_initializer,
        final_bias_init,
    ):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.ndim = ndim
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.param = param
        self.pre_activation = pre_activation
        self.post_activation = post_activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.final_bias_init = final_bias_init

        self.has_norm = norm is not None and norm != "none"
        self.has_dropout = dropout_p != 0
        self.unsqueeze = in_dim == 0

        self.act = get_activation(nonlinearity, param)

    def reset_parameters(self):
        init_(self, self.weight_initializer, self.bias_initializer, self.nonlinearity, self.param)
        final_layer = getattr(self, f"{self.layer_prefix}{self.num_layers}")
        if not self.post_activation:
            init_(
                final_layer,
                self.weight_initializer,
                self.bias_initializer,
                nonlinearity="linear",
                param=None,
            )
        if self.final_bias_init is not None:

            def final_init(m):
                if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                    with torch.no_grad():
                        m.bias.data.fill_(self.final_bias_init)

            final_layer.apply(final_init)

    def forward(self, x):
        if self.unsqueeze:
            x = x.unsqueeze(-self.ndim)

        if x.ndim > self.ndim + 1:
            batch_shape = x.shape[: -self.ndim]
            x = x.reshape(-1, *x.shape[-self.ndim :])
        else:
            batch_shape = None

        if self.pre_activation:
            if self.has_norm:
                x = getattr(self, "norm0")(x)
            x = self.act(x)

        for i in range(self.num_layers - 1):
            x = getattr(self, f"{self.layer_prefix}{i + 1}")(x)
            if self.has_norm:
                x = getattr(self, f"norm{i + 1}")(x)
            x = self.act(x)
            if self.has_dropout:
                x = self.dropout(x)
        x = getattr(self, f"{self.layer_prefix}{self.num_layers}")(x)

        if self.post_activation:
            if self.has_norm:
                x = getattr(self, f"norm{self.num_layers}")(x)
            x = self.act(x)

        if batch_shape is not None:
            x = x.unflatten(0, batch_shape)
        return x


class MLP(_MultilayerModule):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        nonlinearity,
        param=None,
        norm=None,
        dropout_p=0,
        bias=True,
        pre_activation=False,
        post_activation=False,
        weight_initializer="kaiming_uniform",
        bias_initializer="zeros",
        final_bias_init=None,
    ):
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        super().__init__(
            "linear",
            1,
            in_dim,
            len(dims) - 1,
            nonlinearity,
            param,
            norm,
            dropout_p,
            pre_activation,
            post_activation,
            weight_initializer,
            bias_initializer,
            final_bias_init,
        )
        if self.unsqueeze:
            dims = (1,) + dims[1:]

        if pre_activation and self.has_norm:
            norm_layer = get_norm_1d(norm, in_dim)
            self.add_module("norm0", norm_layer)

        for i in range(self.num_layers - 1):
            linear_layer = nn.Linear(dims[i], dims[i + 1], bias=bias)
            self.add_module(f"linear{i + 1}", linear_layer)
            if self.has_norm:
                norm_layer = get_norm_1d(norm, dims[i + 1])
                self.add_module(f"norm{i + 1}", norm_layer)

        linear_layer = nn.Linear(dims[-2], dims[-1], bias=bias)
        self.add_module(f"linear{self.num_layers}", linear_layer)

        if post_activation and self.has_norm:
            norm_layer = get_norm_1d(norm, dims[-1])
            self.add_module(f"norm{self.num_layers}", norm_layer)

        if self.has_dropout:
            self.dropout = nn.Dropout(dropout_p)

        self.reset_parameters()


class CNN(_MultilayerModule):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        kernel_sizes,
        strides,
        paddings,
        nonlinearity,
        param=None,
        norm=None,
        dropout_p=0,
        bias=True,
        padding_mode="zeros",
        in_shape=None,
        pre_activation=False,
        post_activation=False,
        weight_initializer="kaiming_uniform",
        bias_initializer="zeros",
        final_bias_init=None,
    ):
        assert len(kernel_sizes) == len(hidden_dims) + 1
        assert len(strides) == len(kernel_sizes) and len(paddings) == len(kernel_sizes)
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        super().__init__(
            "conv",
            3,
            in_dim,
            len(dims) - 1,
            nonlinearity,
            param,
            norm,
            dropout_p,
            pre_activation,
            post_activation,
            weight_initializer,
            bias_initializer,
            final_bias_init,
        )
        if self.unsqueeze:
            dims = (1,) + dims[1:]

        def to_pair(x):
            if isinstance(x, int):
                return x, x
            assert isinstance(x, tuple) and len(x) == 2
            return x

        def calc_out_shape(shape, kernel_size, stride, padding):
            kernel_size, padding, stride = [to_pair(x) for x in (kernel_size, stride, padding)]
            return tuple(
                (shape[j] + 2 * padding[j] - kernel_size[j]) / stride[j] + 1 for j in [0, 1]
            )

        if pre_activation and self.has_norm:
            norm_layer = get_norm_2d(norm, in_dim, in_shape[0], in_shape[1])
            self.add_module("norm0", norm_layer)

        shape = in_shape
        for i in range(self.num_layers - 1):
            conv_layer = nn.Conv2d(
                dims[i],
                dims[i + 1],
                kernel_sizes[i],
                strides[i],
                paddings[i],
                bias=bias,
                padding_mode=padding_mode,
            )
            self.add_module(f"conv{i + 1}", conv_layer)
            if self.has_norm:
                if shape is not None:
                    shape = calc_out_shape(shape, kernel_sizes[i], strides[i], paddings[i])
                norm_layer = get_norm_2d(norm, dims[i + 1], shape[0], shape[1])
                self.add_module(f"norm{i + 1}", norm_layer)

        conv_layer = nn.Conv2d(
            dims[-2],
            dims[-1],
            kernel_sizes[-1],
            strides[-1],
            paddings[-1],
            bias=bias,
            padding_mode=padding_mode,
        )
        self.add_module(f"conv{self.num_layers}", conv_layer)

        if post_activation and self.has_norm:
            shape = calc_out_shape(shape, kernel_sizes[-1], strides[-1], paddings[-1])
            norm_layer = get_norm_2d(norm, dims[-1], shape[0], shape[1])
            self.add_module(f"norm{self.num_layers}", norm_layer)

        if self.has_dropout:
            self.dropout = nn.Dropout2d(dropout_p)

        self.reset_parameters()


class TransposeCNN(_MultilayerModule):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        kernel_sizes,
        strides,
        paddings,
        nonlinearity,
        param=None,
        norm=None,
        dropout_p=0,
        bias=True,
        padding_mode="zeros",
        in_shape=None,
        pre_activation=False,
        post_activation=False,
        weight_initializer="kaiming_uniform",
        bias_initializer="zeros",
        final_bias_init=None,
    ):
        assert len(kernel_sizes) == len(hidden_dims) + 1
        assert len(strides) == len(kernel_sizes) and len(paddings) == len(kernel_sizes)
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        super().__init__(
            "conv_transpose",
            3,
            in_dim,
            len(dims) - 1,
            nonlinearity,
            param,
            norm,
            dropout_p,
            pre_activation,
            post_activation,
            weight_initializer,
            bias_initializer,
            final_bias_init,
        )
        if self.unsqueeze:
            dims = (1,) + dims[1:]

        def to_pair(x):
            if isinstance(x, int):
                return x, x
            assert isinstance(x, tuple) and len(x) == 2
            return x

        def calc_out_shape(shape, kernel_size, stride, padding):
            kernel_size, padding, stride = [to_pair(x) for x in (kernel_size, stride, padding)]
            return tuple(
                (shape[j] - 1) * stride[j] - 2 * padding[j] + kernel_size[j] for j in [0, 1]
            )

        if pre_activation and self.has_norm:
            norm_layer = get_norm_2d(norm, in_dim, in_shape[0], in_shape[1])
            self.add_module("norm0", norm_layer)

        shape = in_shape
        for i in range(self.num_layers - 1):
            conv_transpose_layer = nn.ConvTranspose2d(
                dims[i],
                dims[i + 1],
                kernel_sizes[i],
                strides[i],
                paddings[i],
                bias=bias,
                padding_mode=padding_mode,
            )
            self.add_module(f"conv_transpose{i + 1}", conv_transpose_layer)
            if self.has_norm:
                if shape is not None:
                    shape = calc_out_shape(shape, kernel_sizes[i], strides[i], paddings[i])
                norm_layer = get_norm_2d(norm, dims[i + 1], shape[0], shape[1])
                self.add_module(f"norm{i + 1}", norm_layer)

        conv_transpose_layer = nn.ConvTranspose2d(
            dims[-2],
            dims[-1],
            kernel_sizes[-1],
            strides[-1],
            paddings[-1],
            bias=bias,
            padding_mode=padding_mode,
        )
        self.add_module(f"conv_transpose{self.num_layers}", conv_transpose_layer)

        if post_activation and self.has_norm:
            shape = calc_out_shape(shape, kernel_sizes[-1], strides[-1], paddings[-1])
            norm_layer = get_norm_2d(norm, dims[-1], shape[0], shape[1])
            self.add_module(f"norm{self.num_layers}", norm_layer)

        if self.has_dropout:
            self.dropout = nn.Dropout2d(dropout_p)

        self.reset_parameters()


# adopted from
# https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
# and https://github.com/sooftware/attentions/blob/master/attentions.py
class TransformerXLDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, max_length, mem_length, batch_first=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.mem_length = mem_length
        self.batch_first = batch_first

        self.pos_enc = PositionalEncoding(
            decoder_layer.dim, max_length, dropout_p=decoder_layer.dropout_p
        )
        self.u_bias = nn.Parameter(torch.Tensor(decoder_layer.num_heads, decoder_layer.head_dim))
        self.v_bias = nn.Parameter(torch.Tensor(decoder_layer.num_heads, decoder_layer.head_dim))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)

    def init_mems(self):
        if self.mem_length > 0:
            param = next(self.parameters())
            dtype, device = param.dtype, param.device
            mems = []
            for i in range(self.num_layers + 1):
                mems.append(torch.empty(0, dtype=dtype, device=device))
            return mems
        else:
            return None

    def forward(self, x, positions, attn_mask, mems=None, tgt_length=None, return_attention=False):
        if self.batch_first:
            x = x.transpose(0, 1)

        if mems is None:
            mems = self.init_mems()

        if tgt_length is None:
            tgt_length = x.shape[0]
        assert tgt_length > 0

        pos_enc = self.pos_enc(positions)
        hiddens = [x]
        attentions = []
        out = x
        for i, layer in enumerate(self.layers):
            out, attention = layer(
                out, pos_enc, self.u_bias, self.v_bias, attn_mask=attn_mask, mems=mems[i]
            )
            hiddens.append(out)
            attentions.append(attention)

        out = out[-tgt_length:]

        if self.batch_first:
            out = out.transpose(0, 1)

        assert len(hiddens) == len(mems)
        with torch.no_grad():
            new_mems = []
            for i in range(len(hiddens)):
                cat = torch.cat([mems[i], hiddens[i]], dim=0)
                new_mems.append(cat[-self.mem_length :].detach())
        if return_attention:
            attention = torch.stack(attentions, dim=-2)
            return out, new_mems, attention
        return out, new_mems


class TransformerXLDecoderLayer(nn.Module):
    def __init__(
        self, dim, feedforward_dim, head_dim, num_heads, activation, dropout_p, layer_norm_eps=1e-5
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.self_attn = RelativeMultiheadSelfAttention(dim, head_dim, num_heads, dropout_p)
        self.linear1 = nn.Linear(dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def _ff(self, x):
        x = self.linear2(self.dropout(self.act(self.linear1(x))))
        return self.dropout(x)

    def forward(self, x, pos_encodings, u_bias, v_bias, attn_mask=None, mems=None):
        out, attention = self.self_attn(x, pos_encodings, u_bias, v_bias, attn_mask, mems)
        out = self.dropout(out)
        out = self.norm1(x + out)
        out = self.norm2(out + self._ff(out))
        return out, attention


class RelativeMultiheadSelfAttention(nn.Module):
    def __init__(self, dim, head_dim, num_heads, dropout_p):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = 1 / (dim**0.5)

        self.qkv_proj = nn.Linear(dim, 3 * num_heads * head_dim, bias=False)
        self.pos_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
        x = x_padded[1:].view_as(x)
        return x

    def forward(self, x, pos_encodings, u_bias, v_bias, attn_mask=None, mems=None):
        tgt_length, batch_size = x.shape[:2]
        pos_len = pos_encodings.shape[0]

        if mems is not None:
            cat = torch.cat([mems, x], dim=0)
            qkv = self.qkv_proj(cat)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = q[-tgt_length:]
        else:
            qkv = self.qkv_proj(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

        pos_encodings = self.pos_proj(pos_encodings)

        src_length = k.shape[0]
        num_heads = self.num_heads
        head_dim = self.head_dim

        q = q.view(tgt_length, batch_size, num_heads, head_dim)
        k = k.view(src_length, batch_size, num_heads, head_dim)
        v = v.view(src_length, batch_size, num_heads, head_dim)
        pos_encodings = pos_encodings.view(pos_len, num_heads, head_dim)

        content_score = torch.einsum("ibnd,jbnd->ijbn", (q + u_bias, k))
        pos_score = torch.einsum("ibnd,jnd->ijbn", (q + v_bias, pos_encodings))
        pos_score = self._rel_shift(pos_score)

        # [tgt_length x src_length x batch_size x num_heads]
        attn_score = content_score + pos_score
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_score = attn_score.masked_fill(attn_mask[:, :, None, None], -float("inf"))
            elif attn_mask.ndim == 3:
                attn_score = attn_score.masked_fill(attn_mask[:, :, :, None], -float("inf"))

        # [tgt_length x src_length x batch_size x num_heads]
        attn = F.softmax(attn_score, dim=1)
        return_attn = attn
        attn = self.dropout(attn)

        context = torch.einsum("ijbn,jbnd->ibnd", (attn, v))
        context = context.reshape(context.shape[0], context.shape[1], num_heads * head_dim)
        return self.out_proj(context), return_attn


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_length, dropout_p=0, batch_first=False):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        encodings = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encodings", encodings)

    def forward(self, positions):
        out = self.encodings[positions]
        out = self.dropout(out)
        return out.unsqueeze(0) if self.batch_first else out.unsqueeze(1)


class PredictionNet(nn.Module):
    def __init__(
        self,
        modality_order,
        num_current,
        embeds,
        out_heads,
        embed_dim,
        activation,
        norm,
        dropout_p,
        feedforward_dim,
        head_dim,
        num_heads,
        num_layers,
        memory_length,
        max_length,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_length = memory_length
        self.modality_order = tuple(modality_order)
        self.num_current = num_current

        self.embeds = nn.ModuleDict(
            {
                name: nn.Embedding(embed["in_dim"], embed_dim)
                if embed.get("categorical", False)
                else MLP(
                    embed["in_dim"],
                    [],
                    embed_dim,
                    activation,
                    norm=norm,
                    dropout_p=dropout_p,
                    post_activation=True,
                )
                for name, embed in embeds.items()
            }
        )

        decoder_layer = TransformerXLDecoderLayer(
            embed_dim, feedforward_dim, head_dim, num_heads, activation, dropout_p
        )

        num_modalities = len(modality_order)
        max_length = max_length * num_modalities + self.num_current
        mem_length = memory_length * num_modalities + self.num_current
        self.transformer = TransformerXLDecoder(
            decoder_layer, num_layers, max_length, mem_length, batch_first=True
        )

        self.out_heads = nn.ModuleDict(
            {
                name: MLP(
                    embed_dim,
                    head["hidden_dims"],
                    head["out_dim"],
                    activation,
                    norm=norm,
                    dropout_p=dropout_p,
                    pre_activation=True,
                    final_bias_init=head.get("final_bias_init", None),
                )
                for name, head in out_heads.items()
            }
        )

    @lru_cache(maxsize=20)
    def _get_base_mask(self, src_length, tgt_length, device):
        src_mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=device)
        num_modalities = len(self.modality_order)
        for tgt_index in range(tgt_length):
            # the last indices are always 'current'
            start_index = src_length - self.num_current
            src_index = src_length - tgt_length + tgt_index
            modality_index = (src_index - start_index) % num_modalities
            if modality_index < self.num_current:
                start = max(src_index - (self.memory_length + 1) * num_modalities, 0)
            else:
                start = max(src_index - modality_index - self.memory_length * num_modalities, 0)
            src_mask[tgt_index, start : src_index + 1] = False
        return src_mask

    def _get_mask(self, src_length, tgt_length, device, stop_mask):
        # prevent attention over episode ends using stop_mask
        num_modalities = len(self.modality_order)
        assert stop_mask.shape[1] * num_modalities + self.num_current == src_length

        src_mask = self._get_base_mask(src_length, tgt_length, device)

        batch_size, seq_length = stop_mask.shape
        stop_mask = stop_mask.t()
        stop_mask_shift_right = torch.cat([stop_mask.new_zeros(1, batch_size), stop_mask], dim=0)
        stop_mask_shift_left = torch.cat([stop_mask, stop_mask.new_zeros(1, batch_size)], dim=0)

        tril = stop_mask.new_ones(seq_length + 1, seq_length + 1).tril()
        src = torch.logical_and(stop_mask_shift_left.unsqueeze(0), tril.unsqueeze(-1))
        src = torch.cummax(src.flip(1), dim=1).values.flip(1)

        shifted_tril = stop_mask.new_ones(seq_length + 1, seq_length + 1).tril(diagonal=-1)
        tgt = torch.logical_and(stop_mask_shift_right.unsqueeze(1), shifted_tril.unsqueeze(-1))
        tgt = torch.cummax(tgt, dim=0).values

        idx = torch.logical_and(src, tgt)

        i, j, k = idx.shape
        idx = (
            idx.reshape(i, 1, j, 1, k)
            .expand(i, num_modalities, j, num_modalities, k)
            .reshape(i * num_modalities, j * num_modalities, k)
        )

        offset = num_modalities - self.num_current
        if offset > 0:
            idx = idx[:-offset, :-offset]
        idx = idx[-tgt_length:]

        src_mask = src_mask.unsqueeze(-1).tile(1, 1, batch_size)
        src_mask[idx] = True
        return src_mask

    def forward(self, inputs, tgt_length, stop_mask, heads=None, mems=None, return_attention=False):
        modality_order = self.modality_order
        num_modalities = len(modality_order)
        num_current = self.num_current

        assert utils.same_batch_shape([inputs[name] for name in modality_order[:num_current]])
        if num_modalities > num_current:
            assert utils.same_batch_shape([inputs[name] for name in modality_order[num_current:]])

        embeds = {name: mod(inputs[name]) for name, mod in self.embeds.items()}

        def cat_modalities(xs):
            batch_size, seq_len, dim = xs[0].shape
            return torch.cat(xs, dim=2).reshape(batch_size, seq_len * len(xs), dim)

        if mems is None:
            history_length = embeds[modality_order[0]].shape[1] - 1
            if num_modalities == num_current:
                inputs = cat_modalities([embeds[name] for name in modality_order])
            else:
                history = cat_modalities(
                    [embeds[name][:, :history_length] for name in modality_order]
                )
                current = cat_modalities(
                    [embeds[name][:, history_length:] for name in modality_order[:num_current]]
                )
                inputs = torch.cat([history, current], dim=1)
            tgt_length = (tgt_length - 1) * num_modalities + num_current
            src_length = history_length * num_modalities + num_current
            assert inputs.shape[1] == src_length
            src_mask = self._get_mask(src_length, src_length, inputs.device, stop_mask)
        else:
            sequence_length = embeds[modality_order[0]].shape[1]
            # switch order so that 'currents' are last
            inputs = cat_modalities(
                [
                    embeds[name]
                    for name in (modality_order[num_current:] + modality_order[:num_current])
                ]
            )
            tgt_length = tgt_length * num_modalities
            mem_length = mems[0].shape[0]
            src_length = mem_length + sequence_length * num_modalities
            src_mask = self._get_mask(src_length, tgt_length, inputs.device, stop_mask)

        positions = torch.arange(src_length - 1, -1, -1, device=inputs.device)
        outputs = self.transformer(
            inputs,
            positions,
            attn_mask=src_mask,
            mems=mems,
            tgt_length=tgt_length,
            return_attention=return_attention,
        )
        hiddens, mems, attention = outputs if return_attention else (outputs + (None,))

        # take outputs at last current
        assert hiddens.shape[1] == tgt_length
        out_idx = torch.arange(tgt_length - 1, -1, -num_modalities, device=inputs.device).flip([0])
        hiddens = hiddens[:, out_idx]
        if return_attention:
            attention = attention[out_idx]

        if heads is None:
            heads = self.out_heads.keys()

        out = {name: self.out_heads[name](hiddens) for name in heads}

        return (out, hiddens, mems) if not return_attention else (out, hiddens, mems, attention)


# ---- end vendored twm/nets.py ----


# ---- twm/world_model.py (vendored verbatim) ----
class WorldModel(nn.Module):
    def __init__(self, config, num_actions):
        super().__init__()
        self.config = config
        self.num_actions = num_actions

        self.obs_model = ObservationModel(config)
        self.dyn_model = DynamicsModel(config, self.obs_model.z_dim, num_actions)

        self.obs_optimizer = utils.AdamOptim(
            self.obs_model.parameters(),
            lr=config["obs_lr"],
            eps=config["obs_eps"],
            weight_decay=config["obs_wd"],
            grad_clip=config["obs_grad_clip"],
        )
        self.dyn_optimizer = utils.AdamOptim(
            self.dyn_model.parameters(),
            lr=config["dyn_lr"],
            eps=config["dyn_eps"],
            weight_decay=config["dyn_wd"],
            grad_clip=config["dyn_grad_clip"],
        )

    @property
    def z_dim(self):
        return self.obs_model.z_dim

    @property
    def h_dim(self):
        return self.dyn_model.h_dim

    def optimize_pretrain_obs(self, o):
        obs_model = self.obs_model
        obs_model.train()

        z_dist = obs_model.encode(o)
        z = obs_model.sample_z(z_dist, reparameterized=True)
        recons = obs_model.decode(z)

        # no consistency loss required for pretraining
        dec_loss, dec_met = obs_model.compute_decoder_loss(recons, o)
        ent_loss, ent_met = obs_model.compute_entropy_loss(z_dist)

        obs_loss = dec_loss + ent_loss
        self.obs_optimizer.step(obs_loss)

        metrics = utils.combine_metrics([ent_met, dec_met])
        metrics["obs_loss"] = obs_loss.detach()
        return metrics

    @torch.no_grad()
    def to_discounts(self, mask):
        assert utils.check_no_grad(mask)
        discount_factor = self.config["env_discount_factor"]
        g = torch.full(mask.shape, discount_factor, device=mask.device)
        g = g * (~mask).float()
        return g


class ObservationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.z_dim = config["z_categoricals"] * config["z_categories"]

        h = config["obs_channels"]
        activation = config["obs_act"]
        norm = config["obs_norm"]
        dropout_p = config["obs_dropout"]

        num_channels = config["env_frame_stack"]
        if not config["env_grayscale"]:
            num_channels *= 3

        self.encoder = nn.Sequential(
            CNN(
                num_channels,
                [h, h * 2, h * 4],
                h * 8,
                [4, 4, 4, 4],
                [2, 2, 2, 2],
                [0, 0, 0, 0],
                activation,
                norm=norm,
                post_activation=True,
            ),
            nn.Flatten(),
            MLP(
                (h * 8) * 2 * 2, [512, 512], self.z_dim, activation, norm=norm, dropout_p=dropout_p
            ),
        )

        # no norm here
        self.decoder = nn.Sequential(
            MLP(
                self.z_dim,
                [],
                (h * 16) * 1 * 1,
                activation,
                dropout_p=dropout_p,
                post_activation=True,
            ),
            nn.Unflatten(1, (h * 16, 1, 1)),
            TransposeCNN(
                h * 16,
                [h * 4, h * 2, h],
                num_channels,
                [5, 5, 6, 6],
                [2, 2, 2, 2],
                [0, 0, 0, 0],
                activation,
                final_bias_init=0.5,
            ),
        )

    @staticmethod
    def create_z_dist(logits, temperature=1):
        assert temperature > 0
        return D.Independent(D.OneHotCategoricalStraightThrough(logits=logits / temperature), 1)

    def encode(self, o):
        assert utils.check_no_grad(o)
        config = self.config
        shape = o.shape[:2]
        o = o.flatten(0, 1)

        if not config["env_grayscale"]:
            o = o.permute(0, 1, 4, 2, 3)
            o = o.flatten(1, 2)

        z_logits = self.encoder(o)
        z_logits = z_logits.unflatten(0, shape)
        z_logits = z_logits.unflatten(-1, (config["z_categoricals"], config["z_categories"]))
        z_dist = ObservationModel.create_z_dist(z_logits)
        return z_dist

    def sample_z(self, z_dist, reparameterized=False, temperature=1, idx=None, return_logits=False):
        logits = z_dist.base_dist.logits
        assert (not reparameterized) == utils.check_no_grad(logits)
        if temperature == 0:
            assert not reparameterized
            with torch.no_grad():
                if idx is not None:
                    logits = logits[idx]
                indices = torch.argmax(logits, dim=-1)
                z = (
                    F.one_hot(indices, num_classes=self.config["z_categories"])
                    .flatten(2, 3)
                    .float()
                )
            if return_logits:
                return z, logits  # actually wrong logits for temperature = 0
            return z

        if temperature != 1 or idx is not None:
            if idx is not None:
                logits = logits[idx]
            z_dist = ObservationModel.create_z_dist(logits, temperature)
            if return_logits:
                logits = z_dist.base_dist.logits  # return new normalized logits

        z = z_dist.rsample() if reparameterized else z_dist.sample()
        z = z.flatten(2, 3)
        if return_logits:
            return z, logits
        return z

    def decode(self, z):
        config = self.config
        shape = z.shape[:2]
        z = z.flatten(0, 1)
        recons = self.decoder(z)
        if not config["env_grayscale"]:
            recons = recons.unflatten(1, (config["env_frame_stack"], 3))
            recons = recons.permute(0, 1, 3, 4, 2)
        recons = recons.unflatten(0, shape)
        return recons


class DynamicsModel(nn.Module):
    def __init__(self, config, z_dim, num_actions):
        super().__init__()
        self.config = config

        embeds = {
            "z": {"in_dim": z_dim, "categorical": False},
            "a": {"in_dim": num_actions, "categorical": True},
        }
        modality_order = ["z", "a"]
        num_current = 2

        if config["dyn_input_rewards"]:
            embeds["r"] = {"in_dim": 0, "categorical": False}
            modality_order.append("r")

        if config["dyn_input_discounts"]:
            embeds["g"] = {"in_dim": 0, "categorical": False}
            modality_order.append("g")

        self.modality_order = modality_order

        out_heads = {
            "z": {"hidden_dims": config["dyn_z_dims"], "out_dim": z_dim},
            "r": {"hidden_dims": config["dyn_reward_dims"], "out_dim": 1, "final_bias_init": 0.0},
            "g": {
                "hidden_dims": config["dyn_discount_dims"],
                "out_dim": 1,
                "final_bias_init": config["env_discount_factor"],
            },
        }

        memory_length = config["wm_memory_length"]
        max_length = 1 + config["wm_sequence_length"]  # 1 for context
        self.prediction_net = PredictionNet(
            modality_order,
            num_current,
            embeds,
            out_heads,
            embed_dim=config["dyn_embed_dim"],
            activation=config["dyn_act"],
            norm=config["dyn_norm"],
            dropout_p=config["dyn_dropout"],
            feedforward_dim=config["dyn_feedforward_dim"],
            head_dim=config["dyn_head_dim"],
            num_heads=config["dyn_num_heads"],
            num_layers=config["dyn_num_layers"],
            memory_length=memory_length,
            max_length=max_length,
        )

    @property
    def h_dim(self):
        return self.prediction_net.embed_dim

    def predict(
        self,
        z,
        a,
        r,
        g,
        d,
        tgt_length,
        heads=None,
        mems=None,
        return_attention=False,
        compute_consistency=False,
    ):
        assert utils.check_no_grad(z, a, r, g, d)
        assert mems is None or utils.check_no_grad(*mems)
        config = self.config

        if compute_consistency:
            tgt_length += 1  # add 1 timestep for context

        inputs = {"z": z, "a": a, "r": r, "g": g}
        heads = tuple(heads) if heads is not None else ("z", "r", "g")

        outputs = self.prediction_net(
            inputs,
            tgt_length,
            stop_mask=d,
            heads=heads,
            mems=mems,
            return_attention=return_attention,
        )
        out, h, mems, attention = outputs if return_attention else (outputs + (None,))

        preds = {}

        if "z" in heads:  # latent states
            z_categoricals = config["z_categoricals"]
            z_categories = config["z_categories"]
            z_logits = out["z"].unflatten(-1, (z_categoricals, z_categories))

            if compute_consistency:
                # used for consistency loss
                preds["z_hat_probs"] = ObservationModel.create_z_dist(
                    z_logits[:, :-1].detach()
                ).base_dist.probs
                z_logits = z_logits[:, 1:]  # remove context

            z_dist = ObservationModel.create_z_dist(z_logits)
            preds["z_dist"] = z_dist

        if "r" in heads:  # rewards
            r_params = out["r"]
            if compute_consistency:
                r_params = r_params[:, 1:]  # remove context
            r_mean = r_params.squeeze(-1)
            r_dist = D.Normal(r_mean, torch.ones_like(r_mean))

            r_pred = r_dist.mean
            preds["r_dist"] = r_dist  # used for dynamics loss
            preds["r"] = r_pred

        if "g" in heads:  # discounts
            g_params = out["g"]
            if compute_consistency:
                g_params = g_params[:, 1:]  # remove context
            g_mean = g_params.squeeze(-1)
            g_dist = D.Bernoulli(logits=g_mean)

            g_pred = torch.clip(g_dist.mean, 0, 1)
            preds["g_dist"] = g_dist  # used for dynamics loss
            preds["g"] = g_pred

        return (preds, h, mems) if not return_attention else (preds, h, mems, attention)


# ---- end vendored twm/world_model.py ----


# ---- twm/actor_critic.py (vendored verbatim) ----
class ActorCritic(nn.Module):
    def __init__(self, config, num_actions, z_dim, h_dim):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        activation = config["ac_act"]
        norm = config["ac_norm"]
        dropout_p = config["ac_dropout"]

        input_dim = z_dim
        if config["ac_input_h"]:
            input_dim += h_dim

        self.h_norm = get_norm_1d(config["ac_h_norm"], h_dim)
        self.trunk = nn.Identity()
        self.actor_model = MLP(
            input_dim,
            config["actor_dims"],
            num_actions,
            activation,
            norm=norm,
            dropout_p=dropout_p,
            weight_initializer="orthogonal",
            bias_initializer="zeros",
        )
        self.critic_model = MLP(
            input_dim,
            config["critic_dims"],
            1,
            activation,
            norm=norm,
            dropout_p=dropout_p,
            weight_initializer="orthogonal",
            bias_initializer="zeros",
        )
        if config["critic_target_interval"] > 1:
            self.target_critic_model = copy.deepcopy(self.critic_model).requires_grad_(False)
            self.register_buffer("target_critic_lag", torch.zeros(1, dtype=torch.long))

        self.actor_optimizer = utils.AdamOptim(
            self.actor_model.parameters(),
            lr=config["actor_lr"],
            eps=config["actor_eps"],
            weight_decay=config["actor_wd"],
            grad_clip=config["actor_grad_clip"],
        )
        self.critic_optimizer = utils.AdamOptim(
            self.critic_model.parameters(),
            lr=config["critic_lr"],
            eps=config["critic_eps"],
            weight_decay=config["critic_wd"],
            grad_clip=config["critic_grad_clip"],
        )

        self.sync_target()

    @torch.no_grad()
    def _prepare_inputs(self, z, h):
        assert utils.check_no_grad(z, h)
        assert h is None or utils.same_batch_shape([z, h])
        config = self.config
        if config["ac_input_h"]:
            h = self.h_norm(h)
            x = torch.cat([z, h], dim=-1)
        else:
            x = z
        shape = x.shape[:2]
        x = self.trunk(x.flatten(0, 1)).unflatten(0, shape)
        return x

    def actor(self, x):
        shape = x.shape[:2]
        logits = self.actor_model(x.flatten(0, 1)).unflatten(0, shape)
        return logits

    def critic(self, x):
        shape = x.shape[:2]
        values = self.critic_model(x.flatten(0, 1)).squeeze(-1).unflatten(0, shape)
        return values

    def sync_target(self):
        if self.config["critic_target_interval"] > 1:
            self.target_critic_lag[:] = 0
            self.target_critic_model.load_state_dict(self.critic_model.state_dict())

    @torch.no_grad()
    def policy(self, z, h, temperature=1):
        assert utils.check_no_grad(z, h)
        self.eval()
        x = self._prepare_inputs(z, h)
        logits = self.actor(x)

        if temperature == 0:
            actions = logits.argmax(dim=-1)
        else:
            if temperature != 1 or True:
                logits = logits / temperature
            actions = D.Categorical(logits=logits / temperature).sample()
        return actions


# ---- end vendored twm/actor_critic.py ----


# ---- staging wrapper (new code, not in original repo) ----
class TWMNet(nn.Module):
    """Staging wrapper exercising the real WorldModel construction
    (ObservationModel CNN-encoder -> categorical latent z, DynamicsModel's
    TransformerXL PredictionNet consuming interleaved z/a/r/g token streams,
    and ActorCritic's MLP actor+critic reading the model's (z, h) state) as a
    single traceable module. Reproduces `WorldModel.optimize`'s real forward
    path (encode -> sample_z -> dyn_model.predict -> decode) at reduced scale,
    followed by `ActorCritic.actor`/`critic` on the resulting latent state --
    matching the exact real-code call sequence, at tiny widths/depths so the
    trace runs quickly."""

    def __init__(self):
        super().__init__()
        config = dict(
            obs_channels=8,
            obs_act="relu",
            obs_norm="none",
            obs_dropout=0.0,
            env_frame_stack=1,
            env_grayscale=True,
            z_categoricals=4,
            z_categories=4,
            dyn_input_rewards=True,
            dyn_input_discounts=True,
            dyn_z_dims=[],
            dyn_reward_dims=[],
            dyn_discount_dims=[],
            dyn_embed_dim=16,
            dyn_act="relu",
            dyn_norm="none",
            dyn_dropout=0.0,
            dyn_feedforward_dim=32,
            dyn_head_dim=8,
            dyn_num_heads=2,
            dyn_num_layers=1,
            wm_memory_length=2,
            wm_sequence_length=2,
            env_discount_factor=0.995,
            obs_lr=1e-4,
            obs_eps=1e-5,
            obs_wd=0.0,
            obs_grad_clip=0,
            dyn_lr=1e-4,
            dyn_eps=1e-5,
            dyn_wd=0.0,
            dyn_grad_clip=0,
            ac_act="relu",
            ac_norm="none",
            ac_dropout=0.0,
            ac_input_h=True,
            ac_h_norm="none",
            actor_dims=[16],
            critic_dims=[16],
            actor_lr=1e-4,
            actor_eps=1e-5,
            actor_wd=0.0,
            actor_grad_clip=0,
            critic_lr=1e-4,
            critic_eps=1e-5,
            critic_wd=0.0,
            critic_grad_clip=0,
            critic_target_interval=1,
        )
        self.config = config
        num_actions = 4
        self.world_model = WorldModel(config, num_actions)
        self.actor_critic = ActorCritic(
            config, num_actions, self.world_model.z_dim, self.world_model.h_dim
        )

    def forward(self, o, a, r, terminated, truncated):
        # Mirrors WorldModel.optimize()'s real shape contract exactly:
        # o: (batch, seq+2, C, H, W) frame stack (context + middle + next);
        # a, r, terminated, truncated: (batch, seq+1) -- one longer than the
        # "middle" observation slice `o[:, 1:-1]` (same_batch_shape_time_offset).
        obs_model = self.world_model.obs_model
        dyn_model = self.world_model.dyn_model

        with torch.no_grad():
            context_z_dist = obs_model.encode(o[:, :1])
            context_z = obs_model.sample_z(context_z_dist)
            next_z_dist = obs_model.encode(o[:, -1:])
            next_logits = next_z_dist.base_dist.logits

        o_mid = o[:, 1:-1]
        z_dist = obs_model.encode(o_mid)
        z = obs_model.sample_z(z_dist, reparameterized=True)
        recons = obs_model.decode(z)

        z = z.detach()
        z = torch.cat([context_z, z], dim=1)
        z_logits = z_dist.base_dist.logits
        target_logits = torch.cat([z_logits[:, 1:].detach(), next_logits.detach()], dim=1)
        # `a` is fed straight to a categorical `nn.Embedding` (as in
        # ReplayBuffer.get_actions -> WorldModel.optimize's real dtype
        # contract: raw long action indices, no one-hot encoding).
        d = torch.logical_or(terminated, truncated)
        g = self.world_model.to_discounts(terminated)
        tgt_length = target_logits.shape[1]

        preds, h, mems = dyn_model.predict(
            z, a, r[:, :-1], g[:, :-1], d[:, :-1], tgt_length, compute_consistency=True
        )

        # In the real training loop, `z`/`h` returned by WorldModel.optimize()
        # only reach ActorCritic via the Dreamer imagination rollout
        # (trainer.py's `dreamer.imagine_reset`/`imagine_step` loop, which is
        # rollout/training-loop logic, not part of the traced network
        # architecture). ActorCritic's own real, self-contained inference
        # entry point is `policy(z, h)` (as `Dreamer.act()` calls it), which
        # requires detached (no-grad) inputs -- matching its real
        # `@torch.no_grad()` contract. We exercise ActorCritic here via its
        # real `policy` method on the detached world-model latents, plus a
        # direct `actor`/`critic` head call for full head coverage.
        z_detached = z.detach()
        h_detached = h.detach()
        actions = self.actor_critic.policy(z_detached, h_detached)
        x = self.actor_critic._prepare_inputs(z_detached, h_detached)
        action_logits = self.actor_critic.actor(x)
        values = self.actor_critic.critic(x)

        return (
            recons,
            preds["z_dist"].base_dist.logits,
            preds["r"],
            preds["g"],
            actions,
            action_logits,
            values,
        )


def build_twm():
    return TWMNet()


def example_input_twm():
    batch, seq = 2, 3
    o = torch.rand(batch, seq + 2, 1, 64, 64)
    a = torch.randint(0, 4, (batch, seq + 1))
    r = torch.randn(batch, seq + 1)
    terminated = torch.zeros(batch, seq + 1, dtype=torch.bool)
    truncated = torch.zeros(batch, seq + 1, dtype=torch.bool)
    return o, a, r, terminated, truncated


MENAGERIE_ENTRIES = [
    ("TWM", build_twm, example_input_twm, 2023, "vendored-pytorch"),
]
