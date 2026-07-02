# FAITHFUL PORT of https://github.com/ICSResearch/TransCS @ master
# (models/module.py + models/demo.py, commit 363287f5ab83f018c09a0b4d62b71c54c378c95;
# original framework: PyTorch, but pinned to a removed private API)
#
# TransCS: A Transformer-Based Hybrid Architecture for Image Compressed
# Sensing (Shen, Gan, Fu, Zeng, IEEE Transactions on Image Processing 2022).
# Official ICSResearch/TransCS repo.
#
# TransCS is a deep-unrolled ISTA-style compressed-sensing reconstruction
# network (``HybridNet``): a learned linear sampling operator ``phi``/``Q``,
# followed by ``num_layers`` unrolling stages each combining (1) a residual
# gradient-descent data-consistency step, (2) a small CNN-based "sparsifying"
# pre/post block (custom ``Conv`` module using a fixed identity-plus-learned
# residual 3x3 kernel), and (3) a windowed Transformer encoder/decoder
# (``Trans``) that operates on 8x8 patches with soft-thresholding in between.
#
# This is a FAITHFUL PORT rather than a straight vendor because the repo's
# ``models/module.py`` imports ``from torch._six import container_abcs``,
# which CPython/PyTorch removed years ago (torch._six was deleted); every
# current torch raises ImportError on that line before any model code runs.
# The only change made below is replacing that dead import with the
# std-lib equivalent it aliased (``collections.abc.Iterable``) -- same
# runtime behavior, zero architectural change. Every layer/mechanism
# (Conv's identity-plus-residual kernel trick, pre_layer/post_layer,
# Trans/PositionalEmbedding/MultiHeadSelfAttention/EncoderLayer/DecoderLayer,
# and HybridNet's sampling()/recon()/size8to256()/size256to8() block
# reshuffling) is transcribed verbatim from the real repo code.

import math
from collections.abc import Iterable
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn import init
from torch.nn.modules.module import Module

MENAGERIE_ZOO = "ported-pytorch"


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


# ---------------------------------------------------------------------------
# models/module.py (Transformer building blocks)
# ---------------------------------------------------------------------------


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :, :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.ModuleList()
        for i in range(3):
            proj = nn.Sequential(nn.Linear(dim, dim, bias=True))
            self.projection.append(proj)

    def forward(self, x, extra=None):
        if extra is None:
            q, k, v = self.projection[0](x), self.projection[1](x), self.projection[2](x)
        else:
            q, k, v = self.projection[0](x), self.projection[1](extra), self.projection[2](extra)
        return q, k, v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, head=None):
        super().__init__()
        self.dim = dim
        self.head = head or 8
        if self.dim % self.head != 0:
            raise NotImplementedError(
                "Dimensions cannot be divisible by head, get dim {}, but head {}".format(dim, head)
            )
        self.attention = Attention(dim)

    def forward(self, x, extra=None):
        q, k, v = self.attention(x, extra)
        _temp = torch.chunk(q, self.head, dim=2)
        q = torch.cat(_temp, dim=0)
        _temp = torch.chunk(k, self.head, dim=2)
        k = torch.cat(_temp, dim=0)
        _temp = torch.chunk(v, self.head, dim=2)
        v = torch.cat(_temp, dim=0)

        weights = torch.softmax(torch.matmul(q, k.permute(0, 2, 1)), dim=2)
        y = torch.matmul(weights, v)

        _temp = torch.chunk(y, self.head, dim=0)
        y = torch.cat(_temp, dim=2)
        return y


class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, n_head, drop):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MHA = MultiHeadSelfAttention(dim=d_model, head=n_head)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.ELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, d_model, bias=True),
        )

    def forward(self, x):
        y = x + self.norm1(self.MHA(x))
        y = y + self.norm2(self.MLP(y))
        return y


class DecoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, n_head, drop):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.MHA1 = MultiHeadSelfAttention(dim=d_model, head=n_head)
        self.MHA2 = MultiHeadSelfAttention(dim=d_model, head=n_head)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.ELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, d_model, bias=True),
        )

    def forward(self, x, extra_x):
        y = x + self.norm1(self.MHA1(x))
        y = y + self.norm2(self.MHA2(y, extra_x))
        y = y + self.norm3(self.MLP(y))
        return y


class Encoder(nn.Module):
    def __init__(self, dim, hidden_dim=None, num_head=8, factor=4, dropout=0.0):
        super(Encoder, self).__init__()
        hidden_dim = hidden_dim or dim * factor
        assert dim % num_head == 0, f"dim {dim} should be divided by num_heads {num_head}."
        self.dim = dim
        self.size = int(dim**0.5)
        self.pos_embed = PositionalEmbedding(d_model=dim)
        self.layer = EncoderLayer(d_model=dim, hidden_dim=hidden_dim, n_head=num_head, drop=dropout)

    def forward(self, x):
        x = Rearrange("b c h w -> b c (h w)")(x)
        x = x * math.sqrt(self.dim)
        x = self.pos_embed(x)
        y = self.layer(x)
        y = Rearrange("b c (h w)-> b c h w", h=self.size, w=self.size)(y)
        return y


class Decoder(nn.Module):
    def __init__(self, dim, hidden_dim=None, num_head=8, factor=4, dropout=0.0):
        super(Decoder, self).__init__()
        hidden_dim = hidden_dim or dim * factor
        assert dim % num_head == 0, f"dim {dim} should be divided by num_heads {num_head}."
        self.dim = dim
        self.size = int(dim**0.5)
        self.pos_embed = PositionalEmbedding(d_model=dim)
        self.layer = DecoderLayer(d_model=dim, hidden_dim=hidden_dim, n_head=num_head, drop=dropout)

    def forward(self, x, x_extra):
        x_extra = Rearrange("b c h w -> b c (h w)")(x_extra)
        x = Rearrange("b c h w -> b c (h w)")(x)
        y = x * math.sqrt(self.dim)
        y = self.pos_embed(y)
        y = self.layer(y, x_extra)
        y = Rearrange("b c (h w)-> b c h w", h=self.size, w=self.size)(y)
        return y


# ---------------------------------------------------------------------------
# models/demo.py (HybridNet: sampling + unrolled reconstruction)
# ---------------------------------------------------------------------------


class Conv(Module):
    def __init__(self, config, ic, oc):
        super(Conv, self).__init__()
        self.config = config
        self.ic = ic
        self.oc = oc
        self.w = nn.Parameter(torch.Tensor(oc, ic, 9))
        self.padding = _pair(1)
        self.init = nn.Parameter(torch.zeros([ic, 9, 9], dtype=torch.float32))
        init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, inputs):
        init_ = self.init + torch.eye(9, dtype=torch.float32).unsqueeze(0).repeat(
            (self.ic, 1, 1)
        ).to(self.config.device)
        weight = torch.reshape(
            torch.einsum("abc, dac->dab", init_, self.w), (self.oc, self.ic, 3, 3)
        )
        outputs = F.conv2d(inputs, weight, None, 1, self.padding)
        return outputs


class pre_layer(nn.Module):
    def __init__(self, config):
        super(pre_layer, self).__init__()

        self.num = 4

        self.conv_in = nn.Sequential(Conv(config, 1, 32), nn.BatchNorm2d(32), nn.ELU())

        self.conv = nn.ModuleList()
        for i in range(self.num):
            self.conv.append(nn.Sequential(Conv(config, 32, 32), nn.BatchNorm2d(32), nn.ELU()))

        self.conv_out = nn.Sequential(Conv(config, 32, 1))

    def forward(self, x_recon):
        x_recon = torch.transpose(x_recon, 0, 1).reshape([-1, 1, 32, 32])
        x_input = self.conv_in(x_recon)
        x_mid = x_input
        for i in range(self.num):
            x_mid = self.conv[i](x_mid)
        x_output = self.conv_out(x_mid)
        x_output = torch.transpose(x_output.reshape(-1, 1024), 0, 1)
        return x_output


class post_layer(nn.Module):
    def __init__(self, config):
        super(post_layer, self).__init__()

        self.num = 4

        self.conv_in = nn.Sequential(Conv(config, 1, 32), nn.BatchNorm2d(32), nn.ELU())

        self.conv = nn.ModuleList()
        for i in range(self.num):
            self.conv.append(nn.Sequential(Conv(config, 32, 32), nn.BatchNorm2d(32), nn.ELU()))

        self.conv_out = nn.Sequential(Conv(config, 32, 1))

    def forward(self, x_recon):
        x_input = self.conv_in(x_recon)
        x_mid = x_input
        for i in range(self.num):
            x_mid = self.conv[i](x_mid)
        x_output = self.conv_out(x_mid)
        return x_output


class Trans(nn.Module):
    def __init__(self, config, dim):
        super(Trans, self).__init__()
        self.config = config
        self.threshold = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
        self.encoder = Encoder(dim=dim)
        self.decoder = Decoder(dim=dim)

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        outputs = torch.mul(torch.sign(outputs), F.relu(torch.abs(outputs) - self.threshold))
        outputs = self.decoder(inputs, outputs)
        return outputs


class HybridNet(nn.Module):
    def __init__(self, config):
        super(HybridNet, self).__init__()
        self.config = config
        self.phi_size = 32
        points = self.phi_size**2
        phi_init = np.random.normal(
            0.0, (1 / points) ** 0.5, size=(int(config.ratio * points), points)
        )
        self.phi = nn.Parameter(torch.from_numpy(phi_init).float(), requires_grad=True)
        self.Q = nn.Parameter(torch.from_numpy(np.transpose(phi_init)).float(), requires_grad=True)

        self.num_layers = 6
        self.pre_block = nn.ModuleList()
        for i in range(self.num_layers):
            self.pre_block.append(pre_layer(config))

        self.post_block = nn.ModuleList()
        for i in range(self.num_layers):
            self.post_block.append(post_layer(config))

        self.trans = nn.ModuleList()
        for i in range(self.num_layers):
            self.trans.append(Trans(config, dim=8**2))

        self.weights = []
        self.etas = []
        for i in range(self.num_layers):
            self.weights.append(nn.Parameter(torch.tensor(1.0), requires_grad=True))
            self.register_parameter(
                "eta_" + str(i + 1), nn.Parameter(torch.tensor(0.1), requires_grad=True)
            )
            self.etas.append(eval("self.eta_" + str(i + 1)))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        y = self.sampling(inputs, self.phi_size)
        recon = self.recon(y, self.phi_size, batch_size)
        return recon

    def sampling(self, inputs, init_block):
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=init_block, dim=3), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=init_block, dim=2), dim=0)
        inputs = torch.reshape(inputs, [-1, init_block**2])
        inputs = torch.transpose(inputs, 0, 1)
        y = torch.matmul(self.phi, inputs)
        return y

    def recon(self, y, init_block, batch_size):
        idx = int(self.config.block_size / init_block)

        recon = torch.matmul(self.Q, y)
        for i in range(self.num_layers):
            recon = recon - self.weights[i] * torch.mm(
                torch.transpose(self.phi, 0, 1), (torch.mm(self.phi, recon) - y)
            )
            recon = recon - self.pre_block[i](recon)
            recon = torch.reshape(torch.transpose(recon, 0, 1), [-1, 1, init_block, init_block])
            recon = torch.cat(
                torch.split(recon, split_size_or_sections=idx * batch_size, dim=0), dim=2
            )
            recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)
            recon = self.size256to8(recon)
            recon = recon - self.etas[i] * self.trans[i](recon)
            recon = self.size8to256(recon)
            recon = recon - self.post_block[i](recon)

            recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=3), dim=0)
            recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=2), dim=0)
            recon = torch.reshape(recon, [-1, init_block**2])
            recon = torch.transpose(recon, 0, 1)

        recon = torch.reshape(torch.transpose(recon, 0, 1), [-1, 1, init_block, init_block])
        recon = torch.cat(torch.split(recon, split_size_or_sections=idx * batch_size, dim=0), dim=2)
        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)
        return recon

    def size8to256(self, inputs):
        idx = int(self.config.block_size / 8)
        outputs = torch.cat(torch.split(inputs, split_size_or_sections=idx, dim=1), dim=2)
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=1, dim=1), dim=3)
        return outputs

    def size256to8(self, inputs):
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=8, dim=3), dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=8, dim=2), dim=1)
        return inputs


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------


class _Config:
    """Same field names/roles as utils/config.py's GetConfig, minus the
    filesystem path setup (irrelevant to model construction)."""

    def __init__(self, ratio=0.25, device="cpu", block_size=32):
        self.ratio = ratio
        self.channel = 1
        self.block_size = block_size  # must be a multiple of 32 (phi_size) and 8
        self.device = torch.device(device)


def build_transcs():
    torch.manual_seed(0)
    np.random.seed(0)
    config = _Config(ratio=0.25, device="cpu", block_size=32)
    model = HybridNet(config)
    model.eval()
    return model


def example_input_transcs():
    torch.manual_seed(0)
    return torch.randn(1, 1, 32, 32)


MENAGERIE_ENTRIES = [
    ("TransCS", "build_transcs", "example_input_transcs", 2022, MENAGERIE_ZOO),
]
