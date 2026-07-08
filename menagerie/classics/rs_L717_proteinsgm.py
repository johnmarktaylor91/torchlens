# SOURCE: vendored from https://gitlab.com/mjslee0921/proteinsgm @ main
# Files: score_sde_pytorch/models/ncsnpp.py, score_sde_pytorch/models/layers.py,
#        score_sde_pytorch/models/normalization.py
#
# ProteinSGM (Lee, Kim 2022) -- a score-based (diffusion) generative model for de novo
# protein backbone design. The network is an NCSN++ (score_sde_pytorch, Song et al. 2021)
# 2D conv U-Net -- BigGAN-style up/down residual blocks with dilated GroupNorm, sinusoidal
# timestep embedding, and channel-wise self-attention at low resolution -- applied over a
# per-residue-pair 6D-coordinate feature map (distance + 3 dihedral/2 planar angle channels,
# `data.num_channels` in {5, 8} depending on length-conditioning) representing a protein
# backbone. Vendored verbatim from the real repo's `score_sde_pytorch/models/` package
# (itself an adaptation of the official google-research/score_sde_pytorch that the
# ProteinSGM authors ship in-repo) -- kept: NCSNpp top-level model, all layers.py building
# blocks it uses (AttnBlockpp, ResnetBlockDDPMpp/ResnetBlockBigGANpp, Upsample/Downsample,
# NIN, get_timestep_embedding, variance_scaling/default_init), and normalization.py's
# `get_normalization` dispatcher (GroupNorm is the only branch this config exercises).
# Config values below (nf, ch_mult, num_res_blocks, attn_resolutions, resblock_type,
# num_channels, max_res_num) reproduce the repo's `configs/no_cond.yml` shipped config.
# Dropped: the SDE/sampling/training machinery (losses.py, sampling.py, sde_lib.py, train.py)
# and the Rosetta-based structure-realization pipeline -- data/training utilities, not the
# score-network architecture itself.
import functools
import math
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- score_sde_pytorch/models/layers.py ---


def get_act(config):
    """Get activation functions from the config file."""
    if config.model.nonlinearity.lower() == "elu":
        return nn.ELU()
    elif config.model.nonlinearity.lower() == "relu":
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == "lrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == "swish":
        return nn.SiLU()
    else:
        raise NotImplementedError("activation function does not exist!")


def variance_scaling(
    scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"
):
    """Ported from JAX."""

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0) * np.sqrt(
                3 * variance
            )
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "uniform")


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.0, padding=0):
    """1x1 convolution with DDPM initialization."""
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def _einsum(a, b, c, x, y):
    einsum_str = f"{''.join(a)},{''.join(b)}->{''.join(c)}"
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape) : len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(
            default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True
        )
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3


class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    def __init__(self, channels, skip_rescale=False, init_scale=0.0):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6
        )
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


def naive_upsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))


class Upsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if with_conv:
            self.Conv_0 = conv3x3(in_ch, out_ch)

        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        h = F.interpolate(x, (H * 2, W * 2), "nearest")
        if self.with_conv:
            h = self.Conv_0(h)

        return h


class Downsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if with_conv:
            self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)

        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1))
            x = self.Conv_0(x)
        else:
            x = F.avg_pool2d(x, 2, stride=2)

        return x


class ResnetBlockDDPMpp(nn.Module):
    """ResBlock adapted from DDPM."""

    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        conv_shortcut=False,
        dropout=0.1,
        skip_rescale=False,
        init_scale=0.0,
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class ResnetBlockBigGANpp(nn.Module):
    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        up=False,
        down=False,
        dropout=0.1,
        skip_rescale=True,
        init_scale=0.0,
    ):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.up = up
        self.down = down

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))

        if self.up:
            h = naive_upsample_2d(h, factor=2)
            x = naive_upsample_2d(x, factor=2)
        elif self.down:
            h = naive_downsample_2d(h, factor=2)
            x = naive_downsample_2d(x, factor=2)

        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


# --- score_sde_pytorch/models/normalization.py ---


def get_normalization(config, conditional=False):
    """Obtain normalization modules from the config file."""
    norm = config.model.normalization
    if conditional:
        raise NotImplementedError(f"{norm} conditional normalization not needed by this config.")
    else:
        if norm == "GroupNorm":
            return nn.GroupNorm
        else:
            raise ValueError(f"Unknown normalization: {norm}")


# --- score_sde_pytorch/models/ncsnpp.py ---
# Adapted from score_sde_pytorch: https://github.com/yang-song/score_sde_pytorch
# Removed progressive module and Fourier timestep embeddings and FIR kernel

ResnetBlockDDPM = ResnetBlockDDPMpp
ResnetBlockBigGAN = ResnetBlockBigGANpp


def _get_sigmas(config):
    sigmas = np.exp(
        np.linspace(
            np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales
        )
    )
    return sigmas


class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        self.register_buffer("sigmas", torch.tensor(_get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            config.data.max_res_num // (2**i) for i in range(num_resolutions)
        ]

        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        init_scale = config.model.init_scale

        self.embedding_type = embedding_type = config.model.embedding_type.lower()

        assert embedding_type in ["fourier", "positional"]

        modules = []
        embed_dim = nf
        modules.append(nn.Linear(embed_dim, nf * 4))
        modules[-1].weight.data = default_init()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(nf * 4, nf * 4))
        modules[-1].weight.data = default_init()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample_ = functools.partial(Upsample, with_conv=resamp_with_conv)

        Downsample_ = functools.partial(Downsample, with_conv=resamp_with_conv)

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Downsampling block

        channels = config.data.num_channels
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    modules.append(Downsample_(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample_(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond):
        modules = self.all_modules
        m_idx = 0
        # Sinusoidal positional embeddings.
        timesteps = time_cond
        used_sigmas = self.sigmas[time_cond.long()]
        temb = get_timestep_embedding(timesteps, self.nf)

        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        # Downsampling block
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)
        if self.config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        return h


class _NS:
    """Minimal attribute-namespace standing in for the repo's ml_collections.ConfigDict."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _make_config():
    # Reproduces the real repo's configs/no_cond.yml (unconditional NCSN++ over an
    # 8-channel per-residue-pair 6D feature map), with nf/ch_mult/num_res_blocks/
    # max_res_num shrunk for a fast CPU trace -- same architecture, smaller instance.
    model = _NS(
        sigma_max=100.0,
        sigma_min=0.01,
        num_scales=10,
        dropout=0.1,
        embedding_type="positional",
        name="ncsnpp",
        scale_by_sigma=True,
        normalization="GroupNorm",
        nonlinearity="swish",
        nf=8,
        ch_mult=(1, 1, 2),
        num_res_blocks=1,
        attn_resolutions=(4,),
        resamp_with_conv=True,
        skip_rescale=True,
        resblock_type="biggan",
        init_scale=0.0,
    )
    data = _NS(max_res_num=16, num_channels=8)
    return _NS(model=model, data=data)


def build_proteinsgm():
    torch.manual_seed(0)
    config = _make_config()
    return NCSNpp(config)


def example_input_proteinsgm():
    torch.manual_seed(0)
    batch, channels, res = 1, 8, 16
    x = torch.randn(batch, channels, res, res)
    time_cond = torch.randint(0, 10, (batch,))
    return (x, time_cond)


MENAGERIE_ENTRIES = [
    ("ProteinSGM", "build_proteinsgm", "example_input_proteinsgm", 2022, "SOURCE_AVAILABLE"),
]
