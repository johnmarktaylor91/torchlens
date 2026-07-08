# FAITHFUL PORT of mindspore/models research/cv/BiMLP @ master (original framework: MindSpore)
# https://gitee.com/mindspore/models/raw/master/research/cv/BiMLP/src/bimlp.py
# https://gitee.com/mindspore/models/raw/master/research/cv/BiMLP/src/quan_conv.py
#
# "BiMLP: Compact Binary Architectures for Vision Multi-Layer Perceptrons"
# (Xu et al., NeurIPS 2022, Huawei Noah's Ark Lab). No official PyTorch release; the
# official code is MindSpore (`mindspore.nn.Cell`/`mindspore.ops`), hosted on Gitee
# only. MindSpore is not a base lib here and is not reasonably installable alongside
# torch (separate deep-learning framework, GPU/Ascend-targeted build matrix), so this
# is a faithful transcription of the real forward-pass architecture rather than a
# vendor. The architecture is transcribed module-for-module from the two real source
# files above, translating MindSpore `nn.Cell.construct()` -> torch `nn.Module.forward()`
# and MindSpore ops -> their exact torch equivalents (`ops.Sign`->`torch.sign`,
# `ops.Rint`->`torch.round`, `mindspore.nn.Conv2d(pad_mode="pad", group=...)`->
# `torch.nn.Conv2d(padding=..., groups=...)`, `mnp.tile`->`torch.tile`, `ops.Concat`->
# `torch.cat`, MindSpore's `bprop` straight-through custom backward for `Signer`/
# `XnorM`/`ScaleSigner`/`BirealActivation` -> `torch.autograd.Function` with an
# identity/pass-through `backward`):
#   - `QuanConv` (quan_conv.py): a Conv2d whose forward path (`construct`) binarizes
#     BOTH the weight (`quan_w`) and the input activation (`quan_a`) before convolving,
#     when `nbit_w < 32` / `nbit_a < 32`. BiMLP's model file only ever instantiates it
#     with `quan_name_w="xnor", quan_name_a="xnor", nbit_w=1, nbit_a=1` (see every
#     `QuanConv(...)` call in bimlp.py), so only the XNOR-Net weight path (`xnor_w` ->
#     `XnorM`: `sign(x_c) * mean(|x_c|, dim=1, keepdim=True)`) and the XNOR-Net
#     activation path (`quan_name_a="xnor"` maps to `dorefa_a` in `name_a_dict`, i.e.
#     `Quantizer()(clip(inp,-1,1), nbit_a=1)` = round-to-nearest-of-{-1,1} after
#     clipping) are needed/ported; the other quan_name_w/quan_name_a branches
#     (bnn/dorefa/pact/wrpn/bireal) are omitted as genuinely dead code for this
#     specific model (never selected by any BiMLP_S/BiMLP_M call site).
#   - `LearnableBias`, `prelu`, `RPReLU` (bimlp.py): the ReCU-style "reparameterized
#     PReLU" activation used everywhere in place of a plain nonlinearity -- a learnable
#     per-channel additive bias, then a learnable per-channel PReLU, then a second
#     learnable per-channel additive bias.
#   - `Mlp`, `PATM` (Phase-Aware Token Mixing, from WaveMLP), `WaveBlock`, `Downsample`,
#     `PatchEmbedOverlapping`, `basic_blocks`, `WaveNet`: ported verbatim, including the
#     4-way multi-scale-pool `Downsample` (kernel 2/3/5/7, all stride 2, averaged), the
#     4-channel-group tiling/averaging in `Mlp.construct` (`hidden_features //
#     in_features == 4` -> tile x4 else average groups of 4 channels), and `PATM`'s
#     cos/theta phase-modulated token mixing (`x_h*cos(theta_h)` / `x_h*sin(theta_h)`
#     concatenated then re-summed via strided slicing, then 1D depthwise 7-taps along H
#     and W). `BiMLP_S` (the smallest published variant: layers=[2,2,4,2],
#     embed_dims=[64,128,320,512], BatchNorm2d norm_layer) is the entry point built
#     here, at reduced embed_dims/layers for a fast trace (architecture unchanged --
#     WaveNet is fully shape-parametric over `embed_dims`/`layers`/`mlp_ratios`, exactly
#     as the real BiMLP_S/BiMLP_M factory functions are themselves thin wrappers
#     selecting those same constructor arguments).
import collections.abc
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, 2))


# ---- quan_conv.py (ported) ----


class _Signer(torch.autograd.Function):
    """standard sign with STE (quan_conv.py's `Signer`)."""

    @staticmethod
    def forward(ctx, inp):
        return torch.sign(inp)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _XnorM(torch.autograd.Function):
    """sign in xnor-net for weights (quan_conv.py's `XnorM`):
    sign(x_c) * E(|x_c|) with E over dim=1, keepdim."""

    @staticmethod
    def forward(ctx, inp):
        return torch.sign(inp) * torch.mean(torch.abs(inp), dim=1, keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _xnor_w(w, nbit_w):
    if nbit_w != 1:
        raise ValueError("nbit_w must be 1 in XNOR-Net.")
    return _XnorM.apply(w)


class _Quantizer(nn.Module):
    """quan_conv.py's `Quantizer`: round-to-nearest at `nbit`-precision within
    alpha*[0,1] or alpha*[-1,1]."""

    def forward(self, inp, nbit, alpha=None, offset=None):
        scale = (2**nbit - 1) if alpha is None else (2**nbit - 1) / float(alpha)
        if offset is None:
            return torch.round(inp * scale) / scale
        return (torch.round(inp * scale) + torch.round(offset)) / scale


def _dorefa_a(inp, nbit_a):
    # quan_conv.py's `dorefa_a`, reached via quan_name_a="xnor" (name_a_dict maps
    # "xnor" -> dorefa_a).
    return _Quantizer()(torch.clamp(inp, -1.0, 1.0), nbit_a)


class QuanConv(nn.Conv2d):
    """Port of quan_conv.py's `QuanConv` (extends mindspore.nn.Conv2d), restricted to
    the (quan_name_w="xnor", quan_name_a="xnor", nbit_w=1, nbit_a=1) path -- the only
    configuration BiMLP's model file ever instantiates."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        nbit_w=1,
        nbit_a=1,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a

    def forward(self, inp):
        w = _xnor_w(self.weight, self.nbit_w) if self.nbit_w < 32 else self.weight
        x = _dorefa_a(inp, self.nbit_a) if self.nbit_a < 32 else F.relu(inp)
        return self._conv_forward(x, w, self.bias)


# ---- bimlp.py (ported) ----


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1))

    def forward(self, x):
        return x + self.bias.expand_as(x)


class Prelu(nn.Module):
    """bimlp.py's `prelu`: split positive/negative parts, scale the negative part by a
    learnable per-channel weight (equivalent to PReLU with a (1,C,1,1)-shaped weight)."""

    def __init__(self, out_chn):
        super().__init__()
        self.w = nn.Parameter(0.25 * torch.ones(1, out_chn, 1, 1))

    def forward(self, x):
        pos = torch.clamp(x, min=0)
        neg = torch.clamp(x, max=0)
        return pos + self.w * neg


class RPReLU(nn.Module):
    def __init__(self, out_chn):
        super().__init__()
        self.move1 = LearnableBias(out_chn)
        self.move2 = LearnableBias(out_chn)
        self.act = Prelu(out_chn)

    def forward(self, x):
        x = self.move1(x)
        x = self.act(x)
        x = self.move2(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.move1 = LearnableBias(in_features)
        self.fc1 = QuanConv(in_features, hidden_features, kernel_size=1, bias=False)
        self.act1 = RPReLU(hidden_features)
        self.act2 = RPReLU(hidden_features)
        self.act3 = RPReLU(out_features)
        self.move2 = LearnableBias(hidden_features)
        self.fc2 = QuanConv(hidden_features, out_features, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)
        self.norm0 = nn.BatchNorm2d(hidden_features)

        self.hidden_features = hidden_features
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        if self.hidden_features // self.in_features == 4:
            inp = x.tile((1, 4, 1, 1))
        else:
            inp = (x[:, ::4, :, :] + x[:, 1::4, :, :] + x[:, 2::4, :, :] + x[:, 3::4, :, :]) / 4.0
        inp = self.norm0(inp)
        inp = self.act1(inp)
        x = self.move1(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = x + inp
        x = self.act2(x)

        if self.out_features // self.hidden_features == 12:
            inp = x.tile((1, 12, 1, 1))
        else:
            inp = (x[:, ::4, :, :] + x[:, 1::4, :, :] + x[:, 2::4, :, :] + x[:, 3::4, :, :]) / 4.0
        x = self.move2(x)
        x2 = self.fc2(x)
        x2 = self.norm2(x2)
        x2 = x2 + inp
        x2 = self.act3(x2)
        return x2


class PATM(nn.Module):
    """Phase-Aware Token Mixing (PATM), from WaveMLP, ported verbatim from bimlp.py."""

    def __init__(self, dim, mode="fc"):
        super().__init__()
        self.fc_h = QuanConv(dim, dim, kernel_size=1, bias=False)
        self.fc_w = QuanConv(dim, dim, kernel_size=1, bias=False)
        self.fc_c = QuanConv(dim, dim, kernel_size=1, bias=False)
        self.move_fch = LearnableBias(dim)
        self.move_fcw = LearnableBias(dim)
        self.move_fcc = LearnableBias(dim)
        self.fc_h_bn = nn.BatchNorm2d(dim)
        self.fc_w_bn = nn.BatchNorm2d(dim)
        self.fc_h_act = RPReLU(dim)
        self.fc_w_act = RPReLU(dim)

        self.tfc_h = QuanConv(
            2 * dim, dim, kernel_size=(1, 7), stride=1, padding=(0, 3), groups=dim, bias=False
        )
        self.tfc_w = QuanConv(
            2 * dim, dim, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=dim, bias=False
        )
        self.move_tfch = LearnableBias(2 * dim)
        self.move_tfcw = LearnableBias(2 * dim)

        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = QuanConv(dim, dim, kernel_size=1, bias=False)
        self.move_proj = LearnableBias(dim)
        self.mode = mode

        if mode == "fc":
            self.move_thetah = LearnableBias(dim)
            self.theta_h_conv = QuanConv(dim, dim, kernel_size=1, bias=False)
            self.theta_h_bn = nn.BatchNorm2d(dim)
            self.theta_h_act = RPReLU(dim)
            self.move_thetaw = LearnableBias(dim)
            self.theta_w_conv = QuanConv(dim, dim, kernel_size=1, bias=False)
            self.theta_w_bn = nn.BatchNorm2d(dim)
            self.theta_w_act = RPReLU(dim)
        else:
            self.theta_h_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
            )
            self.theta_w_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
            )

        self.act_c = RPReLU(dim)
        self.act_h = RPReLU(dim)
        self.act_w = RPReLU(dim)
        self.act_proj = RPReLU(dim)
        self.norm_c = nn.BatchNorm2d(dim)
        self.norm_h = nn.BatchNorm2d(dim)
        self.norm_w = nn.BatchNorm2d(dim)
        self.norm_proj = nn.BatchNorm2d(dim)

    def forward(self, inp):
        b, c, _, _ = inp.shape
        theta_h_in = self.move_thetah(inp)
        theta_h = self.theta_h_conv(theta_h_in)
        theta_h = self.theta_h_bn(theta_h)
        theta_h = self.theta_h_act(theta_h + theta_h_in)

        theta_w_in = self.move_thetaw(inp)
        theta_w = self.theta_w_conv(theta_w_in)
        theta_w = self.theta_w_bn(theta_w)
        theta_w = self.theta_w_act(theta_w + theta_w_in)

        x_h_in = self.move_fch(inp)
        x_h = self.fc_h(x_h_in)
        x_h = self.fc_h_bn(x_h)
        x_h = self.fc_h_act(x_h + x_h_in)

        x_w_in = self.move_fcw(inp)
        x_w = self.fc_w(x_w_in)
        x_w = self.fc_w_bn(x_w)
        x_w = self.fc_w_act(x_w + x_w_in)

        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)

        x_h = self.move_tfch(x_h)
        x_w = self.move_tfcw(x_w)

        fc_c_in = self.move_fcc(inp)
        th = x_h[:, ::2, :, :] + x_h[:, 1::2, :, :]
        tw = x_w[:, ::2, :, :] + x_w[:, 1::2, :, :]

        ah = self.norm_h(self.tfc_h(x_h)) + (th / 2.0)
        aw = self.norm_w(self.tfc_w(x_w)) + (tw / 2.0)

        h = self.act_h(ah)
        w = self.act_w(aw)
        c_out = self.act_c(self.norm_c(self.fc_c(fc_c_in)) + fc_c_in)
        a = F.adaptive_avg_pool2d(h + w + c_out + inp, (1, 1))

        a = self.reweight(a).reshape(b, c, 3)
        a = a.permute(2, 0, 1)
        a = torch.softmax(a, dim=0)
        a = a.unsqueeze(-1).unsqueeze(-1)

        x = h * a[0] + w * a[1] + c_out * a[2] + inp
        x_in = self.move_proj(x)
        x = self.act_proj(self.norm_proj(self.proj(x_in)) + x_in)
        return x


class WaveBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, norm_layer=nn.BatchNorm2d, mode="fc"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, mode=mode)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

        self.move_h = LearnableBias(dim)
        self.move_w = LearnableBias(dim)
        self.sfc_h = QuanConv(
            dim, dim, kernel_size=(1, 7), stride=1, padding=(0, 3), groups=dim, bias=False
        )
        self.sfc_w = QuanConv(
            dim, dim, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=dim, bias=False
        )
        self.act1 = RPReLU(dim)
        self.act2 = RPReLU(dim)
        self.norm_h = nn.BatchNorm2d(dim)
        self.norm_w = nn.BatchNorm2d(dim)

        self.weight = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))

        mid = self.norm2(x)
        a = self.mlp(mid)
        mid_h = self.move_h(mid)
        mid_w = self.move_w(mid)
        b = self.act1(self.norm_h(self.sfc_h(mid_h)) + mid_h)
        c = self.act2(self.norm_w(self.sfc_w(mid_w)) + mid_w)
        x = x + (
            self.weight[0] * b + self.weight[1] * c + (1 - self.weight[0] - self.weight[1]) * a
        )
        return x


class PatchEmbedOverlapping(nn.Module):
    def __init__(
        self,
        patch_size=16,
        stride=16,
        padding=0,
        in_chans=3,
        embed_dim=768,
        norm_layer=nn.BatchNorm2d,
        groups=1,
        use_norm=True,
    ):
        super().__init__()
        patch_size = _to_2tuple(patch_size)
        stride = _to_2tuple(stride)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=True,
        )
        self.norm = norm_layer(embed_dim) if use_norm else nn.Identity()
        self.act = RPReLU(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Downsample(nn.Module):
    def __init__(
        self, in_embed_dim, out_embed_dim, patch_size, norm_layer=nn.BatchNorm2d, use_norm=True
    ):
        super().__init__()
        assert patch_size == 2, patch_size
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=7, stride=2, padding=3)
        self.proj = nn.Conv2d(
            in_embed_dim, out_embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.norm = norm_layer(out_embed_dim) if use_norm else nn.Identity()
        self.act = RPReLU(out_embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = (self.pool0(x) + self.pool1(x) + self.pool2(x) + self.pool3(x)) / 4.0
        return x


def basic_blocks(dim, index, layers, mlp_ratio=3.0, norm_layer=nn.BatchNorm2d, mode="fc"):
    blocks = []
    for _block_idx in range(layers[index]):
        blocks.append(WaveBlock(dim, mlp_ratio=mlp_ratio, norm_layer=norm_layer, mode=mode))
    return nn.Sequential(*blocks)


class WaveNet(nn.Module):
    """Port of bimlp.py's `WaveNet` (the BiMLP backbone)."""

    def __init__(
        self,
        layers,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dims=None,
        transitions=None,
        mlp_ratios=None,
        norm_layer=nn.BatchNorm2d,
        mode="fc",
        ds_use_norm=True,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbedOverlapping(
            patch_size=7,
            stride=4,
            padding=2,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer,
            use_norm=ds_use_norm,
        )

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                mlp_ratio=mlp_ratios[i],
                norm_layer=norm_layer,
                mode=mode,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    Downsample(
                        embed_dims[i],
                        embed_dims[i + 1],
                        2,
                        norm_layer=norm_layer,
                        use_norm=ds_use_norm,
                    )
                )

        self.network = nn.ModuleList(network)
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        return self.patch_embed(x)

    def forward_tokens(self, x):
        for block in self.network:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        return self.head(x)


def _bimlp_s_tiny(num_classes=10):
    # BiMLP_S factory (bimlp.py): transitions=[True]*4, layers=[2,2,4,2],
    # mlp_ratios=[4,4,4,4], embed_dims=[64,128,320,512] at full scale. Shrunk here to a
    # depth-1-per-stage, narrow-channel configuration for a fast trace -- WaveNet is
    # fully shape-parametric, so this is the identical architecture at reduced size,
    # not an architectural change.
    transitions = [True, True, True, True]
    layers = [1, 1, 1, 1]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [8, 16, 24, 32]
    return WaveNet(
        layers,
        embed_dims=embed_dims,
        patch_size=7,
        transitions=transitions,
        mlp_ratios=mlp_ratios,
        num_classes=num_classes,
    )


def build_bimlp():
    model = _bimlp_s_tiny(num_classes=10)
    model.eval()
    return model


def example_input_bimlp():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    ("BiMLP_S", "build_bimlp", "example_input_bimlp", 2022, "ported"),
]
