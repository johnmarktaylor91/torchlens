# SOURCE: vendored from 2y7c3/Super-Resolution-Neural-Operator @ main
# https://raw.githubusercontent.com/2y7c3/Super-Resolution-Neural-Operator/main/models/sronet.py
# https://raw.githubusercontent.com/2y7c3/Super-Resolution-Neural-Operator/main/models/galerkin.py
# https://raw.githubusercontent.com/2y7c3/Super-Resolution-Neural-Operator/main/models/edsr.py
# https://raw.githubusercontent.com/2y7c3/Super-Resolution-Neural-Operator/main/utils.py
#
# SRNO (Wei & Zhang, CVPR 2023, "Super-Resolution Neural Operator") -- official
# reference PyTorch implementation. Continuous-coordinate arbitrary-scale image
# super-resolution: an EDSR-baseline CNN feature encoder followed by SRNO's core
# `simple_attn` "Galerkin-type" linear-attention operator layers (2 stacked blocks in
# this repo's default config) that map local feature neighborhoods to a continuous
# (coord, cell)-queried RGB field, evaluated at query coordinates via a 4-neighbor
# bilinear-area-weighted local ensemble (`query_rgb`) and added back to a bilinearly
# up-sampled base image (a residual global skip).
#
# Transcribed verbatim from the four real source files (sronet.py, galerkin.py,
# edsr.py, utils.py's `make_coord`): every Conv2d/LayerNorm/`simple_attn`
# attention-head arithmetic, the EDSR encoder body (ResBlock stack, head/body/tail),
# and SRNO's `gen_feat`/`query_rgb`/`forward` logic (4-corner coordinate shift,
# grid_sample feature/area gathering, area-weight swap, Galerkin attention blocks,
# final residual add of the bilinear-upsampled input) is unchanged. Only:
#   (1) the `models.register`/`models.make` mini-registry indirection is flattened --
#       `SRNO.__init__` here directly builds the real `make_edsr_baseline(...)` EDSR
#       encoder instead of going through `models.make(encoder_spec)`;
#   (2) the two hardcoded `.cuda()` calls in `query_rgb` (`make_coord(...).cuda()`) are
#       replaced with `.to(x.device)`-equivalent (both example construction and forward
#       state stay on the same device as the input) so the module runs on CPU, since no
#       other part of the architecture's math changes;
#   (3) `show_feature_map` (a debug-only plotting import, unused in the traced path) is
#       dropped;
#   (4) `conv00`'s in-channel count, hardcoded in the repo as `(64 + 2)*4+2` (tied to
#       the paper's default EDSR width of 64), is parametrized on the encoder's actual
#       `n_feats` so a tiny-width build stays self-consistent -- same
#       "(feat_dim + 2 rel-coord chans) x 4 corners + 2 rel-cell chans" formula, just
#       not hardcoded to 64.
# No architectural code was altered.

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# utils.py :: make_coord
# ---------------------------------------------------------------------------
def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


# ---------------------------------------------------------------------------
# edsr.py
# ---------------------------------------------------------------------------
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1
    ):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size),
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        return x


def make_edsr_baseline(
    n_resblocks=16, n_feats=64, res_scale=1, scale=2, no_upsampling=False, rgb_range=1
):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)


# ---------------------------------------------------------------------------
# galerkin.py
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out


class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()

    def forward(self, x, name="0"):
        B, C, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H * W, self.heads, 3 * self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (H * W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias

        return bias


# ---------------------------------------------------------------------------
# sronet.py
# ---------------------------------------------------------------------------
class SRNO(nn.Module):
    def __init__(self, width=256, blocks=16, encoder_n_resblocks=16, encoder_n_feats=64):
        super().__init__()
        self.width = width
        # The real repo builds this via `models.make(encoder_spec)` off a config file
        # pointing at `edsr-baseline`; instantiated directly here (same real class).
        self.encoder = make_edsr_baseline(
            n_resblocks=encoder_n_resblocks, n_feats=encoder_n_feats, no_upsampling=True
        )
        # Real repo hardcodes this to the paper's default EDSR width (64):
        # `nn.Conv2d((64 + 2)*4+2, self.width, 1)`. Parametrized on `encoder_n_feats`
        # so the tiny build's shrunken encoder output-channel count still matches --
        # same "(feat_dim + 2 rel-coord chans) x 4 corners + 2 rel-cell chans" formula.
        self.conv00 = nn.Conv2d((encoder_n_feats + 2) * 4 + 2, self.width, 1)

        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)

        self.fc1 = nn.Conv2d(self.width, 256, 1)
        self.fc2 = nn.Conv2d(256, 3, 1)

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell):
        feat = self.feat
        grid = 0

        pos_lr = (
            make_coord(feat.shape[-2:], flatten=False)
            .to(feat.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode="nearest", align_corners=False)

                old_coord = F.grid_sample(
                    pos_lr, coord_.flip(-1), mode="nearest", align_corners=False
                )
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)

        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)

        grid = torch.cat(
            [
                *rel_coords,
                *feat_s,
                rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, coord.shape[1], coord.shape[2]),
            ],
            dim=1,
        )

        x = self.conv00(grid)
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)

        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))

        ret = ret + F.grid_sample(
            self.inp, coord.flip(-1), mode="bilinear", padding_mode="border", align_corners=False
        )
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


def build_srno():
    torch.manual_seed(0)
    # Tiny config: fewer EDSR residual blocks / narrower channels + fewer attention
    # heads than the paper's defaults (width=256, blocks=16, EDSR n_resblocks=16,
    # n_feats=64), keeping every real architectural mechanism (EDSR encoder, the two
    # stacked Galerkin `simple_attn` operator blocks, the 4-corner local-ensemble
    # coordinate query, the bilinear residual skip) intact.
    model = SRNO(width=16, blocks=2, encoder_n_resblocks=2, encoder_n_feats=8)
    model.eval()
    return model


def example_input_srno():
    torch.manual_seed(0)
    # Low-res input image (n, 3, h, w) plus a batch of continuous query coordinates
    # (n, hq, wq, 2) and per-query cell sizes (n, 2), matching the repo's
    # coord/cell-conditioned continuous super-resolution query convention.
    inp = torch.rand(1, 3, 12, 12)
    coord = make_coord((16, 16), flatten=False).unsqueeze(0)
    cell = torch.ones(1, 2)
    cell[:, 0] *= 2.0 / 16
    cell[:, 1] *= 2.0 / 16
    return (inp, coord, cell)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SRNO", "build_srno", "example_input_srno", 2023, MENAGERIE_ZOO),
]
