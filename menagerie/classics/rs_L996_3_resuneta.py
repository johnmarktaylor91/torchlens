# FAITHFUL PORT of feevos/resuneta @ master (original framework: MXNet/Gluon)
# https://github.com/feevos/resuneta
# Files transcribed (real math kept verbatim, only MXNet Gluon -> PyTorch nn.Module
# tensor-op translation):
#   https://raw.githubusercontent.com/feevos/resuneta/master/nn/BBlocks/resnet_blocks.py
#   https://raw.githubusercontent.com/feevos/resuneta/master/nn/Units/resnet_units.py
#   https://raw.githubusercontent.com/feevos/resuneta/master/nn/Units/resnet_atrous_units.py
#   https://raw.githubusercontent.com/feevos/resuneta/master/nn/layers/conv2Dnormed.py
#   https://raw.githubusercontent.com/feevos/resuneta/master/nn/layers/scale.py
#   https://raw.githubusercontent.com/feevos/resuneta/master/nn/layers/combine.py
#   https://raw.githubusercontent.com/feevos/resuneta/master/nn/pooling/psp_pooling.py
#   https://raw.githubusercontent.com/feevos/resuneta/master/models/resunet_d6_encoder.py
#   https://raw.githubusercontent.com/feevos/resuneta/master/models/resunet_d6_causal_mtskcolor_ddist.py
#
# Diakogiannis et al. 2020 (ISPRS J. Photogrammetry & Remote Sensing)
# "ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed
# data" -- a multi-task (MTSK) ResUNet with (1) an encoder of stacked dilated-ResNet-v2
# "atrous" units (parallel dilation-rate branches summed together, following the
# published `ResNet_atrous_unit`/`ResNet_atrous_2_unit` dilation schedule that shrinks
# with depth) and strided-conv downsampling, (2) a PSP (pyramid scene parsing) pooling
# bottleneck built via a RECURSIVE quad-split/global-pool/quad-stitch scheme
# (`PSP_Pooling.SplitPooling`, exactly the published multi-resolution context pooling,
# not a plain adaptive-avg-pool PSP), (3) a symmetric decoder of `combine_layers`
# (nearest-neighbor upsample + 1x1-conv + concat-with-skip) followed by the same atrous
# ResNet units, and (4) FOUR coupled output heads sharing the penultimate feature map:
# segmentation `logits`, boundary `bound_logits`, distance-transform `distance_logits`,
# and an HSV-color-reconstruction `color_logits` head -- with `bound`/`logits`
# deliberately built by concatenating the *other* heads' outputs (`bound` consumes
# `dist`; `logits` consumes `bound`+`dist`), the paper's explicit task-coupling design,
# not four independent heads. This four-head coupling plus the recursive PSP pooling
# and the atrous-unit dilation schedule are genuine architectural content with no
# PyTorch equivalent to reuse, hence a full port (not a base-lib reuse).
#
# `ResNet_v2_block`, `ResNet_v2_unit`, `ResNet_atrous_unit`, `ResNet_atrous_2_unit`,
# `Conv2DNormed`, `DownSample`, `UpSample`, `combine_layers`, `PSP_Pooling`,
# `ResUNet_d6_encoder`, `ResUNet_d6` mirror the upstream Gluon `HybridBlock` graph
# op-for-op (same BN-ReLU-Conv pre-activation order, same "SAME"-padding dilation math,
# same broadcast-add residual summation across parallel dilation branches, same
# recursive half-split/global-pool/quad-stitch PSP algorithm, same four-head wiring and
# concat order). Only mechanical framework substitutions were made:
#   - `mxnet.gluon.HybridBlock`/`hybrid_forward(self, F, x)` -> `torch.nn.Module`/
#     `forward(self, x)` (the `F` NDArray/Symbol dispatch argument is a Gluon
#     hybridization artifact with no PyTorch analogue).
#   - `gluon.nn.Conv2D(..., padding=p, dilation=d, use_bias=False)` ->
#     `nn.Conv2d(..., padding=p, dilation=d, bias=False)`; MXNet's SAME-for-odd-kernel
#     padding formula `p = dilation * (kernel - 1) // 2` is kept verbatim.
#   - `gluon.nn.BatchNorm(axis=1)` -> `nn.BatchNorm2d` (MXNet's default channel axis 1
#     matches PyTorch's NCHW convention exactly).
#   - `F.broadcast_add(a, b)` -> `a + b` (plain elementwise add; MXNet's broadcast
#     variant is required only because of Gluon's Symbol/NDArray duality, not because
#     the tensors differ in shape here).
#   - `F.UpSampling(x, scale=f, sample_type='nearest')` -> `F.interpolate(x,
#     scale_factor=f, mode='nearest')`.
#   - `F.split(...)`/`F.concat(...)` -> `torch.chunk`/`torch.cat` in `PSP_Pooling`'s
#     `HalfSplit`/`QuarterStitch`/recursive `SplitPooling` (kept as the same recursive
#     algorithm, not flattened into a single adaptive pool).
#   - `F.Pooling(x, global_pool=True)` + `F.broadcast_mul(F.ones_like(x), ...)`
#     (MXNet's idiom for "global-average-pool, then broadcast back to the input's
#     spatial size") -> `F.adaptive_avg_pool2d(x, 1).expand_as(x)`.
#   - `gluon.nn.HybridLambda(lambda F, x: F.softmax(x, axis=1))` /
#     `F.sigmoid(x, axis=1)` last-activation dispatch on `NClasses` -> the equivalent
#     plain `torch.sigmoid`/`torch.softmax(dim=1)` branch in `forward`.
# No architecture code (filter-count doubling schedule, dilation-rate schedule per
# depth, the four-head concat wiring, or the recursive PSP algorithm) was altered.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# nn/layers/conv2Dnormed.py
# ---------------------------------------------------------------------------
class Conv2DNormed(nn.Module):
    """Conv2D (no bias) followed by BatchNorm2d."""

    def __init__(
        self, in_channels, channels, kernel_size, strides=(1, 1), padding=(0, 0), dilation=(1, 1)
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm_layer = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm_layer(x)
        return x


# ---------------------------------------------------------------------------
# nn/BBlocks/resnet_blocks.py
# ---------------------------------------------------------------------------
class ResNet_v2_block(nn.Module):
    """ResNet v2 pre-activation building block (odd kernel, SAME padding)."""

    def __init__(self, in_channels, nfilters, kernel_size=(3, 3), dilation_rate=(1, 1)):
        super().__init__()
        p0 = dilation_rate[0] * (kernel_size[0] - 1) // 2
        p1 = dilation_rate[1] * (kernel_size[1] - 1) // 2
        p = (p0, p1)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            nfilters,
            kernel_size=kernel_size,
            padding=p,
            dilation=dilation_rate,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(nfilters)
        self.conv2 = nn.Conv2d(
            nfilters,
            nfilters,
            kernel_size=kernel_size,
            padding=p,
            dilation=dilation_rate,
            bias=True,
        )

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out


# ---------------------------------------------------------------------------
# nn/Units/resnet_units.py, nn/Units/resnet_atrous_units.py
# ---------------------------------------------------------------------------
class ResNet_v2_unit(nn.Module):
    """A single ResNet_v2_block with a residual (dilation_rate=1) shortcut."""

    def __init__(self, nfilters, kernel_size=(3, 3), dilation_rate=(1, 1)):
        super().__init__()
        self.ResBlock1 = ResNet_v2_block(nfilters, nfilters, kernel_size, dilation_rate)

    def forward(self, x):
        return self.ResBlock1(x) + x


class ResNet_atrous_unit(nn.Module):
    """4 parallel dilated ResNet_v2_blocks (dilations 1, d0, d1, d2) summed together."""

    def __init__(self, nfilters, kernel_size=(3, 3), dilation_rates=(3, 15, 31)):
        super().__init__()
        self.ResBlock1 = ResNet_v2_block(nfilters, nfilters, kernel_size, (1, 1))
        d = dilation_rates[0]
        self.ResBlock2 = ResNet_v2_block(nfilters, nfilters, kernel_size, (d, d))
        d = dilation_rates[1]
        self.ResBlock3 = ResNet_v2_block(nfilters, nfilters, kernel_size, (d, d))
        d = dilation_rates[2]
        self.ResBlock4 = ResNet_v2_block(nfilters, nfilters, kernel_size, (d, d))

    def forward(self, x):
        out = x
        out = out + self.ResBlock1(x)
        out = out + self.ResBlock2(x)
        out = out + self.ResBlock3(x)
        out = out + self.ResBlock4(x)
        return out


class ResNet_atrous_2_unit(nn.Module):
    """3 parallel dilated ResNet_v2_blocks (dilations 1, d0, d1) summed together."""

    def __init__(self, nfilters, kernel_size=(3, 3), dilation_rates=(3, 15)):
        super().__init__()
        self.ResBlock1 = ResNet_v2_block(nfilters, nfilters, kernel_size, (1, 1))
        d = dilation_rates[0]
        self.ResBlock2 = ResNet_v2_block(nfilters, nfilters, kernel_size, (d, d))
        d = dilation_rates[1]
        self.ResBlock3 = ResNet_v2_block(nfilters, nfilters, kernel_size, (d, d))

    def forward(self, x):
        out = x
        out = out + self.ResBlock1(x)
        out = out + self.ResBlock2(x)
        out = out + self.ResBlock3(x)
        return out


# ---------------------------------------------------------------------------
# nn/layers/scale.py
# ---------------------------------------------------------------------------
class DownSample(nn.Module):
    """Halve spatial size, double channel count (stride-2 3x3 conv)."""

    def __init__(self, in_channels, factor=2):
        super().__init__()
        nfilters = in_channels * factor
        self.convdn = nn.Conv2d(
            in_channels, nfilters, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    def forward(self, x):
        return self.convdn(x)


class UpSample(nn.Module):
    """Nearest-neighbor upsample by `factor`, then 1x1 Conv2DNormed to halve channels."""

    def __init__(self, in_channels, factor=2):
        super().__init__()
        self.factor = factor
        nfilters = in_channels // factor
        self.convup_normed = Conv2DNormed(in_channels, nfilters, kernel_size=(1, 1))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode="nearest")
        x = self.convup_normed(x)
        return x


# ---------------------------------------------------------------------------
# nn/layers/combine.py
# ---------------------------------------------------------------------------
class combine_layers(nn.Module):
    """Upsample the low-res branch, concat with the high-res skip, 1x1-conv-normed."""

    def __init__(self, lo_channels, hi_channels, nfilters):
        super().__init__()
        self.up = UpSample(lo_channels)
        self.conv_normed = Conv2DNormed(
            lo_channels // 2 + hi_channels, nfilters, kernel_size=(1, 1), padding=(0, 0)
        )

    def forward(self, layer_lo, layer_hi):
        up = self.up(layer_lo)
        up = F.relu(up)
        x = torch.cat([up, layer_hi], dim=1)
        x = self.conv_normed(x)
        return x


# ---------------------------------------------------------------------------
# nn/pooling/psp_pooling.py
# ---------------------------------------------------------------------------
class PSP_Pooling(nn.Module):
    """Pyramid-scene-parsing pooling via recursive quad-split/global-pool/stitch."""

    def __init__(self, nfilters, depth=4):
        super().__init__()
        self.nfilters = nfilters
        self.depth = depth
        self.convs = nn.ModuleList(
            [
                Conv2DNormed(nfilters, nfilters // depth, kernel_size=(1, 1), padding=(0, 0))
                for _ in range(depth)
            ]
        )
        # cat([x (nfilters ch), *convs outputs (depth * (nfilters // depth) ch)]) -> 2*nfilters
        self.conv_norm_final = Conv2DNormed(
            2 * nfilters, nfilters, kernel_size=(1, 1), padding=(0, 0)
        )

    @staticmethod
    def _half_split(a):
        b0, b1 = torch.chunk(a, 2, dim=2)
        c1 = torch.chunk(b0, 2, dim=3)
        c2 = torch.chunk(b1, 2, dim=3)
        return [c1[0], c1[1], c2[0], c2[1]]

    @staticmethod
    def _quarter_stitch(dss):
        temp1 = torch.cat([dss[0], dss[1]], dim=-1)
        temp2 = torch.cat([dss[2], dss[3]], dim=-1)
        return torch.cat([temp1, temp2], dim=2)

    @staticmethod
    def _half_pooling(a):
        ds = PSP_Pooling._half_split(a)
        dss = [F.adaptive_avg_pool2d(x, 1).expand_as(x) for x in ds]
        return PSP_Pooling._quarter_stitch(dss)

    def _split_pooling(self, a, depth):
        if depth == 1:
            return self._half_pooling(a)
        d = self._half_split(a)
        return self._quarter_stitch([self._split_pooling(x, depth - 1) for x in d])

    def forward(self, x):
        p = [x]
        p.append(self.convs[0](F.adaptive_avg_pool2d(x, 1).expand_as(x)))
        p += [self.convs[d](self._split_pooling(x, d)) for d in range(1, self.depth)]
        out = torch.cat(p, dim=1)
        out = self.conv_norm_final(out)
        return out


# ---------------------------------------------------------------------------
# models/resunet_d6_encoder.py
# ---------------------------------------------------------------------------
class ResUNet_d6_encoder(nn.Module):
    """6-level dilated-ResNet-atrous encoder with a PSP-pooling bottleneck."""

    def __init__(self, in_channels, nfilters_init):
        super().__init__()
        self.nfilters = nfilters_init
        self.conv_first_normed = Conv2DNormed(in_channels, self.nfilters, kernel_size=(1, 1))

        nf0 = self.nfilters * 2**0
        self.Dn1 = ResNet_atrous_unit(nf0)
        self.pool1 = DownSample(nf0)

        nf1 = self.nfilters * 2**1
        self.Dn2 = ResNet_atrous_unit(nf1)
        self.pool2 = DownSample(nf1)

        nf2 = self.nfilters * 2**2
        self.Dn3 = ResNet_atrous_2_unit(nf2)
        self.pool3 = DownSample(nf2)

        nf3 = self.nfilters * 2**3
        self.Dn4 = ResNet_atrous_2_unit(nf3, dilation_rates=(3, 5))
        self.pool4 = DownSample(nf3)

        nf4 = self.nfilters * 2**4
        self.Dn5 = ResNet_v2_unit(nf4)
        self.pool5 = DownSample(nf4)

        nf5 = self.nfilters * 2**5
        self.Dn6 = ResNet_v2_unit(nf5)

        self.middle = PSP_Pooling(nf5)

    def forward(self, x):
        conv1 = F.relu(self.conv_first_normed(x))

        dn1 = self.Dn1(conv1)
        pool1 = self.pool1(dn1)

        dn2 = self.Dn2(pool1)
        pool2 = self.pool2(dn2)

        dn3 = self.Dn3(pool2)
        pool3 = self.pool3(dn3)

        dn4 = self.Dn4(pool3)
        pool4 = self.pool4(dn4)

        dn5 = self.Dn5(pool4)
        pool5 = self.pool5(dn5)

        dn6 = self.Dn6(pool5)

        middle = F.relu(self.middle(dn6))
        return conv1, dn1, dn2, dn3, dn4, dn5, dn6, middle


# ---------------------------------------------------------------------------
# models/resunet_d6_causal_mtskcolor_ddist.py: ResUNet_d6 (4-headed MTSK model)
# ---------------------------------------------------------------------------
class ResUNet_d6(nn.Module):
    """Multi-task ResUNet-a: segmentation + boundary + distance + HSV-color heads."""

    def __init__(self, in_channels, nfilters_init, n_classes):
        super().__init__()
        self.nfilters = nfilters_init
        self.n_classes = n_classes
        depth = 6

        self.encoder = ResUNet_d6_encoder(in_channels, nfilters_init)

        nf = self.nfilters * 2 ** (depth - 1 - 1)  # 16x
        self.UpComb1 = combine_layers(self.nfilters * 2**5, self.nfilters * 2**4, nf)
        self.UpConv1 = ResNet_atrous_2_unit(nf, dilation_rates=(3, 5))

        nf = self.nfilters * 2 ** (depth - 1 - 2)  # 8x
        self.UpComb2 = combine_layers(self.nfilters * 2**4, self.nfilters * 2**3, nf)
        self.UpConv2 = ResNet_atrous_2_unit(nf)

        nf = self.nfilters * 2 ** (depth - 1 - 3)  # 4x
        self.UpComb3 = combine_layers(self.nfilters * 2**3, self.nfilters * 2**2, nf)
        self.UpConv3 = ResNet_atrous_unit(nf)

        nf = self.nfilters * 2 ** (depth - 1 - 4)  # 2x
        self.UpComb4 = combine_layers(self.nfilters * 2**2, self.nfilters * 2**1, nf)
        self.UpConv4 = ResNet_atrous_unit(nf)

        nf = self.nfilters * 2 ** (depth - 1 - 5)  # 1x
        self.UpComb5 = combine_layers(self.nfilters * 2**1, self.nfilters * 2**0, nf)
        self.UpConv5 = ResNet_atrous_unit(nf)

        self.psp_2ndlast = PSP_Pooling(2 * self.nfilters)

        self.logits = nn.Sequential(
            Conv2DNormed(3 * self.nfilters, self.nfilters, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            Conv2DNormed(self.nfilters, self.nfilters, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nfilters, n_classes, kernel_size=1, padding=0),
        )

        self.bound_logits = nn.Sequential(
            Conv2DNormed(
                2 * self.nfilters + n_classes, self.nfilters, kernel_size=(3, 3), padding=(1, 1)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nfilters, n_classes, kernel_size=1, padding=0),
        )

        self.distance_logits = nn.Sequential(
            Conv2DNormed(2 * self.nfilters, self.nfilters, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            Conv2DNormed(self.nfilters, self.nfilters, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.nfilters, n_classes, kernel_size=1, padding=0),
        )

        self.color_logits = nn.Conv2d(2 * self.nfilters, 3, kernel_size=1, padding=0)

    def _channel_act(self, x):
        if self.n_classes == 1:
            return torch.sigmoid(x)
        return torch.softmax(x, dim=1)

    def forward(self, x):
        conv1, dn1, dn2, dn3, dn4, dn5, dn6, middle = self.encoder(x)

        up1 = self.UpConv1(self.UpComb1(middle, dn5))
        up2 = self.UpConv2(self.UpComb2(up1, dn4))
        up3 = self.UpConv3(self.UpComb3(up2, dn3))
        up4 = self.UpConv4(self.UpComb4(up3, dn2))
        up5 = self.UpConv5(self.UpComb5(up4, dn1))

        convl = torch.cat([conv1, up5], dim=1)
        conv = F.relu(self.psp_2ndlast(convl))

        # 1st: distance map (topology info)
        dist = self.distance_logits(convl)
        dist = self._channel_act(dist)

        # then boundaries, conditioned on the distance map
        bound = torch.cat([conv, dist], dim=1)
        bound = self.bound_logits(bound)
        bound = torch.sigmoid(bound)

        # HSV color prediction (self-supervised auxiliary head)
        convc = self.color_logits(convl)
        convc = torch.sigmoid(convc)

        # finally, segmentation mask, conditioned on boundary + distance
        logits = torch.cat([conv, bound, dist], dim=1)
        logits = self.logits(logits)
        logits = self._channel_act(logits)

        return logits, bound, dist, convc


def build_resuneta_d6():
    # Small (nfilters_init=4) 2-class config for a tractable menagerie trace; the real
    # model as used in the paper's ISPRS demo is ResUNet_d7(nfilters_init=32,
    # NClasses=6) on 256x256 tiles -- same architecture, larger width/depth/resolution.
    return ResUNet_d6(in_channels=3, nfilters_init=4, n_classes=2)


def example_input_resuneta_d6():
    # 512x512 RGB tile: divisible by 2**5 (5 downsample stages in the depth-6 encoder)
    # down to a 16x16 PSP bottleneck, itself divisible enough for the recursive
    # quad-split PSP pooling (default depth=4 -> 1x1 leaf tiles at the bottleneck).
    return (torch.randn(1, 3, 512, 512),)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    (
        "ResUNet-a (d6, multi-task seg+bound+dist+color)",
        "build_resuneta_d6",
        "example_input_resuneta_d6",
        2020,
        "ported-pytorch",
    ),
]
