# SOURCE: vendored from https://github.com/liyunsheng13/micronet @ main
# (backbone/micronet.py: StemLayer, GroupConv, ChannelShuffle, ChannelShuffle2,
#  SpatialSepConvSF, DepthConv, DepthSpatialSepConv, DYMicroBlock, MicroNet;
#  backbone/activation.py: h_sigmoid, h_tanh, get_act_layer, SELayer, DYShiftMax;
#  backbone/microconfig.py: msnx_dy6_exp4_4M_221_cfgs (the "MicroNet-M0" config))
#
# MicroNets (Li, Zhou, Chen, Wang, Yuan, Sun, Yan; ICCV 2021, "Micronet: Improving image
# recognition with extremely low flops", arXiv:2108.05894) is a family of extremely
# low-FLOP image classifiers built from two architectural contributions vendored
# verbatim below: (1) "micro-factorized" convolution -- pointwise (1x1) and depthwise
# convs are each split into a low-rank product of a grouped conv followed by a channel
# shuffle (`SpatialSepConvSF`/`DepthSpatialSepConv` for the depthwise side,
# `GroupConv`/`get_pointwise_conv` for the pointwise side), so the layer factorizes
# through a small "bottleneck" of groups*groups rather than a full C_in*C_out matrix;
# (2) Dynamic Shift-Max (`DYShiftMax` in activation.py) -- a dynamic activation that
# computes input-conditioned affine coefficients (a1,b1,a2,b2) per group via a
# squeeze-excitation-style FC, circularly group-shifts the feature map
# (`x2 = x_out[:, self.index, :, :]`), and takes an elementwise max of two affine
# combinations `max(x*a1 + x_shift*b1, x*a2 + x_shift*b2)` -- fusing cross-group
# information exchange into the activation itself so the micro-factorized groups
# don't act as fully independent sub-networks. `DYMicroBlock` composes these two
# ideas into the paper's basic inverted-residual-style block, and `MicroNet` stacks
# blocks per a stage config table (`microconfig.py`) behind a `StemLayer` and a
# `SwishLinear` classifier head. Vendored verbatim from `backbone/{micronet,
# activation}.py`; only the `microconfig.py`/`utils.defaults` yacs `CfgNode` plumbing
# is collapsed into a plain-namespace `_MicroNetCfg` shim carrying the exact same
# field names+values used by the repo's `scripts/train_micronet_m0_2gpu.sh` (the
# smallest published "M0" variant, ~4M-FLOPs config `msnx_dy6_exp4_4M_221`), so this
# file has no `yacs` dependency; the debug `print(...)` calls inside
# `SELayer`/`DYShiftMax`/`GroupConv.__init__` are dropped (cosmetic training-script
# logging only, no effect on the traced architecture).

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- backbone/activation.py (vendored, verbatim math; debug prints dropped) ----


def _make_divisible(v, divisor, min_value=None):
    """This function is taken from the original tf repo. It ensures that all layers
    have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max / 6

    def forward(self, x):
        return self.relu(x + 3) * self.h_max


class h_tanh(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 3 - self.h_max


def get_squeeze_channels(inp, reduction):
    if reduction == 4:
        squeeze = inp // reduction
    else:
        squeeze = _make_divisible(inp // reduction, 4)
    return squeeze


def get_act_layer(
    inp,
    oup,
    mode="SE1",
    act_relu=True,
    act_max=2,
    act_bias=True,
    init_a=(1.0, 0.0),
    reduction=4,
    init_b=(0.0, 0.0),
    g=None,
    act="relu",
    expansion=True,
):
    layer = None
    if mode == "SE1":
        layer = nn.Sequential(
            SELayer(inp, oup, reduction=reduction),
            nn.ReLU6(inplace=True) if act_relu else nn.Sequential(),
        )
    elif mode == "SE0":
        layer = nn.Sequential(SELayer(inp, oup, reduction=reduction))
    elif mode == "NA":
        layer = nn.ReLU6(inplace=True) if act_relu else nn.Sequential()
    elif mode == "LeakyReLU":
        layer = nn.LeakyReLU(inplace=True) if act_relu else nn.Sequential()
    elif mode == "RReLU":
        layer = nn.RReLU(inplace=True) if act_relu else nn.Sequential()
    elif mode == "PReLU":
        layer = nn.PReLU() if act_relu else nn.Sequential()
    elif mode == "DYShiftMax":
        layer = DYShiftMax(
            inp,
            oup,
            act_max=act_max,
            act_relu=act_relu,
            init_a=init_a,
            reduction=reduction,
            init_b=init_b,
            g=g,
            expansion=expansion,
        )
    return layer


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.oup = oup
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        squeeze = get_squeeze_channels(inp, reduction)
        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup),
            h_sigmoid(),
        )

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup, 1, 1)
        return x_out * y


class DYShiftMax(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        reduction=4,
        act_max=1.0,
        act_relu=True,
        init_a=(0.0, 0.0),
        init_b=(0.0, 0.0),
        relu_before_pool=False,
        g=None,
        expansion=False,
    ):
        super().__init__()
        self.oup = oup
        self.act_max = act_max * 2
        self.act_relu = act_relu
        self.avg_pool = nn.Sequential(
            nn.ReLU(inplace=True) if relu_before_pool else nn.Sequential(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.exp = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b

        squeeze = _make_divisible(inp // reduction, 4)
        if squeeze < 4:
            squeeze = 4

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup * self.exp),
            h_sigmoid(),
        )
        if g is None:
            g = 1
        self.g = g[1]
        if self.g != 1 and expansion:
            self.g = inp // self.g
        self.gc = inp // self.g
        index = torch.Tensor(range(inp)).view(1, inp, 1, 1)
        index = index.view(1, self.g, self.gc, 1, 1)
        indexgs = torch.split(index, [1, self.g - 1], dim=1)
        indexgs = torch.cat((indexgs[1], indexgs[0]), dim=1)
        indexs = torch.split(indexgs, [1, self.gc - 1], dim=2)
        indexs = torch.cat((indexs[1], indexs[0]), dim=2)
        self.index = indexs.view(inp).type(torch.LongTensor)
        self.expansion = expansion

    def forward(self, x):
        x_in = x
        x_out = x

        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        y = (y - 0.5) * self.act_max

        n2, c2, h2, w2 = x_out.size()
        x2 = x_out[:, self.index, :, :]

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_a[1]

            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = torch.max(z1, z2)

        elif self.exp == 2:
            a1, b1 = torch.split(y, self.oup, dim=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1
        else:
            out = x_out

        return out


# ---- backbone/micronet.py (vendored, verbatim math; debug prints dropped) ----


class MaxGroupPooling(nn.Module):
    def __init__(self, channel_per_group=2):
        super().__init__()
        self.channel_per_group = channel_per_group

    def forward(self, x):
        if self.channel_per_group == 1:
            return x
        b, c, h, w = x.size()
        y = x.view(b, c // self.channel_per_group, -1, h, w)
        out, _ = torch.max(y, dim=2)
        return out


class SwishLinear(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(inp, oup),
            nn.BatchNorm1d(oup),
            _h_swish(),
        )

    def forward(self, x):
        return self.linear(x)


class _h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class StemLayer(nn.Module):
    def __init__(self, inp, oup, stride, dilation=1, mode="default", groups=(4, 4)):
        super().__init__()

        self.exp = 1 if mode == "default" else 2
        g1, g2 = groups
        if mode == "default":
            self.stem = nn.Sequential(
                nn.Conv2d(inp, oup * self.exp, 3, stride, 1, bias=False, dilation=dilation),
                nn.BatchNorm2d(oup * self.exp),
                nn.ReLU6(inplace=True) if self.exp == 1 else MaxGroupPooling(self.exp),
            )
        elif mode == "spatialsepsf":
            self.stem = nn.Sequential(
                SpatialSepConvSF(inp, groups, 3, stride),
                MaxGroupPooling(2) if g1 * g2 == 2 * oup else nn.ReLU6(inplace=True),
            )
        else:
            raise ValueError("Undefined stem layer")

    def forward(self, x):
        return self.stem(x)


class GroupConv(nn.Module):
    def __init__(self, inp, oup, groups=2):
        super().__init__()
        self.inp = inp
        self.oup = oup
        self.groups = groups
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=self.groups[0]),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.conv(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(b, -1, h, w)


class ChannelShuffle2(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(b, -1, h, w)


class SpatialSepConvSF(nn.Module):
    def __init__(self, inp, oups, kernel_size, stride):
        super().__init__()
        oup1, oup2 = oups
        self.conv = nn.Sequential(
            nn.Conv2d(
                inp,
                oup1,
                (kernel_size, 1),
                (stride, 1),
                (kernel_size // 2, 0),
                bias=False,
                groups=1,
            ),
            nn.BatchNorm2d(oup1),
            nn.Conv2d(
                oup1,
                oup1 * oup2,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size // 2),
                bias=False,
                groups=oup1,
            ),
            nn.BatchNorm2d(oup1 * oup2),
            ChannelShuffle(oup1),
        )

    def forward(self, x):
        return self.conv(x)


class DepthConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False, groups=inp),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.conv(x)


class DepthSpatialSepConv(nn.Module):
    def __init__(self, inp, expand, kernel_size, stride):
        super().__init__()
        exp1, exp2 = expand
        hidden_dim = inp * exp1
        oup = inp * exp1 * exp2
        self.conv = nn.Sequential(
            nn.Conv2d(
                inp,
                inp * exp1,
                (kernel_size, 1),
                (stride, 1),
                (kernel_size // 2, 0),
                bias=False,
                groups=inp,
            ),
            nn.BatchNorm2d(inp * exp1),
            nn.Conv2d(
                hidden_dim,
                oup,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size // 2),
                bias=False,
                groups=hidden_dim,
            ),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.conv(x)


def get_pointwise_conv(mode, inp, oup, hiddendim, groups):
    if mode == "group":
        return GroupConv(inp, oup, groups)
    elif mode == "1x1":
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
    else:
        return None


class DYMicroBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size=3,
        stride=1,
        ch_exp=(2, 2),
        ch_per_group=4,
        groups_1x1=(1, 1),
        depthsep=True,
        shuffle=False,
        pointwise="fft",
        activation_cfg=None,
    ):
        super().__init__()

        self.identity = stride == 1 and inp == oup

        y1, y2, y3 = activation_cfg.dy
        act = activation_cfg.MODULE
        act_max = activation_cfg.ACT_MAX
        act_bias = activation_cfg.LINEARSE_BIAS
        act_reduction = activation_cfg.REDUCTION * activation_cfg.ratio
        init_a = activation_cfg.INIT_A
        init_b = activation_cfg.INIT_B
        init_ab3 = activation_cfg.INIT_A_BLOCK3

        t1 = ch_exp
        gs1 = ch_per_group
        hidden_fft, g1, g2 = groups_1x1

        hidden_dim1 = inp * t1[0]
        hidden_dim2 = inp * t1[0] * t1[1]

        if gs1[0] == 0:
            self.layers = nn.Sequential(
                DepthSpatialSepConv(inp, t1, kernel_size, stride),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g=gs1,
                    expansion=False,
                )
                if y2 > 0
                else nn.ReLU6(inplace=True),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                ChannelShuffle2(hidden_dim2 // 2) if shuffle and y2 != 0 else nn.Sequential(),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)),
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction // 2,
                    init_b=[init_ab3[1], 0.0],
                    g=(g1, g2),
                    expansion=False,
                )
                if y3 > 0
                else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle2(oup // 2)
                if shuffle and oup % 2 == 0 and y3 != 0
                else nn.Sequential(),
            )
        elif g2 == 0:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g=gs1,
                    expansion=False,
                )
                if y3 > 0
                else nn.Sequential(),
            )
        else:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y1 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g=gs1,
                    expansion=False,
                )
                if y1 > 0
                else nn.ReLU6(inplace=True),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                DepthSpatialSepConv(hidden_dim2, (1, 1), kernel_size, stride)
                if depthsep
                else DepthConv(hidden_dim2, hidden_dim2, kernel_size, stride),
                nn.Sequential(),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g=gs1,
                    expansion=True,
                )
                if y2 > 0
                else nn.ReLU6(inplace=True),
                ChannelShuffle2(hidden_dim2 // 4)
                if shuffle and y1 != 0 and y2 != 0
                else nn.Sequential()
                if y1 == 0 and y2 == 0
                else ChannelShuffle2(hidden_dim2 // 2),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)),
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction // 2 if oup < hidden_dim2 else act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g=(g1, g2),
                    expansion=False,
                )
                if y3 > 0
                else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle2(oup // 2) if shuffle and y3 != 0 else nn.Sequential(),
            )

    def forward(self, x):
        identity = x
        out = self.layers(x)
        if self.identity:
            out = out + identity
        return out


class MicroNet(nn.Module):
    """Vendored `backbone.micronet.MicroNet`, with the yacs `CfgNode` config
    replaced by a plain namespace shim (`_MicroNetCfg`) carrying identical field
    names/values -- no architectural change, only config-plumbing simplification."""

    def __init__(self, cfg, input_size=224, num_classes=1000):
        super().__init__()

        mode = cfg.MODEL.MICRONETS.NET_CONFIG
        self.cfgs = _get_micronet_config(mode)

        block = DYMicroBlock
        stem_mode = cfg.MODEL.MICRONETS.STEM_MODE
        stem_ch = cfg.MODEL.MICRONETS.STEM_CH
        stem_dilation = cfg.MODEL.MICRONETS.STEM_DILATION
        stem_groups = cfg.MODEL.MICRONETS.STEM_GROUPS
        out_ch = cfg.MODEL.MICRONETS.OUT_CH
        depthsep = cfg.MODEL.MICRONETS.DEPTHSEP
        shuffle = cfg.MODEL.MICRONETS.SHUFFLE
        pointwise = cfg.MODEL.MICRONETS.POINTWISE
        dropout_rate = cfg.MODEL.MICRONETS.DROPOUT

        # act_max/act_bias are unused here (matches the real upstream MicroNet.__init__
        # verbatim -- dead reads there too; activation_cfg carries the values instead).
        act_max = cfg.MODEL.ACTIVATION.ACT_MAX  # noqa: F841
        act_bias = cfg.MODEL.ACTIVATION.LINEARSE_BIAS  # noqa: F841
        activation_cfg = cfg.MODEL.ACTIVATION

        assert input_size % 32 == 0
        input_channel = stem_ch
        layers = [
            StemLayer(
                3,
                input_channel,
                stride=2,
                dilation=stem_dilation,
                mode=stem_mode,
                groups=stem_groups,
            )
        ]

        for val in self.cfgs:
            s, n, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r = val

            t1 = (c1, c2)
            gs1 = (g1, g2)
            gs2 = (c3, g3, g4)
            activation_cfg.dy = [y1, y2, y3]
            activation_cfg.ratio = r

            output_channel = c
            layers.append(
                block(
                    input_channel,
                    output_channel,
                    kernel_size=ks,
                    stride=s,
                    ch_exp=t1,
                    ch_per_group=gs1,
                    groups_1x1=gs2,
                    depthsep=depthsep,
                    shuffle=shuffle,
                    pointwise=pointwise,
                    activation_cfg=activation_cfg,
                )
            )
            input_channel = output_channel
            for _ in range(1, n):
                layers.append(
                    block(
                        input_channel,
                        output_channel,
                        kernel_size=ks,
                        stride=1,
                        ch_exp=t1,
                        ch_per_group=gs1,
                        groups_1x1=gs2,
                        depthsep=depthsep,
                        shuffle=shuffle,
                        pointwise=pointwise,
                        activation_cfg=activation_cfg,
                    )
                )
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.Sequential(
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            _h_swish(),
        )

        output_channel = out_ch
        self.classifier = nn.Sequential(
            SwishLinear(input_channel, output_channel),
            nn.Dropout(dropout_rate),
            SwishLinear(output_channel, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


# ---- backbone/microconfig.py (vendored: M0's msnx_dy6_exp4_4M_221 stage table) ----

_msnx_dy6_exp4_4M_221_cfgs = [
    # s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
    [2, 1, 8, 3, 2, 2, 0, 4, 8, 2, 2, 2, 0, 1, 1],
    [2, 1, 12, 3, 2, 2, 0, 8, 12, 4, 4, 2, 2, 1, 1],
    [2, 1, 16, 5, 2, 2, 0, 12, 16, 4, 4, 2, 2, 1, 1],
    [1, 1, 32, 5, 1, 4, 4, 4, 32, 4, 4, 2, 2, 1, 1],
    [2, 1, 64, 5, 1, 4, 8, 8, 64, 8, 8, 2, 2, 1, 1],
    [1, 1, 96, 3, 1, 4, 8, 8, 96, 8, 8, 2, 2, 1, 2],
    [1, 1, 384, 3, 1, 4, 12, 12, 0, 0, 0, 2, 2, 1, 2],
]


def _get_micronet_config(mode):
    if mode == "msnx_dy6_exp4_4M_221":
        return _msnx_dy6_exp4_4M_221_cfgs
    raise ValueError(f"Unknown MicroNet config mode: {mode}")


# ---- utils/defaults.py analog: plain-namespace cfg shim (field names/values match
# the repo's scripts/train_micronet_m0_2gpu.sh MicroNet-M0 command-line overrides) ----


class _Namespace:
    """Minimal attribute-bag standing in for yacs' `CfgNode` (no `yacs` dependency)."""


def _build_m0_cfg():
    cfg = _Namespace()
    cfg.MODEL = _Namespace()

    cfg.MODEL.ACTIVATION = _Namespace()
    cfg.MODEL.ACTIVATION.MODULE = "DYShiftMax"
    cfg.MODEL.ACTIVATION.ACT_MAX = 2.0
    cfg.MODEL.ACTIVATION.LINEARSE_BIAS = False
    cfg.MODEL.ACTIVATION.INIT_A_BLOCK3 = [1.0, 0.0]
    cfg.MODEL.ACTIVATION.INIT_A = [1.0, 1.0]
    cfg.MODEL.ACTIVATION.INIT_B = [0.0, 0.0]
    cfg.MODEL.ACTIVATION.REDUCTION = 8
    cfg.MODEL.ACTIVATION.FC = False
    cfg.MODEL.ACTIVATION.ACT = "relu"

    cfg.MODEL.MICRONETS = _Namespace()
    cfg.MODEL.MICRONETS.NET_CONFIG = "msnx_dy6_exp4_4M_221"
    cfg.MODEL.MICRONETS.STEM_CH = 4
    cfg.MODEL.MICRONETS.STEM_DILATION = 1
    cfg.MODEL.MICRONETS.STEM_GROUPS = [2, 2]
    cfg.MODEL.MICRONETS.STEM_MODE = "spatialsepsf"
    cfg.MODEL.MICRONETS.BLOCK = "DYMicroBlock"
    cfg.MODEL.MICRONETS.POINTWISE = "group"
    cfg.MODEL.MICRONETS.DEPTHSEP = True
    cfg.MODEL.MICRONETS.SHUFFLE = True
    cfg.MODEL.MICRONETS.OUT_CH = 640
    cfg.MODEL.MICRONETS.DROPOUT = 0.05
    return cfg


def build_micronet_m0():
    torch.manual_seed(0)
    cfg = _build_m0_cfg()
    # input_size=32 (smallest multiple of 32 satisfying the repo's `assert
    # input_size % 32 == 0`) + num_classes=10 for a menagerie-tiny random-init instance;
    # architecture/config values otherwise match the real MicroNet-M0 training script.
    return MicroNet(cfg, input_size=32, num_classes=10)


def example_input_micronet_m0():
    torch.manual_seed(0)
    return torch.randn(2, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("MicroNet-M0", "build_micronet_m0", "example_input_micronet_m0", 2021, MENAGERIE_ZOO),
]
