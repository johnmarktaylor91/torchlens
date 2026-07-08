# SOURCE: vendored from Zheng222/IMDN @ master
# https://raw.githubusercontent.com/Zheng222/IMDN/master/model/architecture.py
# https://raw.githubusercontent.com/Zheng222/IMDN/master/model/block.py
#
# Hui, Gao, Yang, Wang, 2019 (ACM MM) "Lightweight Image Super-Resolution with
# Information Multi-distillation Network". IMDN's architectural contribution is
# the Information Multi-Distillation Block (IMDModule): a cascade of 3x3 convs
# where each stage channel-splits its output into a small "distilled" slice
# (kept) and a larger "remaining" slice (fed to the next stage), the four
# distilled slices are concatenated and refined by a contrast-aware channel
# attention layer (CCALayer) before a 1x1 fusion conv and a residual add. Six
# stacked IMDModules feed a pixel-shuffle upsampler for x4 super-resolution.
# Vendored as real code (not a stock library block), from `model/architecture.py`
# (top-level `IMDN`) + `model/block.py` (the `IMDModule`/`CCALayer`/helper
# builders it depends on), reproduced verbatim below.

import torch
import torch.nn as nn
from collections import OrderedDict

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# model/block.py (verbatim, trimmed to symbols IMDN actually uses:
# conv_layer/conv_block/activation/sequential/CCALayer/IMDModule/pixelshuffle_block)
# ============================================================================


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=padding,
        bias=bias,
        dilation=dilation,
        groups=groups,
    )


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError("activation layer [{:s}] is not found".format(act_type))
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def mean_channels(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (
        F.size(2) * F.size(3)
    )
    return F_variance.pow(0.5)


# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation("lrelu", neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(
            out_c1, (self.distilled_channels, self.remaining_channels), dim=1
        )
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(
            out_c2, (self.distilled_channels, self.remaining_channels), dim=1
        )
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(
            out_c3, (self.distilled_channels, self.remaining_channels), dim=1
        )
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def conv_block(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
):
    padding = int((kernel_size - 1) / 2) * dilation if pad_type == "zero" else 0
    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = activation(act_type) if act_type else None
    return sequential(c, a)


# ============================================================================
# model/architecture.py :: IMDN (verbatim)
# ============================================================================


class IMDN(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(IMDN, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=nf)
        self.IMDB2 = IMDModule(in_channels=nf)
        self.IMDB3 = IMDModule(in_channels=nf)
        self.IMDB4 = IMDModule(in_channels=nf)
        self.IMDB5 = IMDModule(in_channels=nf)
        self.IMDB6 = IMDModule(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type="lrelu")

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_imdn():
    # Real repo defaults (model/architecture.py): nf=64, 6 IMDModules, x4 upscale.
    model = IMDN(in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4)
    model.eval()
    return model


def example_input_imdn():
    torch.manual_seed(0)
    # Small LR patch; IMDN is fully convolutional so any spatial size works.
    return torch.randn(1, 3, 24, 24)


MENAGERIE_ENTRIES = [
    ("IMDN", build_imdn, example_input_imdn, 2019, "vendored-pytorch"),
]
