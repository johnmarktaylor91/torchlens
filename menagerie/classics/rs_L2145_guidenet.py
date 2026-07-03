# FAITHFUL PORT of kakaxi314/GuideNet @ master (models.py) (original framework: PyTorch
# + custom CUDA extension `GuideConv` + `PyTorch-Encoding` (`encoding.nn.SyncBatchNorm`))
# https://github.com/kakaxi314/GuideNet
#
# GuideNet (guided convolutional network for depth completion, IEEE TIP 2020): dual
# image/lidar ResNet encoder-decoder towers fused at 4 scales by a "Guide" module that
# predicts a per-pixel, per-channel local convolution kernel and a per-pixel 1x1 channel
# mixing matrix, applied to the lidar branch features.
#
# The upstream `models.py` cannot run in a base-lib env as-is: it depends on a custom
# compiled CUDA extension (`GuideConv`, built from `exts/guideconv.cpp` +
# `exts/guideconv_kernel.cu`, no prebuilt wheel) for the per-pixel local convolution
# `Conv2dLocal_F`/`Conv2dLocal_B`, and on `PyTorch-Encoding`'s `encoding.nn.SyncBatchNorm`
# (a distributed-training multi-GPU BatchNorm variant, not an architectural difference
# from `nn.BatchNorm2d` in a single-process trace). Per the ladder this becomes a faithful
# PORT: every layer/mechanism below is transcribed unchanged from `models.py`, with only
# two substitutions:
#   1. `Conv2dLocal`/`Conv2dLocal_F` (the custom CUDA op) is re-implemented in pure torch
#      via `F.unfold`, reproducing the exact math of `guideconv_kernel.cu`'s
#      `conv2d_kernel_lf`: for each batch/channel/pixel, output[b,c,h,w] =
#      sum_{i,j in KxK} input[b,c,h+i,w+j] * weight[b,c,i,j,h,w] (zero-padded), i.e. a
#      per-pixel-varying depthwise KxK convolution -- this is a drop-in `Conv2dLocal_F`
#      forward-equivalent (autograd handled by `unfold`'s own backward, no custom
#      `Conv2dLocal_B` needed).
#   2. `encoding.nn.SyncBatchNorm` -> `torch.nn.BatchNorm2d` (identical single-process
#      forward math; SyncBatchNorm only changes multi-GPU statistics synchronization).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:
    from scipy.stats import truncnorm

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a base-lib dependency here
    _HAS_SCIPY = False

__all__ = [
    "GN",
    "GNS",
]


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


# ---------------------------------------------------------------------------
# Pure-torch port of the custom `GuideConv.Conv2dLocal_F` CUDA kernel.
# Reference semantics (exs/guideconv_kernel.cu, conv2d_kernel_lf):
#   x: (B, C, H, W); y (weight): (B, C*K*K, H, W) viewed as (B, C, K, K, H, W)
#   z[b,c,row,col] = sum_{i,j} x[b,c,row+i,col+j] * y[b,c,i,j,row,col] (zero outside bounds)
# i.e. a per-pixel-varying depthwise KxK convolution (Co == Ci by construction upstream).
# ---------------------------------------------------------------------------
def _conv2d_local_forward(x, weight):
    """Pure-torch equivalent of GuideConv.Conv2dLocal_F.

    Args:
        x: (B, C, H, W) input.
        weight: (B, C*K*K, H, W) per-pixel local kernel weights.
    Returns:
        (B, C, H, W) output.
    """
    B, C, H, W = x.shape
    K = int(round(math.sqrt(weight.shape[1] / C)))
    pad = (K - 1) // 2
    # unfold x into (B, C*K*K, H*W) sliding local blocks (zero-padded), matching the
    # CUDA kernel's zero-outside-bounds behavior.
    x_unf = F.unfold(x, kernel_size=K, padding=pad)  # (B, C*K*K, H*W)
    x_unf = x_unf.view(B, C, K * K, H * W)
    w = weight.view(B, C, K * K, H * W)
    out = (x_unf * w).sum(dim=2)  # (B, C, H*W)
    return out.view(B, C, H, W)


class Conv2dLocal_F(Function):
    @staticmethod
    def forward(ctx, input, weight):
        return _conv2d_local_forward(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        # Not needed for forward tracing/capture; the pure-torch forward above is
        # itself autograd-differentiable via unfold, so this custom Function is kept
        # only as a structural stand-in for the original `Conv2dLocal_F.apply` call
        # site and simply re-dispatches to autograd if ever invoked directly.
        raise NotImplementedError(
            "Conv2dLocal_F.backward is unused; Conv2dLocal uses the autograd-native "
            "unfold-based forward instead of torch.autograd.Function.apply."
        )


class Conv2dLocal(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, input, weight):
        # Directly call the differentiable pure-torch implementation (equivalent to
        # `Conv2dLocal_F.apply(input, weight)` in the original, but autograd-native).
        output = _conv2d_local_forward(input, weight)
        return output


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            )
        else:
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=True,
            )
        self.conv = nn.Sequential(
            conv,
        )
        if norm_layer:
            self.conv.add_module("bn", norm_layer(out_channels))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Basic2dLocal(nn.Module):
    def __init__(self, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = Conv2dLocal()
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, weight):
        out = self.conv(input, weight)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Guide(nn.Module):
    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.local = Basic2dLocal(input_planes, norm_layer)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv11 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv12 = nn.Conv2d(
            input_planes, input_planes * 9, kernel_size=weight_ks, padding=weight_ks // 2
        )
        self.conv21 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv22 = nn.Conv2d(input_planes, input_planes * input_planes, kernel_size=1, padding=0)
        self.br = nn.Sequential(
            norm_layer(num_features=input_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic2d(input_planes, input_planes, norm_layer)

    def forward(self, input, weight):
        B, Ci, H, W = input.shape
        weight = torch.cat([input, weight], 1)
        weight11 = self.conv11(weight)
        weight12 = self.conv12(weight11)
        weight21 = self.conv21(weight)
        weight21 = self.pool(weight21)
        weight22 = self.conv22(weight21).view(B, -1, Ci)
        out = self.local(input, weight12).view(B, Ci, -1)
        out = torch.bmm(weight22, out).view(B, Ci, H, W)
        out = self.br(out)
        out = self.conv3(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out


class GuideNet(nn.Module):
    """
    Not activate at the ref
    Init change to trunctated norm
    """

    def __init__(
        self,
        block=BasicBlock,
        bc=16,
        img_layers=[2, 2, 2, 2, 2],
        depth_layers=[2, 2, 2, 2, 2],
        norm_layer=nn.BatchNorm2d,
        guide=Guide,
        weight_ks=3,
    ):
        super().__init__()
        self._norm_layer = norm_layer

        self.conv_img = Basic2d(3, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)
        in_channels = bc * 2
        self.inplanes = in_channels
        self.layer1_img = self._make_layer(block, in_channels * 2, img_layers[0], stride=2)

        self.guide1 = guide(in_channels * 2, in_channels * 2, norm_layer, weight_ks)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_img = self._make_layer(block, in_channels * 4, img_layers[1], stride=2)

        self.guide2 = guide(in_channels * 4, in_channels * 4, norm_layer, weight_ks)
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_img = self._make_layer(block, in_channels * 8, img_layers[2], stride=2)

        self.guide3 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_img = self._make_layer(block, in_channels * 8, img_layers[3], stride=2)

        self.guide4 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_img = self._make_layer(block, in_channels * 8, img_layers[4], stride=2)

        self.layer2d_img = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.layer3d_img = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer4d_img = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer5d_img = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)

        self.conv_lidar = Basic2d(1, bc * 2, norm_layer=None, kernel_size=5, padding=2)

        self.inplanes = in_channels
        self.layer1_lidar = self._make_layer(block, in_channels * 2, depth_layers[0], stride=2)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_lidar = self._make_layer(block, in_channels * 4, depth_layers[1], stride=2)
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_lidar = self._make_layer(block, in_channels * 8, depth_layers[2], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_lidar = self._make_layer(block, in_channels * 8, depth_layers[3], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_lidar = self._make_layer(block, in_channels * 8, depth_layers[4], stride=2)

        self.layer1d = Basic2dTrans(in_channels * 2, in_channels, norm_layer)
        self.layer2d = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.layer3d = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer4d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer5d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)

        self.conv = nn.Conv2d(bc * 2, 1, kernel_size=3, stride=1, padding=1)
        self.ref = block(bc * 2, bc * 2, norm_layer=norm_layer, act=False)

        self._initialize_weights()

    def forward(self, img, lidar):
        c0_img = self.conv_img(img)
        c1_img = self.layer1_img(c0_img)
        c2_img = self.layer2_img(c1_img)
        c3_img = self.layer3_img(c2_img)
        c4_img = self.layer4_img(c3_img)
        c5_img = self.layer5_img(c4_img)
        dc5_img = self.layer5d_img(c5_img)
        c4_mix = dc5_img + c4_img
        dc4_img = self.layer4d_img(c4_mix)
        c3_mix = dc4_img + c3_img
        dc3_img = self.layer3d_img(c3_mix)
        c2_mix = dc3_img + c2_img
        dc2_img = self.layer2d_img(c2_mix)
        c1_mix = dc2_img + c1_img

        c0_lidar = self.conv_lidar(lidar)
        c1_lidar = self.layer1_lidar(c0_lidar)
        c1_lidar_dyn = self.guide1(c1_lidar, c1_mix)
        c2_lidar = self.layer2_lidar(c1_lidar_dyn)
        c2_lidar_dyn = self.guide2(c2_lidar, c2_mix)
        c3_lidar = self.layer3_lidar(c2_lidar_dyn)
        c3_lidar_dyn = self.guide3(c3_lidar, c3_mix)
        c4_lidar = self.layer4_lidar(c3_lidar_dyn)
        c4_lidar_dyn = self.guide4(c4_lidar, c4_mix)
        c5_lidar = self.layer5_lidar(c4_lidar_dyn)
        c5 = c5_img + c5_lidar
        dc5 = self.layer5d(c5)
        c4 = dc5 + c4_lidar_dyn
        dc4 = self.layer4d(c4)
        c3 = dc4 + c3_lidar_dyn
        dc3 = self.layer3d(c3)
        c2 = dc3 + c2_lidar_dyn
        dc2 = self.layer2d(c2)
        c1 = dc2 + c1_lidar_dyn
        dc1 = self.layer1d(c1)
        c0 = dc1 + c0_lidar
        output = self.ref(c0)
        output = self.conv(output)
        return (output,)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        def truncated_normal_(num, mean=0.0, std=1.0):
            lower = -2 * std
            upper = 2 * std
            if _HAS_SCIPY:
                X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
                samples = X.rvs(num)
                output = torch.from_numpy(samples)
            else:  # pragma: no cover - scipy is available in the target env
                output = torch.empty(num).normal_(mean, std).clamp_(lower, upper)
            return output

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2.0 / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def GN():
    return GuideNet(norm_layer=nn.BatchNorm2d, guide=Guide)


def GNS():
    return GuideNet(norm_layer=nn.BatchNorm2d, guide=Guide, weight_ks=1)


# ---------------------------------------------------------------------------
# Menagerie staging helpers
# ---------------------------------------------------------------------------

MENAGERIE_ZOO = "ported-pytorch"


def build_guidenet():
    # Small bc keeps the traced graph tiny while exercising the full dual-tower
    # encoder/decoder + all 4 Guide fusion modules.
    return GuideNet(
        bc=4,
        img_layers=[1, 1, 1, 1, 1],
        depth_layers=[1, 1, 1, 1, 1],
        norm_layer=nn.BatchNorm2d,
        guide=Guide,
        weight_ks=3,
    )


def example_input_guidenet():
    return {"img": torch.zeros(1, 3, 64, 64), "lidar": torch.zeros(1, 1, 64, 64)}


def _build_guidenet_module_wrapper():
    class _GuideNetWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = build_guidenet()

        def forward(self, x):
            return self.net(x["img"], x["lidar"])

    return _GuideNetWrapper()


MENAGERIE_ENTRIES = [
    (
        "GuideNet (guided depth completion)",
        _build_guidenet_module_wrapper,
        example_input_guidenet,
        2020,
        "ported",
    ),
]
