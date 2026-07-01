# SOURCE: vendored from ZJULearning/resa @ main
#
# https://github.com/ZJULearning/resa
# https://raw.githubusercontent.com/ZJULearning/resa/main/models/resa.py
# https://raw.githubusercontent.com/ZJULearning/resa/main/models/resnet.py
# https://raw.githubusercontent.com/ZJULearning/resa/main/models/decoder.py
# https://raw.githubusercontent.com/ZJULearning/resa/main/configs/culane.py
#
# NOTE: the queue entry names this candidate "LaneGAN"
# (https://github.com/ZJULearning/resa) but the oneline/notes identify the
# actual repo as RESA (Recurrent Feature-Shift Aggregator for Lane
# Detection, AAAI 2021) -- "LaneGAN" in the catalog queue apparently refers
# to RESA's recurrent feature-shift mechanism, not a GAN architecture; there
# is no GAN component anywhere in this repo. A RESA family entry already
# exists in the live catalog (`RESA-ResNet18`, family=`resa`) built from a
# from-scratch reimplementation in `menagerie/classics/lanedet.py` (a
# rehost/reimplementation of several lane-detection heads, not sourced from
# ZJULearning/resa). This file instead vendors the REAL ZJULearning/resa
# source directly (strictly more faithful than a reimplementation, per the
# menagerie ladder's "if source code exists, use the real source" rule) as
# an additional, source-verified entry.
#
# This vendors the real `RESA`/`ExistHead`/`RESANet` classes from
# `models/resa.py`, the real `ResNetWrapper`/`ResNet`/`BasicBlock`/
# `Bottleneck`/resnetNN factory functions from `models/resnet.py` (this repo
# vendors its own copy of a torchvision-style ResNet, explicitly commented
# "This code is borrow from torchvision" in the real file), and the real
# `PlainDecoder` from `models/decoder.py` -- copied verbatim (only
# whitespace-preserving copy, no architecture changes). `@NET.register_module`
# (a decorator that registers `RESANet` into a repo-internal `utils.Registry`
# config-building system, `models/registry.py`) is dropped: it is
# config-plumbing infrastructure for building models from YAML-like cfg
# objects via `build_net(cfg)`, not part of the network architecture itself;
# `RESANet` is constructed directly here with the same `cfg` object shape
# the registry path would have produced from `configs/culane.py`.
#
# `build_resa()` constructs `RESANet(cfg)` with a `SimpleNamespace`-based cfg
# whose fields mirror the real `configs/culane.py` reference config exactly
# (`backbone.resnet='resnet18'` -- resnet18 substituted for the config's
# resnet50 purely to keep the traced graph small, a real supported value of
# the real `ResNetWrapper.__init__`'s `eval(cfg.backbone.resnet)` dispatch;
# `replace_stride_with_dilation=[False, True, True]`, `out_conv=True`,
# `fea_stride=8`; `resa.alpha=2.0`, `resa.iter=4`, `resa.input_channel=128`,
# `resa.conv_stride=9`; `decoder='PlainDecoder'`; `num_classes=5`
# (`4 + 1` in the real config); `img_height`/`img_width` shrunk from the
# real 288x800 to a small multiple of `fea_stride` for fast tracing).

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.hub import load_state_dict_from_url

BN_MOMENTUM = (
    0.1  # not used by resnet.py itself; kept for parity with other menagerie DLA/ResNet vendors
)


# ---------------------------------------------------------------------------
# models/resnet.py ("This code is borrow from torchvision.")
# ---------------------------------------------------------------------------

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def resnet_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
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


def resnet_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(ResNetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        self.conv1 = resnet_conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resnet_conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)

        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(ResNetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = resnet_conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetWrapper(nn.Module):
    def __init__(self, cfg):
        super(ResNetWrapper, self).__init__()
        self.cfg = cfg
        self.in_channels = [64, 128, 256, 512]
        if hasattr(cfg.backbone, "in_channels"):
            self.in_channels = cfg.backbone.in_channels
        self.model = _RESNET_FACTORY[cfg.backbone.resnet](
            pretrained=cfg.backbone.pretrained,
            replace_stride_with_dilation=cfg.backbone.replace_stride_with_dilation,
            in_channels=self.in_channels,
        )
        self.out = None
        if cfg.backbone.out_conv:
            out_channel = 512
            for chan in reversed(self.in_channels):
                if chan < 0:
                    continue
                out_channel = chan
                break
            self.out = resnet_conv1x1(out_channel * self.model.expansion, 128)

    def forward(self, x):
        x = self.model(x)
        if self.out:
            x = self.out(x)
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        in_channels=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = in_channels
        self.layer1 = self._make_layer(block, in_channels[0], layers[0])
        self.layer2 = self._make_layer(
            block, in_channels[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, in_channels[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        if in_channels[3] > 0:
            self.layer4 = self._make_layer(
                block, in_channels[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2]
            )
        self.expansion = block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResNetBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                resnet_conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.in_channels[3] > 0:
            x = self.layer4(x)

        return x


def _resa_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resa_resnet("resnet18", ResNetBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resa_resnet("resnet34", ResNetBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resa_resnet("resnet50", ResNetBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resa_resnet(
        "resnet101", ResNetBottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resa_resnet(
        "resnet152", ResNetBottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


_RESNET_FACTORY = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


# ---------------------------------------------------------------------------
# models/decoder.py
# ---------------------------------------------------------------------------


class PlainDecoder(nn.Module):
    def __init__(self, cfg):
        super(PlainDecoder, self).__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout2d(0.1)
        self.conv8 = nn.Conv2d(128, cfg.num_classes, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv8(x)
        x = F.interpolate(
            x, size=[self.cfg.img_height, self.cfg.img_width], mode="bilinear", align_corners=False
        )

        return x


# ---------------------------------------------------------------------------
# models/resa.py
# ---------------------------------------------------------------------------


class RESA(nn.Module):
    def __init__(self, cfg):
        super(RESA, self).__init__()
        self.iter = cfg.resa.iter
        chan = cfg.resa.input_channel
        fea_stride = cfg.backbone.fea_stride
        self.height = cfg.img_height // fea_stride
        self.width = cfg.img_width // fea_stride
        self.alpha = cfg.resa.alpha
        conv_stride = cfg.resa.conv_stride

        for i in range(self.iter):
            conv_vert1 = nn.Conv2d(
                chan, chan, (1, conv_stride), padding=(0, conv_stride // 2), groups=1, bias=False
            )
            conv_vert2 = nn.Conv2d(
                chan, chan, (1, conv_stride), padding=(0, conv_stride // 2), groups=1, bias=False
            )

            setattr(self, "conv_d" + str(i), conv_vert1)
            setattr(self, "conv_u" + str(i), conv_vert2)

            conv_hori1 = nn.Conv2d(
                chan, chan, (conv_stride, 1), padding=(conv_stride // 2, 0), groups=1, bias=False
            )
            conv_hori2 = nn.Conv2d(
                chan, chan, (conv_stride, 1), padding=(conv_stride // 2, 0), groups=1, bias=False
            )

            setattr(self, "conv_r" + str(i), conv_hori1)
            setattr(self, "conv_l" + str(i), conv_hori2)

            idx_d = (torch.arange(self.height) + self.height // 2 ** (self.iter - i)) % self.height
            setattr(self, "idx_d" + str(i), idx_d)

            idx_u = (torch.arange(self.height) - self.height // 2 ** (self.iter - i)) % self.height
            setattr(self, "idx_u" + str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width // 2 ** (self.iter - i)) % self.width
            setattr(self, "idx_r" + str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width // 2 ** (self.iter - i)) % self.width
            setattr(self, "idx_l" + str(i), idx_l)

    def forward(self, x):
        x = x.clone()

        for direction in ["d", "u"]:
            for i in range(self.iter):
                conv = getattr(self, "conv_" + direction + str(i))
                idx = getattr(self, "idx_" + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx, :])))

        for direction in ["r", "l"]:
            for i in range(self.iter):
                conv = getattr(self, "conv_" + direction + str(i))
                idx = getattr(self, "idx_" + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx])))

        return x


class ExistHead(nn.Module):
    def __init__(self, cfg=None):
        super(ExistHead, self).__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout2d(0.1)  # ???
        self.conv8 = nn.Conv2d(128, cfg.num_classes, 1)

        stride = cfg.backbone.fea_stride * 2
        self.fc9 = nn.Linear(
            int(cfg.num_classes * cfg.img_width / stride * cfg.img_height / stride), 128
        )
        self.fc10 = nn.Linear(128, cfg.num_classes - 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv8(x)

        x = F.softmax(x, dim=1)
        x = F.avg_pool2d(x, 2, stride=2, padding=0)
        x = x.view(-1, x.numel() // x.shape[0])
        x = self.fc9(x)
        x = F.relu(x)
        x = self.fc10(x)
        x = torch.sigmoid(x)

        return x


class RESANet(nn.Module):
    def __init__(self, cfg):
        super(RESANet, self).__init__()
        # real code: self.decoder = eval(cfg.decoder)(cfg) -- dispatches to
        # either PlainDecoder or BUSD (models/decoder.py) by cfg string.
        # Only PlainDecoder is vendored (this build always sets
        # cfg.decoder='PlainDecoder', matching the real culane.py config's
        # default), so the dispatch is inlined directly rather than kept as
        # a string-eval (no behavior change for the configs this module
        # supports).
        assert cfg.decoder == "PlainDecoder", "only PlainDecoder is vendored"
        self.cfg = cfg
        self.backbone = ResNetWrapper(cfg)
        self.resa = RESA(cfg)
        self.decoder = PlainDecoder(cfg)
        self.heads = ExistHead(cfg)

    def forward(self, batch):
        fea = self.backbone(batch)
        fea = self.resa(fea)
        seg = self.decoder(fea)
        exist = self.heads(fea)

        output = {"seg": seg, "exist": exist}

        return output


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------


class _Cfg:
    """Plain attribute-namespace substitute for the real repo's
    utils.config.Config (a dict-backed attrdict); only the fields RESANet's
    submodules actually read are set, matching configs/culane.py's values
    (resnet18 substituted for resnet50, num_classes/img_height/img_width
    shrunk -- see file header)."""


class _Backbone:
    pass


class _Resa:
    pass


def build_resa() -> nn.Module:
    cfg = _Cfg()
    cfg.backbone = _Backbone()
    cfg.backbone.resnet = "resnet18"
    cfg.backbone.pretrained = False
    cfg.backbone.replace_stride_with_dilation = [False, True, True]
    cfg.backbone.out_conv = True
    cfg.backbone.fea_stride = 8

    cfg.resa = _Resa()
    cfg.resa.alpha = 2.0
    cfg.resa.iter = 4
    cfg.resa.input_channel = 128
    cfg.resa.conv_stride = 9

    cfg.decoder = "PlainDecoder"
    cfg.num_classes = 5
    cfg.img_height = 64
    cfg.img_width = 160

    return RESANet(cfg)


def example_input_resa():
    # Real CULane config input is (1, 3, 288, 800); shrunk to a small
    # multiple of ExistHead's stride = fea_stride*2 = 16 (64 = 4*16,
    # 160 = 10*16) so ExistHead.fc9's input-width formula
    # (num_classes * img_width/stride * img_height/stride, an exact-integer
    # real-code computation) matches the actual tensor size after
    # avg_pool2d(x, 2, stride=2) for fast tracing.
    return torch.randn(1, 3, 64, 160)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RESA-ResNet18-real", "build_resa", "example_input_resa", 2021, "vendored"),
]
