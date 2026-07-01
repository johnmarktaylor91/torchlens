# SOURCE: vendored from zhangyp15/MonoFlex @ ec6da017c325451b7d997d89e323083fa8430ada
# (https://github.com/zhangyp15/MonoFlex, CVPR 2021)
#
# Files vendored (near-verbatim, only import paths/formatting adjusted):
#   - model/backbone/dla_dcn.py   (DLA-34 backbone + DLAUp/IDAUp with deformable convs)
#   - model/head/detector_predictor.py (`_predictor` / Base_Predictor head)
#   - model/make_layers.py        (group_norm / _fill_fc_weights helpers)
#   - model/layers/utils.py       (sigmoid_hm)
#   - model/detector.py           (KeypointDetector top-level module)
#
# Environment substitution (NOT an architecture change): the upstream `DCN` module
# (model/backbone/DCNv2/dcn_v2.py) wraps a hand-written CUDA extension (`_ext`,
# built via `python setup.py build` inside DCNv2/) implementing modulated deformable
# convolution v2 (offset + mask). That extension cannot be compiled in this base
# environment. torchvision ships the identical op natively as
# `torchvision.ops.DeformConv2d` / `deform_conv2d` (same forward signature: input,
# offset shape `2*deformable_groups*kh*kw`, mask shape `deformable_groups*kh*kw`).
# We use torchvision's native op in place of the CUDA extension; the surrounding
# `DeformConv` wrapper module (conv_offset_mask -> chunk -> sigmoid(mask) -> deform
# conv) is otherwise unchanged from the original `DCN.forward`.
#
# Config choice for tracing (both are existing cfg flags in the real repo, default
# values from config/defaults.py -- not an architecture change):
#   - MODEL.INPLACE_ABN = False (default)          -> plain BatchNorm2d path, avoids
#     the `inplace_abn` CUDA extension used only when INPLACE_ABN=True.
#   - MODEL.HEAD.ENABLE_EDGE_FUSION = False (default) -> skips the KITTI-specific
#     edge-fusion branch in `_predictor.forward`, which otherwise requires dataset
#     `targets` objects (edge_indices/edge_len fields from the KITTI dataloader) that
#     are orthogonal to the network architecture itself.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d

MENAGERIE_ZOO = "vendored-pytorch"

BN_MOMENTUM = 0.1


# --------------------------------------------------------------------------- #
# model/make_layers.py + model/layers/utils.py (vendored verbatim, minus the
# global `cfg` GroupNorm helper which is unused with the default USE_NORMALIZATION="BN")
# --------------------------------------------------------------------------- #


def _fill_fc_weights(layers, value=0):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, value)


def sigmoid_hm(hm_features):
    x = hm_features.sigmoid()
    x = x.clamp(min=1e-4, max=1 - 1e-4)
    return x


# --------------------------------------------------------------------------- #
# model/backbone/dla_dcn.py (vendored, DCN CUDA ext swapped for torchvision native op)
# --------------------------------------------------------------------------- #


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
    ):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(
        self,
        levels,
        channels,
        num_classes=1000,
        block=BasicBlock,
        residual_root=False,
        linear_root=False,
    ):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
        )

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, "level{}".format(i))(x)
            y.append(x)

        return y


def dla34(pretrained=False, **kwargs):  # DLA-34, random init (pretrained ImageNet
    # download disabled here: menagerie captures tiny random-init nets only)
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    return model


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DCN(nn.Module):
    """Modulated deformable conv (DCNv2), same forward contract as the upstream
    CUDA-extension `DCN` (model/backbone/DCNv2/dcn_v2.py): a conv predicts the
    per-location offset+mask, which drives a deformable convolution. Here the
    deformable-conv primitive itself is `torchvision.ops.DeformConv2d` (native,
    no custom CUDA build required) instead of the hand-written `_ext` kernel."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCN, self).__init__()
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.deformable_groups = deformable_groups
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=True,
        )

        channels_ = deformable_groups * 3 * kernel_size[0] * kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            channels_,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.deform_conv(input, offset, mask)


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.conv = DCN(
            chi,
            cho,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(
                o,
                o,
                f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=o,
                bias=False,
            )
            fill_up_weights(up)

            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)
            setattr(self, "node_" + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, "up_" + str(i - startp))
            project = getattr(self, "proj_" + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, "node_" + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self,
                "ida_{}".format(i),
                IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]),
            )
            scales[j + 1 :] = scales[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, "ida_{}".format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class DLASeg(nn.Module):
    def __init__(self, base_name, down_ratio, last_level, pretrained=False):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)

        channels = self.base.channels
        scales = [2**i for i in range(len(channels[self.first_level :]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level :], scales)

        self.out_channels = channels[self.first_level]

        self.ida_up = IDAUp(
            self.out_channels,
            channels[self.first_level : self.last_level],
            [2**i for i in range(self.last_level - self.first_level)],
        )

    def forward(self, x):
        # x: list of features with stride = 1, 2, 4, 8, 16, 32
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return y[-1]


def build_backbone(down_ratio=4):
    return DLASeg(base_name="dla34", pretrained=False, down_ratio=down_ratio, last_level=5)


# --------------------------------------------------------------------------- #
# model/head/detector_predictor.py `_predictor` (Base_Predictor), vendored with
# defaults MODEL.INPLACE_ABN=False, MODEL.HEAD.ENABLE_EDGE_FUSION=False (both are
# the real repo's default cfg values, see config/defaults.py)
# --------------------------------------------------------------------------- #


class Predictor(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=3,
        head_conv=256,
        regression_head_cfg=(
            ("2d_dim",),
            ("3d_offset",),
            ("3d_dim",),
            ("ori_cls", "ori_offset"),
            ("depth",),
        ),
        regression_channel_cfg=((4,), (2,), (3,), (4, 2), (1,)),
        init_p=0.01,
        bn_momentum=0.1,
        uncertainty_init=True,
    ):
        super(Predictor, self).__init__()
        classes = num_classes

        self.regression_head_cfg = regression_head_cfg
        self.regression_channel_cfg = regression_channel_cfg
        self.head_conv = head_conv

        norm_func = nn.BatchNorm2d  # MODEL.HEAD.USE_NORMALIZATION default "BN"

        # MODEL.INPLACE_ABN default False -> plain conv/bn/relu class head
        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
            norm_func(self.head_conv, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
        )

        self.class_head[-1].bias.data.fill_(-np.log(1 / init_p - 1))

        self.reg_features = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        for idx, regress_head_key in enumerate(self.regression_head_cfg):
            feat_layer = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                norm_func(self.head_conv, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            )

            self.reg_features.append(feat_layer)
            head_channels = self.regression_channel_cfg[idx]
            head_list = nn.ModuleList()
            for key_index, key in enumerate(regress_head_key):
                key_channel = head_channels[key_index]
                output_head = nn.Conv2d(
                    self.head_conv, key_channel, kernel_size=1, padding=1 // 2, bias=True
                )

                if key.find("uncertainty") >= 0 and uncertainty_init:
                    torch.nn.init.xavier_normal_(output_head.weight, gain=0.01)

                _fill_fc_weights(output_head, 0)
                head_list.append(output_head)

            self.reg_heads.append(head_list)

        # MODEL.HEAD.ENABLE_EDGE_FUSION default False -> no edge-fusion branch built

    def forward(self, features):
        # output classification
        feature_cls = self.class_head[:-1](features)
        output_cls = self.class_head[-1](feature_cls)

        output_regs = []
        for i, reg_feature_head in enumerate(self.reg_features):
            reg_feature = reg_feature_head(features)
            for reg_output_head in self.reg_heads[i]:
                output_reg = reg_output_head(reg_feature)
                output_regs.append(output_reg)

        output_cls = sigmoid_hm(output_cls)
        output_regs = torch.cat(output_regs, dim=1)

        return {"cls": output_cls, "reg": output_regs}


# --------------------------------------------------------------------------- #
# model/detector.py `KeypointDetector`, vendored (inference-only: the real class
# also branches into `self.heads` -> loss_evaluator/post_processor for train/test
# post-processing, which are dataset/box-decoding utilities layered on top of this
# same backbone+predictor forward; those are not part of the network architecture
# proper and are out of scope for a forward-pass activation capture).
# --------------------------------------------------------------------------- #


class MonoFlexDetector(nn.Module):
    def __init__(self, num_classes=3, down_ratio=4):
        super(MonoFlexDetector, self).__init__()
        self.backbone = build_backbone(down_ratio=down_ratio)
        self.predictor = Predictor(self.backbone.out_channels, num_classes=num_classes)

    def forward(self, images):
        features = self.backbone(images)
        return self.predictor(features)


# --------------------------------------------------------------------------- #
# menagerie staging entry points
# --------------------------------------------------------------------------- #


def build_monoflex():
    return MonoFlexDetector(num_classes=3, down_ratio=4)


def example_input_monoflex():
    # KITTI-style input; kept tiny (down_ratio=4 gives an output stride of 4, so a
    # 64x160 input is small but still a multiple of 32 for the DLA stem).
    return torch.rand(1, 3, 64, 160)


MENAGERIE_ENTRIES = [
    ("MonoFlex", "build_monoflex", "example_input_monoflex", 2021, "vendored-pytorch"),
]
