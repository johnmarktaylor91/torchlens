# FAITHFUL PORT of lzccccc/SMOKE @ master (original framework: PyTorch + custom CUDA
# DCNv2 extension built via smoke/csrc, compiled at install time by setup.py).
#
# SMOKE (CVPRW 2020): "SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint
# Estimation". Single-stage monocular 3D detector: a DLA-34-DCN backbone (Deep Layer
# Aggregation with deformable-convolution upsampling) feeds a lightweight class/regression
# head predicting a keypoint heatmap plus depth/dimension/orientation regression channels.
#
# Source files transcribed:
#   - smoke/modeling/backbone/dla.py (DLABase tree/BasicBlock/Root, DLAUp, IDAUp)
#   - smoke/modeling/backbone/backbone.py (DLA-34-DCN backbone assembly)
#   - smoke/modeling/make_layers.py (_make_conv_level, _fill_up_weights, _fill_fc_weights,
#     group_norm)
#   - smoke/modeling/heads/smoke_head/smoke_predictor.py (SMOKEPredictor: class_head +
#     regression_head convs, dim/ori channel post-processing)
#   - smoke/modeling/heads/smoke_head/smoke_head.py (SMOKEHead.forward: predictor is the
#     only network computation on the traced/inference path; loss_evaluator and
#     post_processor consume GT `targets` / do NMS-style decoding and are not part of the
#     forward network graph)
#   - smoke/modeling/detector/keypoint_detector.py (KeypointDetector: backbone -> heads)
#   - smoke/layers/utils.py (sigmoid_hm)
#   - smoke/config/defaults.py (DLA34DCN stage spec, default channel/head config values
#     used below: DOWN_RATIO=4, BACKBONE_OUT_CHANNELS=64, NUM_CHANNEL=256,
#     REGRESSION_CHANNEL=(1, 2, 3, 2), DETECT_CLASSES=("Car",) by default -> 1 class)
#
# The ONLY deviation from the original source: the custom CUDA `DCN` module in
# smoke/layers/dcn_v2.py (a hand-written autograd.Function calling a compiled
# `smoke._ext.dcn_v2_forward` kernel built from smoke/csrc/*.cu at install time) is
# replaced with torchvision.ops.deform_conv2d, the native base-lib modulated deformable
# convolution kernel -- mathematically the same operator (conv_offset_mask produces
# offset_x/offset_y/mask via a plain nn.Conv2d, mask passed through sigmoid, then applied
# via deform_conv2d), just without needing a private compiled extension. This is the same
# substitution already used for the RTM3D port in this staging tree (L211_1_rtm3d.py),
# which also depends on a compiled DCNv2 extension.
#
# Norm choice: SMOKE's registered `GN` norm variant is used for both backbone and head
# (cfg.MODEL.BACKBONE.USE_NORMALIZATION / cfg.MODEL.SMOKE_HEAD.USE_NORMALIZATION default
# to "GN" in defaults.py), transcribing `group_norm()` verbatim (NUM_GROUPS=32, halved
# when out_channels is not divisible by 32).

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


# --------------------------------------------------------------------------------------
# smoke/layers/dcn_v2.py -- DCN module, offset/mask conv transcribed verbatim; the actual
# deformable-conv kernel call is swapped for torchvision's native deform_conv2d (see
# header note above).
# --------------------------------------------------------------------------------------
class DCN(nn.Module):
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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


# --------------------------------------------------------------------------------------
# smoke/layers/deform_conv.py -- DeformConv (DCN + norm + relu), verbatim.
# --------------------------------------------------------------------------------------
class DeformConv(nn.Module):
    def __init__(self, in_channel, out_channel, norm_func):
        super().__init__()
        self.norm = norm_func(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.deform_conv = DCN(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1,
        )

    def forward(self, x):
        x = self.deform_conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


# --------------------------------------------------------------------------------------
# smoke/modeling/make_layers.py, verbatim (group_norm reads NUM_GROUPS=32 default).
# --------------------------------------------------------------------------------------
def _make_conv_level(in_channels, out_channels, num_convs, norm_func, stride=1, dilation=1):
    modules = []
    for i in range(num_convs):
        modules.extend(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=False,
                    dilation=dilation,
                ),
                norm_func(out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        in_channels = out_channels
    return nn.Sequential(*modules)


def group_norm(out_channels, num_groups=32):
    if out_channels % 32 == 0:
        return nn.GroupNorm(num_groups, out_channels)
    else:
        return nn.GroupNorm(num_groups // 2, out_channels)


def _fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def _fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# --------------------------------------------------------------------------------------
# smoke/modeling/backbone/dla.py -- BasicBlock, Tree, Root, DLABase, DLAUp, IDAUp,
# verbatim (DLA34DCN stage spec: levels=[1,1,1,2,2,1], channels=[16,32,64,128,256,512],
# block=BasicBlock, from defaults.py's registered "DLA-34-DCN" CONV_BODY).
# --------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_func, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.norm1 = norm_func(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.norm2 = norm_func(out_channels)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, norm_func, kernel_size, residual):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.norm = norm_func(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.norm(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Module):
    def __init__(
        self,
        level,
        block,
        in_channels,
        out_channels,
        norm_func,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
    ):
        super().__init__()

        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels

        if level == 1:
            self.tree1 = block(in_channels, out_channels, norm_func, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, norm_func, stride=1, dilation=dilation)
        else:
            new_level = level - 1
            self.tree1 = Tree(
                new_level,
                block,
                in_channels,
                out_channels,
                norm_func,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                new_level,
                block,
                out_channels,
                out_channels,
                norm_func,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        if level == 1:
            self.root = Root(root_dim, out_channels, norm_func, root_kernel_size, root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.level = level

        self.downsample = None
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                norm_func(out_channels),
            )

    def forward(self, x, residual=None, children=None):
        if children is None:
            children = []

        if self.downsample:
            bottom = self.downsample(x)
        else:
            bottom = x

        if self.project:
            residual = self.project(bottom)
        else:
            residual = bottom

        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)

        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLABase(nn.Module):
    def __init__(
        self, levels, channels, block=BasicBlock, residual_root=False, norm_func=nn.BatchNorm2d
    ):
        super().__init__()
        self.channels = channels
        self.level_length = len(levels)

        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            norm_func(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.level0 = _make_conv_level(
            in_channels=channels[0],
            out_channels=channels[0],
            num_convs=levels[0],
            norm_func=norm_func,
        )
        self.level1 = _make_conv_level(
            in_channels=channels[0],
            out_channels=channels[1],
            num_convs=levels[0],
            norm_func=norm_func,
            stride=2,
        )
        self.level2 = Tree(
            level=levels[2],
            block=block,
            in_channels=channels[1],
            out_channels=channels[2],
            norm_func=norm_func,
            stride=2,
            level_root=False,
            root_residual=residual_root,
        )
        self.level3 = Tree(
            level=levels[3],
            block=block,
            in_channels=channels[2],
            out_channels=channels[3],
            norm_func=norm_func,
            stride=2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            level=levels[4],
            block=block,
            in_channels=channels[3],
            out_channels=channels[4],
            norm_func=norm_func,
            stride=2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            level=levels[5],
            block=block,
            in_channels=channels[4],
            out_channels=channels[5],
            norm_func=norm_func,
            stride=2,
            level_root=True,
            root_residual=residual_root,
        )

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(self.level_length):
            x = getattr(self, "level{}".format(i))(x)
            y.append(x)
        return y


class IDAUp(nn.Module):
    def __init__(self, in_channels, out_channel, up_f, norm_func):
        super().__init__()
        for i in range(1, len(in_channels)):
            in_channel = in_channels[i]
            f = int(up_f[i])
            proj = DeformConv(in_channel, out_channel, norm_func)
            node = DeformConv(out_channel, out_channel, norm_func)

            up = nn.ConvTranspose2d(
                out_channel,
                out_channel,
                kernel_size=f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=out_channel,
                bias=False,
            )
            _fill_up_weights(up)

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
    def __init__(self, startp, channels, scales, in_channels=None, norm_func=nn.BatchNorm2d):
        super().__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)

        import numpy as np

        scales = np.array(scales, dtype=int)

        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self,
                "ida_{}".format(i),
                IDAUp(in_channels[j:], channels[j], scales[j:] // scales[j], norm_func),
            )
            scales[j + 1 :] = scales[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, "ida_{}".format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class DLA(nn.Module):
    """smoke/modeling/backbone/dla.py: DLA (full backbone: DLABase -> DLAUp -> IDAUp).

    cfg fields replaced by their DLA-34-DCN / defaults.py values directly (down_ratio=4,
    last_level=5, out_channel=0 -> channels[first_level]).
    """

    def __init__(self, down_ratio=4, last_level=5, out_channel=0):
        super().__init__()
        assert down_ratio in [2, 4, 8, 16]

        self.first_level = int(math.log2(down_ratio))
        self.last_level = last_level

        levels = [1, 1, 1, 2, 2, 1]
        channels = [16, 32, 64, 128, 256, 512]
        norm_func = group_norm  # cfg.MODEL.BACKBONE.USE_NORMALIZATION default "GN"

        self.base = DLABase(levels=levels, channels=channels, block=BasicBlock, norm_func=norm_func)

        scales = [2**i for i in range(len(channels[self.first_level :]))]
        self.dla_up = DLAUp(
            startp=self.first_level,
            channels=channels[self.first_level :],
            scales=scales,
            norm_func=norm_func,
        )

        if out_channel == 0:
            out_channel = channels[self.first_level]

        up_scales = [2**i for i in range(self.last_level - self.first_level)]
        self.ida_up = IDAUp(
            in_channels=channels[self.first_level : self.last_level],
            out_channel=out_channel,
            up_f=up_scales,
            norm_func=norm_func,
        )

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        return y[-1]


# --------------------------------------------------------------------------------------
# smoke/modeling/heads/smoke_head/smoke_predictor.py, verbatim. Defaults from
# defaults.py: DETECT_CLASSES=("Car",) -> classes=1, REGRESSION_HEADS=8,
# REGRESSION_CHANNEL=(1, 2, 3, 2) (depth_offset, keypoint_offset, dimension_offset,
# orientation), NUM_CHANNEL=256, USE_NORMALIZATION="GN".
# --------------------------------------------------------------------------------------
def _get_channel_spec(reg_channels, name):
    if name == "dim":
        s = sum(reg_channels[:2])
        e = sum(reg_channels[:3])
    elif name == "ori":
        s = sum(reg_channels[:3])
        e = sum(reg_channels)
    return slice(s, e, 1)


def sigmoid_hm(hm_features):
    x = hm_features.sigmoid()
    x = x.clamp(min=1e-4, max=1 - 1e-4)
    return x


class SMOKEPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        classes=1,
        regression=8,
        regression_channels=(1, 2, 3, 2),
        head_conv=256,
        norm_func=group_norm,
    ):
        super().__init__()

        assert sum(regression_channels) == regression

        self.dim_channel = _get_channel_spec(regression_channels, name="dim")
        self.ori_channel = _get_channel_spec(regression_channels, name="ori")

        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            norm_func(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
        )
        self.class_head[-1].bias.data.fill_(-2.19)

        self.regression_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            norm_func(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, regression, kernel_size=1, padding=1 // 2, bias=True),
        )
        _fill_fc_weights(self.regression_head)

    def forward(self, features):
        head_class = self.class_head(features)
        head_regression = self.regression_head(features)

        head_class = sigmoid_hm(head_class)

        offset_dims = head_regression[:, self.dim_channel, ...].clone()
        head_regression[:, self.dim_channel, ...] = torch.sigmoid(offset_dims) - 0.5

        vector_ori = head_regression[:, self.ori_channel, ...].clone()
        head_regression[:, self.ori_channel, ...] = F.normalize(vector_ori)

        return [head_class, head_regression]


# --------------------------------------------------------------------------------------
# smoke/modeling/detector/keypoint_detector.py: KeypointDetector (backbone -> heads).
# smoke/modeling/heads/smoke_head/smoke_head.py: SMOKEHead.forward -- the predictor call
# is the only network computation on the forward path (loss_evaluator/post_processor
# consume GT `targets` / heatmap decoding and are not part of the traced network graph).
# --------------------------------------------------------------------------------------
class SMOKE(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone: DLA-34-DCN, BACKBONE_OUT_CHANNELS=64 (defaults.py)
        self.backbone = DLA(down_ratio=4, last_level=5, out_channel=0)
        backbone_out_channels = 64
        self.predictor = SMOKEPredictor(
            in_channels=backbone_out_channels,
            classes=1,
            regression=8,
            regression_channels=(1, 2, 3, 2),
            head_conv=256,
            norm_func=group_norm,
        )

    def forward(self, images):
        features = self.backbone(images)
        head_class, head_regression = self.predictor(features)
        return head_class, head_regression


# --------------------------------------------------------------------------------------
# Staging entry points
# --------------------------------------------------------------------------------------
def build_smoke():
    return SMOKE()


def example_input_smoke():
    # INPUT.HEIGHT_TRAIN=384, INPUT.WIDTH_TRAIN=1280 (defaults.py); shrunk for tracing.
    return torch.randn(1, 3, 96, 320)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("SMOKE", "build_smoke", "example_input_smoke", 2020, "ported"),
]
