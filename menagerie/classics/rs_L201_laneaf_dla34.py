# SOURCE: vendored from sel118/LaneAF @ main
#
# https://github.com/sel118/LaneAF
# https://raw.githubusercontent.com/sel118/LaneAF/main/models/dla/pose_dla_dcn.py
# https://raw.githubusercontent.com/sel118/LaneAF/main/infer_culane.py
#
# LaneAF ("LaneAF: Robust Multi-Lane Detection with Affinity Fields") -- this
# vendors the real DLA-34-based pose/heatmap network from
# `models/dla/pose_dla_dcn.py`: `DLA` (the tree-recursive Deep Layer
# Aggregation backbone, `BasicBlock`/`Bottleneck`/`BottleneckX`/`Root`/`Tree`),
# `DLAUp`/`IDAUp` (the iterative deformable-conv-based up-sampling/aggregation
# stages), and `DLASeg` (backbone -> DLAUp -> IDAUp -> per-head heatmap/vaf/haf
# conv heads) -- copied verbatim (only whitespace-preserving copy, no
# architecture changes). `heads={'hm': 1, 'vaf': 2, 'haf': 1}` and
# `get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)`
# match the exact real construction call in `infer_culane.py` for the
# `dla34` backbone (LaneAF's reference/default backbone).
#
# `models/dla/pose_dla_dcn.py` imports `from .DCNv2.dcn_v2 import DCN` --
# CharlesShang/DCNv2, a custom CUDA torch extension implementing "Modulated
# Deformable Convolution" (DCNv2, https://arxiv.org/abs/1811.11168) via a
# hand-written CUDA kernel (`_ext.dcn_v2_forward`/`dcn_v2_backward`), not a
# pure-Python pip package -- not usable per RUNG 2 (needs_env territory, not
# something to install/compile here). The real `DCN` class (fetched from
# CharlesShang/DCNv2 for reference) is a thin, standard wrapper: a
# `conv_offset_mask` Conv2d generates `offset`/`mask` from the block input,
# then feeds `(input, offset, mask, weight, bias, ...)` into the deformable
# convolution primitive -- exactly the interface described by the DCNv2
# paper and implemented natively by `torchvision.ops.deform_conv2d`
# (`mask is not None` selects the DCNv2 modulated variant, same op
# semantics). This file's `DCN` class below reproduces that exact real
# wrapper pattern (own `conv_offset_mask`, same offset/mask split/chunk,
# same zero-init, same forward contract used by `DeformConv`/`IDAUp` here)
# but calls `torchvision.ops.deform_conv2d` for the actual deformable
# convolution op instead of the CUDA extension -- a base-library substitution
# of the identical primitive, not an architecture change.

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as tv_ops
from torch import nn

BN_MOMENTUM = 0.1


# ---------------------------------------------------------------------------
# DCNv2-equivalent DCN wrapper (torchvision.ops.deform_conv2d instead of the
# CharlesShang/DCNv2 CUDA extension -- see header note)
# ---------------------------------------------------------------------------


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
        return tv_ops.deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


# ---------------------------------------------------------------------------
# models/dla/pose_dla_dcn.py
# ---------------------------------------------------------------------------


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
            planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation
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


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
            groups=cardinality,
        )
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2
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

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

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


def dla34(pretrained=False, **kwargs):  # DLA-34
    # pretrained=False (menagerie default): the real dla34() default is
    # pretrained=True, downloading ImageNet weights from dl.yf.io via
    # load_pretrained_model()/model_zoo.load_url(); menagerie recipes use
    # random init, not network downloads.
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    return model


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.conv = DCN(
            chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1
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
                o, o, f * 2, stride=f, padding=f // 2, output_padding=0, groups=o, bias=False
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
    def __init__(
        self,
        base_name,
        heads,
        pretrained,
        down_ratio,
        final_kernel,
        last_level,
        head_conv,
        out_channel=0,
    ):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2**i for i in range(len(channels[self.first_level :]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level :], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel,
            channels[self.first_level : self.last_level],
            [2**i for i in range(self.last_level - self.first_level)],
        )

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(
                        channels[self.first_level], head_conv, kernel_size=3, padding=1, bias=True
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        head_conv,
                        classes,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True,
                    ),
                )
                if "hm" in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(
                    channels[self.first_level],
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                )
                if "hm" in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
    model = DLASeg(
        "dla{}".format(num_layers),
        heads,
        pretrained=False,
        down_ratio=down_ratio,
        final_kernel=1,
        last_level=5,
        head_conv=head_conv,
    )
    return model


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------


def build_laneaf_dla34():
    # heads/backbone/head_conv/down_ratio match the exact real construction
    # call in infer_culane.py: heads = {'hm': 1, 'vaf': 2, 'haf': 1};
    # get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4).
    heads = {"hm": 1, "vaf": 2, "haf": 1}
    return get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)


def example_input_laneaf_dla34():
    # Real CULane/TuSimple inputs are ~(1, 3, 288, 800)/(1, 3, 360, 640);
    # shrunk to a small multiple of the down_ratio=4 total stride (and the
    # DLA-34 backbone's own internal stride-32 total downsampling) for fast
    # tracing.
    return torch.randn(1, 3, 64, 128)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("LaneAF-DLA34", "build_laneaf_dla34", "example_input_laneaf_dla34", 2021, "vendored"),
]
