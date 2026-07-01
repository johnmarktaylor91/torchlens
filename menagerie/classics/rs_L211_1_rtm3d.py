# FAITHFUL PORT of Banconxuan/RTM3D @ master (original framework: PyTorch + custom CUDA DCNv2 extension)
#
# RTM3D (ECCV 2020): "RTM3D: Real-time Monocular 3D Detection from Object Keypoints
# for Autonomous Driving". Keypoint-based one-stage monocular 3D object detector.
#
# Source files transcribed:
#   - src/lib/models/networks/resnet_dcn.py (PoseResNet + DCN-based deconv decoder)
#   - src/lib/models/networks/DCNv2/dcn_v2.py (DCN offset/mask head; original relies on
#     a compiled `_ext` CUDA extension that cannot build in this base environment)
#   - src/lib/models/model.py (_model_factory['resdcn'] entry point + heads assembly)
#   - src/lib/opts.py (default heads dict for the `kitti` dataset: hm/wh/hps/rot/dim/prob
#     plus reg/hm_hp/hp_offset, all enabled by default since --not_reg_offset,
#     --not_hm_hp, --not_reg_hp_offset default to False)
#   - src/lib/datasets/dataset/kittihp.py (num_classes=3 for KITTI car/pedestrian/cyclist)
#
# The ONLY deviation from the original source: the custom CUDA `DCN` module (which calls
# a compiled `_ext.dcn_v2_forward` kernel via a hand-written autograd.Function) is replaced
# with torchvision.ops.deform_conv2d, which is the native, base-lib, modulated deformable
# convolution kernel -- mathematically the same operator (offset + sigmoid mask, applied via
# `torch.ops.torchvision.deform_conv2d`), just without needing a private compiled extension.
# The offset/mask computation (a plain nn.Conv2d producing 3*k*k*deformable_groups channels,
# split into offset_x/offset_y/mask, mask passed through sigmoid) is transcribed verbatim
# from `DCN.forward` in dcn_v2.py. All ResNet block code, deconv layer construction, and
# head assembly are transcribed verbatim from resnet_dcn.py / model.py.
#
# Official repo only ships `res_18`/`res_101` (no-DCN, msra_resnet.py), `resdcn_18`/
# `resdcn_101` (DCN deconv, resnet_dcn.py -- this file), `dla_34` (DCN deconv,
# pose_dla_dcn.py), and `hourglass`. We port the `resdcn_18` variant -- a real,
# officially-supported `_model_factory` entry (see model.py) -- since it is the
# architecturally-representative (ResNet+DCN-deconv+multi-head) member of the family
# without transcribing the much larger DLA-34 tree backbone.

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

BN_MOMENTUM = 0.1


# --------------------------------------------------------------------------------------
# ResNet backbone blocks (verbatim from resnet_dcn.py)
# --------------------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# --------------------------------------------------------------------------------------
# DCN: modulated deformable convolution (offset/mask head verbatim from dcn_v2.py;
# the compiled `_ext` kernel call is swapped for torchvision.ops.deform_conv2d, the
# native base-lib equivalent of the same modulated-deformable-conv operator).
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
# PoseResNet: ResNet-18/34/50/101/152 backbone + DCN deconv decoder + multi-task heads
# (verbatim from resnet_dcn.py, random init instead of downloading torchvision weights).
# --------------------------------------------------------------------------------------
class PoseResNet(nn.Module):
    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False

        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.deconv_layers = self._make_deconv_layer(3, [256, 128, 64], [4, 4, 4])

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True),
                )
                if "hm" in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes, kernel_size=1, stride=1, padding=0, bias=True)
                if "hm" in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters)
        assert num_layers == len(num_kernels)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(
                self.inplanes,
                planes,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                deformable_groups=1,
            )
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias,
            )
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}


def get_pose_net(num_layers, heads, head_conv=256):
    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
    return model


# --------------------------------------------------------------------------------------
# Staging entry points
# --------------------------------------------------------------------------------------
# Default KITTI heads dict from opts.update_dataset_info_and_set_heads (num_classes=3
# for KITTI car/pedestrian/cyclist; reg/hm_hp/hp_offset all enabled by default since
# --not_reg_offset / --not_hm_hp / --not_reg_hp_offset default to False in opts.py).
_KITTI_HEADS = {
    "hm": 3,
    "wh": 2,
    "hps": 18,
    "rot": 8,
    "dim": 3,
    "prob": 1,
    "reg": 2,
    "hm_hp": 9,
    "hp_offset": 2,
}


def build_rtm3d():
    # resdcn_18: real, officially-supported _model_factory['resdcn'] arch with
    # num_layers=18 (see model.py: create_model('resdcn_18', ...)). head_conv=64
    # matches opts.py's documented default for resnets ("-1 for default setting:
    # 64 for resnets and 256 for dla").
    return get_pose_net(num_layers=18, heads=_KITTI_HEADS, head_conv=64)


def example_input_rtm3d():
    # KITTI default_resolution = [384, 1280] (kittihp.py); down_ratio=4 -> output 96x320.
    return torch.randn(1, 3, 384, 1280)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("RTM3D", "build_rtm3d", "example_input_rtm3d", 2020, "ported"),
]
