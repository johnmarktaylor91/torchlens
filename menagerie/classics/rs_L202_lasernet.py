# SOURCE: vendored from kareemalsawah/Modified_LaserNet_Pytorch @ master
#
# https://github.com/kareemalsawah/Modified_LaserNet_Pytorch
# https://raw.githubusercontent.com/kareemalsawah/Modified_LaserNet_Pytorch/master/models.py
#
# Community PyTorch implementation of LaserNet (Meyer et al. 2019, CVPR,
# "LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous
# Driving"); the original Uber ATG paper never released official code. This
# vendors the actual `BasicBlock` / `DownSample` / `Deconv` / `Feature_Extractor`
# / `Feature_Aggregator` / `Deep_Aggregation` / `AuxNet` / `LaserNet` classes
# from `models.py` verbatim (only whitespace-preserving copy, no architecture
# changes). The real `LaserNet` class (the deep-aggregation range-image CNN
# fusing an RGB auxiliary branch with a LiDAR range-image branch) is 100%
# pure torch. The file's top-level `import cupy as cp` and the
# `Radius_NMS` / `Bounding_Box_Detector` classes that follow `LaserNet` in the
# original file are LiDAR-postprocessing-only (radius-NMS box refinement +
# k-means, CUDA/cupy-only, `.cuda()`-hardcoded) and are NOT part of the
# LaserNet network itself, so they are dropped here rather than vendored,
# since `cupy` is not an installed base lib per the discovery ladder's rung-2
# rule (non-base deps -> needs_env, not silent install) and they are provably
# outside the traced network's forward pass.

import torch
import torch.nn.functional as F
from torch import nn

BatchNorm = nn.BatchNorm2d


class BasicBlock(nn.Module):
    """Basic Resnet Block: Conv2d-BN-ReLU-Conv2d-BN-ReLU with residual connection."""

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = BatchNorm(out_channels)
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
        self.bn2 = BatchNorm(out_channels)
        self.stride = stride
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.project is not None:
            residual = self.project(residual)
        out += residual
        out = self.relu(out)

        return out


class Deconv(nn.Module):
    """Deconvolution Layer for upsampling: TransposeConv2d-BN-ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Feature_Aggregator(nn.Module):
    """Feature Aggregator Module described in the LaserNet paper."""

    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super().__init__()
        self.deconv = Deconv(in_channels_2, out_channels)
        self.block_1 = BasicBlock(in_channels_1 + in_channels_2, out_channels)
        self.block_2 = BasicBlock(out_channels, out_channels)

    def forward(self, x1, x2):
        x2 = self.deconv(x2)
        x1 = torch.cat([x1, x2], 1)
        x1 = self.block_1(x1)
        x1 = self.block_2(x1)
        return x1


class DownSample(nn.Module):
    """DownSample module using Conv2d with stride > 1."""

    def __init__(self, in_channels, out_channels, stride=2, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = BatchNorm(out_channels)
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
        self.bn2 = BatchNorm(out_channels)
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.project(residual)
        out += residual
        out = self.relu(out)
        return out


class Feature_Extractor(nn.Module):
    """Feature Extractor module described in LaserNet paper. DownSamples input if not stage 1a."""

    def __init__(self, in_channels, out_channels, num_blocks=6, down_sample_input=False):
        super().__init__()
        self.down_sample = None
        self.down_sample_input = down_sample_input
        if down_sample_input:
            self.down_sample = DownSample(in_channels, out_channels)

        blocks_modules = []
        for i in range(num_blocks):
            if i == 0 and not down_sample_input:
                blocks_modules.append(BasicBlock(in_channels, out_channels))
            else:
                blocks_modules.append(BasicBlock(out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks_modules)

    def forward(self, x):
        if self.down_sample_input:
            x = self.down_sample(x)
        x = self.blocks(x)
        return x


class Deep_Aggregation(nn.Module):
    """Main Deep Aggregation class described in LaserNet paper.

    num_outputs is the number of channels of the output image; output image
    has the same width and height as input image.
    """

    def __init__(self, num_inputs, channels, num_outputs):
        super().__init__()
        self.extract_1a = Feature_Extractor(num_inputs, channels[0])
        self.extract_2a = Feature_Extractor(channels[0], channels[1], down_sample_input=True)
        self.extract_3a = Feature_Extractor(channels[1], channels[2], down_sample_input=True)
        self.aggregate_1b = Feature_Aggregator(channels[0], channels[1], channels[1])
        self.aggregate_1c = Feature_Aggregator(channels[1], channels[2], channels[2])
        self.aggregate_2b = Feature_Aggregator(channels[1], channels[2], channels[2])
        self.conv_1x1 = nn.Conv2d(channels[2], num_outputs, kernel_size=1, stride=1)

    def forward(self, x):
        x_1a = self.extract_1a(x)
        x_2a = self.extract_2a(x_1a)
        x_3a = self.extract_3a(x_2a)
        x_1b = self.aggregate_1b(x_1a, x_2a)
        x_2b = self.aggregate_2b(x_2a, x_3a)
        x_1c = self.aggregate_1c(x_1b, x_2b)
        out = self.conv_1x1(x_1c)
        return out


class ResUnit(nn.Module):
    def __init__(
        self, in_channels, channels_num, filter_size=3, dim_change=False, custom_stride=(2, 2)
    ):
        super().__init__()
        self.stride = 1
        if dim_change:
            self.stride = custom_stride
        self.conv1 = nn.Conv2d(
            in_channels, channels_num, filter_size, stride=self.stride, padding=1
        )
        self.conv2 = nn.Conv2d(channels_num, channels_num, filter_size, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, channels_num, dim_change=True, custom_stride=(2, 2)):
        super().__init__()
        self.in_channels = in_channels
        self.channels_num = channels_num
        self.dim_change = dim_change
        self.resUnit1 = ResUnit(
            self.in_channels,
            channels_num=self.channels_num,
            filter_size=3,
            dim_change=dim_change,
            custom_stride=custom_stride,
        )
        self.resUnit2 = ResUnit(
            self.channels_num, channels_num=self.channels_num, filter_size=3, dim_change=False
        )
        self.resUnit3 = ResUnit(
            self.channels_num, channels_num=self.channels_num, filter_size=3, dim_change=False
        )
        self.resUnit4 = ResUnit(
            self.channels_num, channels_num=self.channels_num, filter_size=3, dim_change=False
        )
        if self.dim_change:
            self.reshaping_conv = nn.Conv2d(
                self.in_channels, self.channels_num, 1, stride=custom_stride, padding=0
            )
        else:
            self.reshaping_conv = nn.Conv2d(
                self.in_channels, self.channels_num, 1, stride=1, padding=0
            )

    def forward(self, x):
        residue = self.reshaping_conv(x)
        x = self.resUnit1(x)
        x = x + residue
        residue = x
        x = self.resUnit2(x)
        x = x + residue
        residue = x
        x = self.resUnit3(x)
        x = x + residue
        residue = x
        x = self.resUnit4(x)
        x = x + residue
        return x


class AuxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resBlock1 = ResBlock(in_channels=3, channels_num=16, dim_change=True)
        self.resBlock2 = ResBlock(
            in_channels=16, channels_num=24, dim_change=True, custom_stride=(1, 2)
        )
        self.resBlock3 = ResBlock(
            in_channels=24, channels_num=32, dim_change=True, custom_stride=(1, 2)
        )

    def forward(self, x):
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        return x


class LaserNet(nn.Module):
    """LaserNet network as described in the original paper, built on the Deep
    Aggregation Network. The RGB/Lidar fusion mirrors the LaserNet++ paper."""

    def __init__(self, deep_aggregation_num_channels=[64, 128, 128], num_out_channels=4):
        super().__init__()
        self.RGB_CNN = AuxNet()
        self.DL = Deep_Aggregation(64, deep_aggregation_num_channels, num_out_channels)
        self.Lidar_CNN = nn.Conv2d(6, 32, kernel_size=3, padding=1)

    def forward(self, rgb_img, lidar):
        rgb_semantics = self.RGB_CNN.forward(rgb_img)
        lidar_semantics = self.Lidar_CNN(lidar)
        fused_semantics = torch.cat((rgb_semantics, lidar_semantics), dim=1)
        out = self.DL.forward(fused_semantics)
        return out


def build_lasernet():
    return LaserNet(deep_aggregation_num_channels=[8, 16, 16], num_out_channels=4)


def example_input_lasernet():
    # RGB branch (AuxNet): resBlock1 uses stride (2,2), resBlock2/3 use stride
    # (1,2) -> overall H downsamples by 2x, W downsamples by 8x.
    rgb_img = torch.randn(1, 3, 16, 64)
    # Lidar branch output spatial size must match RGB_CNN output (8 x 8) for concat.
    lidar = torch.randn(1, 6, 8, 8)
    return rgb_img, lidar


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("LaserNet", "build_lasernet", "example_input_lasernet", 2019, "vendored"),
]
