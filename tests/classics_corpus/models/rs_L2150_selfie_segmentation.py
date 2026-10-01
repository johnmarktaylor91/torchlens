# SOURCE: vendored from drumichiro/selfie-segmentation-pytorch @ master (d7450bbc)
# https://raw.githubusercontent.com/drumichiro/selfie-segmentation-pytorch/master/selfie_segmentation.py
"""MediaPipe Selfie Segmentation model (PyTorch re-implementation faithfully matching
Google's MediaPipe TFLite graph topology, layer-by-layer).

Vendored real nn.Module code with only import/formatting touch-ups (no
architectural changes). MobileNetV3-style depthwise-separable encoder-decoder
that produces a binary (person) segmentation mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfieBlock(nn.Module):
    def __init__(self, io_channels, exp_channels, mid_channels):
        super(SelfieBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=io_channels, out_channels=exp_channels, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=exp_channels,
            out_channels=exp_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=exp_channels,
        )
        self.sub_block = SelfieSubBlock(
            in_channels=exp_channels, out_channels=io_channels, mid_channels=mid_channels
        )

    def forward(self, x):
        h = F.hardswish(self.conv1(x))
        h = self.conv2(h)
        h = self.sub_block(h)
        return x + h


class SelfieSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(SelfieSubBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1
        )

    def forward(self, x):
        x = F.hardswish(x)
        h = torch.mean(x, dim=[2, 3], keepdim=True)
        h = F.relu(self.conv1(h))
        h = torch.sigmoid(self.conv2(h))
        return self.conv3(x * h)


class SelfieFinalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, interp_size):
        super(SelfieFinalBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            groups=out_channels,
            padding=1,
        )
        self.interp = nn.Upsample(size=interp_size, mode="bilinear", align_corners=False)

    def forward(self, x, h):
        x = self.interp(x)
        x = self.conv1(x)
        h1 = x + h
        h1 = torch.mean(h1, dim=[2, 3], keepdim=True)
        h1 = F.relu(self.conv2(h1))
        h1 = torch.sigmoid(self.conv3(h1))
        h1 = h * h1
        h1 = F.relu(self.conv4(x + h1))
        h2 = F.relu(self.conv5(h1))
        return h1 + h2


class SelfieSegmentation(nn.Module):
    def __init__(self, width, height):
        super(SelfieSegmentation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=2, groups=16
        )
        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.conv3_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)
        self.conv4_1 = nn.Conv2d(in_channels=16, out_channels=72, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(
            in_channels=72, out_channels=72, kernel_size=3, stride=2, groups=72
        )
        self.conv4_3 = nn.Conv2d(in_channels=72, out_channels=24, kernel_size=1, stride=1)
        self.conv5_1 = nn.Conv2d(in_channels=24, out_channels=88, kernel_size=1, stride=1)
        self.conv5_2 = nn.Conv2d(
            in_channels=88, out_channels=88, kernel_size=3, stride=1, padding=1, groups=88
        )
        self.conv5_3 = nn.Conv2d(in_channels=88, out_channels=24, kernel_size=1, stride=1)
        self.conv6_1 = nn.Conv2d(in_channels=24, out_channels=96, kernel_size=1, stride=1)
        self.conv6_2 = nn.Conv2d(
            in_channels=96, out_channels=96, kernel_size=5, stride=2, groups=96
        )
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1)

        self.selfie_sub_block1 = SelfieSubBlock(in_channels=96, out_channels=32, mid_channels=24)
        self.selfie_block1 = SelfieBlock(io_channels=32, exp_channels=128, mid_channels=32)
        self.selfie_block2 = SelfieBlock(io_channels=32, exp_channels=128, mid_channels=32)
        self.selfie_block3 = SelfieBlock(io_channels=32, exp_channels=96, mid_channels=24)
        self.selfie_block4 = SelfieBlock(io_channels=32, exp_channels=96, mid_channels=24)
        self.selfie_final_block1 = SelfieFinalBlock(
            in_channels=128, out_channels=24, interp_size=(int(height / 8), int(width / 8))
        )
        self.selfie_final_block2 = SelfieFinalBlock(
            in_channels=24, out_channels=16, interp_size=(int(height / 4), int(width / 4))
        )
        self.selfie_final_block3 = SelfieFinalBlock(
            in_channels=16, out_channels=16, interp_size=(int(height / 2), int(width / 2))
        )
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=16, out_channels=1, kernel_size=2, stride=2, bias=True
        )

    def forward(self, x):
        h1 = F.hardswish(self.conv1(F.pad(x, (0, 1, 0, 1))))
        ######################
        h2 = F.relu(self.conv2_1(h1))
        h2 = F.relu(self.conv2_2(F.pad(h2, (0, 1, 0, 1))))
        ######################
        h3 = torch.mean(h2, dim=[2, 3], keepdim=True)
        h3 = F.relu(self.conv3_1(h3))
        h3 = torch.sigmoid(self.conv3_2(h3))
        h3 = self.conv3_3(h2 * h3)
        ######################
        h4 = F.relu(self.conv4_1(h3))
        h4 = F.relu(self.conv4_2(F.pad(h4, (0, 1, 0, 1))))
        h4 = self.conv4_3(h4)
        ######################
        h5 = F.relu(self.conv5_1(h4))
        h5 = F.relu(self.conv5_2(h5))
        h5 = self.conv5_3(h5)
        h5 = h4 + h5
        ######################
        h6 = F.hardswish(self.conv6_1(h5))
        h6 = self.conv6_2(F.pad(h6, (1, 2, 1, 2)))
        h6 = self.selfie_sub_block1(h6)
        h6 = self.selfie_block1(h6)
        h6 = self.selfie_block2(h6)
        h6 = self.selfie_block3(h6)
        h6 = self.selfie_block4(h6)
        ######################
        h7 = F.relu(self.conv7(h6))
        h8 = torch.mean(h6, dim=[2, 3], keepdim=True)
        h8 = torch.sigmoid(self.conv8(h8))
        ######################
        fin1 = self.selfie_final_block1(h7 * h8, h5)
        fin2 = self.selfie_final_block2(fin1, h3)
        fin3 = self.selfie_final_block3(fin2, h1)
        return torch.sigmoid(self.conv_transpose(fin3))


def build_selfie_segmentation():
    return SelfieSegmentation(width=256, height=256)


def example_input_selfie_segmentation():
    return torch.randn(1, 3, 256, 256)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "MediaPipe Selfie Segmentation",
        "build_selfie_segmentation",
        "example_input_selfie_segmentation",
        2020,
        "vendored",
    ),
]
