# SOURCE: vendored from msracver/Deep-Exemplar-based-Colorization @ master
# File: colorization_subnet/models/ExampleColorNet.py
# https://github.com/msracver/Deep-Exemplar-based-Colorization
#
# Minimal changes from the original source:
#   - None to the architecture. Only added a tiny build_/example_input_
#     harness at the bottom to construct an instance and drive the real
#     forward pass. `F.tanh` (deprecated alias) kept as in the original.
#
# Architecture (unmodified from source): ExampleColorNet is the
# "colorization subnet" of Deep Exemplar-based Colorization (SIGGRAPH 2018),
# a U-Net-like fully convolutional network with dilated convolutions in the
# bottleneck (conv5/conv6, matching the Zhang et al. "Colorful Image
# Colorization" backbone) and three long skip connections
# (conv3_3->conv8_1, conv2_2->conv9_1, conv1_2->conv10_1) that fuse
# encoder features into the decoder before predicting the two ab chroma
# channels. In the full pipeline this subnet is driven by per-pixel
# similarity maps computed by a separate Caffe-based "similarity subnet"
# (deep image analogy correspondence network); test.py in the original
# repo constructs it with ic=13 (1 luminance channel + 2 warped_ba ab
# channels + 4 err_ba + 4 warped_aba + 2 err_ab), reproduced here
# verbatim.

import torch.nn as nn
import torch.nn.functional as F


class ExampleColorNet(nn.Module):
    def __init__(self, ic):
        super(ExampleColorNet, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(ic, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1)
        )
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_2norm = nn.BatchNorm2d(64, affine=False)
        self.conv1_2norm_ss = nn.Conv2d(64, 64, 1, 2, bias=False, groups=64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv2_2norm_ss = nn.Conv2d(128, 128, 1, 2, bias=False, groups=128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv3_3norm_ss = nn.Conv2d(256, 256, 1, 2, bias=False, groups=256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv6_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv7_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv8_1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv3_3_short = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv9_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv2_2_short = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv10_1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.conv1_2_short = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv10_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv10_ab = nn.Conv2d(128, 2, 1, 1)

        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        conv1_1 = F.relu(self.conv1_1(x))
        conv1_2 = F.relu(self.conv1_2(conv1_1))
        conv1_2norm = self.conv1_2norm(conv1_2)
        conv1_2norm_ss = self.conv1_2norm_ss(conv1_2norm)
        conv2_1 = F.relu(self.conv2_1(conv1_2norm_ss))
        conv2_2 = F.relu(self.conv2_2(conv2_1))
        conv2_2norm = self.conv2_2norm(conv2_2)
        conv2_2norm_ss = self.conv2_2norm_ss(conv2_2norm)
        conv3_1 = F.relu(self.conv3_1(conv2_2norm_ss))
        conv3_2 = F.relu(self.conv3_2(conv3_1))
        conv3_3 = F.relu(self.conv3_3(conv3_2))
        conv3_3norm = self.conv3_3norm(conv3_3)
        conv3_3norm_ss = self.conv3_3norm_ss(conv3_3norm)
        conv4_1 = F.relu(self.conv4_1(conv3_3norm_ss))
        conv4_2 = F.relu(self.conv4_2(conv4_1))
        conv4_3 = F.relu(self.conv4_3(conv4_2))
        conv4_3norm = self.conv4_3norm(conv4_3)
        conv5_1 = F.relu(self.conv5_1(conv4_3norm))
        conv5_2 = F.relu(self.conv5_2(conv5_1))
        conv5_3 = F.relu(self.conv5_3(conv5_2))
        conv5_3norm = self.conv5_3norm(conv5_3)
        conv6_1 = F.relu(self.conv6_1(conv5_3norm))
        conv6_2 = F.relu(self.conv6_2(conv6_1))
        conv6_3 = F.relu(self.conv6_3(conv6_2))
        conv6_3norm = self.conv6_3norm(conv6_3)
        conv7_1 = F.relu(self.conv7_1(conv6_3norm))
        conv7_2 = F.relu(self.conv7_2(conv7_1))
        conv7_3 = F.relu(self.conv7_3(conv7_2))
        conv7_3norm = self.conv7_3norm(conv7_3)
        conv8_1 = self.conv8_1(conv7_3norm)
        conv3_3_short = self.conv3_3_short(conv3_3norm)

        conv8_1_comb = F.relu(conv8_1 + conv3_3_short)
        conv8_2 = F.relu(self.conv8_2(conv8_1_comb))
        conv8_3 = F.relu(self.conv8_3(conv8_2))
        conv8_3norm = self.conv8_3norm(conv8_3)
        conv9_1 = self.conv9_1(conv8_3norm)
        conv2_2_short = self.conv2_2_short(conv2_2norm)

        conv9_1_comb = F.relu(conv9_1 + conv2_2_short)
        conv9_2 = F.relu(self.conv9_2(conv9_1_comb))
        conv9_2norm = self.conv9_2norm(conv9_2)
        conv10_1 = self.conv10_1(conv9_2norm)
        conv1_2_short = self.conv1_2_short(conv1_2norm)

        conv10_1_comb = F.relu(conv10_1 + conv1_2_short)
        conv10_2 = self.leaky_relu(self.conv10_2(conv10_1_comb))
        conv10_ab = self.conv10_ab(conv10_2)
        pred_ab = F.tanh(conv10_ab) * 100

        return pred_ab


def build_deep_exemplar_colorization():
    return ExampleColorNet(ic=13)


def example_input_deep_exemplar_colorization():
    import torch

    return torch.randn(1, 13, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Deep Exemplar-based Colorization",
        "build_deep_exemplar_colorization",
        "example_input_deep_exemplar_colorization",
        2018,
        "vendored",
    ),
]
