# FAITHFUL PORT of coldrainyht/caffe_dr2 @ master (original framework: Caffe)
# https://github.com/coldrainyht/caffe_dr2
# Deploy spec transcribed from:
#   DR2/test/deploy_prototxt_files/reconnet_0_25.prototxt
#
# DR2-Net (Deep Residual Reconstruction Network for image compressive sensing, Yao,
# Dai, Zhang, Ma, Zhang, Zhang, Tian, arXiv:1702.05743 / Neurocomputing 2019). Given a
# compressively-sensed measurement vector for a 33x33 image patch, a linear-mapping
# fully-connected layer ("fc1") produces a preliminary 33x33 reconstruction, which is
# then refined by 4 stacked residual blocks (the official repo's ReconNet-style
# "reconnet_0_XX.prototxt" naming refers to the measurement-rate-specific fc1 input
# size, not a separate architecture). No PyTorch/portable-framework code exists
# anywhere for DR2-Net (only the official Caffe repo and a mirror `AtenaKid/Caffe-DCS`,
# both Caffe/C++); the Caffe .prototxt IS the literal, unambiguous layer-by-layer
# architecture spec, so it is transcribed here faithfully rather than reimplemented
# from the paper's prose description.
#
# Each residual block (conv1_r..conv3_r / conv4_r..conv6_r / conv7_r..conv9_r /
# conv10_r..conv12_r in the prototxt) is: Conv2d(11x11, pad=5, no bias) -> BatchNorm2d
# -> (affine) Scale -> ReLU -> Conv2d(1x1, 32 out, no bias) -> BatchNorm2d -> Scale ->
# ReLU -> Conv2d(7x7, pad=3, 1 out, no bias) -> ReLU -> Eltwise-sum with the block's
# input ("res1".."res4" in the prototxt). Caffe's BatchNorm + separate affine Scale
# layer pair is exactly `nn.BatchNorm2d(affine=True)` in PyTorch (BatchNorm2d fuses
# the running-stats normalization and the learned affine scale/shift into one module),
# so each `BatchNorm -> Scale` pair collapses to a single `nn.BatchNorm2d` here with no
# behavioral change.

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    One DR2-Net residual block, i.e. one of the four repeated conv*_r..conv*_r+2
    groups in the deploy prototxt (num_output 64 -> 32 -> 1, kernel_size 11 -> 1 -> 7,
    pad 5 -> 0 -> 3, all bias_term: false), followed by an elementwise ("Eltwise") sum
    with the block's own input.
    """

    def __init__(self, channels=1):
        super().__init__()
        self.conv_a = nn.Conv2d(channels, 64, kernel_size=11, padding=5, bias=False)
        self.bn_a = nn.BatchNorm2d(64)
        self.relu_a = nn.ReLU(inplace=True)

        self.conv_b = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)
        self.bn_b = nn.BatchNorm2d(32)
        self.relu_b = nn.ReLU(inplace=True)

        self.conv_c = nn.Conv2d(32, channels, kernel_size=7, padding=3, bias=False)
        self.relu_c = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu_a(self.bn_a(self.conv_a(x)))
        out = self.relu_b(self.bn_b(self.conv_b(out)))
        out = self.relu_c(self.conv_c(out))
        return out + x


class DR2Net(nn.Module):
    """
    DR2-Net: linear-mapping fc1 (measurement vector -> 33x33 preliminary
    reconstruction) followed by `num_blocks` stacked residual blocks (the deploy
    prototxt for the 0.25 measurement-rate variant uses 4 blocks / measurement
    dimension 272; other measurement-rate prototxts in the repo only change the fc1
    input dimension, not the residual-block architecture).
    """

    def __init__(self, measurement_dim=272, patch_size=33, num_blocks=4):
        super().__init__()
        self.patch_size = patch_size
        self.fc1 = nn.Linear(measurement_dim, patch_size * patch_size)
        self.blocks = nn.ModuleList([ResidualBlock(channels=1) for _ in range(num_blocks)])

    def forward(self, measurements):
        # measurements: [N, measurement_dim]
        x = self.fc1(measurements)
        x = x.view(-1, 1, self.patch_size, self.patch_size)
        for block in self.blocks:
            x = block(x)
        return x


MENAGERIE_ZOO = "ported-pytorch"


def build_dr2net():
    # Repo default (reconnet_0_25.prototxt, MR=0.25): measurement_dim=272,
    # patch_size=33, num_blocks=4. Kept at repo scale since the network is already
    # small (a handful of 33x33 convs); only the measurement_dim / batch size need not
    # be shrunk further for a fast trace.
    model = DR2Net(measurement_dim=272, patch_size=33, num_blocks=4)
    model.eval()
    return model


def example_input_dr2net():
    return torch.randn(2, 272)


MENAGERIE_ENTRIES = [
    ("DR2-Net", "build_dr2net", "example_input_dr2net", 2017, MENAGERIE_ZOO),
]
