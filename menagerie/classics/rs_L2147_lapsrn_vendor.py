# SOURCE: vendored from twtygqyy/pytorch-LapSRN @ master
# https://raw.githubusercontent.com/twtygqyy/pytorch-LapSRN/master/lapsrn.py
#
# Lai, Huang, Ahuja, Yang 2017 "Deep Laplacian Pyramid Networks for Fast and Accurate
# Super-Resolution" (CVPR 2017). Progressive Laplacian-pyramid super-resolution network:
# a shared 10-conv "feature embedding" block (_Conv_Block) run recursively at each
# pyramid level, each level producing a residual image added to a transposed-conv
# upsample of the previous level's HR estimate (2x -> 4x cascaded reconstruction).
# Net and _Conv_Block are copied verbatim from lapsrn.py; only the bilinear-upsample-
# filter initialization helper (get_upsample_filter, numpy-based, weight-init only,
# not part of forward) is kept for fidelity though it doesn't affect graph structure.
"""LapSRN: Laplacian Pyramid Super-Resolution Network (progressive 2x/4x upsampling)."""

import math

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from lapsrn.py ---
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()

        self.cov_block = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.cov_block(x)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.convt_I1 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.convt_R1 = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.convt_F1 = self.make_layer(_Conv_Block)

        self.convt_I2 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.convt_R2 = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.convt_F2 = self.make_layer(_Conv_Block)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        convt_F2 = self.convt_F2(convt_F1)
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)
        HR_4x = convt_I2 + convt_R2

        return HR_2x, HR_4x


def build_lapsrn():
    return Net()


def example_input_lapsrn():
    torch.manual_seed(0)
    return (torch.randn(1, 1, 32, 32),)


MENAGERIE_ENTRIES = [
    ("LapSRN", "build_lapsrn", "example_input_lapsrn", 2017, "vendored"),
]
