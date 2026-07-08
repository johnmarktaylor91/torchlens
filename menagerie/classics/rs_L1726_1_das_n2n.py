# SOURCE: vendored from https://github.com/sachalapins/DAS-N2N-torch @ main
# (dasn2n/model.py :: DASN2N)
#
# DAS-N2N (Lapins et al., 2024, GJI, "DAS-N2N: machine learning-based denoising
# of distributed acoustic sensing data without clean labels"): a self-supervised
# Noise2Noise denoiser for Distributed Acoustic Sensing (DAS) seismic data. A
# small fully-convolutional 2D encoder-decoder (skip-connected U-Net-style,
# single down/up level) maps a noisy (time x channel) DAS patch to a denoised
# patch; trained Noise2Noise-style on independently-noisy repeat recordings of
# the same signal (no clean reference required).
#
# The original repo (sachalapins/DAS-N2N) ships only a TensorFlow SavedModel
# (no Python source, weights-only). The author's companion repo
# sachalapins/DAS-N2N-torch is a from-scratch PyTorch re-implementation of the
# same published architecture (with matching pretrained weights) -- that repo's
# dasn2n/model.py::DASN2N class is vendored verbatim below (only the
# `denoise_numpy` data-processing convenience method -- unused for tracing --
# and the package-relative weights-loading helper were left in place;
# architecture layers/forward are untouched).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class DASN2N(nn.Module):
    def __init__(self):
        super().__init__()

        # Define DAS-N2N model layers
        self.INPUT_SHAPE = [
            128,
            96,
        ]  # Hard code input shape from paper: 128 time samples, 96 DAS channels
        self.conv00 = nn.Conv2d(
            in_channels=1, out_channels=24, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.conv10 = nn.Conv2d(
            in_channels=24, out_channels=24, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.conv01a = nn.Conv2d(
            in_channels=48, out_channels=48, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.conv01b = nn.Conv2d(
            in_channels=48, out_channels=48, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.out01 = nn.Conv2d(
            in_channels=48, out_channels=1, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )

        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.down = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, inp):
        # Forward pass of DAS-N2N model
        x = torch.unsqueeze(inp, dim=1)  # Expand dims

        # Encoder layer
        x = self.act(self.conv00(x))
        enc_skip = x

        # Middle layer
        x = self.act(self.conv10(self.down(x)))

        # Decoder layer
        x = torch.cat((enc_skip, self.up(x)), -3)
        x = self.act(self.conv01a(x))
        x = self.act(self.conv01b(x))
        x = self.out01(x)

        return torch.reshape(x, (inp.shape))


def build_das_n2n():
    return DASN2N()


def example_input_das_n2n():
    # Real repo's fixed patch shape: 128 time samples x 96 DAS channels.
    return torch.randn(2, 128, 96)


MENAGERIE_ENTRIES = [
    (
        "DAS-N2N",
        build_das_n2n,
        example_input_das_n2n,
        2024,
        MENAGERIE_ZOO,
    ),
]
