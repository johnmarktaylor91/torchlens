# FAITHFUL REIMPLEMENTATION from Sanchez-Garcia et al. 2021, "DeepEMhancer: a
# deep learning solution for cryo-EM volume post-processing", Communications
# Biology 4:874 (https://doi.org/10.1038/s42003-021-02399-1); methods
# section, no public training/architecture code.
#
# The public repo (rsanchezgarc/deepEMhancer) ships ONLY inference tooling
# that loads a pretrained Keras `.hd5` checkpoint via
# `keras.models.load_model(...)` (deepEMhancer/utils/loadModel.py); the
# actual model-definition source (referenced internally as
# `devel_code.trainNet.defaultNet`) was never published in this or any other
# repo (verified via GitHub code/repo search: no `defaultNet`/
# `getCustomObjects`/DeepEMhancer-training hits outside this inference-only
# repo). This is therefore a rung-4 candidate: no code exists to vendor or
# port, but the paper's methods section gives a sufficiently detailed
# architecture description to reimplement faithfully:
#
#   "a 3D U-net consisting of three downsampling blocks and three
#    upsampling blocks that process cubic chunks of the input map"
#   -- each block: "three convolutional layers followed by group
#      normalization and PRelu activation"
#   -- filter counts per block: "3 x 32, 3 x 64, and 3 x 128, respectively"
#      (three conv layers per block, channel width 32/64/128 across the
#      three depth levels)
#   -- "Downsampling is carried out using strided convolution and
#      upsampling is performed via transposed convolution"
#   -- U-Net skip connections between matching encoder/decoder depths
#   -- inference operates on "64 x 64 x 64" voxel cubic chunks with
#      overlapping strides
#
# This reimplementation follows that description mechanism-for-mechanism:
# a symmetric 3D U-Net with 3 encoder stages (32/64/128 channels, 3 conv
# layers per stage, strided-conv downsampling) and 3 decoder stages
# (128/64/32 channels, 3 conv layers per stage, transposed-conv upsampling,
# concatenated skip connections from the matching encoder stage), every
# conv followed by GroupNorm + PReLU, ending in a 1x1x1 conv projecting to a
# single-channel post-processed density map.

import torch
import torch.nn as nn


def _group_norm(channels, max_groups=8):
    # GroupNorm requires num_groups to divide num_channels; fall back
    # gracefully for the small toy channel counts used at trace time.
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class ConvGNPReLU3D(nn.Module):
    """One "conv -> group norm -> PReLU" unit, the repeated primitive
    inside each of DeepEMhancer's six U-Net blocks."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = _group_norm(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class EncoderBlock3D(nn.Module):
    """Downsampling block: strided conv (downsample + project) followed by
    two more conv-GN-PReLU layers, three conv layers total per block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = ConvGNPReLU3D(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv2 = ConvGNPReLU3D(out_channels, out_channels, kernel_size=3, stride=1)
        self.conv3 = ConvGNPReLU3D(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.down(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DecoderBlock3D(nn.Module):
    """Upsampling block: transposed conv (upsample + project), concatenate
    the matching encoder skip connection, then two more conv-GN-PReLU
    layers, three conv layers total per block."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_norm = _group_norm(out_channels)
        self.up_act = nn.PReLU(out_channels)
        self.conv2 = ConvGNPReLU3D(
            out_channels + skip_channels, out_channels, kernel_size=3, stride=1
        )
        self.conv3 = ConvGNPReLU3D(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x, skip):
        x = self.up_act(self.up_norm(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DeepEMhancer(nn.Module):
    """Faithful reimplementation of the DeepEMhancer 3D U-Net: three
    downsampling blocks (32/64/128 channels) and three upsampling blocks
    (128/64/32 channels) with skip connections, operating on cubic cryo-EM
    map chunks and producing a single-channel post-processed density map.
    """

    def __init__(self, in_channels=1, base_channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = base_channels

        self.stem = ConvGNPReLU3D(in_channels, c1, kernel_size=3, stride=1)

        self.enc1 = EncoderBlock3D(c1, c1)
        self.enc2 = EncoderBlock3D(c1, c2)
        self.enc3 = EncoderBlock3D(c2, c3)

        self.dec3 = DecoderBlock3D(in_channels=c3, skip_channels=c2, out_channels=c2)
        self.dec2 = DecoderBlock3D(in_channels=c2, skip_channels=c1, out_channels=c1)
        self.dec1 = DecoderBlock3D(in_channels=c1, skip_channels=c1, out_channels=c1)

        self.head = nn.Conv3d(c1, 1, kernel_size=1)

    def forward(self, volume):
        # volume: [N, 1, D, H, W] cubic cryo-EM map chunk.
        s0 = self.stem(volume)  # skip for the final decoder stage
        s1 = self.enc1(s0)  # skip for dec2
        s2 = self.enc2(s1)  # skip for dec3
        s3 = self.enc3(s2)  # bottleneck

        d3 = self.dec3(s3, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)

        return self.head(d1)


# ---------------------------------------------------------------------------
# Menagerie staging entries
# ---------------------------------------------------------------------------


def build_deepemhancer():
    torch.manual_seed(0)
    model = DeepEMhancer(in_channels=1, base_channels=(4, 8, 16))
    model.eval()
    return model


def example_input_deepemhancer():
    torch.manual_seed(0)
    return (torch.randn(1, 1, 16, 16, 16),)


MENAGERIE_ZOO = "reimpl-pytorch"

MENAGERIE_ENTRIES = [
    ("DeepEMhancer", "build_deepemhancer", "example_input_deepemhancer", 2021, "reimpl-pytorch"),
]
