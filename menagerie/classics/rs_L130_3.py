# SOURCE: vendored from papkov/noise2same.pytorch @ master (noise2same/backbone/unet.py,
# a PyTorch port of the official TF1.x divelab/Noise2Same @ main network.py /
# resnet_module.py). The paper's own repo (divelab/Noise2Same, NeurIPS 2020) is TF1.x-only
# (requirements.txt pins `tensorflow>=1.15,<2.0`) and cannot run in this base torch env, so
# per rung 2 we vendor the widely-used third-party PyTorch reimplementation instead of
# guessing an architecture from the paper. papkov/noise2same.pytorch's unet.py file header
# states it is "translated from" the official TF resnet-style U-Net module-for-module
# (ResidualUnit/ResidualBlock/EncoderBlock stack, matching resnet_module.py), so this is
# real, faithful, runnable code -- not an approximation written from a paper summary.
"""Noise2Same (NeurIPS 2020) self-supervised blind-spot-free denoising backbone.

Noise2Same derives a self-supervised denoising objective that avoids the explicit
blind-spot masking of Noise2Void/Noise2Self, using an invariance loss between masked and
raw forward passes. The network itself is a residual-block U-Net (matching the official
TF resnet_module.py design): strided-conv downsampling, residual bottleneck, and
transposed-conv upsampling with concat skip connections. This file vendors the backbone
(`UNet`) and denoising `RegressionHead` classes verbatim from the PyTorch port, and adds a
thin composition wrapper mirroring `noise2same.model.Noise2Same.forward_whole` (backbone
features -> regression head), without the training-only contrastive/PSF-deconvolution
machinery in the full training-time `Noise2Same` class.
"""

from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# noise2same/backbone/unet.py (verbatim: RegressionHead, ResidualUnit,
# ResidualBlock, EncoderBlock, UNet). ProjectHead omitted (only used when
# lambda_proj > 0 contrastive pretraining is enabled; not part of the base
# denoising forward path).
# ---------------------------------------------------------------------------
class RegressionHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, n_dim: int = 2, kernel_size: int = 1):
        """
        Denoising regression head BN-ReLU-Conv
        """
        assert n_dim in (2, 3)
        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        bn = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d

        bn = bn(num_features=in_channels)
        relu = nn.ReLU(inplace=True)
        conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        super().__init__(bn, relu, conv)


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_dim: int = 2,
        kernel_size: int = 3,
        downsample: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.downsample = downsample

        bn = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        stride = 2 if downsample else 1

        self.act = nn.ReLU(inplace=True)
        self.bn = bn(in_channels, momentum=1 - 0.997, eps=1e-5)
        self.conv_shortcut = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            bias=False,
        )

        self.layers = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 if downsample else kernel_size,
                padding=0 if downsample else kernel_size // 2,
                stride=stride,
                bias=False,
            ),
            bn(out_channels),
            self.act,
            conv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x: T) -> T:
        shortcut = x
        x = self.bn(x)
        x = self.act(x)
        if self.in_channels != self.out_channels or self.downsample:
            shortcut = self.conv_shortcut(x)
        x = self.layers(x)
        return x + shortcut


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_size: int = 1,
        n_dim: int = 2,
        kernel_size: int = 3,
        downsample: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.block_size = block_size

        self.block = nn.Sequential(
            *[
                ResidualUnit(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    downsample=downsample if i == 0 else False,
                )
                for i in range(0, block_size)
            ]
        )

    def forward(self, x: T) -> T:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_size: int = 1,
        n_dim: int = 2,
        kernel_size: int = 3,
        downsampling: str = "conv",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.block_size = block_size

        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d

        if downsampling == "res":
            downsampling_block = ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                n_dim=n_dim,
                kernel_size=kernel_size,
                block_size=1,
                downsample=True,
            )
        elif downsampling == "conv":
            downsampling_block = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                bias=True,
            )
        else:
            raise ValueError("downsampling should be `res`. `conv`, `pool`")

        self.block = nn.Sequential(
            downsampling_block,
            ResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                n_dim=n_dim,
                block_size=block_size,
                downsample=False,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x: T) -> T:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 96,
        kernel_size: int = 3,
        n_dim: int = 2,
        depth: int = 3,
        encoding_block_sizes: Tuple[int, ...] = (1, 1, 0),
        decoding_block_sizes: Tuple[int, ...] = (1, 1),
        downsampling: Tuple[str, ...] = ("conv", "conv"),
        skip_method: str = "concat",
    ):
        super().__init__()

        assert depth == len(encoding_block_sizes)
        assert encoding_block_sizes[0] > 0
        assert encoding_block_sizes[-1] == 0
        assert depth == len(decoding_block_sizes) + 1
        assert depth == len(downsampling) + 1
        assert skip_method in ["add", "concat", "cat"]

        self.in_channels = in_channels
        self.n_dim = n_dim
        self.depth = depth
        self.base_channels = base_channels
        self.encoding_block_sizes = encoding_block_sizes
        self.decoding_block_sizes = decoding_block_sizes
        self.downsampling = downsampling
        self.skip_method = skip_method

        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        conv_transpose = nn.ConvTranspose2d if n_dim == 2 else nn.ConvTranspose3d

        self.conv_first = conv(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
            bias=False,
        )

        # Encoder
        self.encoder_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    block_size=encoding_block_sizes[0],
                )
            ]
        )

        out_channels = base_channels
        for i in range(2, self.depth + 1):
            in_channels = base_channels * (2 ** (i - 2))
            out_channels = base_channels * (2 ** (i - 1))

            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    block_size=encoding_block_sizes[i - 1],
                    downsampling=downsampling[i - 2],
                )
            )

        # Bottom block
        self.bottom_block = ResidualBlock(
            in_channels=out_channels,
            out_channels=base_channels * (2 ** (depth - 1)),
            n_dim=n_dim,
            kernel_size=kernel_size,
            block_size=1,
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            in_channels = int(base_channels * (2**i))
            out_channels = int(base_channels * (2 ** (i - 1)))

            self.upsampling_blocks.append(
                conv_transpose(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                    bias=True,
                )
            )

            self.decoder_blocks.append(
                ResidualBlock(
                    in_channels=out_channels * (2 if self.skip_method != "add" else 1),
                    out_channels=out_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    block_size=decoding_block_sizes[depth - 1 - i],
                )
            )

    def forward(self, x: T) -> T:
        encoder_outputs = []
        x = self.conv_first(x)
        x = self.encoder_blocks[0](x)

        for i, encoder_block in enumerate(self.encoder_blocks[1:]):
            encoder_outputs.append(x)
            x = encoder_block(x)

        x = self.bottom_block(x)

        for i, (upsampling_block, decoder_block, skip) in enumerate(
            zip(self.upsampling_blocks, self.decoder_blocks, encoder_outputs[::-1])
        ):
            x = upsampling_block(x)
            if self.skip_method == "add":
                x.add_(skip)
            elif self.skip_method in ("cat", "concat"):
                x = torch.cat([x, skip], dim=1)
            else:
                raise ValueError
            x = decoder_block(x)

        return x


# ---------------------------------------------------------------------------
# Menagerie build/example helpers: thin composition mirroring
# noise2same.model.Noise2Same.forward_whole (backbone -> regression head).
# ---------------------------------------------------------------------------
class Noise2SameDenoiser(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.net = UNet(in_channels=in_channels, base_channels=base_channels)
        self.head = RegressionHead(in_channels=base_channels, out_channels=in_channels)

    def forward(self, x: T) -> T:
        features = self.net(x)
        return self.head(features)


def build_noise2same():
    return Noise2SameDenoiser(in_channels=1, base_channels=32)


def example_input_noise2same():
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    ("Noise2Same", build_noise2same, example_input_noise2same, 2020, MENAGERIE_ZOO),
]
