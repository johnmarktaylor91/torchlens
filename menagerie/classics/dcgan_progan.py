"""DCGAN and ProGAN generators from Facebook Research pytorch_GAN_zoo.

DCGAN: Radford, Metz, Chintala 2015, arXiv:1511.06434.
ProGAN: Karras, Laine, Aila (NVIDIA) 2018, arXiv:1710.10196.
Source repo: https://github.com/facebookresearch/pytorch_GAN_zoo

DCGAN distinctive primitive: transposed-conv upsampling stack with BatchNorm
and ReLU activations, Tanh at output -- the baseline deep convolutional generator.

ProGAN distinctive primitives:
  - Progressive-growing conv blocks (upsample-then-conv pairs at each resolution);
  - Pixel-norm (normalize each pixel's feature vector to unit length) replacing BN;
  - Equalized learning rate (LR-equalized linear/conv via runtime weight scaling by
    sqrt(fan_in));
  - Per-level toRGB 1x1 conv that converts features to a 3-channel image;
  - Smooth alpha-blend during progressive growing (not reproduced here as it is
    training-phase-only; the static inference path is the pure stacked block form).

Both models: random init, CPU, small channels/spatial for compact tracing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DCGAN generator
# ---------------------------------------------------------------------------


class DCGANGenerator(nn.Module):
    """DCGAN generator: project noise z, then upsample with transposed convolutions."""

    def __init__(
        self,
        nz: int = 32,
        ngf: int = 16,
        nc: int = 3,
    ) -> None:
        super().__init__()
        # z -> (ngf*8) x 4 x 4
        self.project = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )
        # upsample blocks: 4->8->16->32
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )
        self.out = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = self.up1(x)
        x = self.up2(x)
        return self.out(x)


def build_fb_pytorch_gan_zoo_dcgan() -> nn.Module:
    return DCGANGenerator(nz=32, ngf=16, nc=3)


def example_input_dcgan() -> torch.Tensor:
    return torch.randn(1, 32, 1, 1)


# ---------------------------------------------------------------------------
# ProGAN generator
# ---------------------------------------------------------------------------


class PixelNorm(nn.Module):
    """Pixel-norm: normalize each spatial position's feature vector to unit length."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(1, keepdim=True).add(1e-8).sqrt())


class EqConv2d(nn.Module):
    """Conv2d with equalized learning rate (weight scaled by sqrt(fan_in) at runtime)."""

    def __init__(self, in_c: int, out_c: int, k: int, stride: int = 1, pad: int = 0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_c, in_c, k, k))
        self.bias = nn.Parameter(torch.zeros(out_c))
        fan_in = in_c * k * k
        self.scale = math.sqrt(2.0 / fan_in)
        self.stride = stride
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.pad)


class ProGANBlock(nn.Module):
    """One ProGAN generator resolution block: upsample + 2x EqConv + PixelNorm + LReLU."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = EqConv2d(in_c, out_c, 3, pad=1)
        self.conv2 = EqConv2d(out_c, out_c, 3, pad=1)
        self.pnorm = PixelNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = F.leaky_relu(self.pnorm(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.pnorm(self.conv2(x)), 0.2)
        return x


class ToRGB(nn.Module):
    """1x1 conv that maps feature channels to a 3-channel RGB image."""

    def __init__(self, in_c: int) -> None:
        super().__init__()
        self.conv = EqConv2d(in_c, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ProGANGenerator(nn.Module):
    """ProGAN generator (static inference path, 4x4->32x32, small channels).

    The full model grows from 4x4 to 1024x1024 progressively during training.
    Here we reproduce the final stacked form (post-growing) with compact dims.
    """

    def __init__(self, nz: int = 32, nf: int = 16) -> None:
        super().__init__()
        # Initial block: const 4x4 feature map from z
        self.initial_conv = nn.Sequential(
            EqConv2d(nz, nf * 8, 4, pad=3),  # 1x1 -> 4x4 via pad
            nn.LeakyReLU(0.2),
            PixelNorm(),
            EqConv2d(nf * 8, nf * 8, 3, pad=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )
        # Progressive blocks: 4->8->16->32
        self.block1 = ProGANBlock(nf * 8, nf * 4)  # 4->8
        self.block2 = ProGANBlock(nf * 4, nf * 2)  # 8->16
        self.block3 = ProGANBlock(nf * 2, nf)  # 16->32
        self.to_rgb = ToRGB(nf)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, nz, 1, 1)
        x = self.initial_conv(z)  # (B, nf*8, 4, 4)
        x = self.block1(x)  # (B, nf*4, 8, 8)
        x = self.block2(x)  # (B, nf*2, 16, 16)
        x = self.block3(x)  # (B, nf, 32, 32)
        return torch.tanh(self.to_rgb(x))


def build_fb_pytorch_gan_zoo_progan() -> nn.Module:
    return ProGANGenerator(nz=32, nf=16)


def example_input_progan() -> torch.Tensor:
    return torch.randn(1, 32, 1, 1)


MENAGERIE_ENTRIES = [
    (
        "DCGAN Generator (deep convolutional GAN transposed-conv upsampling stack)",
        "build_fb_pytorch_gan_zoo_dcgan",
        "example_input_dcgan",
        "2015",
        "DC",
    ),
    (
        "ProGAN Generator (progressive-growing GAN with pixel-norm and equalized LR)",
        "build_fb_pytorch_gan_zoo_progan",
        "example_input_progan",
        "2018",
        "DC",
    ),
]
