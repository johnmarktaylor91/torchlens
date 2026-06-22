"""Progressive Growing GAN (PGGAN) / Facebook pytorch_GAN_zoo StyleGAN beta.

PGGAN: Progressive Growing of GANs for Improved Quality, Stability, and Variation.
  Karras et al., ICLR 2018.  arXiv:1710.10196.
  Source: https://github.com/tkarras/progressive_growing_of_gans

pytorch_GAN_zoo (Facebook): A mixed bag of GAN implementations.
  Source: https://github.com/facebookresearch/pytorch_GAN_zoo
  The `stylegan_beta` in pytorch_GAN_zoo is an early PyTorch port of PGGAN with
  StyleGAN-like modulation added (pre-StyleGAN2 era); essentially PGGAN architecture
  with AdaIN-style conditioning (StyleGAN1 mapping network + AdaIN, no weight demod).

Distinctive primitives:
  - **Progressive growing**: training starts at low resolution (4x4) and progressively
    adds new layers (fading in) to reach full resolution. At inference, all layers
    are active. The graph shows the full final-resolution structure.
  - **Equalized learning rate**: weights are divided by sqrt(fan_in) at runtime
    (not at init). All nn.Linear/Conv weights are stored un-scaled; the scale factor
    is applied at each forward call. This is the PGGAN signature runtime op.
  - **Pixel norm**: normalises each spatial pixel's feature vector to unit length.
  - **Minibatch std**: appended as an extra feature map in the discriminator.
  - For the beta_pytorch_gan_zoo variant: additionally includes an **AdaIN
    modulation** (style mapping network + per-layer AdaIN, no weight demodulation --
    this is StyleGAN1 style, distinct from StyleGAN2's weight demodulation).

Compact: 32x32 output, base_ch=32, 3 progressive stages (all active at inference).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PGGAN building blocks
# ============================================================


class EqualizedLinear(nn.Module):
    """Linear with equalized learning rate (runtime weight scaling by 1/sqrt(fan_in))."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(in_dim)
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x * self.scale)


class EqualizedConv2d(nn.Module):
    """Conv2d with equalized learning rate."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0
    ) -> None:
        super().__init__()
        fan_in = in_ch * kernel_size * kernel_size
        self.scale = 1.0 / math.sqrt(fan_in)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x * self.scale)


class PixelNorm(nn.Module):
    """Per-pixel L2 normalisation (PGGAN generator feature normalisation)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return x / (x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())


class MinibatchStd(nn.Module):
    """Minibatch standard deviation appended as extra channel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=0, keepdim=True).mean(dim=[1, 2, 3], keepdim=True)
        std = std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std], dim=1)


# ============================================================
# PGGAN generator blocks (progressive stages)
# ============================================================


class PGGANGenBlock(nn.Module):
    """One progressive resolution stage of the PGGAN generator.

    Upsample -> equalized conv -> pixel norm -> equalized conv -> pixel norm.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)
        self.pnorm1 = PixelNorm()
        self.conv2 = EqualizedConv2d(out_ch, out_ch, 3, padding=1)
        self.pnorm2 = PixelNorm()
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.pnorm1(self.act(self.conv1(x)))
        x = self.pnorm2(self.act(self.conv2(x)))
        return x


# ============================================================
# PGGAN Generator
# ============================================================


class PGGANGenerator(nn.Module):
    """PGGAN generator (all stages active, compact 32x32 output).

    Progressive stages (all active at inference):
      - 4x4 base block (constant noise input)
      - 4x8: stage 0
      - 8x16: stage 1
      - 16x32: stage 2
    """

    def __init__(self, z_dim: int = 64, base_ch: int = 32) -> None:
        super().__init__()
        # 4x4 base block: z -> const 4x4 feature map
        self.fc = EqualizedLinear(z_dim, base_ch * 4 * 4 * 4)
        self.base_pnorm = PixelNorm()
        self.base_conv = EqualizedConv2d(base_ch * 4, base_ch * 4, 3, padding=1)
        self.base_pnorm2 = PixelNorm()
        # Progressive stages
        self.stage0 = PGGANGenBlock(base_ch * 4, base_ch * 2)  # 4->8
        self.stage1 = PGGANGenBlock(base_ch * 2, base_ch)  # 8->16
        self.stage2 = PGGANGenBlock(base_ch, base_ch)  # 16->32
        # to-RGB (1x1 equalized conv)
        self.to_rgb = EqualizedConv2d(base_ch, 3, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        x = self.base_pnorm(F.leaky_relu(self.fc(z), 0.2))
        x = x.view(B, -1, 4, 4)
        x = self.base_pnorm2(F.leaky_relu(self.base_conv(x), 0.2))
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return torch.tanh(self.to_rgb(x))


def build_stylegan_beta_pytorch_gan_zoo_generator() -> nn.Module:
    """PGGAN / pytorch_GAN_zoo stylegan_beta generator (equalized lr + pixel norm)."""
    return PGGANGenerator()


def example_input_stylegan_beta_pytorch_gan_zoo_generator() -> torch.Tensor:
    return torch.randn(1, 64)


# ============================================================
# PGGAN Discriminator
# ============================================================


class PGGANDiscBlock(nn.Module):
    """Downsampling block for the PGGAN discriminator."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = EqualizedConv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = EqualizedConv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        return self.act(self.conv2(x))


class PGGANDiscriminator(nn.Module):
    """PGGAN discriminator (compact 32x32 input)."""

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        self.from_rgb = EqualizedConv2d(3, base_ch, 1)
        self.block0 = PGGANDiscBlock(base_ch, base_ch * 2)  # 32->16
        self.block1 = PGGANDiscBlock(base_ch * 2, base_ch * 4)  # 16->8
        self.mbstd = MinibatchStd()
        self.conv_final = EqualizedConv2d(base_ch * 4 + 1, base_ch * 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = EqualizedLinear(base_ch * 4, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.from_rgb(img), 0.2)
        x = self.block0(x)
        x = self.block1(x)
        x = self.mbstd(x)
        x = F.leaky_relu(self.conv_final(x), 0.2)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_pggan_discriminator() -> nn.Module:
    return PGGANDiscriminator()


def example_input_pggan_discriminator() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "stylegan_beta_pytorch_gan_zoo_generator",
        "build_stylegan_beta_pytorch_gan_zoo_generator",
        "example_input_stylegan_beta_pytorch_gan_zoo_generator",
        "2018",
        "DC",
    ),
    (
        "pggan_discriminator",
        "build_pggan_discriminator",
        "example_input_pggan_discriminator",
        "2018",
        "DC",
    ),
]
