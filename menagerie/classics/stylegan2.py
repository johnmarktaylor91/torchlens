"""StyleGAN2 (+ ADA): Analyzing and Improving the Image Quality of StyleGAN.

Karras et al., CVPR 2020.  arXiv:1912.04958.
ADA: Training Generative Adversarial Networks with Limited Data.
  Karras et al., NeurIPS 2020.  arXiv:2006.06676.
Source: https://github.com/NVlabs/stylegan2-ada-pytorch

Distinctive primitives:
  - Mapping network: z -> w (8-layer MLP with LeakyReLU, pixel norm)
  - Weight-demodulated modulated convolution: weight_ij *= style_i / sqrt(sum_i style_i^2 * w_ij^2)
    replaces AdaIN from StyleGAN1. Removes "water-droplet" artifacts.
  - Per-pixel noise injection: output += broadcast noise * learned amplitude.
  - Skip/residual architecture with to-RGB projections summed at each scale.
  - Discriminator: residual downsampling blocks with minibatch std, final linear.

ADA (Adaptive Discriminator Augmentation) is a TRAINING-TIME augmentation pipeline
that adjusts augmentation probability dynamically. It does not change the architecture --
stylegan2_ada_generator / stylegan2_ada_pytorch_generator / stylegan2_ada_discriminator /
stylegan2_ada_pytorch_discriminator all share this same architecture.

StyleGAN-Human (human body generation) uses exactly this architecture with a taller
aspect ratio (512x256 or similar). Provided as a separate builder showing a non-square
aspect-ratio variant.

Compact: 32x32 images, z_dim=64, w_dim=64, base_ch=32, 2 synthesis blocks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared building blocks
# ============================================================


class PixelNorm(nn.Module):
    """Per-pixel L2 normalisation used in the mapping network."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())


class MappingNetwork(nn.Module):
    """z -> w mapping MLP with LeakyReLU and optional pixel norm."""

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        n_layers: int = 4,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.pixel_norm = PixelNorm()
        in_dim = z_dim + cond_dim
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        z = self.pixel_norm(z)
        if c is not None:
            z = torch.cat([z, c], dim=1)
        return self.net(z)


class ModulatedConv2d(nn.Module):
    """Weight-demodulated modulated convolution (StyleGAN2 signature op).

    Per sample:  w'_ijk = w_ijk * s_i  (modulate by style scale)
    Demod:       w''_ijk = w'_ijk / sqrt(sum_{i,k} w'^2_ijk + eps)
    Then standard conv2d with w''.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        w_dim: int,
        kernel_size: int = 3,
        upsample: bool = False,
        demodulate: bool = True,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.demodulate = demodulate
        # Affine: w vector -> per-channel scale
        self.affine = nn.Linear(w_dim, in_ch, bias=True)
        nn.init.ones_(self.affine.bias)
        # Convolutional kernel
        self.weight = nn.Parameter(
            torch.randn(1, out_ch, in_ch, kernel_size, kernel_size) / math.sqrt(in_ch)
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel_size
        # style scale: (B, in_ch)
        s = self.affine(w)  # (B, in_ch)
        # modulate weight: (B, out_ch, in_ch, k, k)
        weight = self.weight * s.view(B, 1, self.in_ch, 1, 1)
        if self.demodulate:
            d = weight.pow(2).sum(dim=[2, 3, 4], keepdim=True).add(1e-8).sqrt()
            weight = weight / d
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # Apply per-sample (fold batch into groups)
        x = x.reshape(1, B * C, H * (2 if self.upsample else 1), W * (2 if self.upsample else 1))
        weight = weight.view(B * self.out_ch, self.in_ch, k, k)
        out = F.conv2d(x, weight, padding=k // 2, groups=B)
        out = out.view(B, self.out_ch, out.shape[-2], out.shape[-1])
        out = out + self.bias.view(1, -1, 1, 1)
        return out


class NoiseInjection(nn.Module):
    """Per-pixel additive noise with learned amplitude."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.amplitude = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        noise = torch.randn(B, 1, H, W, device=x.device)
        return x + self.amplitude * noise


class SynthesisBlock(nn.Module):
    """One scale of the StyleGAN2 synthesis network.

    upsample modulated conv -> noise -> act -> to-RGB skip.
    """

    def __init__(self, in_ch: int, out_ch: int, w_dim: int) -> None:
        super().__init__()
        self.conv1 = ModulatedConv2d(in_ch, out_ch, w_dim, upsample=True)
        self.noise1 = NoiseInjection(out_ch)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = ModulatedConv2d(out_ch, out_ch, w_dim, upsample=False)
        self.noise2 = NoiseInjection(out_ch)
        self.act2 = nn.LeakyReLU(0.2)
        self.to_rgb = ModulatedConv2d(out_ch, 3, w_dim, kernel_size=1, demodulate=False)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.act1(self.noise1(self.conv1(x, w)))
        x = self.act2(self.noise2(self.conv2(x, w)))
        rgb = self.to_rgb(x, w)
        return x, rgb


# ============================================================
# 1. StyleGAN2 Generator
# ============================================================


class StyleGAN2Generator(nn.Module):
    """StyleGAN2 generator (compact).

    Mapping network (z -> w) + synthesis network with 2 skip blocks.
    Outputs tiny 32x32 images (paper: 1024x1024).
    """

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        base_ch: int = 32,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim, n_layers=4, cond_dim=cond_dim)
        # Learned constant 4x4 input
        self.const = nn.Parameter(torch.randn(1, base_ch * 4, 4, 4))
        # 4->8: block 0
        self.block0 = SynthesisBlock(base_ch * 4, base_ch * 2, w_dim)
        # 8->16: block 1
        self.block1 = SynthesisBlock(base_ch * 2, base_ch, w_dim)
        # 16->32: block 2
        self.block2 = SynthesisBlock(base_ch, base_ch, w_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        x, rgb0 = self.block0(x, w)
        x, rgb1 = self.block1(x, w)
        x, rgb2 = self.block2(x, w)
        # Skip connections: sum upsampled rgbs
        out = self.upsample(self.upsample(rgb0)) + self.upsample(rgb1) + rgb2
        return torch.tanh(out)


def build_stylegan2_generator() -> nn.Module:
    return StyleGAN2Generator()


def example_input_stylegan2_generator() -> torch.Tensor:
    return torch.randn(1, 64)


# ============================================================
# 2. StyleGAN2 Discriminator
# ============================================================


class ResBlock(nn.Module):
    """Residual downsampling block for the StyleGAN2 discriminator."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2)
        self.act2 = nn.LeakyReLU(0.2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act2(self.conv2(self.act1(self.conv1(x))))
        return (h + self.skip(x)) / math.sqrt(2)


class MinibatchStd(nn.Module):
    """Minibatch standard deviation feature (appended as extra channel)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=0, keepdim=True).mean(dim=[1, 2, 3], keepdim=True)
        std = std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std], dim=1)


class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 discriminator (compact).

    Residual downsampling blocks -> minibatch std -> final linear.
    Input: 32x32 RGB.
    """

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        self.from_rgb = nn.Sequential(nn.Conv2d(3, base_ch, 1), nn.LeakyReLU(0.2))
        self.block0 = ResBlock(base_ch, base_ch * 2)  # 32->16
        self.block1 = ResBlock(base_ch * 2, base_ch * 4)  # 16->8
        self.block2 = ResBlock(base_ch * 4, base_ch * 4)  # 8->4
        self.mbstd = MinibatchStd()
        self.conv_final = nn.Conv2d(base_ch * 4 + 1, base_ch * 4, 3, padding=1)
        self.act_final = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(base_ch * 4 * 4 * 4, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.from_rgb(img)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.mbstd(x)
        x = self.act_final(self.conv_final(x))
        x = x.flatten(1)
        return self.fc(x)


def build_stylegan2_discriminator() -> nn.Module:
    return StyleGAN2Discriminator()


def example_input_stylegan2_discriminator() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ============================================================
# ADA alias builders (same architecture, separate catalog entries)
# ============================================================


def build_stylegan2_ada_generator() -> nn.Module:
    """StyleGAN2-ADA generator (ADA = training augmentation, same arch)."""
    return StyleGAN2Generator()


def example_input_stylegan2_ada_generator() -> torch.Tensor:
    return torch.randn(1, 64)


def build_stylegan2_ada_discriminator() -> nn.Module:
    """StyleGAN2-ADA discriminator (same architecture as StyleGAN2)."""
    return StyleGAN2Discriminator()


def example_input_stylegan2_ada_discriminator() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


def build_stylegan2_ada_pytorch_generator() -> nn.Module:
    """StyleGAN2-ADA-PyTorch generator (NVlabs official PyTorch port, same arch)."""
    return StyleGAN2Generator()


def example_input_stylegan2_ada_pytorch_generator() -> torch.Tensor:
    return torch.randn(1, 64)


def build_stylegan2_ada_pytorch_discriminator() -> nn.Module:
    """StyleGAN2-ADA-PyTorch discriminator (same architecture)."""
    return StyleGAN2Discriminator()


def example_input_stylegan2_ada_pytorch_discriminator() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ============================================================
# StyleGAN-Human (StyleGAN2 arch applied to full-body generation)
# ============================================================


class StyleGAN2GeneratorTall(nn.Module):
    """StyleGAN-Human generator: StyleGAN2 synthesis with taller aspect ratio (32x16).

    StyleGAN-Human (Fu et al., ECCV 2022, arXiv:2204.11823) applies StyleGAN2
    architecture to full human body generation with tall aspect-ratio images
    (512x256 in the paper). This compact version outputs 32x16.
    """

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        base_ch: int = 32,
    ) -> None:
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim, n_layers=4)
        # Tall constant seed (8x4)
        self.const = nn.Parameter(torch.randn(1, base_ch * 4, 8, 4))
        self.block0 = SynthesisBlock(base_ch * 4, base_ch * 2, w_dim)  # 16x8
        self.block1 = SynthesisBlock(base_ch * 2, base_ch, w_dim)  # 32x16
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        x, rgb0 = self.block0(x, w)
        x, rgb1 = self.block1(x, w)
        out = self.upsample(rgb0) + rgb1
        return torch.tanh(out)


def build_stylegan_human_generator() -> nn.Module:
    return StyleGAN2GeneratorTall()


def example_input_stylegan_human_generator() -> torch.Tensor:
    return torch.randn(1, 64)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "stylegan2_ada_generator",
        "build_stylegan2_ada_generator",
        "example_input_stylegan2_ada_generator",
        "2020",
        "DC",
    ),
    (
        "stylegan2_ada_discriminator",
        "build_stylegan2_ada_discriminator",
        "example_input_stylegan2_ada_discriminator",
        "2020",
        "DC",
    ),
    (
        "stylegan2_ada_pytorch_generator",
        "build_stylegan2_ada_pytorch_generator",
        "example_input_stylegan2_ada_pytorch_generator",
        "2020",
        "DC",
    ),
    (
        "stylegan2_ada_pytorch_discriminator",
        "build_stylegan2_ada_pytorch_discriminator",
        "example_input_stylegan2_ada_pytorch_discriminator",
        "2020",
        "DC",
    ),
    (
        "stylegan_human_generator",
        "build_stylegan_human_generator",
        "example_input_stylegan_human_generator",
        "2022",
        "DC",
    ),
]
