"""StyleGAN3: Alias-Free Generative Adversarial Networks.

Karras et al., NeurIPS 2021.  arXiv:2106.12423.
Source: https://github.com/NVlabs/stylegan3

Distinctive primitives vs StyleGAN2:
  - No learned constant input; instead a **Fourier-feature sinusoidal input**
    (frequencies learned, phases randomized per sample) to remove texture sticking.
  - **Alias-free filtered upsampling / downsampling**: every activation is preceded
    by an ideal low-pass filter (implemented as a Kaiser-windowed FIR), ensuring the
    network is equivariant to continuous sub-pixel translations (and rotations for
    StyleGAN3-R). The signature primitive is `filtered_lrelu`: upsample -> leaky_relu
    -> low-pass filter -> optional downsample, all as one fused op.
  - **Modulated 1x1 convolutions** (no spatial extent at base resolution; spatial
    processing is entirely handled by the filter bank). Full-size kernels only after
    sufficient upsampling stages.
  - **Two configs** that share one synthesis architecture:
      stylegan3_t (translation equivariant): rotation-variant FIR filters, more stages.
      stylegan3_r (rotation equivariant): rotation-equivariant FIR via radial filters.
  - **Discriminator**: same residual downsampling design as StyleGAN2 (shared here).

Compact: z_dim=64, w_dim=64, 3 Fourier-feature + filtered synthesis layers, output 32x32.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Mapping network (shared with StyleGAN2 style)
# ============================================================


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int = 64, w_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.pixel_norm = PixelNorm()
        layers: list[nn.Module] = []
        in_dim = z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.pixel_norm(z))


# ============================================================
# Fourier-feature input (StyleGAN3 signature: replaces learned constant)
# ============================================================


class FourierFeatureInput(nn.Module):
    """Sinusoidal Fourier-feature input grid.

    StyleGAN3 replaces the learned constant 4x4 tensor with a spatial grid of
    sinusoidal Fourier features: F(x,y) = sum_k A_k * cos(2pi * f_k . [x,y] + phi_k).
    Frequencies f_k are learned; amplitudes A_k are learned. Phases phi_k can be
    randomised per sample (translation equivariance) or fixed.
    """

    def __init__(self, out_ch: int, n_freqs: int = 8) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.n_freqs = n_freqs
        # Learned frequency vectors (n_freqs, 2)
        self.freqs = nn.Parameter(torch.randn(n_freqs, 2) * 2.0)
        # Learned amplitudes and phases (out_ch mapped from n_freqs via linear)
        self.amplitude = nn.Parameter(torch.ones(n_freqs))
        self.proj = nn.Linear(n_freqs, out_ch)

    def forward(self, z: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = z.size(0)
        device = z.device
        # Spatial coordinates grid: (H*W, 2)
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (H*W, 2)
        # Fourier features: (H*W, n_freqs)
        phase = (coords @ self.freqs.T) * (2 * math.pi)
        features = torch.cos(phase) * self.amplitude.unsqueeze(0)  # (H*W, n_freqs)
        # Project to channels
        features = self.proj(features)  # (H*W, out_ch)
        features = features.T.view(1, self.out_ch, H, W).expand(B, -1, -1, -1)
        return features


# ============================================================
# Alias-free filtered synthesis layer (StyleGAN3 signature op)
# ============================================================


class FilteredLReLU(nn.Module):
    """Alias-free filtered leaky ReLU: upsample -> act -> low-pass filter.

    Implements the core alias-free primitive from StyleGAN3:
    each nonlinearity is wrapped in low-pass filters to prevent aliasing.
    Here we use a simple average-pool based low-pass (faithful to the concept;
    full paper uses a Kaiser-windowed FIR with learned cutoff).
    """

    def __init__(self, channels: int, upsample: int = 2, downsample: int = 2) -> None:
        super().__init__()
        self.upsample = upsample
        self.downsample = downsample
        # Low-pass filter approximation: depthwise conv with Gaussian kernel
        k = 5
        sigma = 1.5
        g = torch.tensor(
            [math.exp(-((i - k // 2) ** 2) / (2 * sigma**2)) for i in range(k)],
            dtype=torch.float32,
        )
        kernel = g[:, None] * g[None, :]
        kernel = kernel / kernel.sum()
        # Depthwise per-channel filter
        self.register_buffer("lpf", kernel.view(1, 1, k, k).expand(channels, 1, k, k).clone())
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample, mode="bilinear", align_corners=False)
        x = F.leaky_relu(x, 0.2)
        # Low-pass filter (alias suppression)
        x = F.conv2d(x, self.lpf, padding=2, groups=self.channels)
        if self.downsample > 1:
            x = F.avg_pool2d(x, self.downsample)
        return x


class StyleGAN3SynthesisLayer(nn.Module):
    """StyleGAN3 alias-free synthesis layer.

    Modulated 1x1 conv (or 3x3) -> filtered leaky relu.
    Style modulation: weight scaled per sample by affine projection of w.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        w_dim: int,
        kernel_size: int = 3,
        upsample: bool = True,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.upsample = upsample
        # Affine: w -> per-channel style scale
        self.affine = nn.Linear(w_dim, in_ch)
        nn.init.ones_(self.affine.bias)
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size) / math.sqrt(in_ch)
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))
        # Alias-free activation: upsample inside if requested
        self.filtered_act = FilteredLReLU(
            out_ch, upsample=2 if upsample else 1, downsample=2 if upsample else 1
        )

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        s = self.affine(w)  # (B, in_ch)
        # Modulate weight
        k = self.weight.size(-1)
        weight = self.weight.unsqueeze(0) * s.view(B, 1, self.in_ch, 1, 1)
        # Demodulate
        d = weight.pow(2).sum(dim=[2, 3, 4], keepdim=True).add(1e-8).sqrt()
        weight = weight / d
        # Per-sample grouped conv
        x_in = x.reshape(1, B * self.in_ch, x.size(2), x.size(3))
        w_flat = weight.view(B * self.out_ch, self.in_ch, k, k)
        out = F.conv2d(x_in, w_flat, padding=k // 2, groups=B)
        out = out.view(B, self.out_ch, out.size(2), out.size(3))
        out = out + self.bias.view(1, -1, 1, 1)
        # Alias-free filtered activation
        out = self.filtered_act(out)
        return out


# ============================================================
# Discriminator (same architecture as StyleGAN2)
# ============================================================


class ResBlock(nn.Module):
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=0, keepdim=True).mean(dim=[1, 2, 3], keepdim=True)
        std = std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std], dim=1)


# ============================================================
# StyleGAN3-T Generator (translation-equivariant)
# ============================================================


class StyleGAN3Generator(nn.Module):
    """StyleGAN3 alias-free generator (compact).

    z -> mapping -> w -> Fourier-feature input ->
    N alias-free filtered synthesis layers -> to-RGB.

    stylegan3_t (translation): standard filtered lrelu, no rotation constraint.
    stylegan3_r (rotation): same architecture but with radial-symmetric filters
    (not separately parametrized here -- both use the same alias-free primitive).
    """

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        base_ch: int = 32,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim, n_layers=4)
        self.fourier_input = FourierFeatureInput(base_ch * 4, n_freqs=8)
        # Synthesis layers: gradually reduce channels + upsample
        ch = [base_ch * 4, base_ch * 2, base_ch, base_ch]
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                StyleGAN3SynthesisLayer(ch[i], ch[i + 1], w_dim, upsample=(i < n_layers - 1))
            )
        self.to_rgb = nn.Conv2d(ch[-1], 3, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        w = self.mapping(z)
        # Fourier-feature input at base 4x4
        x = self.fourier_input(z, 4, 4)
        for layer in self.layers:
            x = layer(x, w)
        return torch.tanh(self.to_rgb(x))


def build_stylegan3_t_generator() -> nn.Module:
    """StyleGAN3-T (translation-equivariant) generator."""
    return StyleGAN3Generator()


def example_input_stylegan3_t_generator() -> torch.Tensor:
    return torch.randn(1, 64)


def build_stylegan3_r_generator() -> nn.Module:
    """StyleGAN3-R (rotation-equivariant) generator (same compact arch)."""
    return StyleGAN3Generator()


def example_input_stylegan3_r_generator() -> torch.Tensor:
    return torch.randn(1, 64)


# ============================================================
# StyleGAN3 Discriminator
# ============================================================


class StyleGAN3Discriminator(nn.Module):
    """StyleGAN3 discriminator: same residual structure as StyleGAN2."""

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        self.from_rgb = nn.Sequential(nn.Conv2d(3, base_ch, 1), nn.LeakyReLU(0.2))
        self.block0 = ResBlock(base_ch, base_ch * 2)
        self.block1 = ResBlock(base_ch * 2, base_ch * 4)
        self.block2 = ResBlock(base_ch * 4, base_ch * 4)
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
        return self.fc(x.flatten(1))


def build_stylegan3_discriminator() -> nn.Module:
    return StyleGAN3Discriminator()


def example_input_stylegan3_discriminator() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "stylegan3_t_generator",
        "build_stylegan3_t_generator",
        "example_input_stylegan3_t_generator",
        "2021",
        "DC",
    ),
    (
        "stylegan3_r_generator",
        "build_stylegan3_r_generator",
        "example_input_stylegan3_r_generator",
        "2021",
        "DC",
    ),
    (
        "stylegan3_discriminator",
        "build_stylegan3_discriminator",
        "example_input_stylegan3_discriminator",
        "2021",
        "DC",
    ),
]
