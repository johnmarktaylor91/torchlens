"""StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2.

Skorokhodov et al., CVPR 2022.  arXiv:2112.14683.
Source: https://github.com/universome/stylegan-v

StyleSV (StyleSV: Style-based Video Generation):
  Guo et al.  arXiv:2311.10014.  (successor/concurrent video StyleGAN variant)
  Also mapped here as a separate entry showing continuous motion conditioning.

Distinctive primitives vs StyleGAN2:
  - **Continuous-time positional encoding (temporal embedding)**: a sinusoidal or
    Fourier-based embedding of a continuous time value t in [0,1] is injected into
    the mapping network, causing the style w to vary smoothly with time.
    This is the KEY contribution: w(z, t) allows temporally smooth video generation
    by interpolating in w-space.
  - **Time-conditioned mapping network**: t_emb = positional_encoding(t) is concatenated
    to z before the mapping MLP. No architectural changes to the synthesis blocks.
  - **Motion codes**: StyleSV additionally uses a separate motion code m (z_m -> w_m)
    to disentangle appearance (z_a -> w_a) from motion (z_m -> w_m). The synthesis
    network receives both w_a and w_m via separate modulation paths.
  - Synthesis network is StyleGAN2-based (weight-demodulated modulated conv + noise).

Compact: z_dim=64, w_dim=64, time_emb_dim=16, output 32x32, 2 synthesis blocks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Temporal / continuous-time positional encoding
# ============================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal encoding of continuous time t in [0, 1] -> R^d.

    The signature temporal primitive in StyleGAN-V: inject time as a positional
    code so that w(z, t) varies smoothly with t, enabling video frame generation.
    """

    def __init__(self, emb_dim: int = 16) -> None:
        super().__init__()
        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim
        # Learned frequency scales for the sinusoidal encoding
        self.freq_proj = nn.Linear(1, emb_dim // 2)
        self.out = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1)  in range [0, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (B, 1)
        freqs = self.freq_proj(t)  # (B, emb_dim//2)
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=1)  # (B, emb_dim)
        return self.out(emb)


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())


class TemporalMappingNetwork(nn.Module):
    """Mapping network conditioned on (z, time_emb) -> w.

    StyleGAN-V's core: time is positionally encoded and concatenated to z,
    making w a smooth function of continuous time t.
    """

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        time_emb_dim: int = 16,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.pixel_norm = PixelNorm()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        in_dim = z_dim + time_emb_dim
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = self.pixel_norm(z)
        te = self.time_emb(t)
        return self.net(torch.cat([z, te], dim=1))


# ============================================================
# StyleGAN2-style synthesis (modulated conv + noise)
# ============================================================


class ModulatedConv2d(nn.Module):
    """Weight-demodulated modulated convolution (StyleGAN2 op)."""

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
        self.affine = nn.Linear(w_dim, in_ch, bias=True)
        nn.init.ones_(self.affine.bias)
        self.weight = nn.Parameter(
            torch.randn(1, out_ch, in_ch, kernel_size, kernel_size) / math.sqrt(in_ch)
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel_size
        s = self.affine(w)
        weight = self.weight * s.view(B, 1, self.in_ch, 1, 1)
        if self.demodulate:
            d = weight.pow(2).sum(dim=[2, 3, 4], keepdim=True).add(1e-8).sqrt()
            weight = weight / d
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = x.reshape(1, B * C, x.size(2), x.size(3))
        weight = weight.view(B * self.out_ch, self.in_ch, k, k)
        out = F.conv2d(x, weight, padding=k // 2, groups=B)
        out = out.view(B, self.out_ch, out.size(2), out.size(3))
        return out + self.bias.view(1, -1, 1, 1)


class NoiseInjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.amplitude = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return x + self.amplitude * torch.randn(B, 1, H, W, device=x.device)


class SynthesisBlock(nn.Module):
    """StyleGAN2-style synthesis block: upsample modulated conv + noise."""

    def __init__(self, in_ch: int, out_ch: int, w_dim: int) -> None:
        super().__init__()
        self.conv1 = ModulatedConv2d(in_ch, out_ch, w_dim, upsample=True)
        self.noise1 = NoiseInjection(out_ch)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = ModulatedConv2d(out_ch, out_ch, w_dim)
        self.noise2 = NoiseInjection(out_ch)
        self.act2 = nn.LeakyReLU(0.2)
        self.to_rgb = ModulatedConv2d(out_ch, 3, w_dim, kernel_size=1, demodulate=False)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.act1(self.noise1(self.conv1(x, w)))
        x = self.act2(self.noise2(self.conv2(x, w)))
        rgb = self.to_rgb(x, w)
        return x, rgb


# ============================================================
# StyleGAN-V Generator
# ============================================================


class StyleGANVGenerator(nn.Module):
    """StyleGAN-V continuous-time video generator (compact).

    Inputs: z (appearance noise) + t (continuous time scalar in [0, 1]).
    w(z, t) is smooth in t -> temporally coherent video frames.
    Synthesis: StyleGAN2-style blocks with skip RGB connections.
    """

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        time_emb_dim: int = 16,
        base_ch: int = 32,
    ) -> None:
        super().__init__()
        self.mapping = TemporalMappingNetwork(z_dim, w_dim, time_emb_dim)
        self.const = nn.Parameter(torch.randn(1, base_ch * 4, 4, 4))
        self.block0 = SynthesisBlock(base_ch * 4, base_ch * 2, w_dim)  # 4->8
        self.block1 = SynthesisBlock(base_ch * 2, base_ch, w_dim)  # 8->16
        self.block2 = SynthesisBlock(base_ch, base_ch, w_dim)  # 16->32
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, z_dim) noise latent
            t: (B,) or (B,1) continuous time in [0,1]
        Returns:
            (B, 3, 32, 32) generated frame
        """
        w = self.mapping(z, t)
        x = self.const.expand(z.size(0), -1, -1, -1)
        x, rgb0 = self.block0(x, w)
        x, rgb1 = self.block1(x, w)
        x, rgb2 = self.block2(x, w)
        out = self.upsample(self.upsample(rgb0)) + self.upsample(rgb1) + rgb2
        return torch.tanh(out)


def build_stylegan_v_generator() -> nn.Module:
    return StyleGANVGenerator()


def example_input_stylegan_v_generator() -> tuple:
    """(z, t): z is noise latent, t is continuous time in [0,1]."""
    z = torch.randn(1, 64)
    t = torch.rand(1)
    return (z, t)


# ============================================================
# StyleSV Generator (appearance + motion disentanglement)
# ============================================================


class StyleSVGenerator(nn.Module):
    """StyleSV generator: separate appearance (z_a) and motion (z_m) codes.

    StyleSV disentangles appearance and motion via two separate mapping paths:
      w_a = mapping_a(z_a)   -- appearance style
      w_m = mapping_m(z_m, t) -- motion style (time-conditioned)
    The synthesis layers receive concatenated (w_a, w_m) via a fused linear.
    """

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        time_emb_dim: int = 16,
        base_ch: int = 32,
    ) -> None:
        super().__init__()
        # Appearance mapping (no time conditioning)
        self.pixel_norm = PixelNorm()
        _layers_a: list[nn.Module] = []
        in_d = z_dim
        for _ in range(3):
            _layers_a += [nn.Linear(in_d, w_dim), nn.LeakyReLU(0.2)]
            in_d = w_dim
        self.mapping_appearance = nn.Sequential(*_layers_a)
        # Motion mapping (time-conditioned)
        self.mapping_motion = TemporalMappingNetwork(z_dim, w_dim, time_emb_dim, n_layers=3)
        # Fuse appearance + motion -> single w for synthesis
        self.w_fuse = nn.Linear(w_dim * 2, w_dim)
        # Synthesis (StyleGAN2-style)
        self.const = nn.Parameter(torch.randn(1, base_ch * 4, 4, 4))
        self.block0 = SynthesisBlock(base_ch * 4, base_ch * 2, w_dim)
        self.block1 = SynthesisBlock(base_ch * 2, base_ch, w_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, z_a: torch.Tensor, z_m: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        w_a = self.mapping_appearance(self.pixel_norm(z_a))
        w_m = self.mapping_motion(z_m, t)
        w = self.w_fuse(torch.cat([w_a, w_m], dim=1))
        x = self.const.expand(z_a.size(0), -1, -1, -1)
        x, rgb0 = self.block0(x, w)
        x, rgb1 = self.block1(x, w)
        out = self.upsample(rgb0) + rgb1
        return torch.tanh(out)


def build_stylesv_generator() -> nn.Module:
    return StyleSVGenerator()


def example_input_stylesv_generator() -> tuple:
    """(z_a, z_m, t): appearance noise, motion noise, continuous time."""
    z_a = torch.randn(1, 64)
    z_m = torch.randn(1, 64)
    t = torch.rand(1)
    return (z_a, z_m, t)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "stylegan_v_generator",
        "build_stylegan_v_generator",
        "example_input_stylegan_v_generator",
        "2022",
        "DC",
    ),
    (
        "stylesv_generator",
        "build_stylesv_generator",
        "example_input_stylesv_generator",
        "2023",
        "DC",
    ),
]
