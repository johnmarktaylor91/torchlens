"""GALIP: Generative Adversarial CLIPs (CLIP-conditioned text-to-image GAN generator).

Tao et al. 2023.  arXiv:2301.12959.
Source: https://github.com/tobran/GALIP

GALIP's distinctive primitives:
  - **CLIP-conditioned generator**: a frozen CLIP image encoder provides visual features
    that are concatenated with a noise code z to form the conditioning signal.
  - **GAN generator backbone**: CLIP-feature + noise -> MLP mapping to style code w,
    then a series of affine-conditioned conv-upsample blocks (similar in spirit to
    StyleGAN2's synthesis network but using CLIP features rather than learned W+ codes).
  - **Affine modulation blocks**: per-block learned affine transform of instance-norm
    features conditioned on the style vector (same pattern as SPADE/AdaIN-style cond).
  - **Multi-scale discriminator** (not reproduced here; generator only).

Here we reproduce the generator: noise z + CLIP conditioning vector -> style MLP ->
stack of AffineBlock upsamplers -> RGB image.  CLIP encoder replaced by a small random
linear projection for compact tracing (no CLIP weights needed).

Random init, CPU, small channels/spatial for clean tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPConditioner(nn.Module):
    """Stub: random-projection stand-in for a frozen CLIP image/text encoder."""

    def __init__(self, clip_dim: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(clip_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MappingNetwork(nn.Module):
    """Maps (noise z || clip_feat) -> style code w."""

    def __init__(self, z_dim: int, clip_dim: int, w_dim: int, n_layers: int = 3) -> None:
        super().__init__()
        in_dim = z_dim + clip_dim
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, clip_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, clip_feat], dim=-1))


class AffineModBlock(nn.Module):
    """Upsample + conv + affine-style modulation conditioned on style code w."""

    def __init__(self, in_c: int, out_c: int, w_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_c, affine=False)
        self.style_scale = nn.Linear(w_dim, out_c)
        self.style_bias = nn.Linear(w_dim, out_c)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        scale = self.style_scale(w).unsqueeze(-1).unsqueeze(-1)
        bias = self.style_bias(w).unsqueeze(-1).unsqueeze(-1)
        x = x * (1.0 + scale) + bias
        return F.leaky_relu(x, 0.2)


class GALIPGenerator(nn.Module):
    """GALIP generator: CLIP-cond + noise -> style MLP -> affine-mod upsample stack."""

    def __init__(
        self,
        z_dim: int = 32,
        clip_dim: int = 64,
        w_dim: int = 64,
        nf: int = 16,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.clip_cond = CLIPConditioner(clip_dim, clip_dim)
        self.mapping = MappingNetwork(z_dim, clip_dim, w_dim)
        # Learned constant 4x4 starting feature map
        self.const = nn.Parameter(torch.randn(1, nf * 8, 4, 4))
        # Upsample blocks: 4->8->16->32
        self.block1 = AffineModBlock(nf * 8, nf * 4, w_dim)
        self.block2 = AffineModBlock(nf * 4, nf * 2, w_dim)
        self.block3 = AffineModBlock(nf * 2, nf, w_dim)
        self.to_rgb = nn.Conv2d(nf, 3, 1)

    def forward(self, z: torch.Tensor, clip_feat: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        clip_feat = self.clip_cond(clip_feat)  # (B, clip_dim)
        w = self.mapping(z, clip_feat)  # (B, w_dim)
        x = self.const.expand(B, -1, -1, -1)  # (B, nf*8, 4, 4)
        x = self.block1(x, w)
        x = self.block2(x, w)
        x = self.block3(x, w)
        return torch.tanh(self.to_rgb(x))


class _GALIPWrapper(nn.Module):
    """Single-input wrapper: expects (z || clip_feat) concatenated on dim=-1."""

    def __init__(self, model: nn.Module, z_dim: int, clip_dim: int) -> None:
        super().__init__()
        self.model = model
        self.z_dim = z_dim
        self.clip_dim = clip_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x[..., : self.z_dim]
        clip_feat = x[..., self.z_dim :]
        return self.model(z, clip_feat)


def build_galip_netg() -> nn.Module:
    z_dim, clip_dim = 32, 64
    gen = GALIPGenerator(z_dim=z_dim, clip_dim=clip_dim, w_dim=64, nf=16)
    return _GALIPWrapper(gen, z_dim, clip_dim)


def example_input() -> torch.Tensor:
    # Concatenated (z || clip_feat) vector, batch size 1
    return torch.randn(1, 32 + 64)


MENAGERIE_ENTRIES = [
    (
        "GALIP Generator (CLIP-conditioned affine-mod GAN text-to-image generator)",
        "build_galip_netg",
        "example_input",
        "2023",
        "DC",
    ),
]
