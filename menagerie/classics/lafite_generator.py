"""LAFITE: Towards Language-Free Training for Text-to-Image Generation.

Zhou et al. 2022.  arXiv:2111.13792.
Source: https://github.com/drboog/Lafite

LAFITE's distinctive primitives:
  - **Language-free training**: trains a text-to-image GAN *without* text labels.
    Instead of real text captions, LAFITE uses CLIP's aligned embedding space:
    the generator is conditioned on CLIP image features of the *real* image during
    training (via projection).  At inference, text embeddings can be substituted.
  - **StyleGAN2-style synthesis network**: the backbone is a StyleGAN2 generator
    (mapping network z + clip_code -> w; const feature map; modulated conv blocks).
  - **Dual-path conditioning**: the style code w is derived from both noise z *and*
    a CLIP feature code (projected to match w dimension), mixed via learned affine.
  - No custom discriminator change; the CLIP conditioning is solely on the generator.

Here we reproduce:
  - Mapping network: (z || clip_code) -> w.
  - Const feature map -> modulated-conv synthesis blocks (3 blocks, 4->8->16->32).
  - CLIP feature conditioner (stub random projection).
  - RGB head.

Random init, CPU, compact dims for clean tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingNetwork(nn.Module):
    """(z || clip_code) -> style code w."""

    def __init__(self, z_dim: int, clip_dim: int, w_dim: int, n_layers: int = 3) -> None:
        super().__init__()
        in_dim = z_dim + clip_dim
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, clip_code: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, clip_code], dim=-1))


class ModConv2d(nn.Module):
    """Modulated conv2d (StyleGAN2 style): weight scaled per-sample by style vector."""

    def __init__(self, in_c: int, out_c: int, k: int, w_dim: int, pad: int = 1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_c, in_c, k, k))
        self.style = nn.Linear(w_dim, in_c)
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.pad = pad

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        s = self.style(w) + 1.0  # (B, in_c)
        wt = self.weight.unsqueeze(0) * s.view(B, 1, self.in_c, 1, 1)
        denom = wt.pow(2).sum([2, 3, 4], keepdim=True).add(1e-8).sqrt()
        wt = wt / denom
        x = x.reshape(1, B * self.in_c, x.shape[2], x.shape[3])
        wt = wt.view(B * self.out_c, self.in_c, self.k, self.k)
        out = F.conv2d(x, wt, padding=self.pad, groups=B)
        return out.view(B, self.out_c, out.shape[2], out.shape[3]) + self.bias.view(1, -1, 1, 1)


class SynthBlock(nn.Module):
    """StyleGAN2 synthesis block: upsample + modulated conv + noise."""

    def __init__(self, in_c: int, out_c: int, w_dim: int) -> None:
        super().__init__()
        self.conv = ModConv2d(in_c, out_c, 3, w_dim)
        self.noise_w = nn.Parameter(torch.zeros(1, out_c, 1, 1))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv(x, w)
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        return F.leaky_relu(x + self.noise_w * noise, 0.2)


class LAFITEGenerator(nn.Module):
    """LAFITE generator: CLIP-conditioned StyleGAN2-style synthesis."""

    def __init__(
        self,
        z_dim: int = 32,
        clip_dim: int = 64,
        w_dim: int = 64,
        nf: int = 16,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.clip_dim = clip_dim
        # CLIP feature projector (stub: random linear)
        self.clip_proj = nn.Linear(clip_dim, clip_dim)
        self.mapping = MappingNetwork(z_dim, clip_dim, w_dim)
        # Const feature map
        self.const = nn.Parameter(torch.randn(1, nf * 8, 4, 4))
        # Synthesis blocks: 4->8->16->32
        self.blk1 = SynthBlock(nf * 8, nf * 4, w_dim)
        self.blk2 = SynthBlock(nf * 4, nf * 2, w_dim)
        self.blk3 = SynthBlock(nf * 2, nf, w_dim)
        self.to_rgb = nn.Conv2d(nf, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, z_dim + clip_dim)
        z = x[:, : self.z_dim]
        clip_raw = x[:, self.z_dim :]
        clip_code = self.clip_proj(clip_raw)
        w = self.mapping(z, clip_code)
        B = z.shape[0]
        feat = self.const.expand(B, -1, -1, -1)
        feat = self.blk1(feat, w)
        feat = self.blk2(feat, w)
        feat = self.blk3(feat, w)
        return torch.tanh(self.to_rgb(feat))


def build_lafite_generator() -> nn.Module:
    return LAFITEGenerator()


def example_input() -> torch.Tensor:
    # z (32) || clip_code (64)
    return torch.randn(1, 32 + 64)


MENAGERIE_ENTRIES = [
    (
        "LAFITE Generator (language-free CLIP-conditioned StyleGAN2-style synthesis)",
        "build_lafite_generator",
        "example_input",
        "2022",
        "DC",
    ),
]
