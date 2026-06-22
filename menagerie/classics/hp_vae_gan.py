"""HP-VAE-GAN: Hierarchical Patch VAE-GAN for Diverse and Coherent Image Generation.

Shoshan et al. 2021.  arXiv:2012.03751.
Source: https://github.com/shikoG/HP-VAE-GAN

HP-VAE-GAN's distinctive primitives:
  - **Hierarchical patch pyramid**: generation proceeds coarse-to-fine across
    multiple scales.  At each scale, a patch-level VAE-GAN produces residuals
    that are added to an upsampled version of the previous scale's output.
  - **Patch VAE at each level**: a scale-specific encoder compresses the real
    image patches to a stochastic code (mu, logvar); the decoder generates
    additive residuals conditioned on the noise sample.
  - **Residual synthesis**: each level's generated image = upsample(prev) + residual,
    giving the characteristic multi-scale additive pyramid structure.
  - **Adversarial training at every scale** (discriminators not reproduced here;
    generator only).

Here we reproduce the generator pyramid:
  - Three scales (coarse, mid, fine): 8x8 -> 16x16 -> 32x32.
  - Each scale: patch encoder -> reparameterize -> patch decoder -> residual.
  - Output: upsampled base + accumulated residuals at finest scale.

Random init, CPU, small channels for compact tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchVAEEncoder(nn.Module):
    """Per-scale patch encoder: image patches -> (mu, logvar)."""

    def __init__(self, in_c: int, nf: int, z_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, nf, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf * 2, 3, 2, 1),  # downsample x2
            nn.LeakyReLU(0.2),
        )
        self.mu_head = nn.Conv2d(nf * 2, z_dim, 1)
        self.lv_head = nn.Conv2d(nf * 2, z_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu_head(h), self.lv_head(h)


class PatchVAEDecoder(nn.Module):
    """Per-scale patch decoder: z -> image residual (same spatial size as input)."""

    def __init__(self, z_dim: int, nf: int, out_c: int, target_h: int, target_w: int) -> None:
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, nf * 2, 3, 2, 1, output_padding=1),  # upsample x2
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            nn.Conv2d(nf * 2, out_c, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return F.interpolate(
            out, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False
        )


class HPVAEGANScale(nn.Module):
    """One scale in the HP-VAE-GAN hierarchy: encoder + reparameterize + decoder."""

    def __init__(self, in_c: int, nf: int, z_dim: int, h: int, w: int) -> None:
        super().__init__()
        self.encoder = PatchVAEEncoder(in_c, nf, z_dim)
        self.decoder = PatchVAEDecoder(z_dim, nf, in_c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        return self.decoder(z)


class HPVAEGANGenerator(nn.Module):
    """HP-VAE-GAN generator pyramid: coarse->mid->fine residual synthesis.

    At each scale the image is upsampled from the previous scale and a learned
    patch-VAE contributes an additive residual.
    """

    def __init__(self, nc: int = 3, nf: int = 16, z_dim: int = 8) -> None:
        super().__init__()
        # Scale 0: coarse 8x8 -- learned const start
        self.coarse_const = nn.Parameter(torch.randn(1, nc, 8, 8))
        # Scale 1: 8->16, adds residual at 16x16
        self.scale1 = HPVAEGANScale(nc, nf, z_dim, 16, 16)
        # Scale 2: 16->32, adds residual at 32x32
        self.scale2 = HPVAEGANScale(nc, nf, z_dim, 32, 32)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        # noise is unused at inference (reparameterization is inside each scale)
        # Start from coarse constant
        B = noise.shape[0]
        x = self.coarse_const.expand(B, -1, -1, -1)  # (B, nc, 8, 8)
        # Scale 1: upsample to 16, add residual
        x16 = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
        x16 = x16 + self.scale1(x16)
        # Scale 2: upsample to 32, add residual
        x32 = F.interpolate(x16, size=(32, 32), mode="bilinear", align_corners=False)
        x32 = x32 + self.scale2(x32)
        return x32


def build_hp_vae_gan_generator() -> nn.Module:
    return HPVAEGANGenerator()


def example_input() -> torch.Tensor:
    # Dummy noise token (generator ignores it; reparameterization is internal)
    return torch.randn(1, 1)


MENAGERIE_ENTRIES = [
    (
        "HP-VAE-GAN Generator (hierarchical patch VAE-GAN multi-scale residual pyramid)",
        "build_hp_vae_gan_generator",
        "example_input",
        "2021",
        "DC",
    ),
]
