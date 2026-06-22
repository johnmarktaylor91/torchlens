"""MUNIT: Multimodal Unsupervised Image-to-Image Translation.

Huang, Liu, Belongie, Kautz (Cornell / NVIDIA) ECCV 2018.  arXiv:1804.04732.
Source: https://github.com/NVlabs/MUNIT

MUNIT's distinctive primitives:
  - **Disentangled content + style representation**: an image is encoded into a
    domain-invariant *content code* c (spatial feature map, shared across domains)
    and a domain-specific *style code* s (global vector).
  - **Content encoder**: image -> downsampled spatial feature map (residual blocks
    with Instance Norm).
  - **Style encoder**: image -> global style vector (strided conv + global avg pool).
  - **AdaIN (Adaptive Instance Normalization) decoder**: the style code is fed through
    a learned MLP to produce per-layer AdaIN parameters (scale gamma, bias beta).
    The decoder uses AdaIN blocks: IN normalization + affine transform from style MLP.
  - **Cross-domain translation**: content from domain A + style from domain B -> image
    in domain B.

Here we reproduce the generator (encoder pair + decoder) for a single domain,
wrapped to take a single input image and return a reconstructed image.
For clean tracing, the wrapper concatenates (z_content from enc_content,
z_style from enc_style) internally.

Random init, CPU, compact dims.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentEncoder(nn.Module):
    """MUNIT content encoder: image -> domain-invariant spatial feature map."""

    def __init__(self, in_c: int = 3, nf: int = 16, n_res: int = 2) -> None:
        super().__init__()
        # Downsampling: 32->16->8
        self.down = nn.Sequential(
            nn.Conv2d(in_c, nf, 7, 1, 3),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )
        # Residual blocks
        res = []
        for _ in range(n_res):
            res += [
                nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
                nn.InstanceNorm2d(nf * 2),
                nn.ReLU(True),
                nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
                nn.InstanceNorm2d(nf * 2),
            ]
        self.res = nn.Sequential(*res)
        self.nf2 = nf * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        r = self.res(h)
        return h + r  # simplified residual (each block adds)


class StyleEncoder(nn.Module):
    """MUNIT style encoder: image -> global style vector via strided convs + avg pool."""

    def __init__(self, in_c: int = 3, nf: int = 16, style_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(nf * 2, style_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)


class AdaINBlock(nn.Module):
    """Adaptive Instance Normalization block: IN + affine from style MLP."""

    def __init__(self, c: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(c, affine=False)
        self.fc_gamma = nn.Linear(style_dim, c)
        self.fc_beta = nn.Linear(style_dim, c)
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        gamma = self.fc_gamma(s).unsqueeze(-1).unsqueeze(-1)
        beta = self.fc_beta(s).unsqueeze(-1).unsqueeze(-1)
        # First AdaIN conv
        out = self.conv1(x)
        out = self.norm(out)
        out = out * (1.0 + gamma) + beta
        out = F.relu(out)
        # Second conv
        out = self.conv2(out)
        out = self.norm(out)
        out = out * (1.0 + gamma) + beta
        return x + out  # residual


class MUNITDecoder(nn.Module):
    """MUNIT decoder: content feature + style code -> image via AdaIN residual blocks + upsample."""

    def __init__(self, content_c: int, style_dim: int, nf: int = 16, n_res: int = 2) -> None:
        super().__init__()
        # AdaIN residual blocks
        self.res_blocks = nn.ModuleList([AdaINBlock(content_c, style_dim) for _ in range(n_res)])
        # Upsampling: 8->16->32
        self.up1 = nn.Sequential(
            nn.Conv2d(content_c, nf, 3, 1, 1),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(nf, 3, 7, 1, 3),
            nn.Tanh(),
        )

    def forward(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = c
        for blk in self.res_blocks:
            x = blk(x, s)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.up2(x)
        return self.to_rgb(x)


class MUNITGenerator(nn.Module):
    """Full MUNIT generator: content encoder + style encoder + AdaIN decoder."""

    def __init__(
        self,
        in_c: int = 3,
        nf: int = 16,
        style_dim: int = 16,
        n_res: int = 2,
    ) -> None:
        super().__init__()
        self.enc_content = ContentEncoder(in_c, nf, n_res)
        content_c = nf * 2
        self.enc_style = StyleEncoder(in_c, nf, style_dim)
        self.decoder = MUNITDecoder(content_c, style_dim, nf, n_res)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.enc_content(x)
        s = self.enc_style(x)
        return self.decoder(c, s)


def build_munit_generator() -> nn.Module:
    return MUNITGenerator()


def example_input() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "MUNIT Generator (content encoder + style encoder + AdaIN residual decoder)",
        "build_munit_generator",
        "example_input",
        "2018",
        "DC",
    ),
]
