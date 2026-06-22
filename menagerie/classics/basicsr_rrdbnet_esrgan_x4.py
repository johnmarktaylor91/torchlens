"""RRDBNet / ESRGAN / Real-ESRGAN compact generator family.

Paper: "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks",
Wang et al., ECCVW 2018.  Real-ESRGAN (Wang et al., ICCVW 2021) keeps the
RRDBNet generator and trains it with synthetic real-world degradations.

The compact reconstruction keeps the generator's defining primitive:
Residual-in-Residual Dense Blocks without batch normalization.  Scale x2 and x4
variants share the same trunk with different pixel-shuffle tails; anime/video
variants are represented as narrower/fewer-block RRDBNet configurations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Five-convolution dense block used inside RRDB."""

    def __init__(self, channels: int, growth: int = 16) -> None:
        """Initialize a residual dense block."""

        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + growth * 2, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + growth * 3, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + growth * 4, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dense concatenation and local residual scaling."""

        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(torch.cat([x, x1], dim=1)), 0.2)
        x3 = F.leaky_relu(self.conv3(torch.cat([x, x1, x2], dim=1)), 0.2)
        x4 = F.leaky_relu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)), 0.2)
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * 0.2


class RRDB(nn.Module):
    """Residual-in-residual dense block."""

    def __init__(self, channels: int, growth: int = 16) -> None:
        """Initialize an RRDB block."""

        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth)
        self.rdb2 = ResidualDenseBlock(channels, growth)
        self.rdb3 = ResidualDenseBlock(channels, growth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply three dense blocks with outer residual scaling."""

        return x + self.rdb3(self.rdb2(self.rdb1(x))) * 0.2


class RRDBNetCompact(nn.Module):
    """Compact RRDBNet generator."""

    def __init__(self, scale: int = 4, channels: int = 32, blocks: int = 2) -> None:
        """Initialize compact RRDBNet."""

        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.trunk = nn.Sequential(*[RRDB(channels) for _ in range(blocks)])
        self.trunk_conv = nn.Conv2d(channels, channels, 3, padding=1)
        up_layers: list[nn.Module] = []
        for _ in range(1 if scale == 2 else 2):
            up_layers.extend(
                [
                    nn.Conv2d(channels, channels * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2),
                ]
            )
        up_layers.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.up = nn.Sequential(*up_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve an RGB image."""

        feat = self.head(x)
        trunk = self.trunk_conv(self.trunk(feat)) + feat
        return self.up(trunk)


def build_basicsr_rrdbnet_esrgan_x4() -> nn.Module:
    """Build compact ESRGAN RRDBNet x4."""

    return RRDBNetCompact(scale=4, channels=32, blocks=2)


def build_realesrgan_x4plus_rrdbnet() -> nn.Module:
    """Build compact Real-ESRGAN RRDBNet x4plus."""

    return RRDBNetCompact(scale=4, channels=32, blocks=2)


def build_realesrgan_x2plus_rrdbnet() -> nn.Module:
    """Build compact Real-ESRGAN RRDBNet x2plus."""

    return RRDBNetCompact(scale=2, channels=32, blocks=2)


def build_realesrgan_anime_6b() -> nn.Module:
    """Build compact six-block-style anime RRDBNet variant."""

    return RRDBNetCompact(scale=4, channels=24, blocks=3)


def example_input() -> torch.Tensor:
    """Return a small low-resolution RGB image."""

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "ESRGAN RRDBNet x4 (residual-in-residual dense SR generator)",
        "build_basicsr_rrdbnet_esrgan_x4",
        "example_input",
        "2018",
        "E5",
    ),
    (
        "Real-ESRGAN RRDBNet x4plus (synthetic-degradation RRDBNet generator)",
        "build_realesrgan_x4plus_rrdbnet",
        "example_input",
        "2021",
        "E7",
    ),
    (
        "Real-ESRGAN RRDBNet x2plus (x2 RRDBNet generator)",
        "build_realesrgan_x2plus_rrdbnet",
        "example_input",
        "2021",
        "E7",
    ),
    (
        "Real-ESRGAN anime 6B (compact anime RRDBNet generator)",
        "build_realesrgan_anime_6b",
        "example_input",
        "2021",
        "E7",
    ),
]
