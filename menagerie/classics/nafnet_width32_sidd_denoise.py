"""NAFNet width32 SIDD denoising: nonlinear-activation-free restoration.

Paper: "Simple Baselines for Image Restoration", Chen et al., ECCV 2022.

NAFNet removes conventional nonlinear activations and keeps restoration power
through LayerNorm, SimpleGate channel splitting, Simplified Channel Attention
(SCA), depthwise convolution, and learned residual scaling.  This compact model
uses width 32 and a small encoder-decoder denoising topology.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for image tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        """Initialize 2D layer normalization."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize channel statistics at each pixel."""

        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    """NAFNet SimpleGate: split channels and multiply halves."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multiplicative gate without an activation function."""

        left, right = x.chunk(2, dim=1)
        return left * right


class NAFBlock(nn.Module):
    """Nonlinear-activation-free restoration block."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        """Initialize a NAFBlock."""

        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden, hidden, 1))
        self.conv2 = nn.Conv2d(hidden, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn2 = nn.Conv2d(hidden, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the NAFBlock with learned residual scaling."""

        y = self.sg(self.dw(self.conv1(self.norm1(x))))
        y = self.conv2(y * self.sca(y))
        x = x + y * self.beta
        z = self.ffn2(self.sg(self.ffn1(self.norm2(x))))
        return x + z * self.gamma


class NAFNetCompact(nn.Module):
    """Compact width-32 NAFNet denoiser."""

    def __init__(self, width: int = 32) -> None:
        """Initialize the denoising network."""

        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.enc = NAFBlock(width)
        self.down = nn.Conv2d(width, width * 2, 2, stride=2)
        self.mid = NAFBlock(width * 2)
        self.up = nn.Sequential(nn.Conv2d(width * 2, width * 4, 1), nn.PixelShuffle(2))
        self.dec = NAFBlock(width)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise an RGB image."""

        enc = self.enc(self.intro(x))
        mid = self.mid(self.down(enc))
        dec = self.dec(self.up(mid) + enc)
        return x + self.ending(dec)


def build() -> nn.Module:
    """Build compact NAFNet width32 SIDD denoiser."""

    return NAFNetCompact()


def example_input() -> torch.Tensor:
    """Return a small noisy RGB image."""

    return torch.randn(1, 3, 24, 24)


MENAGERIE_ENTRIES = [
    ("NAFNet width32 SIDD denoise (SimpleGate + SCA)", "build", "example_input", "2022", "E5")
]
