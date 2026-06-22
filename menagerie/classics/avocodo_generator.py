"""Avocodo generator: HiFi-GAN-like multi-scale neural vocoder.

Paper: Bak et al., "Avocodo: Generative Adversarial Network for Artifact-Free
Vocoder", AAAI 2023.

Avocodo keeps a HiFi-GAN-style generator but emits intermediate waveforms at
multiple resolutions for collaborative multi-band/sub-band discrimination.
This compact module reconstructs that generator-side graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """HiFi-GAN multi-receptive-field residual block."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize dilated residual convolutions."""

        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual dilated convolutions."""

        y = F.leaky_relu(self.conv1(F.leaky_relu(x, 0.1)), 0.1)
        return x + self.conv2(y)


class UpBlock(nn.Module):
    """Upsampling block with multi-receptive-field residual stack."""

    def __init__(self, in_ch: int, out_ch: int, scale: int) -> None:
        """Initialize transposed convolution and residual blocks."""

        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, scale * 2, stride=scale, padding=scale // 2)
        self.res = nn.ModuleList([ResBlock(out_ch, dilation) for dilation in (1, 3, 5)])
        self.to_wave = nn.Conv1d(out_ch, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Upsample features and emit an intermediate waveform."""

        x = F.leaky_relu(self.up(x), 0.1)
        y = torch.stack([block(x) for block in self.res], dim=0).mean(dim=0)
        return y, torch.tanh(self.to_wave(y))


class AvocodoGenerator(nn.Module):
    """Compact Avocodo multi-scale generator."""

    def __init__(self) -> None:
        """Initialize mel projection and generator sub-blocks."""

        super().__init__()
        self.pre = nn.Conv1d(80, 64, 7, padding=3)
        self.blocks = nn.ModuleList([UpBlock(64, 48, 4), UpBlock(48, 32, 4), UpBlock(32, 16, 2)])

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesize multi-resolution waveforms from mel features."""

        x = self.pre(mel)
        waves = []
        for block in self.blocks:
            x, wave = block(x)
            waves.append(wave)
        return waves[0], waves[1], waves[2]


def build() -> nn.Module:
    """Build compact Avocodo generator."""

    return AvocodoGenerator()


def example_input() -> torch.Tensor:
    """Return a short mel-spectrogram input."""

    return torch.randn(1, 80, 16)


MENAGERIE_ENTRIES = [
    ("avocodo_generator", "build", "example_input", "2023", "audio"),
]
