"""VocGAN: Multi-scale hierarchical vocoder GAN generator.

Yang et al., "VocGAN: A High-Fidelity Real-Time Vocoder with a Hierarchically-nested
Adversarial Network", Interspeech 2020. arXiv:2007.15256.
Source: https://github.com/rishikksh20/VocGAN

VocGAN key contributions:
  1. MULTI-SCALE HIERARCHICAL GENERATOR: the generator progressively upsamples
     mel-spectrograms to waveform through a cascade of upsample stages. At each
     stage, a transposed convolution increases time resolution. A parallel WaveNet
     discriminator hierarchy provides adversarial feedback at each scale.
  2. MULTI-RECEPTIVE FIELD (MRF) STACK: at each upsample stage, instead of a single
     conv, a set of dilated convolutions with different dilation rates run in parallel
     and their outputs are summed. This is the VocGAN/HiFiGAN signature op --
     multi-scale receptive fields capture different temporal patterns.
  3. The discriminator hierarchy mirrors the generator stages, providing adversarial
     loss at each resolution.

Architecture here: input mel (1, 80, 32) -> 1D conv -> 3 upsample stages
(each ConvTranspose1d + MRF stack) -> output waveform (1, 1, T*8).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Multi-Receptive Field (MRF) stack
# ============================================================


class MRFResBlock(nn.Module):
    """MRF residual block: dilated 1D conv with a single dilation factor."""

    def __init__(self, channels: int, dilation: int = 1, kernel_size: int = 3) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(x, 0.2)
        h = F.leaky_relu(self.conv1(h), 0.2)
        h = self.conv2(h)
        return x + h


class MultiReceptiveField(nn.Module):
    """VocGAN Multi-Receptive Field stack: N parallel dilated conv branches summed.

    Signature op: separate dilation-pattern branches run in parallel, outputs summed.
    In VocGAN/HiFiGAN: 3 branches with dilation rates [1,2,4], [2,4,8], [3,6,12].
    Here simplified to 2 branches with 2 blocks each.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Branch 1: dilations [1, 2]
        self.branch1 = nn.Sequential(
            MRFResBlock(channels, dilation=1),
            MRFResBlock(channels, dilation=2),
        )
        # Branch 2: dilations [3, 6]
        self.branch2 = nn.Sequential(
            MRFResBlock(channels, dilation=3),
            MRFResBlock(channels, dilation=6),
        )
        # Branch 3: dilations [2, 4]
        self.branch3 = nn.Sequential(
            MRFResBlock(channels, dilation=2),
            MRFResBlock(channels, dilation=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run all branches in parallel and sum (VocGAN MRF fusion)
        return self.branch1(x) + self.branch2(x) + self.branch3(x)


# ============================================================
# MODULE 26: vocgan_generator
# ============================================================


class VocGANGenerator(nn.Module):
    """VocGAN Generator: mel-spectrogram -> waveform.

    Architecture:
    1. Input conv: (1, 80, T) -> (1, ch, T) via 1D conv
    2. Three upsample stages, each:
       - ConvTranspose1d: doubles T, halves ch
       - MultiReceptiveField (MRF): dilated conv branches in parallel
    3. Output conv: -> (1, 1, T*8) waveform with Tanh

    Signature: multi-scale upsampling with MRF (parallel dilated conv branches) at each stage.
    """

    def __init__(self, mel_channels: int = 80, ch: int = 32) -> None:
        super().__init__()
        # Input projection
        self.in_conv = nn.Conv1d(mel_channels, ch, kernel_size=7, padding=3)
        # Upsample stage 1: T -> 2T, ch -> ch//2
        self.up1 = nn.ConvTranspose1d(ch, ch // 2, kernel_size=8, stride=2, padding=3)
        self.mrf1 = MultiReceptiveField(ch // 2)
        # Upsample stage 2: 2T -> 4T, ch//2 -> ch//4
        self.up2 = nn.ConvTranspose1d(ch // 2, ch // 4, kernel_size=8, stride=2, padding=3)
        self.mrf2 = MultiReceptiveField(ch // 4)
        # Upsample stage 3: 4T -> 8T, ch//4 -> ch//8
        self.up3 = nn.ConvTranspose1d(ch // 4, ch // 8, kernel_size=8, stride=2, padding=3)
        self.mrf3 = MultiReceptiveField(ch // 8)
        # Output conv: -> waveform
        self.out_conv = nn.Conv1d(ch // 8, 1, kernel_size=7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, mel_channels, T)
        x = self.in_conv(mel)
        # Upsample stage 1
        x = F.leaky_relu(self.up1(x), 0.2)
        x = self.mrf1(x)
        # Upsample stage 2
        x = F.leaky_relu(self.up2(x), 0.2)
        x = self.mrf2(x)
        # Upsample stage 3
        x = F.leaky_relu(self.up3(x), 0.2)
        x = self.mrf3(x)
        # Output: waveform
        return torch.tanh(self.out_conv(F.leaky_relu(x, 0.2)))


def build_vocgan_generator() -> nn.Module:
    return VocGANGenerator(mel_channels=80, ch=32)


def example_vocgan_generator() -> torch.Tensor:
    return torch.randn(1, 80, 32)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "vocgan_generator",
        "build_vocgan_generator",
        "example_vocgan_generator",
        "2020",
        "DC",
    ),
]
