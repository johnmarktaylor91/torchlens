"""MelGAN generator for non-autoregressive waveform synthesis.

Kumar et al. (2019), "MelGAN: Generative Adversarial Networks for Conditional
Waveform Synthesis."  The generator is a fully convolutional mel-to-waveform
network: a mel-spectrogram is projected to channels, repeatedly upsampled by
transposed convolutions, and refined after each upsampling step by residual
stacks with dilated 1-D convolutions.  This compact version keeps that generator
topology with small channels and two upsampling stages.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDilatedStack(nn.Module):
    """MelGAN residual stack with dilated temporal convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        """Initialize the residual stack.

        Parameters
        ----------
        channels:
            Number of waveform feature channels.
        kernel_size:
            Temporal convolution kernel size.
        """

        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=False),
                    nn.ReflectionPad1d(dilation * (kernel_size - 1) // 2),
                    nn.Conv1d(channels, channels, kernel_size, dilation=dilation),
                    nn.LeakyReLU(0.2, inplace=False),
                    nn.Conv1d(channels, channels, 1),
                )
                for dilation in (1, 3, 9)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual dilated convolutions.

        Parameters
        ----------
        x:
            Temporal feature map.

        Returns
        -------
        torch.Tensor
            Refined temporal feature map.
        """

        for layer in self.layers:
            x = x + layer(x)
        return x


class MelGANGenerator(nn.Module):
    """Compact MelGAN mel-spectrogram to waveform generator."""

    def __init__(self, mel_channels: int = 80, channels: int = 48) -> None:
        """Initialize the generator.

        Parameters
        ----------
        mel_channels:
            Number of mel-spectrogram bins.
        channels:
            Base hidden channel width.
        """

        super().__init__()
        self.pre = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(mel_channels, channels, 7),
        )
        self.up1 = nn.ConvTranspose1d(channels, channels // 2, 8, stride=4, padding=2)
        self.res1 = ResidualDilatedStack(channels // 2)
        self.up2 = nn.ConvTranspose1d(channels // 2, channels // 4, 8, stride=4, padding=2)
        self.res2 = ResidualDilatedStack(channels // 4)
        self.post = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.ReflectionPad1d(3),
            nn.Conv1d(channels // 4, 1, 7),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize waveform samples from mel frames.

        Parameters
        ----------
        mel:
            Mel-spectrogram tensor of shape ``(B, 80, T)``.

        Returns
        -------
        torch.Tensor
            Waveform tensor of shape ``(B, 1, T * 16)``.
        """

        x = self.pre(mel)
        x = self.res1(F.leaky_relu(self.up1(x), 0.2))
        x = self.res2(F.leaky_relu(self.up2(x), 0.2))
        return self.post(x)


def build() -> nn.Module:
    """Build the compact MelGAN generator.

    Returns
    -------
    nn.Module
        Random-init MelGAN generator in evaluation mode.
    """

    return MelGANGenerator().eval()


def example_input() -> torch.Tensor:
    """Return a short mel-spectrogram.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 80, 12)``.
    """

    return torch.randn(1, 80, 12)


MENAGERIE_ENTRIES = [
    ("MelGANGenerator", "build", "example_input", "2019", "DC"),
]
