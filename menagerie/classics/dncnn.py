"""DnCNN: Beyond a Gaussian Denoiser -- Residual Learning of Deep CNN for Image Denoising.

Zhang et al., IEEE TIP 2017.
Paper: https://arxiv.org/abs/1608.03981
Source: https://github.com/SaoYan/DnCNN-PyTorch (and cszn/DnCNN)

DnCNN is a plain feed-forward residual-learning denoiser: a single Conv+ReLU
head, a body of (Conv + BatchNorm + ReLU) blocks, and a Conv tail that predicts
the residual (noise), which is subtracted from the input. The two canonical
configurations are:

* DnCNN-S (specific noise level): depth 17, grayscale.
* DnCNN-B (blind / unknown noise level): depth 20, grayscale.

Both use 64 feature channels and 3x3 convolutions throughout. This is a faithful
random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """DnCNN residual-learning denoiser.

    Parameters
    ----------
    depth:
        Total number of convolutional layers (17 for DnCNN-S, 20 for DnCNN-B).
    n_channels:
        Number of feature channels in the body (64 in the original).
    image_channels:
        Input/output image channels (1 for grayscale).
    """

    def __init__(self, depth: int = 17, n_channels: int = 64, image_channels: int = 1) -> None:
        super().__init__()
        kernel_size = 3
        padding = 1
        layers: list[nn.Module] = []
        # Head: Conv + ReLU (no BN).
        layers.append(
            nn.Conv2d(image_channels, n_channels, kernel_size, padding=padding, bias=True)
        )
        layers.append(nn.ReLU(inplace=True))
        # Body: (Conv + BN + ReLU) x (depth - 2).
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=False)
            )
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        # Tail: Conv (predicts residual noise).
        layers.append(
            nn.Conv2d(n_channels, image_channels, kernel_size, padding=padding, bias=False)
        )
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.dncnn(x)
        return x - residual


def build_dncnn_s() -> nn.Module:
    """Build DnCNN-S (depth 17, grayscale, fixed-noise denoiser)."""
    return DnCNN(depth=17, n_channels=64, image_channels=1)


def build_dncnn_b() -> nn.Module:
    """Build DnCNN-B (depth 20, grayscale, blind/unknown-noise denoiser)."""
    return DnCNN(depth=20, n_channels=64, image_channels=1)


def build_dncnn_color() -> nn.Module:
    """Build FDnCNN Color (depth 17, 3-channel color denoiser)."""
    return DnCNN(depth=17, n_channels=64, image_channels=3)


def build() -> nn.Module:
    """Default builder (DnCNN-S, depth 17)."""
    return build_dncnn_s()


def example_input() -> torch.Tensor:
    """Example grayscale noisy-image patch tensor ``(1, 1, 64, 64)``."""
    return torch.randn(1, 1, 64, 64)


def example_input_color() -> torch.Tensor:
    """Example color noisy-image patch tensor ``(1, 3, 64, 64)``."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "DnCNN-S (depth-17 residual-learning Gaussian denoiser)",
        "build_dncnn_s",
        "example_input",
        "2017",
        "DC",
    ),
    (
        "DnCNN-B (depth-20 blind residual-learning denoiser)",
        "build_dncnn_b",
        "example_input",
        "2017",
        "DC",
    ),
    (
        "FDnCNN Color (3-channel color residual denoiser)",
        "build_dncnn_color",
        "example_input_color",
        "2017",
        "DC",
    ),
]
