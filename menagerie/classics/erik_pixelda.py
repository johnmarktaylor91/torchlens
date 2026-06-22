"""erik_pixelda: PixelDA generative pixel-level domain adaptation.

Bousmalis et al., CVPR 2017, "Unsupervised Pixel-Level Domain Adaptation with
Generative Adversarial Networks".  PixelDA learns a source-to-target pixel
generator conditioned on source images and noise, while a task classifier keeps
semantic labels usable and a domain discriminator adversarially judges adapted
versus target-style images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-convolution residual block used in the PixelDA generator."""

    def __init__(self, channels: int) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual refinement.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Refined feature map.
        """

        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class PixelDAGenerator(nn.Module):
    """Source-image plus noise generator for target-style images."""

    def __init__(self, noise_channels: int = 4, width: int = 32) -> None:
        """Initialize the PixelDA generator.

        Parameters
        ----------
        noise_channels:
            Number of spatial noise channels concatenated to the source image.
        width:
            Internal convolution width.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3 + noise_channels, width, 7, padding=3),
            nn.InstanceNorm2d(width, affine=True),
            nn.ReLU(inplace=True),
        )
        self.resblocks = nn.Sequential(ResidualBlock(width), ResidualBlock(width))
        self.out = nn.Conv2d(width, 3, 7, padding=3)

    def forward(self, source: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Translate source pixels toward the target domain.

        Parameters
        ----------
        source:
            Source-domain RGB image.
        noise:
            Spatial stochastic code.

        Returns
        -------
        torch.Tensor
            Adapted RGB image in ``[-1, 1]``.
        """

        x = torch.cat([source, noise], dim=1)
        delta = self.out(self.resblocks(self.stem(x)))
        return torch.tanh(source + delta)


class PixelDANet(nn.Module):
    """Full PixelDA graph with generator, task head, and domain head."""

    def __init__(self, num_classes: int = 10, noise_channels: int = 4) -> None:
        """Initialize PixelDA components.

        Parameters
        ----------
        num_classes:
            Number of task classes.
        noise_channels:
            Number of generator noise channels.
        """

        super().__init__()
        self.generator = PixelDAGenerator(noise_channels=noise_channels)
        self.task = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, num_classes),
        )
        self.domain = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(48, 1, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, source: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run PixelDA adaptation and both supervised/adversarial heads.

        Parameters
        ----------
        source:
            Source-domain RGB image.
        noise:
            Spatial noise map.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Adapted image, task logits, and domain logits.
        """

        adapted = self.generator(source, noise)
        return adapted, self.task(adapted), self.domain(adapted)


def build() -> nn.Module:
    """Build a compact PixelDA model.

    Returns
    -------
    nn.Module
        Random-initialized PixelDA generator/classifier/discriminator graph.
    """

    return PixelDANet().eval()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a source image and spatial noise map.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Source image ``(1, 3, 32, 32)`` and noise ``(1, 4, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32), torch.randn(1, 4, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "erik_pixelda",
        "build",
        "example_input",
        "2017",
        "DC",
    ),
]
