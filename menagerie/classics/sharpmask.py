"""SharpMask, 2016, Pinheiro, Lin, Collobert, and Dollar.

Paper: Learning to Refine Object Segments.
DeepMask-style coarse mask features are refined top-down by combining
upsampled coarse activations with progressively shallower skip features.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RefineModule(nn.Module):
    """SharpMask top-down refinement block."""

    def __init__(self, skip_channels: int, coarse_channels: int, out_channels: int) -> None:
        """Initialize vertical and horizontal refinement paths.

        Parameters
        ----------
        skip_channels:
            Channels in the encoder skip feature.
        coarse_channels:
            Channels in the coarse top-down feature.
        out_channels:
            Channels emitted by the refinement block.
        """
        super().__init__()
        self.vertical = nn.Conv2d(skip_channels, out_channels, kernel_size=3, padding=1)
        self.horizontal = nn.Conv2d(coarse_channels, out_channels, kernel_size=3, padding=1)
        self.output = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, skip: Tensor, coarse: Tensor) -> Tensor:
        """Fuse a skip feature with an upsampled coarse feature.

        Parameters
        ----------
        skip:
            Encoder skip tensor.
        coarse:
            Coarser top-down tensor.

        Returns
        -------
        Tensor
            Refined feature tensor.
        """
        up = F.interpolate(coarse, size=skip.shape[-2:], mode="nearest")
        fused = torch.relu(self.vertical(skip) + self.horizontal(up))
        return torch.relu(self.output(fused))


class SharpMask(nn.Module):
    """Small DeepMask trunk with SharpMask refinement path."""

    def __init__(self) -> None:
        """Initialize trunk, refinement modules, and mask head."""
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())
        self.coarse = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.refine2 = RefineModule(16, 32, 16)
        self.refine1 = RefineModule(8, 16, 8)
        self.mask = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict a refined object mask.

        Parameters
        ----------
        x:
            RGB image tensor ``(B, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Mask logits at input resolution.
        """
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        deep = self.enc3(skip2)
        coarse = self.coarse(deep)
        refined = self.refine2(skip2, coarse)
        refined = self.refine1(skip1, refined)
        return self.mask(refined)


def build() -> nn.Module:
    """Build a compact SharpMask module.

    Returns
    -------
    nn.Module
        Random-initialized SharpMask.
    """
    return SharpMask()


def example_input() -> Tensor:
    """Return a traceable RGB image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 224, 224)``.
    """
    return torch.randn(1, 3, 224, 224)
