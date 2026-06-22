"""MAXIM compact multi-axis MLP image-restoration network.

Tu et al., 2022, "MAXIM: Multi-Axis MLP for Image Processing".  MAXIM is a
UNet-shaped restoration backbone with multi-axis gated MLP mixing and
cross-gating blocks.  This compact version traces local/global axis mixing,
encoder-decoder skip fusion, and residual RGB restoration for deblurring.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MultiAxisGatedMLP(nn.Module):
    """Spatial gated MLP mixing horizontal and vertical axes."""

    def __init__(self, channels: int) -> None:
        """Initialize axis mixers.

        Parameters
        ----------
        channels:
            Feature channels.
        """
        super().__init__()
        self.proj = nn.Conv2d(channels, channels * 2, 1)
        self.row = nn.Conv1d(channels, channels, 1)
        self.col = nn.Conv1d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Mix features along image axes.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Mixed feature map.
        """
        u, gate = self.proj(x).chunk(2, dim=1)
        row = self.row(u.mean(dim=2)).unsqueeze(2)
        col = self.col(u.mean(dim=3)).unsqueeze(3)
        return self.out(u * torch.sigmoid(gate + row + col))


class MAXIMRestorer(nn.Module):
    """Compact MAXIM image restorer."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize U-Net and gated MLP blocks.

        Parameters
        ----------
        channels:
            Base channel count.
        """
        super().__init__()
        self.inp = nn.Conv2d(3, channels, 3, padding=1)
        self.enc = MultiAxisGatedMLP(channels)
        self.down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.mid = MultiAxisGatedMLP(channels)
        self.cross = nn.Conv2d(channels * 2, channels, 1)
        self.dec = MultiAxisGatedMLP(channels)
        self.out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        """Restore an RGB image.

        Parameters
        ----------
        image:
            Degraded RGB tensor.

        Returns
        -------
        Tensor
            Restored RGB tensor.
        """
        skip = self.enc(self.inp(image))
        low = self.mid(self.down(skip))
        up = F.interpolate(low, size=skip.shape[-2:], mode="nearest")
        fused = self.cross(torch.cat((skip, up), dim=1))
        return image + self.out(self.dec(fused))


def build() -> nn.Module:
    """Build compact MAXIM deblurring model.

    Returns
    -------
    nn.Module
        Random-initialized MAXIM restorer.
    """
    return MAXIMRestorer().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("maxim_s1_deblurring", "build", "example_input", "2022", "DC"),
]
