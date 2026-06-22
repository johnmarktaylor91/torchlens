"""PaddleSeg U2Net: nested U-structure salient object segmentation.

Paper: Qin et al. 2020, "U2-Net: Going Deeper with Nested U-Structure for
Salient Object Detection".  The compact model keeps residual U-blocks nested
inside an outer U-shaped encoder-decoder with side output fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSU(nn.Module):
    """Small residual U-block."""

    def __init__(self, channels: int) -> None:
        """Initialize the nested encoder-decoder block.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.in_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.up = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual nested U-block.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        xin = F.relu(self.in_conv(x))
        low = F.relu(self.down(xin))
        mid = F.relu(self.mid(low))
        up = F.interpolate(mid, size=xin.shape[-2:], mode="bilinear", align_corners=False)
        return x + F.relu(self.up(torch.cat([xin, up], dim=1)))


class U2Net(nn.Module):
    """Compact U2Net with side-output fusion."""

    def __init__(self) -> None:
        """Initialize outer U-shape and nested RSU blocks."""

        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.enc = RSU(16)
        self.down = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.bottleneck = RSU(16)
        self.dec = RSU(16)
        self.side1 = nn.Conv2d(16, 1, 1)
        self.side2 = nn.Conv2d(16, 1, 1)
        self.fuse = nn.Conv2d(2, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict a salient-object mask with nested U features.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        torch.Tensor
            Saliency mask logits.
        """

        e = self.enc(F.relu(self.stem(image)))
        b = self.bottleneck(F.relu(self.down(e)))
        up = F.interpolate(b, size=e.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec(up + e)
        side2 = F.interpolate(
            self.side2(b), size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.fuse(torch.cat([self.side1(d), side2], dim=1))


def build() -> nn.Module:
    """Build compact PaddleSeg U2Net.

    Returns
    -------
    nn.Module
        Random-initialized U2Net.
    """

    return U2Net()


def example_input() -> torch.Tensor:
    """Create a small RGB image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("paddleseg_u2net", "build", "example_input", "2020", "E5")]
