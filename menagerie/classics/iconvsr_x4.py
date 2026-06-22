"""IconVSR: BasicVSR with information refill and coupled propagation.

Chan et al., 2021.
Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Chan_BasicVSR_The_Search_for_Essential_Components_in_Video_Super-Resolution_and_CVPR_2021_paper.html

IconVSR extends BasicVSR with two distinctive mechanisms: keyframe
information-refill from a local window and coupled backward/forward recurrent
propagation.  This compact model keeps both mechanisms and finishes with x4
pixel-shuffle video super-resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Small residual block for propagated video features."""

    def __init__(self, channels: int) -> None:
        """Initialize two convolutional layers."""

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolutional refinement."""

        return x + self.conv2(F.relu(self.conv1(x)))


class InformationRefill(nn.Module):
    """IconVSR keyframe refill from a three-frame local window."""

    def __init__(self, channels: int) -> None:
        """Initialize refill encoder.

        Parameters
        ----------
        channels:
            Internal feature width.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(9, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, frames: torch.Tensor, index: int) -> torch.Tensor:
        """Extract refill features around a keyframe index."""

        _, time, _, _, _ = frames.shape
        left = frames[:, max(index - 1, 0)]
        center = frames[:, index]
        right = frames[:, min(index + 1, time - 1)]
        return self.encoder(torch.cat([left, center, right], dim=1))


class IconVSRx4(nn.Module):
    """Compact IconVSR x4 video super-resolution model."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize feature, refill, propagation, fusion, and upsampling modules."""

        super().__init__()
        self.feat = nn.Conv2d(3, channels, 3, padding=1)
        self.refill = InformationRefill(channels)
        self.backward_cell = ResidualBlock(channels * 2)
        self.forward_cell = ResidualBlock(channels * 3)
        self.fuse = nn.Conv2d(channels * 3, channels, 1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 3 * 16, 3, padding=1),
            nn.PixelShuffle(4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run refill plus coupled backward/forward recurrent propagation."""

        b, time, _, h, w = x.shape
        shallow = [self.feat(x[:, i]) for i in range(time)]
        backward: list[torch.Tensor] = []
        state = x.new_zeros(b, shallow[0].shape[1], h, w)
        for i in reversed(range(time)):
            refill = self.refill(x, i) if i % 2 == 0 else torch.zeros_like(state)
            state = self.backward_cell(torch.cat([shallow[i] + refill, state], dim=1))[
                :, : shallow[i].shape[1]
            ]
            backward.insert(0, state)

        outputs: list[torch.Tensor] = []
        state = torch.zeros_like(backward[0])
        for i in range(time):
            coupled = torch.cat([shallow[i], backward[i], state], dim=1)
            state = self.forward_cell(coupled)[:, : shallow[i].shape[1]]
            outputs.append(self.up(self.fuse(torch.cat([shallow[i], backward[i], state], dim=1))))
        return torch.stack(outputs, dim=1)


def build_iconvsr_x4() -> nn.Module:
    """Build a compact random-init IconVSR x4 model."""

    return IconVSRx4()


def example_input() -> torch.Tensor:
    """Return a short low-resolution RGB video clip."""

    return torch.randn(1, 4, 3, 12, 12)


MENAGERIE_ENTRIES = [
    (
        "IconVSR x4 (information-refill coupled-propagation VSR)",
        "build_iconvsr_x4",
        "example_input",
        "2021",
        "video-restoration/super-resolution",
    ),
]
