"""BasicVSR: bidirectional propagation, alignment, aggregation, upsampling.

Paper: "BasicVSR: The Search for Essential Components in Video Super-Resolution
and Beyond", Chan et al., CVPR 2021.

The compact reconstruction keeps BasicVSR's defining recurrent structure:
features are propagated backward and forward through a frame sequence, aligned
with simple flow-style warping, fused at each time step, and pixel-shuffle
upsampled.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-convolution residual block without batch normalization."""

    def __init__(self, channels: int) -> None:
        """Initialize the residual block."""

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block."""

        return x + self.conv2(F.relu(self.conv1(x)))


def _make_grid(x: torch.Tensor) -> torch.Tensor:
    """Create a normalized sampling grid for ``grid_sample``."""

    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device),
        torch.linspace(-1.0, 1.0, w, device=x.device),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, h, w, 2)


class FlowAlign(nn.Module):
    """Feature alignment by predicted two-channel flow and bilinear warping."""

    def __init__(self, channels: int) -> None:
        """Initialize the alignment module."""

        super().__init__()
        self.flow = nn.Conv2d(channels + 3, 2, 3, padding=1)

    def forward(self, feat: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """Warp propagated features toward the current frame."""

        flow = torch.tanh(self.flow(torch.cat([feat, frame], dim=1))) * 0.25
        grid = _make_grid(feat) + flow.permute(0, 2, 3, 1)
        return F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=True)


class BasicVSRCompact(nn.Module):
    """Compact BasicVSR x4 video super-resolution network."""

    def __init__(self, channels: int = 24, blocks: int = 2) -> None:
        """Initialize the compact network."""

        super().__init__()
        self.feat = nn.Conv2d(3, channels, 3, padding=1)
        self.align_b = FlowAlign(channels)
        self.align_f = FlowAlign(channels)
        self.backbone = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.fuse = nn.Conv2d(channels * 2, channels, 1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve a short video clip of shape ``(B, T, 3, H, W)``."""

        b, t, c, h, w = x.shape
        frames = x.reshape(b * t, c, h, w)
        feats = self.feat(frames).view(b, t, -1, h, w)
        backward: list[torch.Tensor] = []
        state = torch.zeros_like(feats[:, 0])
        for idx in range(t - 1, -1, -1):
            state = self.backbone(feats[:, idx] + self.align_b(state, x[:, idx]))
            backward.append(state)
        backward = list(reversed(backward))
        outputs = []
        state = torch.zeros_like(feats[:, 0])
        for idx in range(t):
            state = self.backbone(feats[:, idx] + self.align_f(state, x[:, idx]))
            fused = self.fuse(torch.cat([state, backward[idx]], dim=1))
            outputs.append(
                self.up(fused) + F.interpolate(x[:, idx], scale_factor=4, mode="bilinear")
            )
        return torch.stack(outputs, dim=1)


def build_basicvsr_x4() -> nn.Module:
    """Build compact BasicVSR x4."""

    return BasicVSRCompact()


def example_input() -> torch.Tensor:
    """Return a short low-resolution RGB video clip."""

    return torch.randn(1, 3, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "BasicVSR x4 (bidirectional propagation/alignment/aggregation)",
        "build_basicvsr_x4",
        "example_input",
        "2021",
        "E7",
    )
]
