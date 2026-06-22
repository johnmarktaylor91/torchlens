"""EDVR x4: video restoration with PCD alignment and TSA fusion.

Paper: "EDVR: Video Restoration with Enhanced Deformable Convolutional Networks",
Wang et al., CVPRW 2019.

EDVR's load-bearing idea is feature-level multi-frame restoration with
Pyramid-Cascading-Deformable (PCD) alignment followed by Temporal-Spatial
Attention (TSA) fusion.  This compact reconstruction keeps a three-level
coarse-to-fine learned-offset alignment path, temporal correlation attention
against the center frame, spatial attention, residual reconstruction, and late
two-stage pixel-shuffle x4 upsampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockNoBN(nn.Module):
    """Batch-norm-free residual block used by EDVR/BasicSR restorers."""

    def __init__(self, channels: int) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two convolutions with a residual connection."""

        return x + self.conv2(F.relu(self.conv1(x)))


def _warp_with_offset(feat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Warp features with a learned normalized offset field.

    Parameters
    ----------
    feat:
        Feature map to sample, shape ``(B, C, H, W)``.
    offset:
        Two-channel normalized offset, shape ``(B, 2, H, W)``.

    Returns
    -------
    torch.Tensor
        Bilinearly sampled feature map.
    """

    batch, _, height, width = feat.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=feat.device),
        torch.linspace(-1.0, 1.0, width, device=feat.device),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, height, width, 2)
    grid = base + offset.permute(0, 2, 3, 1)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=True)


class PCDAlign(nn.Module):
    """Compact Pyramid-Cascading alignment with learned offsets."""

    def __init__(self, channels: int) -> None:
        """Initialize three-level coarse-to-fine alignment."""

        super().__init__()
        self.offset3 = nn.Conv2d(channels * 2, 2, 3, padding=1)
        self.offset2 = nn.Conv2d(channels * 2 + 2, 2, 3, padding=1)
        self.offset1 = nn.Conv2d(channels * 2 + 2, 2, 3, padding=1)
        self.cascade = nn.Conv2d(channels * 2, 2, 3, padding=1)
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, nbr: list[torch.Tensor], ref: list[torch.Tensor]) -> torch.Tensor:
        """Align neighbor pyramid features to reference pyramid features."""

        off3 = torch.tanh(self.offset3(torch.cat([nbr[2], ref[2]], dim=1))) * 0.25
        up3 = F.interpolate(off3, scale_factor=2.0, mode="bilinear", align_corners=False)
        off2 = torch.tanh(self.offset2(torch.cat([nbr[1], ref[1], up3], dim=1))) * 0.25
        up2 = F.interpolate(off2, scale_factor=2.0, mode="bilinear", align_corners=False)
        off1 = torch.tanh(self.offset1(torch.cat([nbr[0], ref[0], up2], dim=1))) * 0.25
        aligned = _warp_with_offset(nbr[0], off1)
        cascade = torch.tanh(self.cascade(torch.cat([aligned, ref[0]], dim=1))) * 0.125
        return self.refine(_warp_with_offset(aligned, cascade))


class TSAFusion(nn.Module):
    """Temporal correlation attention plus spatial attention fusion."""

    def __init__(self, channels: int, frames: int) -> None:
        """Initialize TSA projections."""

        super().__init__()
        self.frames = frames
        self.temporal = nn.Conv2d(channels * frames, channels, 1)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels * frames, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, aligned: torch.Tensor) -> torch.Tensor:
        """Fuse aligned features of shape ``(B, T, C, H, W)``."""

        center = aligned[:, self.frames // 2]
        weights = []
        for idx in range(self.frames):
            corr = (aligned[:, idx] * center).mean(dim=1, keepdim=True)
            weights.append(torch.sigmoid(corr))
        weighted = aligned * torch.stack(weights, dim=1)
        flat = weighted.flatten(1, 2)
        return self.temporal(flat) * self.spatial(flat)


class EDVRCompact(nn.Module):
    """Compact EDVR x4 network."""

    def __init__(self, channels: int = 20, frames: int = 5) -> None:
        """Initialize EDVR components."""

        super().__init__()
        self.frames = frames
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.trunk = nn.Sequential(ResidualBlockNoBN(channels), ResidualBlockNoBN(channels))
        self.down1 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.align = PCDAlign(channels)
        self.fusion = TSAFusion(channels, frames)
        self.recon = nn.Sequential(ResidualBlockNoBN(channels), ResidualBlockNoBN(channels))
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore the center frame from ``(B, T, 3, H, W)`` low-resolution video."""

        batch, frames, channels, height, width = x.shape
        feat = self.trunk(self.head(x.reshape(batch * frames, channels, height, width)))
        feat = feat.view(batch, frames, -1, height, width)
        pyr1 = feat
        pyr2 = self.down1(feat.flatten(0, 1)).view(batch, frames, -1, height // 2, width // 2)
        pyr3 = self.down2(pyr2.flatten(0, 1)).view(batch, frames, -1, height // 4, width // 4)
        ref = [pyr1[:, frames // 2], pyr2[:, frames // 2], pyr3[:, frames // 2]]
        aligned = [
            self.align([pyr1[:, idx], pyr2[:, idx], pyr3[:, idx]], ref) for idx in range(frames)
        ]
        fused = self.fusion(torch.stack(aligned, dim=1))
        base = F.interpolate(
            x[:, frames // 2], scale_factor=4.0, mode="bilinear", align_corners=False
        )
        return self.up(self.recon(fused)) + base


def build() -> nn.Module:
    """Build compact EDVR x4."""

    return EDVRCompact()


def example_input() -> torch.Tensor:
    """Return a five-frame low-resolution RGB clip."""

    return torch.randn(1, 5, 3, 16, 16)


MENAGERIE_ENTRIES = [
    ("EDVR x4 (PCD alignment + TSA fusion)", "build", "example_input", "2019", "E7")
]
