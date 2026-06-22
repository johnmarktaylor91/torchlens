"""PaddleDetection CLRNet: cross-layer refinement network for lane detection.

Zheng et al. (CVPR 2022), "CLRNet: Cross Layer Refinement Network for Lane
Detection".  CLRNet first detects lanes from high-level semantic features, then
refines them with lower-level detail features and a ROIGather-style context
aggregation step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneBackbone(nn.Module):
    """Backbone that exposes low and high lane features."""

    def __init__(self, width: int = 24) -> None:
        """Initialize the lane backbone.

        Parameters
        ----------
        width:
            Base channel count.
        """

        super().__init__()
        self.low = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=False),
        )
        self.high = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return low-level and high-level feature maps.

        Parameters
        ----------
        x:
            Road image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Low-detail and high-semantic features.
        """

        low = self.low(x)
        return low, self.high(low)


class ROIGather(nn.Module):
    """Attention-style global context gather for lane proposals."""

    def __init__(self, channels: int = 48, lanes: int = 6) -> None:
        """Initialize ROIGather.

        Parameters
        ----------
        channels:
            High-level feature channels.
        lanes:
            Number of lane proposals.
        """

        super().__init__()
        self.query = nn.Parameter(torch.randn(lanes, channels) * 0.02)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Gather global context for each lane proposal.

        Parameters
        ----------
        feat:
            High-level feature map.

        Returns
        -------
        torch.Tensor
            Lane context tensor of shape ``(batch, lanes, channels)``.
        """

        batch, channels, height, width = feat.shape
        key = self.key(feat).flatten(2)
        value = self.value(feat).flatten(2).transpose(1, 2)
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        scores = torch.matmul(query, key) / (channels**0.5)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value).view(batch, -1, channels)


class PaddleDetCLRNet(nn.Module):
    """Compact CLRNet lane detector."""

    def __init__(self, lanes: int = 6, points: int = 8) -> None:
        """Initialize CLRNet.

        Parameters
        ----------
        lanes:
            Number of lane proposals.
        points:
            Number of sampled y-points per lane.
        """

        super().__init__()
        self.backbone = LaneBackbone()
        self.gather = ROIGather(channels=48, lanes=lanes)
        self.coarse = nn.Linear(48, points)
        self.low_proj = nn.Conv2d(24, 48, 1)
        self.refine = nn.Linear(96, points)
        self.score = nn.Linear(48, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict coarse and refined lane coordinates.

        Parameters
        ----------
        x:
            Road image tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Lane confidence, coarse lanes, and refined lanes.
        """

        low, high = self.backbone(x)
        lane_ctx = self.gather(high)
        coarse = self.coarse(lane_ctx)
        low_summary = F.adaptive_avg_pool2d(self.low_proj(low), 1).flatten(1)
        low_summary = low_summary.unsqueeze(1).expand_as(lane_ctx)
        refined = coarse + self.refine(torch.cat([lane_ctx, low_summary], dim=-1))
        return {
            "lane_scores": torch.sigmoid(self.score(lane_ctx)).squeeze(-1),
            "coarse_lanes": coarse,
            "refined_lanes": refined,
        }


def build() -> nn.Module:
    """Build the compact PaddleDetection CLRNet model.

    Returns
    -------
    nn.Module
        Random-init lane detector in evaluation mode.
    """

    return PaddleDetCLRNet().eval()


def example_input() -> torch.Tensor:
    """Return a small road image for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 96)``.
    """

    return torch.randn(1, 3, 64, 96)


MENAGERIE_ENTRIES = [
    ("paddledet_clrnet", "build", "example_input", "2022", "DC"),
]
