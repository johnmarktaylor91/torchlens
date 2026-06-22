"""Transporter Network compact random-init reconstruction.

Paper: Transporter Networks: Rearranging the Visual World for Robotic
Manipulation (Zeng et al., CoRL 2020).

Transporter first predicts a pick/attention affordance, crops a local query
feature around the pick, then predicts place by cross-correlating that query
with deep scene features.  The compact model keeps those pick and
feature-transport correlation primitives.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TransporterNet(nn.Module):
    """Compact pick-and-place Transporter network."""

    def __init__(self, channels: int = 32, kernel: int = 5) -> None:
        """Initialize visual trunk, attention head, and transport projections."""

        super().__init__()
        self.kernel = kernel
        self.trunk = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.attention = nn.Conv2d(channels, 1, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.query = nn.Conv2d(channels, channels, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Return pick affordances and place correlation affordances."""

        feat = self.trunk(image)
        pick_logits = self.attention(feat)
        pick_weights = torch.softmax(pick_logits.flatten(2), dim=-1).view_as(pick_logits)
        pooled_query = (self.query(feat) * pick_weights).sum(dim=(2, 3), keepdim=True)
        local_query = F.avg_pool2d(
            self.query(feat) * pick_weights, self.kernel, stride=1, padding=self.kernel // 2
        )
        query = pooled_query + local_query
        place_logits = (self.key(feat) * query).sum(dim=1, keepdim=True)
        return pick_logits, place_logits


def build() -> nn.Module:
    """Build a compact random-init Transporter Network."""

    return TransporterNet().eval()


def example_input() -> Tensor:
    """Return a small RGB-D top-down heightmap."""

    return torch.randn(1, 4, 32, 32)


MENAGERIE_ENTRIES = [
    ("transporter_network", "build", "example_input", "2020", "DC"),
]
