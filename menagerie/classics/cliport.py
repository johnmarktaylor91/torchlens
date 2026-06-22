"""CLIPort compact random-init reconstruction.

Paper: CLIPort: What and Where Pathways for Robotic Manipulation
(Shridhar, Manuelli, Fox, CoRL 2021/2022).

CLIPort combines a semantic "what" stream inspired by CLIP language-image
features with a spatial "where" Transporter stream.  This compact version keeps
language-conditioned two-stream fusion for both pick attention and place
cross-correlation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CLIPortPolicy(nn.Module):
    """Compact language-conditioned CLIPort pick-place policy."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize semantic and spatial streams."""

        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(4, dim, 3, padding=1), nn.ReLU(), nn.Conv2d(dim, dim, 3, padding=1)
        )
        self.semantic = nn.Sequential(nn.Conv2d(4, dim, 1), nn.ReLU(), nn.Conv2d(dim, dim, 1))
        self.text = nn.Sequential(nn.Linear(16, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.pick = nn.Conv2d(dim, 1, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.query = nn.Conv2d(dim, dim, 1)

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Return language-conditioned pick and place affordance maps."""

        image, language = inputs
        text = self.text(language)[:, :, None, None]
        where = self.spatial(image)
        what = self.semantic(image) * torch.sigmoid(text)
        fused = F.relu(where + what)
        pick_logits = self.pick(fused)
        pick_weights = torch.softmax(pick_logits.flatten(2), dim=-1).view_as(pick_logits)
        query = (self.query(fused) * pick_weights).sum(dim=(2, 3), keepdim=True)
        place_logits = (self.key(fused) * query).sum(dim=1, keepdim=True)
        return pick_logits, place_logits


def build() -> nn.Module:
    """Build a compact random-init CLIPort policy."""

    return CLIPortPolicy().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return a top-down RGB-D map and a compact language embedding."""

    return (torch.randn(1, 4, 32, 32), torch.randn(1, 16))


MENAGERIE_ENTRIES = [
    ("cliport", "build", "example_input", "2022", "DC"),
]
