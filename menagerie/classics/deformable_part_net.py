"""Deformable Part Model, 2008, Pedro Felzenszwalb et al.

Paper: A discriminatively trained, multiscale, deformable part model.
A root filter is combined with part-filter responses after maximizing over a
small displacement window penalized by learned quadratic deformation costs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [
    ("Deformable Part Model (differentiable)", "build", "example_input", "2008", "DC")
]


class DifferentiableDeformablePartModel(nn.Module):
    """Small differentiable DPM-style classifier."""

    def __init__(self, parts: int = 4, num_classes: int = 5) -> None:
        """Initialize feature extractor, root and part filters, and deformation costs.

        Parameters
        ----------
        parts
            Number of part filters.
        num_classes
            Number of classifier outputs.
        """
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 5, stride=8, padding=2), nn.ReLU())
        self.root = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.parts = nn.Conv2d(8, parts, kernel_size=3, padding=1)
        offsets = torch.tensor(
            [
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        self.register_buffer("offsets", offsets)
        self.deformation = nn.Parameter(torch.full((parts, 2), 0.2))
        self.classifier = nn.Linear(1, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Score an image with root and deformable part responses.

        Parameters
        ----------
        x
            RGB image tensor with shape ``(B, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        feat = self.features(x)
        root_score = self.root(feat)
        part_score = self.parts(feat)
        padded = F.pad(part_score, (1, 1, 1, 1), value=-1.0e4)
        candidates = F.unfold(padded, kernel_size=3).view(
            x.shape[0], part_score.shape[1], 9, *part_score.shape[-2:]
        )
        penalty = (
            self.offsets.square().unsqueeze(0) * torch.relu(self.deformation).unsqueeze(1)
        ).sum(dim=-1)
        best_parts = (candidates - penalty.view(1, part_score.shape[1], 9, 1, 1)).amax(dim=2)
        score_map = root_score + best_parts.sum(dim=1, keepdim=True)
        score = F.adaptive_max_pool2d(score_map, (1, 1)).flatten(1)
        return self.classifier(score)


def build() -> nn.Module:
    """Build a compact differentiable DPM.

    Returns
    -------
    nn.Module
        Random-initialized DPM-style classifier.
    """
    return DifferentiableDeformablePartModel()


def example_input() -> Tensor:
    """Return a traceable RGB image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 224, 224)``.
    """
    return torch.randn(1, 3, 224, 224)
