"""LightGlue compact local-feature matcher.

Lindenberger, Sarlin, and Pollefeys, 2023, "LightGlue: Local Feature Matching
at Light Speed".  LightGlue revisits SuperGlue with alternating self/cross
attention, descriptor confidence heads, match-assignment heads, and adaptive
depth/width pruning.  This compact random-init reconstruction keeps the traced
inference primitive: positional keypoint encoding, repeated self/cross
attention over two feature sets, token-confidence gating, and a dual-softmax
assignment matrix.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LightGlueBlock(nn.Module):
    """One LightGlue attention block over two descriptor sets."""

    def __init__(self, dim: int = 32, heads: int = 4) -> None:
        """Initialize self-attention, cross-attention, and confidence heads.

        Parameters
        ----------
        dim:
            Descriptor width.
        heads:
            Number of attention heads.
        """
        super().__init__()
        self.self0 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self1 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross01 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross10 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm0a = nn.LayerNorm(dim)
        self.norm1a = nn.LayerNorm(dim)
        self.norm0b = nn.LayerNorm(dim)
        self.norm1b = nn.LayerNorm(dim)
        self.conf0 = nn.Linear(dim, 1)
        self.conf1 = nn.Linear(dim, 1)

    def forward(self, desc0: Tensor, desc1: Tensor) -> tuple[Tensor, Tensor]:
        """Refine two descriptor sets with self/cross attention.

        Parameters
        ----------
        desc0:
            First descriptor tensor with shape ``(batch, keypoints, dim)``.
        desc1:
            Second descriptor tensor with shape ``(batch, keypoints, dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Refined descriptors for both images.
        """
        desc0 = self.norm0a(desc0 + self.self0(desc0, desc0, desc0, need_weights=False)[0])
        desc1 = self.norm1a(desc1 + self.self1(desc1, desc1, desc1, need_weights=False)[0])
        upd0 = self.cross01(desc0, desc1, desc1, need_weights=False)[0]
        upd1 = self.cross10(desc1, desc0, desc0, need_weights=False)[0]
        desc0 = self.norm0b(desc0 + upd0)
        desc1 = self.norm1b(desc1 + upd1)
        gate0 = torch.sigmoid(self.conf0(desc0))
        gate1 = torch.sigmoid(self.conf1(desc1))
        return desc0 * gate0, desc1 * gate1


class LightGlue(nn.Module):
    """Compact LightGlue-style matcher."""

    def __init__(self, dim: int = 32, layers: int = 2) -> None:
        """Initialize descriptor/keypoint encoders and assignment head.

        Parameters
        ----------
        dim:
            Descriptor width.
        layers:
            Number of alternating attention layers.
        """
        super().__init__()
        self.kpt_encoder = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.input_proj = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([LightGlueBlock(dim) for _ in range(layers)])
        self.final_proj = nn.Linear(dim, dim, bias=False)
        self.bin_score = nn.Parameter(torch.tensor(1.0))

    def forward(self, desc0: Tensor, desc1: Tensor, kpts0: Tensor, kpts1: Tensor) -> Tensor:
        """Compute a dustbin-augmented soft matching matrix.

        Parameters
        ----------
        desc0:
            First image descriptors with shape ``(batch, n0, dim)``.
        desc1:
            Second image descriptors with shape ``(batch, n1, dim)``.
        kpts0:
            First image keypoints in normalized coordinates, shape ``(batch, n0, 2)``.
        kpts1:
            Second image keypoints in normalized coordinates, shape ``(batch, n1, 2)``.

        Returns
        -------
        Tensor
            Assignment log-probabilities with shape ``(batch, n0 + 1, n1 + 1)``.
        """
        desc0 = self.input_proj(desc0) + self.kpt_encoder(kpts0)
        desc1 = self.input_proj(desc1) + self.kpt_encoder(kpts1)
        for block in self.blocks:
            desc0, desc1 = block(desc0, desc1)
        desc0 = F.normalize(self.final_proj(desc0), dim=-1)
        desc1 = F.normalize(self.final_proj(desc1), dim=-1)
        scores = torch.matmul(desc0, desc1.transpose(-1, -2)) / desc0.shape[-1] ** 0.5
        row_bin = self.bin_score.expand(scores.shape[0], scores.shape[1], 1)
        scores = torch.cat((scores, row_bin), dim=-1)
        col_bin = self.bin_score.expand(scores.shape[0], 1, scores.shape[2])
        scores = torch.cat((scores, col_bin), dim=1)
        return F.log_softmax(scores, dim=-1) + F.log_softmax(scores, dim=-2)


def build() -> nn.Module:
    """Build a compact LightGlue matcher.

    Returns
    -------
    nn.Module
        Random-initialized LightGlue reconstruction.
    """
    return LightGlue().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return descriptors and keypoints for two tiny images.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``desc0, desc1, kpts0, kpts1`` tensors.
    """
    return (
        torch.randn(1, 16, 32),
        torch.randn(1, 18, 32),
        torch.rand(1, 16, 2),
        torch.rand(1, 18, 2),
    )


MENAGERIE_ENTRIES = [
    ("LightGlue", "build", "example_input", "2023", "DC"),
]
