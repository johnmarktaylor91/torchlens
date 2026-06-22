"""Sparse R-CNN compact detector with learnable proposals and dynamic heads.

Paper: "Sparse R-CNN: End-to-End Object Detection with Learnable Proposals"
(Sun et al., CVPR 2021).

The faithful primitive is the sparse learned object-query design: a fixed set of
learnable proposal boxes and proposal features iteratively attend to dense image
features, then generate per-proposal dynamic parameters for classification and
box refinement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseRCNNCompact(nn.Module):
    """Compact Sparse R-CNN with iterative dynamic proposal refinement."""

    def __init__(
        self, dim: int = 32, proposals: int = 8, num_classes: int = 5, stages: int = 2
    ) -> None:
        """Initialize the compact detector.

        Parameters
        ----------
        dim:
            Proposal feature dimension.
        proposals:
            Number of learned proposals.
        num_classes:
            Number of output classes.
        stages:
            Number of iterative dynamic-head stages.
        """

        super().__init__()
        self.proposals = proposals
        self.boxes = nn.Parameter(torch.rand(proposals, 4))
        self.query = nn.Parameter(torch.randn(proposals, dim) * 0.02)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(dim, 4, batch_first=True) for _ in range(stages)]
        )
        self.dynamic = nn.ModuleList([nn.Linear(dim, dim * dim) for _ in range(stages)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(stages)])
        self.cls_heads = nn.ModuleList([nn.Linear(dim, num_classes) for _ in range(stages)])
        self.box_heads = nn.ModuleList([nn.Linear(dim, 4) for _ in range(stages)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run sparse set prediction.

        Parameters
        ----------
        x:
            Input image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Class logits and normalized boxes.
        """

        bsz = x.shape[0]
        feat = self.backbone(x).flatten(2).transpose(1, 2)
        query = self.query.unsqueeze(0).expand(bsz, -1, -1)
        boxes = self.boxes.unsqueeze(0).expand(bsz, -1, -1).sigmoid()
        for attn, dyn, norm, cls_head, box_head in zip(
            self.cross_attn, self.dynamic, self.norms, self.cls_heads, self.box_heads, strict=True
        ):
            attended, _ = attn(query, feat, feat)
            weights = dyn(query).view(bsz, self.proposals, query.shape[-1], query.shape[-1])
            dynamic_feat = torch.matmul(attended.unsqueeze(2), weights).squeeze(2)
            query = norm(query + F.silu(dynamic_feat))
            boxes = (boxes + 0.1 * torch.tanh(box_head(query))).sigmoid()
            logits = cls_head(query)
        return logits, boxes


def build_paddledet_sparse_rcnn() -> nn.Module:
    """Build compact random-init Sparse R-CNN.

    Returns
    -------
    nn.Module
        Sparse R-CNN compact model.
    """

    return SparseRCNNCompact().eval()


def example_input() -> torch.Tensor:
    """Create a small RGB image.

    Returns
    -------
    torch.Tensor
        Input tensor.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_sparse_rcnn", "build_paddledet_sparse_rcnn", "example_input", "2021", "DC"),
]
