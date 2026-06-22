"""DEIM compact real-time DETR training-framework reconstruction.

Huang et al., 2024, "DEIM: DETR with Improved Matching for Fast Convergence".
DEIM changes DETR training with dense one-to-one matching and matchability-aware
loss, while deployment uses the underlying real-time DETR inference graph.  This
compact classic traces that inference graph: multi-scale CNN features, encoder
tokens, learned object queries, transformer decoder, and class/box heads.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RTDETRBlock(nn.Module):
    """Pre-norm transformer block used in the compact DEIM detector."""

    def __init__(self, dim: int = 48, heads: int = 4) -> None:
        """Initialize attention and feed-forward layers.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Number of attention heads.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 3), nn.GELU(), nn.Linear(dim * 3, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor, memory: Tensor | None = None) -> Tensor:
        """Apply self- or cross-attention followed by an FFN.

        Parameters
        ----------
        x:
            Query tokens.
        memory:
            Optional key/value memory tokens.

        Returns
        -------
        Tensor
            Updated query tokens.
        """
        kv = x if memory is None else memory
        x = self.norm1(x + self.attn(x, kv, kv, need_weights=False)[0])
        return self.norm2(x + self.ffn(x))


class PaddleDEIM(nn.Module):
    """Compact DEIM/RT-DETR style detector."""

    def __init__(self, dim: int = 48, queries: int = 24, classes: int = 20) -> None:
        """Initialize the detector.

        Parameters
        ----------
        dim:
            Feature width.
        queries:
            Number of object queries.
        classes:
            Number of classes.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.SiLU(),
            nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.aifi = RTDETRBlock(dim)
        self.ccfm_lateral = nn.Conv2d(dim, dim, 1)
        self.ccfm_out = nn.Conv2d(dim, dim, 3, padding=1)
        self.encoder = RTDETRBlock(dim)
        self.query_proj = nn.Linear(dim, dim)
        self.iou_score = nn.Linear(dim, 1)
        self.query_pos = nn.Embedding(queries, dim)
        self.decoder = nn.ModuleList([RTDETRBlock(dim) for _ in range(2)])
        self.cls_head = nn.Linear(dim, classes)
        self.box_head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 4))
        self.matchability = nn.Linear(dim, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict classes and boxes from an RGB image.

        Parameters
        ----------
        image:
            Input tensor with shape ``(batch, 3, height, width)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, normalized boxes, and dense O2O matchability scores.
        """
        c3 = self.stem(image)
        c4 = F.silu(self.down1(c3))
        c5 = F.silu(self.down2(c4))
        c5_tokens = c5.flatten(2).transpose(1, 2)
        aifi = self.aifi(c5_tokens).transpose(1, 2).reshape_as(c5)
        top_down = F.interpolate(self.ccfm_lateral(aifi), size=c4.shape[-2:], mode="nearest") + c4
        bottom_up = F.interpolate(self.ccfm_out(top_down), size=c3.shape[-2:], mode="nearest") + c3
        memory = self.encoder(bottom_up.flatten(2).transpose(1, 2))
        query_scores = torch.sigmoid(self.iou_score(memory)).squeeze(-1)
        _, top_idx = torch.topk(query_scores, k=self.query_pos.num_embeddings, dim=1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, memory.size(-1))
        query = self.query_proj(
            torch.gather(memory, 1, gather_idx)
        ) + self.query_pos.weight.unsqueeze(0)
        for block in self.decoder:
            query = block(query, memory)
        boxes = torch.sigmoid(self.box_head(query))
        dense_o2o = torch.sigmoid(self.matchability(query)) * boxes[..., 2:3] * boxes[..., 3:4]
        return self.cls_head(query), boxes, dense_o2o


def build() -> nn.Module:
    """Build the compact DEIM detector.

    Returns
    -------
    nn.Module
        Random-initialized detector.
    """
    return PaddleDEIM().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_deim", "build", "example_input", "2024", "DC"),
]
