"""Mask RT-DETR compact PaddleDetection exact-name reconstruction.

Paper: RT-DETR: DETRs Beat YOLOs on Real-time Object Detection; Mask RT-DETR
PaddleDetection instance-segmentation extension.

The compact model retains the RT-DETR inference primitive: CNN multi-scale
features, encoder memory, learned/top-k object queries, transformer decoding,
class/box heads, and a query-conditioned mask head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from menagerie.classics.paddledet_deim import RTDETRBlock


class MaskRTDETR(nn.Module):
    """Compact RT-DETR detector with dynamic query masks."""

    def __init__(self, dim: int = 32, queries: int = 12, classes: int = 8) -> None:
        """Initialize feature encoder, query decoder, and mask projection.

        Parameters
        ----------
        dim:
            Token width.
        queries:
            Number of object queries.
        classes:
            Number of classes.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.low = nn.Conv2d(dim, dim, 3, padding=1)
        self.high = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.encoder = RTDETRBlock(dim)
        self.query_pos = nn.Embedding(queries, dim)
        self.decoder = nn.ModuleList([RTDETRBlock(dim) for _ in range(2)])
        self.cls = nn.Linear(dim, classes)
        self.box = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 4))
        self.mask_embed = nn.Linear(dim, dim)
        self.mask_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict classes, boxes, and query-conditioned masks.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, normalized boxes, and mask logits.
        """

        low = F.silu(self.low(self.stem(image)))
        high = F.silu(self.high(low))
        fused = low + F.interpolate(high, size=low.shape[-2:], mode="nearest")
        memory = self.encoder(fused.flatten(2).transpose(1, 2))
        query = self.query_pos.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        for block in self.decoder:
            query = block(query, memory)
        mask_basis = self.mask_proj(fused)
        masks = torch.einsum("bqd,bdhw->bqhw", self.mask_embed(query), mask_basis)
        return self.cls(query), torch.sigmoid(self.box(query)), masks


def build() -> nn.Module:
    """Build compact random-init Mask RT-DETR.

    Returns
    -------
    nn.Module
        Dependency-free Mask RT-DETR-style model.
    """

    return MaskRTDETR().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("ppdet_mask_rtdetr", "build", "example_input", "2024", "DET")]
