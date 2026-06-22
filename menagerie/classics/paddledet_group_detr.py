"""PaddleDetection Group DETR with group-wise object queries.

Chen et al. (ICCV 2023), "Group DETR: Fast DETR Training with Group-Wise
One-to-Many Assignment".  Group DETR partitions object queries into multiple
groups, runs decoder self-attention separately for each group, and shares
prediction heads to support group-wise one-to-many assignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PaddleDetGroupDETR(nn.Module):
    """Compact Group DETR detector."""

    def __init__(self, dim: int = 48, groups: int = 2, queries_per_group: int = 4) -> None:
        """Initialize Group DETR.

        Parameters
        ----------
        dim:
            Transformer width.
        groups:
            Number of query groups.
        queries_per_group:
            Object queries per group.
        """

        super().__init__()
        self.groups = groups
        self.backbone = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False),
        )
        self.pos = nn.Parameter(torch.randn(1, 16 * 16, dim) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True), num_layers=1
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, 4, dim * 2, batch_first=True), num_layers=1
        )
        self.query = nn.Parameter(torch.randn(groups, queries_per_group, dim) * 0.02)
        self.cls = nn.Linear(dim, 5)
        self.box = nn.Linear(dim, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode each query group independently.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Grouped class logits and box predictions.
        """

        batch = x.shape[0]
        feat = self.backbone(x).flatten(2).transpose(1, 2)
        memory = self.encoder(feat + self.pos[:, : feat.shape[1]])
        decoded_groups = []
        for group in range(self.groups):
            query = self.query[group].unsqueeze(0).expand(batch, -1, -1)
            decoded_groups.append(self.decoder(query, memory))
        decoded = torch.stack(decoded_groups, dim=1)
        return {"pred_logits": self.cls(decoded), "pred_boxes": torch.sigmoid(self.box(decoded))}


def build() -> nn.Module:
    """Build the compact PaddleDetection Group DETR model.

    Returns
    -------
    nn.Module
        Random-init detector in evaluation mode.
    """

    return PaddleDetGroupDETR().eval()


def example_input() -> torch.Tensor:
    """Return a small image batch for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("paddledet_group_detr", "build", "example_input", "2023", "DC")]
