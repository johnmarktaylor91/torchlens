"""PaddleDetection DINO: DETR with improved denoising anchor boxes.

Zhang et al. (ICLR 2023), "DINO: DETR with Improved DeNoising Anchor Boxes for
End-to-End Object Detection".  DINO adds contrastive denoising queries, mixed
query selection for anchor initialization, and look-forward-twice box updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PaddleDetDINO(nn.Module):
    """Compact DINO-style DETR detector."""

    def __init__(self, dim: int = 48, queries: int = 8, dn_queries: int = 4) -> None:
        """Initialize DINO.

        Parameters
        ----------
        dim:
            Transformer width.
        queries:
            Number of detection queries.
        dn_queries:
            Number of denoising queries.
        """

        super().__init__()
        self.queries = queries
        self.dn_queries = dn_queries
        self.backbone = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True), num_layers=1
        )
        self.score_memory = nn.Linear(dim, 1)
        self.anchor_embed = nn.Linear(4, dim)
        self.dn_anchor = nn.Parameter(torch.rand(dn_queries, 4))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, 4, dim * 2, batch_first=True), num_layers=1
        )
        self.cls = nn.Linear(dim, 5)
        self.box_delta1 = nn.Linear(dim, 4)
        self.box_delta2 = nn.Linear(dim, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict DINO detection and denoising outputs.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Class logits, refined boxes, and denoising boxes.
        """

        batch = x.shape[0]
        memory = self.backbone(x).flatten(2).transpose(1, 2)
        memory = self.encoder(memory)
        scores = self.score_memory(memory).squeeze(-1)
        _, top_idx = torch.topk(scores, self.queries, dim=1)
        top_memory = torch.gather(memory, 1, top_idx[..., None].expand(-1, -1, memory.shape[-1]))
        mixed_anchors = torch.sigmoid(self.box_delta1(top_memory))
        dn = self.dn_anchor.unsqueeze(0).expand(batch, -1, -1)
        noisy_dn = (dn + 0.05 * torch.sin(dn * 17.0)).clamp(0.0, 1.0)
        all_anchors = torch.cat([noisy_dn, mixed_anchors], dim=1)
        query = self.anchor_embed(all_anchors)
        decoded = self.decoder(query, memory)
        first_boxes = torch.sigmoid(all_anchors + self.box_delta1(decoded))
        second_boxes = torch.sigmoid(first_boxes + self.box_delta2(decoded))
        return {
            "pred_logits": self.cls(decoded[:, self.dn_queries :]),
            "pred_boxes": second_boxes[:, self.dn_queries :],
            "dn_boxes": second_boxes[:, : self.dn_queries],
        }


def build() -> nn.Module:
    """Build the compact PaddleDetection DINO model.

    Returns
    -------
    nn.Module
        Random-init detector in evaluation mode.
    """

    return PaddleDetDINO().eval()


def example_input() -> torch.Tensor:
    """Return a small image batch for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("paddledet_dino", "build", "example_input", "2023", "DC")]
