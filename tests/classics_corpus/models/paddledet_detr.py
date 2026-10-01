"""PaddleDetection DETR: convolutional backbone plus Transformer object queries.

Carion et al. (ECCV 2020), "End-to-End Object Detection with Transformers".
DETR uses image features encoded by a Transformer and a fixed set of learned
object queries decoded into class labels and normalized boxes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PaddleDetDETR(nn.Module):
    """Compact DETR object detector."""

    def __init__(self, dim: int = 48, queries: int = 8, classes: int = 5) -> None:
        """Initialize DETR.

        Parameters
        ----------
        dim:
            Transformer width.
        queries:
            Number of object queries.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False),
        )
        self.pos = nn.Parameter(torch.randn(1, 16 * 16, dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=1)
        self.query = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.cls = nn.Linear(dim, classes)
        self.box = nn.Linear(dim, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict object classes and normalized boxes.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            DETR class logits and box predictions.
        """

        batch = x.shape[0]
        feat = self.backbone(x).flatten(2).transpose(1, 2)
        memory = self.encoder(feat + self.pos[:, : feat.shape[1]])
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(query, memory)
        return {"pred_logits": self.cls(decoded), "pred_boxes": torch.sigmoid(self.box(decoded))}


def build() -> nn.Module:
    """Build the compact PaddleDetection DETR model.

    Returns
    -------
    nn.Module
        Random-init detector in evaluation mode.
    """

    return PaddleDetDETR().eval()


def example_input() -> torch.Tensor:
    """Return a small image batch for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("paddledet_detr", "build", "example_input", "2020", "DC")]
