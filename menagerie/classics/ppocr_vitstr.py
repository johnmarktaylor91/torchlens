"""ViTSTR scene text recognizer.

Atienza (2021), "Vision Transformer for Fast and Efficient Scene Text
Recognition."  ViTSTR removes the usual CNN/RNN/attention recognizer pipeline
and performs single-stage recognition with ViT patch tokens plus a per-position
character classifier.  This compact reconstruction keeps the patch embedding,
learned position embeddings, Transformer encoder, and direct character logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactViTSTR(nn.Module):
    """Small single-stage Vision Transformer for text recognition."""

    def __init__(
        self,
        vocab: int = 38,
        dim: int = 64,
        max_steps: int = 12,
        image_size: tuple[int, int] = (32, 96),
        patch: tuple[int, int] = (8, 8),
    ) -> None:
        """Initialize the compact ViTSTR.

        Parameters
        ----------
        vocab:
            Character vocabulary size.
        dim:
            Transformer width.
        max_steps:
            Number of decoded character positions.
        image_size:
            Input image height and width.
        patch:
            Patch height and width.
        """

        super().__init__()
        self.max_steps = max_steps
        num_patches = (image_size[0] // patch[0]) * (image_size[1] // patch[1])
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=dim * 2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.to_steps = nn.Linear(num_patches, max_steps)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode text logits from image patches.

        Parameters
        ----------
        x:
            Grayscale text image.

        Returns
        -------
        torch.Tensor
            Character logits of shape ``(B, max_steps, vocab)``.
        """

        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens + self.pos)
        steps = self.to_steps(encoded.transpose(1, 2)).transpose(1, 2)
        return self.head(steps)


def build() -> nn.Module:
    """Build the compact ViTSTR.

    Returns
    -------
    nn.Module
        Random-init ViTSTR in evaluation mode.
    """

    return CompactViTSTR().eval()


def example_input() -> torch.Tensor:
    """Return a small text-line image.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 1, 32, 96)``.
    """

    return torch.randn(1, 1, 32, 96)


MENAGERIE_ENTRIES = [
    ("ppocr_vitstr", "build", "example_input", "2021", "DC"),
]
