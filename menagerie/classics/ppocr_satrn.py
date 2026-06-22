"""PaddleOCR SATRN: self-attention text recognition network.

SATRN uses a shallow CNN stem, 2D positional visual self-attention, and a
transformer decoder for arbitrary-shaped text recognition.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SATRN(nn.Module):
    """Compact SATRN encoder-decoder recognizer."""

    def __init__(self, vocab: int = 32, dim: int = 32) -> None:
        """Initialize CNN stem, transformer encoder, and token decoder.

        Parameters
        ----------
        vocab:
            Output vocabulary size.
        dim:
            Transformer width.
        """

        super().__init__()
        self.stem = nn.Conv2d(1, dim, 3, stride=2, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, 128, dim))
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.query = nn.Parameter(torch.zeros(1, 10, dim))
        self.cross = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Linear(dim, vocab)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Decode text with self-attended visual tokens.

        Parameters
        ----------
        image:
            Text-line image.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        tokens = self.stem(image).flatten(2).transpose(1, 2)
        memory = self.encoder(tokens + self.pos[:, : tokens.shape[1]])
        query = self.query.expand(image.shape[0], -1, -1)
        out, _ = self.cross(query, memory, memory)
        return self.head(out)


def build() -> nn.Module:
    """Build compact PaddleOCR SATRN.

    Returns
    -------
    nn.Module
        Random-initialized SATRN.
    """

    return SATRN()


def example_input() -> torch.Tensor:
    """Create a SATRN text-line input.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 1, 16, 32)``.
    """

    return torch.randn(1, 1, 16, 32)


MENAGERIE_ENTRIES = [("ppocr_satrn", "build", "example_input", "2020", "E5")]
