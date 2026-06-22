"""PaddleOCR SAR: show-attend-and-read text recognizer.

SAR combines a convolutional visual encoder, recurrent holistic features, and a
2D attention decoder for irregular scene text recognition.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SAR(nn.Module):
    """Compact SAR recognizer with recurrent attention decoding."""

    def __init__(self, vocab: int = 32, dim: int = 32, steps: int = 10) -> None:
        """Initialize CNN encoder, LSTM context, and attention decoder.

        Parameters
        ----------
        vocab:
            Output vocabulary size.
        dim:
            Hidden feature width.
        steps:
            Number of decoded symbols.
        """

        super().__init__()
        self.cnn = nn.Conv2d(1, dim, 3, padding=1)
        self.encoder = nn.LSTM(dim, dim, batch_first=True)
        self.query = nn.Parameter(torch.zeros(1, steps, dim))
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Linear(dim, vocab)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Recognize scene text with 2D attention over visual features.

        Parameters
        ----------
        image:
            Text-line image.

        Returns
        -------
        torch.Tensor
            Decoder logits.
        """

        feat = self.cnn(image).mean(dim=2).transpose(1, 2)
        memory, _ = self.encoder(feat)
        query = self.query.expand(image.shape[0], -1, -1)
        out, _ = self.attn(query, memory, memory)
        return self.head(out)


def build() -> nn.Module:
    """Build compact PaddleOCR SAR.

    Returns
    -------
    nn.Module
        Random-initialized SAR.
    """

    return SAR()


def example_input() -> torch.Tensor:
    """Create a SAR text-line input.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 1, 32, 96)``.
    """

    return torch.randn(1, 1, 32, 96)


MENAGERIE_ENTRIES = [("ppocr_sar", "build", "example_input", "2018", "E5")]
