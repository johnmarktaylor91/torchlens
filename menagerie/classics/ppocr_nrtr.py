"""PaddleOCR NRTR: transformer encoder-decoder scene text recognizer.

NRTR uses CNN-free transformer recognition with positional token modeling and
autoregressive decoding, as exposed in PaddleOCR's recognition algorithm zoo.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NRTR(nn.Module):
    """Compact NRTR encoder-decoder recognizer."""

    def __init__(self, vocab: int = 32, dim: int = 32) -> None:
        """Initialize image token projection and transformer decoder.

        Parameters
        ----------
        vocab:
            Output vocabulary size.
        dim:
            Transformer width.
        """

        super().__init__()
        self.patch = nn.Conv2d(1, dim, 4, stride=4)
        self.query = nn.Parameter(torch.zeros(1, 12, dim))
        self.transformer = nn.Transformer(
            d_model=dim, nhead=4, num_encoder_layers=1, num_decoder_layers=1, batch_first=True
        )
        self.head = nn.Linear(dim, vocab)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Decode text tokens from image patches.

        Parameters
        ----------
        image:
            Text-line image.

        Returns
        -------
        torch.Tensor
            Autoregressive token logits.
        """

        memory = self.patch(image).flatten(2).transpose(1, 2)
        query = self.query.expand(image.shape[0], -1, -1)
        return self.head(self.transformer(memory, query))


def build() -> nn.Module:
    """Build compact PaddleOCR NRTR.

    Returns
    -------
    nn.Module
        Random-initialized NRTR.
    """

    return NRTR()


def example_input() -> torch.Tensor:
    """Create a text-line input image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 1, 32, 96)``.
    """

    return torch.randn(1, 1, 32, 96)


MENAGERIE_ENTRIES = [("ppocr_nrtr", "build", "example_input", "2019", "E5")]
