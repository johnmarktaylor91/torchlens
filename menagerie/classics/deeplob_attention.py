"""DeepLOB-Attention compact limit-order-book classifier.

Zhang, Zohren, and Roberts, "DeepLOB: Deep Convolutional Neural Networks for
Limit Order Books", IEEE TSP 2019.  DeepLOB extracts local order-book structure
with convolutional filters, uses Inception-style temporal filters at multiple
scales, then models longer dependencies with an LSTM.  The Attention variant
adds temporal attention pooling over LSTM states before classifying future
mid-price movement.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepLOBCompact(nn.Module):
    """Compact DeepLOB with convolutional, inception, and LSTM stages."""

    def __init__(self, classes: int = 3) -> None:
        """Initialize the DeepLOB architecture.

        Parameters
        ----------
        classes:
            Number of price-movement classes.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 16, (4, 1), padding=(3, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 24, (1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
        )
        self.branch1 = nn.Conv2d(24, 16, (1, 1))
        self.branch3 = nn.Conv2d(24, 16, (3, 1), padding=(1, 0))
        self.branch5 = nn.Conv2d(24, 16, (5, 1), padding=(2, 0))
        self.lstm = nn.LSTM(input_size=16 * 3 * 10, hidden_size=32, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, 32) * 0.02)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.head = nn.Linear(32, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify future movement from raw LOB windows.

        Parameters
        ----------
        x:
            Limit-order-book tensor ``(B, T, 40)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        y = self.conv(x.unsqueeze(1))[:, :, : x.shape[1], :]
        y = torch.cat([self.branch1(y), self.branch3(y), self.branch5(y)], dim=1)
        y = y.permute(0, 2, 1, 3).flatten(2)
        out, _ = self.lstm(y)
        query = self.query.expand(x.shape[0], -1, -1)
        pooled = self.attn(query, out, out, need_weights=False)[0].squeeze(1)
        return self.head(pooled)


def build() -> nn.Module:
    """Build a compact DeepLOB model.

    Returns
    -------
    nn.Module
        DeepLOB classifier.
    """

    return DeepLOBCompact()


def example_input() -> torch.Tensor:
    """Create a raw limit-order-book window.

    Returns
    -------
    torch.Tensor
        Example tensor ``(1, 32, 40)``.
    """

    return torch.randn(1, 32, 40)


MENAGERIE_ENTRIES = [
    (
        "DeepLOB-Attention (CNN-Inception-LSTM limit-order-book classifier)",
        "build",
        "example_input",
        "2019",
        "DC",
    ),
]
