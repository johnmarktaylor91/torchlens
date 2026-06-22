"""DeepLOB: CNN-Inception-LSTM model for limit order books.

Zhang, Zohren & Roberts, 2018, arXiv:1808.03668.  DeepLOB applies convolutional
filters to limit-order-book tensors, uses Inception-style parallel temporal
filters, and feeds the resulting sequence to an LSTM classifier.  This compact
version preserves those three blocks for a small synthetic order-book window.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactDeepLOB(nn.Module):
    """Compact DeepLOB classifier."""

    def __init__(self) -> None:
        """Initialize convolutional, inception, and LSTM blocks."""

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 1), padding=(1, 0)),
            nn.ReLU(),
        )
        self.inc1 = nn.Conv2d(16, 16, (1, 1))
        self.inc3 = nn.Conv2d(16, 16, (3, 1), padding=(1, 0))
        self.inc5 = nn.Conv2d(16, 16, (5, 1), padding=(2, 0))
        self.lstm = nn.LSTM(16 * 3 * 5, 32, batch_first=True)
        self.head = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify future price movement from order-book windows.

        Parameters
        ----------
        x:
            Input tensor ``(batch, time, features)``.

        Returns
        -------
        torch.Tensor
            Three-class logits.
        """

        y = self.conv(x.unsqueeze(1))
        y = torch.cat(
            [torch.relu(self.inc1(y)), torch.relu(self.inc3(y)), torch.relu(self.inc5(y))], dim=1
        )
        y = y.permute(0, 2, 1, 3).flatten(2)
        _, (hidden, _) = self.lstm(y)
        return self.head(hidden[-1])


def build() -> nn.Module:
    """Build compact DeepLOB.

    Returns
    -------
    nn.Module
        Random-init DeepLOB reconstruction.
    """

    return CompactDeepLOB()


def example_input() -> torch.Tensor:
    """Create a synthetic limit-order-book window.

    Returns
    -------
    torch.Tensor
        LOB input tensor.
    """

    return torch.randn(1, 20, 10)


MENAGERIE_ENTRIES = [("DeepLOB", "build", "example_input", "2018", "TS")]
