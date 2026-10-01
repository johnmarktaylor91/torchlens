"""DeepSpeech 2, 2015, Amodei et al., "Deep Speech 2".

Two-dimensional spectrogram convolutions feed a deep recurrent stack and a CTC
character classifier. This minimal version keeps the convolutional front end,
bidirectional GRU stack, and framewise log-probability output.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DeepSpeech2(nn.Module):
    """Small DeepSpeech 2 style acoustic recognizer."""

    def __init__(self, n_classes: int = 29, hidden_size: int = 48) -> None:
        """Initialize convolutional, recurrent, and classifier layers.

        Parameters
        ----------
        n_classes:
            Number of CTC output classes.
        hidden_size:
            GRU hidden size per direction.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(11, 7), stride=(2, 2), padding=(5, 3)),
            nn.BatchNorm2d(16),
            nn.Hardtanh(0.0, 20.0),
            nn.Conv2d(16, 24, kernel_size=(11, 5), stride=(2, 2), padding=(5, 2)),
            nn.BatchNorm2d(24),
            nn.Hardtanh(0.0, 20.0),
        )
        self.input_proj = nn.Linear(24 * 32, hidden_size)
        self.rnns = nn.ModuleList(
            [
                nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=False),
                nn.GRU(2 * hidden_size, hidden_size, bidirectional=True, batch_first=False),
            ]
        )
        self.classifier = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, spec: Tensor) -> Tensor:
        """Compute CTC log probabilities from a spectrogram.

        Parameters
        ----------
        spec:
            Spectrogram tensor with shape ``(batch, 1, freq, time)``.

        Returns
        -------
        Tensor
            Log probabilities with shape ``(time, batch, n_classes)``.
        """
        feats = self.conv(spec)
        batch, channels, freq, time = feats.shape
        seq = feats.permute(3, 0, 1, 2).reshape(time, batch, channels * freq)
        seq = torch.relu(self.input_proj(seq))
        for rnn in self.rnns:
            seq, _ = rnn(seq)
        return F.log_softmax(self.classifier(seq), dim=-1)


def build() -> nn.Module:
    """Build a compact DeepSpeech 2 model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """
    return DeepSpeech2()


def example_input() -> Tensor:
    """Return an example spectrogram.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 1, 128, 400)``.
    """
    return torch.randn(1, 1, 128, 400)


MENAGERIE_ENTRIES = [("DeepSpeech 2", "build", "example_input", "2015", "DE")]
