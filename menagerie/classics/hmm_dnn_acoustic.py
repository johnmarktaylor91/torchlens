"""HMM-DNN hybrid acoustic model, 2012, Dahl, Yu, Deng, and Acero.

Paper: "Context-dependent pre-trained deep neural networks for large-vocabulary speech
recognition." A frame-level neural acoustic model predicts HMM senone log
posteriors from spectral features; decoding is omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class HMMDNNAcousticModel(nn.Module):
    """Compact framewise DNN acoustic model for HMM emission posteriors."""

    def __init__(self, n_features: int = 40, hidden_size: int = 64, n_senones: int = 32) -> None:
        """Initialize frame and temporal-delta encoders.

        Parameters
        ----------
        n_features:
            Number of acoustic features per frame.
        hidden_size:
            Hidden representation size.
        n_senones:
            Number of tied HMM state classes.
        """
        super().__init__()
        self.frame = nn.Linear(n_features, hidden_size)
        self.tdnn = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2)
        self.classifier = nn.Linear(hidden_size, n_senones)

    def forward(self, feats: Tensor) -> Tensor:
        """Compute frame log-posteriors for acoustic features.

        Parameters
        ----------
        feats:
            Float tensor of shape ``(batch, time, n_features)``.

        Returns
        -------
        Tensor
            Frame log probabilities with shape ``(batch, time, n_senones)``.
        """
        hidden = torch.relu(self.frame(feats))
        hidden = torch.relu(self.tdnn(hidden.transpose(1, 2)).transpose(1, 2))
        return F.log_softmax(self.classifier(hidden), dim=-1)


def build() -> nn.Module:
    """Build a small HMM-DNN acoustic model.

    Returns
    -------
    nn.Module
        Random-initialized acoustic model.
    """
    return HMMDNNAcousticModel()


def example_input() -> Tensor:
    """Return example acoustic features.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 100, 40)``.
    """
    return torch.randn(1, 100, 40)


MENAGERIE_ENTRIES = [("HMM-DNN Hybrid Acoustic Model", "build", "example_input", "2012", "DE")]
