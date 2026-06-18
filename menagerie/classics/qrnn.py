"""Quasi-Recurrent Neural Network, 2016, Bradbury et al.

Paper: "Quasi-Recurrent Neural Networks." Causal convolutions compute gates in
parallel across time, while a lightweight fo-pooling scan carries recurrence.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QRNN(nn.Module):
    """Single-layer causal-convolution QRNN with fo-pooling."""

    def __init__(self, input_size: int = 128, hidden_size: int = 48, kernel_size: int = 3) -> None:
        """Initialize the QRNN convolutional gates.

        Parameters
        ----------
        input_size:
            Per-step feature size.
        hidden_size:
            Hidden state size.
        kernel_size:
            Causal convolution width.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.conv = nn.Conv1d(input_size, 3 * hidden_size, kernel_size=kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        """Run causal QRNN fo-pooling over a sequence.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Hidden sequence with shape ``(batch, time, hidden_size)``.
        """
        padded = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        z_raw, f_raw, o_raw = self.conv(padded).transpose(1, 2).chunk(3, dim=-1)
        z = torch.tanh(z_raw)
        f_gate = torch.sigmoid(f_raw)
        o_gate = torch.sigmoid(o_raw)
        cell = x.new_zeros(x.shape[0], self.hidden_size)
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            cell = f_gate[:, step] * cell + (1.0 - f_gate[:, step]) * z[:, step]
            outputs.append(o_gate[:, step] * cell)
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a compact QRNN.

    Returns
    -------
    nn.Module
        Random-initialized QRNN.
    """
    return QRNN()


def example_input() -> Tensor:
    """Return an example sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 64, 128)``.
    """
    return torch.randn(1, 64, 128)


MENAGERIE_ENTRIES = [
    ("QRNN Quasi-Recurrent Neural Network", "build", "example_input", "2016", "DE")
]
