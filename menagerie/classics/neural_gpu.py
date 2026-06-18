"""Neural GPU, 2015, Lukasz Kaiser and Ilya Sutskever.

Paper: Neural GPUs Learn Algorithms.
Token embeddings fill an active 2D memory grid that is repeatedly transformed
by a tied convolutional GRU before per-cell symbol classification.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell used by Neural GPU."""

    def __init__(self, channels: int) -> None:
        """Initialize tied convolutional gates.

        Parameters
        ----------
        channels:
            Number of memory channels.
        """
        super().__init__()
        self.gates = nn.Conv2d(channels, 2 * channels, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, state: Tensor) -> Tensor:
        """Update active memory with convolutional GRU equations.

        Parameters
        ----------
        state:
            Memory tensor ``(B, C, L, W)``.

        Returns
        -------
        Tensor
            Updated memory tensor.
        """
        reset, update = torch.sigmoid(self.gates(state)).chunk(2, dim=1)
        proposal = torch.tanh(self.candidate(reset * state))
        return update * state + (1.0 - update) * proposal


class NeuralGPU(nn.Module):
    """Small tied-convolution active-memory algorithm learner."""

    def __init__(
        self,
        vocab_size: int = 12,
        channels: int = 16,
        width: int = 4,
        steps: int = 6,
    ) -> None:
        """Initialize embeddings, tied ConvGRU, and classifier.

        Parameters
        ----------
        vocab_size:
            Number of discrete symbols.
        channels:
            Active memory channels.
        width:
            Width of the 2D memory grid.
        steps:
            Number of tied recurrent updates.
        """
        super().__init__()
        self.width = width
        self.steps = steps
        self.embedding = nn.Embedding(vocab_size, channels)
        self.cell = ConvGRUCell(channels)
        self.classifier = nn.Conv2d(channels, vocab_size, kernel_size=1)

    def forward(self, tokens: Tensor) -> Tensor:
        """Run Neural GPU updates over a token sequence.

        Parameters
        ----------
        tokens:
            Integer token ids with shape ``(B, L)``.

        Returns
        -------
        Tensor
            Per-position symbol logits with shape ``(B, L, vocab_size)``.
        """
        state = self.embedding(tokens).transpose(1, 2).unsqueeze(3)
        state = state.expand(-1, -1, -1, self.width)
        for _ in range(self.steps):
            state = self.cell(state)
        logits = self.classifier(state).mean(dim=3).transpose(1, 2)
        return logits


def build() -> nn.Module:
    """Build a compact Neural GPU.

    Returns
    -------
    nn.Module
        Random-initialized Neural GPU.
    """
    return NeuralGPU()


def example_input() -> Tensor:
    """Return traceable token ids.

    Returns
    -------
    Tensor
        Long tensor with shape ``(1, 8)``.
    """
    return torch.randint(0, 12, (1, 8), dtype=torch.long)
