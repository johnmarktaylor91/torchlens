"""Hyena operator compact faithful reconstruction.

Poli et al. 2023, "Hyena Hierarchy: Towards Larger Convolutional Language
Models". Hyena replaces attention with projections, data-controlled gates, and
implicitly parameterized long causal convolutions. This module keeps that
operator in a compact language-model block.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ImplicitFilter(nn.Module):
    """Small implicit MLP that emits long convolution filters."""

    def __init__(self, dim: int, hidden: int = 32) -> None:
        """Initialize filter network.

        Parameters
        ----------
        dim:
            Number of filter channels.
        hidden:
            Hidden width.
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.decay = nn.Parameter(torch.linspace(0.1, 1.0, dim))

    def forward(self, length: int, device: torch.device) -> Tensor:
        """Generate length-wise filters.

        Parameters
        ----------
        length:
            Sequence length.
        device:
            Target device.

        Returns
        -------
        Tensor
            Filter tensor.
        """
        pos = torch.linspace(0.0, 1.0, length, device=device).unsqueeze(-1)
        features = torch.cat(
            (
                pos,
                torch.sin(2.0 * math.pi * pos),
                torch.cos(2.0 * math.pi * pos),
                torch.sin(8.0 * math.pi * pos),
                torch.cos(8.0 * math.pi * pos),
            ),
            dim=-1,
        )
        return self.net(features) * torch.exp(-pos * self.decay.view(1, -1))


class HyenaOperator(nn.Module):
    """Gated long-convolution Hyena operator."""

    def __init__(self, dim: int = 48, order: int = 2) -> None:
        """Initialize projections and filters.

        Parameters
        ----------
        dim:
            Token dimension.
        order:
            Number of gated long-convolution recurrences.
        """
        super().__init__()
        self.order = order
        self.in_proj = nn.Linear(dim, dim * (order + 1))
        self.short = nn.Conv1d(dim, dim, kernel_size=3, padding=2, groups=dim)
        self.filters = nn.ModuleList([ImplicitFilter(dim) for _ in range(order)])
        self.out_proj = nn.Linear(dim, dim)

    def _long_conv(self, x: Tensor, kernel: Tensor) -> Tensor:
        """Apply channel-wise causal convolution by FFT.

        Parameters
        ----------
        x:
            Sequence tensor.
        kernel:
            Filter tensor.

        Returns
        -------
        Tensor
            Convolved sequence.
        """
        length = x.shape[1]
        fft_len = 2 * length
        x_fft = torch.fft.rfft(x, n=fft_len, dim=1)
        k_fft = torch.fft.rfft(kernel, n=fft_len, dim=0).unsqueeze(0)
        return torch.fft.irfft(x_fft * k_fft, n=fft_len, dim=1)[:, :length]

    def forward(self, x: Tensor) -> Tensor:
        """Apply Hyena recurrence.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        Tensor
            Mixed token tensor.
        """
        length = x.shape[1]
        pieces = self.in_proj(x).chunk(self.order + 1, dim=-1)
        value = self.short(pieces[0].transpose(1, 2))[..., :length].transpose(1, 2)
        y = value
        for idx in range(self.order):
            y = self._long_conv(
                y * torch.sigmoid(pieces[idx + 1]), self.filters[idx](length, x.device)
            )
        return self.out_proj(y)


class HyenaLM(nn.Module):
    """Compact Hyena language model."""

    def __init__(self, vocab: int = 64, dim: int = 48) -> None:
        """Initialize embedding, Hyena operator, and head.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Token dimension.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.norm = nn.LayerNorm(dim)
        self.hyena = HyenaOperator(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens: Tensor) -> Tensor:
        """Run compact Hyena LM.

        Parameters
        ----------
        tokens:
            Token ids.

        Returns
        -------
        Tensor
            Vocabulary logits.
        """
        x = self.embed(tokens)
        x = x + self.hyena(self.norm(x))
        return self.head(x)


def build() -> nn.Module:
    """Build compact random-init Hyena operator model.

    Returns
    -------
    nn.Module
        Compact Hyena model.
    """
    return HyenaLM()


def example_input() -> Tensor:
    """Return token ids.

    Returns
    -------
    Tensor
        Token tensor.
    """
    return torch.randint(0, 64, (1, 16))


MENAGERIE_ENTRIES = [
    ("hyena_operator", "build", "example_input", "2023", "E7"),
]
