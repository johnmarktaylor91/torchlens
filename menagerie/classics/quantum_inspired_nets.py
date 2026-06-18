"""Quantum-inspired neural nets, 2017-2026, reservoir, Born-machine, conv, and attention.

Paper: Fujii 2017, "Harnessing disordered-ensemble quantum dynamics"; Liu 2018,
"Differentiable learning of quantum circuit Born machines"; Henderson 2020,
"Quanvolutional neural networks"; 2026 UQT preprint. These modules are classical
differentiable approximations and omit real quantum simulation or hardware sampling.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QuantumReservoirComputing(nn.Module):
    """Fixed nonlinear recurrent reservoir with Pauli-Z-style readout features."""

    def __init__(self, width: int = 12, steps: int = 50) -> None:
        """Initialize fixed reservoir dynamics.

        Parameters
        ----------
        width
            Number of reservoir observables.
        steps
            Number of time samples to process.
        """
        super().__init__()
        self.steps = steps
        recurrent = torch.randn(width, width) * 0.2
        self.register_buffer("recurrent", recurrent)
        self.register_buffer("drive", torch.randn(width) * 0.4)
        self.readout = nn.Linear(width, 4)

    def forward(self, series: Tensor) -> Tensor:
        """Drive a fixed quantum-reservoir surrogate with a time series.

        Parameters
        ----------
        series
            Input tensor with shape ``(batch, 50)``.

        Returns
        -------
        Tensor
            Linear readout over reservoir observables.
        """
        state = series.new_zeros(series.shape[0], self.recurrent.shape[0])
        for step in range(self.steps):
            phase = series[:, step : step + 1] * self.drive
            state = torch.tanh(state @ self.recurrent + torch.sin(phase))
        return self.readout(state)


class QCBM(nn.Module):
    """Born-rule amplitude surrogate using a small differentiable circuit net."""

    def __init__(self, n_bits: int = 8, hidden_size: int = 16) -> None:
        """Initialize angle encoder and amplitude readout.

        Parameters
        ----------
        n_bits
            Number of bitstring inputs.
        hidden_size
            Hidden amplitude width.
        """
        super().__init__()
        self.theta = nn.Parameter(torch.randn(n_bits) * 0.2)
        self.hidden = nn.Linear(n_bits, hidden_size)
        self.out = nn.Linear(hidden_size, n_bits)

    def forward(self, bitstrings: Tensor) -> Tensor:
        """Compute Born-rule-like probabilities for bitstrings.

        Parameters
        ----------
        bitstrings
            Bitstring feature tensor with shape ``(batch, 8)``.

        Returns
        -------
        Tensor
            Bernoulli probabilities from squared amplitudes.
        """
        angles = bitstrings * math.pi + self.theta
        hidden = torch.sin(self.hidden(torch.cos(angles)))
        amplitude = self.out(hidden)
        probs = amplitude.pow(2)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)


class QuanvolutionalNeuralNetwork(nn.Module):
    """Fixed random nonlinear quanvolutional filter bank plus classical head."""

    def __init__(self, channels: int = 6) -> None:
        """Initialize fixed random patch projections.

        Parameters
        ----------
        channels
            Number of quanvolutional measurement channels.
        """
        super().__init__()
        self.register_buffer("filters", torch.randn(channels, 1, 3, 3) * 0.7)
        self.head = nn.Linear(channels, 10)

    def forward(self, image: Tensor) -> Tensor:
        """Apply random nonlinear quantum-inspired convolutional measurements.

        Parameters
        ----------
        image
            Image tensor with shape ``(batch, 1, 28, 28)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        features = torch.sin(F.conv2d(image, self.filters, padding=1))
        pooled = F.avg_pool2d(features.pow(2), kernel_size=28).flatten(1)
        return self.head(pooled)


class UniversalQuantumTransformer(nn.Module):
    """Classical attention block with fixed unitary-like token mixing."""

    def __init__(self, vocab_size: int = 128, dim: int = 24, seq_len: int = 64) -> None:
        """Initialize embeddings, fixed orthogonal mixing, and readout.

        Parameters
        ----------
        vocab_size
            Number of token ids.
        dim
            Token embedding width.
        seq_len
            Maximum sequence length.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        q, _ = torch.linalg.qr(torch.randn(seq_len, seq_len))
        self.register_buffer("unitary_mix", q)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """Mix token embeddings with interference-style attention.

        Parameters
        ----------
        tokens
            Token ids with shape ``(batch, 64)``.

        Returns
        -------
        Tensor
            Next-token logits for each position.
        """
        emb = self.embedding(tokens)
        mixed = torch.einsum("st,btd->bsd", self.unitary_mix, emb)
        phase = torch.sin(self.query(mixed))
        values = self.value(emb)
        context = phase * torch.cos(values)
        return self.out(context)


MENAGERIE_ENTRIES = [
    ("Quantum Reservoir Computing (QRC)", "build_qrc", "example_input_qrc", "2017", "DA"),
    ("QCBM (Quantum Circuit Born Machine)", "build_qcbm", "example_input_qcbm", "2018", "DA"),
    (
        "Quanvolutional Neural Network",
        "build_quanvolutional",
        "example_input_quanvolutional",
        "2020",
        "DA",
    ),
    ("Universal Quantum Transformer (UQT)", "build_uqt", "example_input_uqt", "2026", "DA"),
]


def build_qrc() -> nn.Module:
    """Build a quantum reservoir surrogate.

    Returns
    -------
    nn.Module
        Configured QRC module.
    """
    return QuantumReservoirComputing()


def example_input_qrc() -> Tensor:
    """Create a time-series example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 50)``.
    """
    return torch.randn(1, 50)


def build_qcbm() -> nn.Module:
    """Build a QCBM surrogate.

    Returns
    -------
    nn.Module
        Configured QCBM module.
    """
    return QCBM()


def example_input_qcbm() -> Tensor:
    """Create example bitstrings.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 8)``.
    """
    return torch.rand(1, 8)


def build_quanvolutional() -> nn.Module:
    """Build a quanvolutional neural network.

    Returns
    -------
    nn.Module
        Configured quanvolutional module.
    """
    return QuanvolutionalNeuralNetwork()


def example_input_quanvolutional() -> Tensor:
    """Create an image example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 1, 28, 28)``.
    """
    return torch.randn(1, 1, 28, 28)


def build_uqt() -> nn.Module:
    """Build a universal quantum transformer surrogate.

    Returns
    -------
    nn.Module
        Configured UQT module.
    """
    return UniversalQuantumTransformer()


def example_input_uqt() -> Tensor:
    """Create token ids.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 64)``.
    """
    return torch.randint(0, 128, (1, 64), dtype=torch.long)
