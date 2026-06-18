"""Classic associative memories from Nakano, Kosko, and Morita.

Paper: Nakano 1972, "Associatron: A Model of Associative Memory."
Paper: Kosko 1988, temporal associative memory sequence recall.
Paper: Morita 1993, nonmonotone associative memory.
These modules keep the characteristic recurrent recall dynamics while replacing
hard signs with smooth saturating transfers for differentiable tracing.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Associatron(nn.Module):
    """Nakano-style local cellular associative recall module."""

    def __init__(self, n_units: int = 256, kernel_size: int = 9, steps: int = 4) -> None:
        """Initialize local recurrent weights.

        Parameters
        ----------
        n_units:
            Number of cells in the one-dimensional cellular memory.
        kernel_size:
            Odd neighborhood width for local recall.
        steps:
            Number of smooth recurrent recall iterations.
        """
        super().__init__()
        self.n_units = n_units
        self.steps = steps
        local = torch.randn(1, 1, kernel_size) * 0.18
        local[..., kernel_size // 2] += 0.85
        self.register_buffer("local_weights", local)

    def forward(self, state: Tensor) -> Tensor:
        """Run local recurrent part-to-whole recall.

        Parameters
        ----------
        state:
            Input cue tensor with shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Recalled bipolar-like state.
        """
        memory = state[:, None, :]
        pad = self.local_weights.shape[-1] // 2
        for _ in range(self.steps):
            neighborhood = F.pad(memory, (pad, pad), mode="circular")
            drive = F.conv1d(neighborhood, self.local_weights)
            memory = torch.tanh(1.4 * drive)
        return memory[:, 0, :]


class MoritaNonmonotoneAssociativeMemory(nn.Module):
    """Hopfield-type associative memory with a nonmonotone transfer."""

    def __init__(self, n_units: int = 256, steps: int = 4, height: float = 0.9) -> None:
        """Initialize symmetric recurrent weights.

        Parameters
        ----------
        n_units:
            Number of recurrent units.
        steps:
            Number of recurrent updates.
        height:
            Threshold where the dome-shaped transfer begins to reverse.
        """
        super().__init__()
        self.steps = steps
        self.height = height
        raw = torch.randn(n_units, n_units) / n_units**0.5
        weights = 0.5 * (raw + raw.T)
        weights.fill_diagonal_(0.0)
        self.register_buffer("weights", weights)

    def _nonmonotone(self, drive: Tensor) -> Tensor:
        """Apply Morita's smooth inverted-U nonmonotone response.

        Parameters
        ----------
        drive:
            Recurrent input drive.

        Returns
        -------
        Tensor
            Smooth nonmonotone activation.
        """
        overshoot = torch.relu(drive.abs() - self.height)
        dome = drive * (1.0 - 0.55 * overshoot)
        return torch.tanh(dome)

    def forward(self, state: Tensor) -> Tensor:
        """Iterate nonmonotone associative recall.

        Parameters
        ----------
        state:
            Input cue tensor with shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Recalled state after recurrent dynamics.
        """
        memory = torch.tanh(state)
        for _ in range(self.steps):
            memory = self._nonmonotone(memory @ self.weights.T)
        return memory


class TemporalAssociativeMemory(nn.Module):
    """Kosko-style asymmetric matrix memory for sequence transitions."""

    def __init__(self, n_units: int = 64, sequence_len: int = 6, steps: int = 5) -> None:
        """Initialize a stored bipolar sequence and transition matrix.

        Parameters
        ----------
        n_units:
            Pattern dimensionality.
        sequence_len:
            Number of stored sequence states.
        steps:
            Number of transition-recall steps.
        """
        super().__init__()
        self.steps = steps
        patterns = torch.sign(torch.randn(sequence_len, n_units))
        transition = patterns[1:].T @ patterns[:-1] / n_units
        transition = transition + torch.outer(patterns[0], patterns[-1]) / n_units
        self.register_buffer("transition", transition)

    def forward(self, state: Tensor) -> Tensor:
        """Recall successive states from an asymmetric temporal association.

        Parameters
        ----------
        state:
            Current sequence cue with shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Smooth recalled state after several sequence transitions.
        """
        memory = state
        for _ in range(self.steps):
            memory = torch.tanh(memory @ self.transition.T)
        return memory


def build_associatron() -> nn.Module:
    """Build a small Associatron.

    Returns
    -------
    nn.Module
        Configured ``Associatron`` instance.
    """
    return Associatron()


def example_input_associatron() -> Tensor:
    """Create an Associatron cue.

    Returns
    -------
    Tensor
        Example cue with shape ``(1, 256)``.
    """
    return torch.sign(torch.randn(1, 256))


def build_morita_nonmonotone() -> nn.Module:
    """Build a small Morita nonmonotone associative memory.

    Returns
    -------
    nn.Module
        Configured ``MoritaNonmonotoneAssociativeMemory`` instance.
    """
    return MoritaNonmonotoneAssociativeMemory()


def example_input_morita_nonmonotone() -> Tensor:
    """Create a Morita associative-memory cue.

    Returns
    -------
    Tensor
        Example cue with shape ``(1, 256)``.
    """
    return torch.randn(1, 256)


def build_temporal_am() -> nn.Module:
    """Build a small temporal associative memory.

    Returns
    -------
    nn.Module
        Configured ``TemporalAssociativeMemory`` instance.
    """
    return TemporalAssociativeMemory()


def example_input_temporal_am() -> Tensor:
    """Create a temporal associative-memory cue.

    Returns
    -------
    Tensor
        Example cue with shape ``(1, 64)``.
    """
    return torch.sign(torch.randn(1, 64))


MENAGERIE_ENTRIES = [
    ("Associatron (Nakano)", "build_associatron", "example_input_associatron", "1972", "MB1"),
    (
        "Morita Nonmonotone Associative Memory",
        "build_morita_nonmonotone",
        "example_input_morita_nonmonotone",
        "1993",
        "MB1",
    ),
    (
        "Temporal Associative Memory (TAM, Kosko)",
        "build_temporal_am",
        "example_input_temporal_am",
        "1988",
        "MB1",
    ),
]
