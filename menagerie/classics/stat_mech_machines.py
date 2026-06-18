"""Statistical-mechanics neural machines, 1980-2002.

Paper: Gardner 1988, "The space of interactions in neural network models";
Kinzel and Kanter 2002, "Interacting neural networks and cryptography".
Tiny sign, erf, parity, and spin-glass modules expose the canonical order
parameters and energies while omitting training protocols.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _bipolar_sign(x: Tensor) -> Tensor:
    """Return a traceable bipolar sign with zeros mapped to one.

    Parameters
    ----------
    x
        Input tensor.

    Returns
    -------
    Tensor
        Tensor containing ``-1`` or ``1`` values.
    """
    return torch.where(x >= 0.0, torch.ones_like(x), -torch.ones_like(x))


class SoftCommitteeMachine(nn.Module):
    """K-unit soft committee machine with visible overlap matrix."""

    def __init__(self, d_in: int = 16, n_hidden: int = 5) -> None:
        """Initialize random student and teacher committee weights.

        Parameters
        ----------
        d_in
            Input dimensionality.
        n_hidden
            Number of committee units.
        """
        super().__init__()
        self.d_in = d_in
        self.weight = nn.Parameter(torch.randn(n_hidden, d_in) * 0.2)
        self.readout = nn.Parameter(torch.randn(n_hidden) * 0.2)
        self.register_buffer("teacher_weight", torch.randn(n_hidden, d_in) * 0.2)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the erf committee and append mean order parameters.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Output, mean student overlap, and mean teacher overlap.
        """
        hidden = torch.erf((x @ self.weight.T) / math.sqrt(self.d_in))
        out = hidden @ self.readout
        q_mean = (self.weight @ self.weight.T / self.d_in).mean().expand_as(out)
        r_mean = (self.weight @ self.teacher_weight.T / self.d_in).mean().expand_as(out)
        return torch.stack((out, q_mean, r_mean), dim=-1)


class HardCommitteeMachine(nn.Module):
    """Majority vote of hard-threshold perceptrons."""

    def __init__(self, d_in: int = 15, n_hidden: int = 5) -> None:
        """Initialize hard committee weights.

        Parameters
        ----------
        d_in
            Input dimensionality.
        n_hidden
            Number of sign perceptrons.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_hidden, d_in))

    def forward(self, x: Tensor) -> Tensor:
        """Compute sign units and a sign-of-sum majority head.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Majority output followed by hidden signs.
        """
        sigma = _bipolar_sign(x @ self.weight.T)
        vote = _bipolar_sign(sigma.sum(dim=-1, keepdim=True))
        return torch.cat((vote, sigma), dim=-1)


class ParityMachine(nn.Module):
    """Overlapping K-unit parity machine with product head."""

    def __init__(self, d_in: int = 16, n_hidden: int = 4) -> None:
        """Initialize parity perceptrons.

        Parameters
        ----------
        d_in
            Input dimensionality.
        n_hidden
            Number of hidden parity units.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_hidden, d_in))

    def forward(self, x: Tensor) -> Tensor:
        """Compute hidden signs and their parity product.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Parity output followed by hidden signs.
        """
        sigma = _bipolar_sign(x @ self.weight.T)
        tau = sigma.prod(dim=-1, keepdim=True)
        return torch.cat((tau, sigma), dim=-1)


class TreeParityMachine(nn.Module):
    """Non-overlapping tree parity machine used in neural cryptography."""

    def __init__(self, n_blocks: int = 3, block_size: int = 5, level: int = 3) -> None:
        """Initialize discrete tree-parity weights as a fixed buffer.

        Parameters
        ----------
        n_blocks
            Number of hidden tree units.
        block_size
            Receptive-field size per hidden unit.
        level
            Integer weight bound from the synchronization protocol.
        """
        super().__init__()
        weight = torch.randint(-level, level + 1, (n_blocks, block_size)).float()
        self.register_buffer("weight", weight)

    def forward(self, x: Tensor) -> Tensor:
        """Compute blockwise sign units and parity.

        Parameters
        ----------
        x
            Bipolar input tensor with shape ``(batch, n_blocks, block_size)``.

        Returns
        -------
        Tensor
            Parity output followed by hidden signs.
        """
        sigma = _bipolar_sign((x * self.weight).sum(dim=-1))
        tau = sigma.prod(dim=-1, keepdim=True)
        return torch.cat((tau, sigma), dim=-1)


class GardnerPerceptron(nn.Module):
    """Spherical perceptron exposing a stability margin proxy."""

    def __init__(self, d_in: int = 18) -> None:
        """Initialize a single perceptron weight.

        Parameters
        ----------
        d_in
            Input dimensionality.
        """
        super().__init__()
        self.d_in = d_in
        self.weight = nn.Parameter(torch.randn(d_in))

    def forward(self, x: Tensor) -> Tensor:
        """Compute sign output and per-pattern stability.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Sign output and normalized preactivation stability.
        """
        w = self.weight / self.weight.norm().clamp_min(1e-6) * math.sqrt(self.d_in)
        stability = (x @ w) / math.sqrt(self.d_in)
        out = _bipolar_sign(stability)
        return torch.stack((out, stability), dim=-1)


class PSpinMemory(nn.Module):
    """Tiny p-spin memory with one traceable zero-temperature descent step."""

    def __init__(self, n_spins: int = 8) -> None:
        """Initialize a third-order spin-glass coupling tensor.

        Parameters
        ----------
        n_spins
            Number of bipolar spins.
        """
        super().__init__()
        scale = math.sqrt(3.0 / (n_spins**2))
        self.register_buffer("couplings", torch.randn(n_spins, n_spins, n_spins) * scale)

    def forward(self, spins: Tensor) -> Tensor:
        """Return energy and a one-step relaxed spin state.

        Parameters
        ----------
        spins
            Bipolar spin tensor with shape ``(batch, n_spins)``.

        Returns
        -------
        Tensor
            Energy followed by relaxed spins.
        """
        s = _bipolar_sign(spins)
        energy = -torch.einsum("bi,bj,bk,ijk->b", s, s, s, self.couplings)
        field = torch.einsum("bj,bk,ijk->bi", s, s, self.couplings)
        delta = 2.0 * s * field
        relaxed = torch.where(delta < 0.0, -s, s)
        return torch.cat((energy.unsqueeze(-1), relaxed), dim=-1)


def build_soft_committee() -> nn.Module:
    """Build a soft committee machine.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return SoftCommitteeMachine()


def example_input_soft_committee() -> Tensor:
    """Return an example soft-committee input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return torch.randn(2, 16)


def build_hard_committee() -> nn.Module:
    """Build a hard committee machine.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return HardCommitteeMachine()


def example_input_hard_committee() -> Tensor:
    """Return an example hard-committee input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return torch.randn(2, 15)


def build_parity_machine() -> nn.Module:
    """Build a parity machine.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return ParityMachine()


def example_input_parity_machine() -> Tensor:
    """Return an example parity-machine input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return torch.randn(2, 16)


def build_tree_parity_machine() -> nn.Module:
    """Build a tree parity machine.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return TreeParityMachine()


def example_input_tree_parity_machine() -> Tensor:
    """Return an example tree-parity input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return _bipolar_sign(torch.randn(2, 3, 5))


def build_gardner_perceptron() -> nn.Module:
    """Build a Gardner-capacity perceptron.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return GardnerPerceptron()


def example_input_gardner_perceptron() -> Tensor:
    """Return an example Gardner perceptron input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return torch.randn(3, 18)


def build_p_spin_memory() -> nn.Module:
    """Build a p-spin memory.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return PSpinMemory()


def example_input_p_spin_memory() -> Tensor:
    """Return an example p-spin state.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return _bipolar_sign(torch.randn(2, 8))


MENAGERIE_ENTRIES = [
    (
        "Soft committee machine",
        "build_soft_committee",
        "example_input_soft_committee",
        "1995",
        "CH-A",
    ),
    (
        "Hard committee machine",
        "build_hard_committee",
        "example_input_hard_committee",
        "1990",
        "CH-A",
    ),
    ("Parity machine", "build_parity_machine", "example_input_parity_machine", "1989", "CH-A"),
    (
        "Tree parity machine",
        "build_tree_parity_machine",
        "example_input_tree_parity_machine",
        "2002",
        "CH-A",
    ),
    (
        "Gardner-capacity perceptron",
        "build_gardner_perceptron",
        "example_input_gardner_perceptron",
        "1988",
        "CH-A",
    ),
    (
        "p-spin / Random Energy Model memory",
        "build_p_spin_memory",
        "example_input_p_spin_memory",
        "1980",
        "CH-A",
    ),
]
