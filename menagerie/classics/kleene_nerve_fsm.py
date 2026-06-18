"""Kleene nerve-net finite automaton, 1956, as threshold state transitions.

Paper: Kleene 1956, "Representation of Events in Nerve Nets and Finite Automata."
A one-hot state register follows deterministic transitions driven by an input
symbol stream, illustrating how McCulloch-Pitts-style nets realize regular languages.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Kleene nerve-net finite automaton", "build", "example_input", "1956", "CA")]


class NerveNetFSM(nn.Module):
    """Finite-state recognizer implemented as tensor transition dynamics."""

    def __init__(self, n_states: int = 3, n_symbols: int = 2) -> None:
        """Initialize transition and accepting-state tables.

        Parameters
        ----------
        n_states
            Number of automaton states.
        n_symbols
            Number of input symbols.
        """
        super().__init__()
        transition = torch.zeros(n_symbols, n_states, n_states)
        transition[0, 0, 1] = 1.0
        transition[1, 0, 0] = 1.0
        transition[0, 1, 1] = 1.0
        transition[1, 1, 2] = 1.0
        transition[0, 2, 1] = 1.0
        transition[1, 2, 0] = 1.0
        self.register_buffer("transition", transition[:, :n_states, :n_states])
        self.register_buffer("start", torch.eye(n_states)[0])
        self.register_buffer("accepting", torch.eye(n_states)[2])
        self.n_symbols = n_symbols

    def forward(self, symbols: Tensor) -> Tensor:
        """Run the automaton over a batch of symbol sequences.

        Parameters
        ----------
        symbols
            Symbol indices with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Accepting score for each sequence.
        """
        batch = symbols.shape[0]
        state = self.start.unsqueeze(0).repeat(batch, 1)
        one_hot = torch.nn.functional.one_hot(symbols, num_classes=self.n_symbols).to(state.dtype)
        for step in range(symbols.shape[1]):
            matrix = torch.einsum("bs,sij->bij", one_hot[:, step], self.transition)
            state = torch.bmm(state.unsqueeze(1), matrix).squeeze(1)
        return state @ self.accepting


def build() -> nn.Module:
    """Build a small Kleene finite automaton net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return NerveNetFSM()


def example_input() -> Tensor:
    """Create a symbol sequence example.

    Returns
    -------
    Tensor
        Example symbols with shape ``(2, 5)``.
    """
    return torch.tensor([[0, 1, 0, 1, 1], [1, 0, 0, 1, 0]], dtype=torch.long)
