"""NEAT phenotype network, 2002, Stanley and Miikkulainen.

Paper: "Evolving Neural Networks through Augmenting Topologies."
This minimal phenotype uses a fixed enabled feedforward topology with learned edge weights;
the evolutionary speciation, mutation, and crossover machinery is omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NEATPhenotypeNetwork(nn.Module):
    """Trace-clean evolved-topology-style directed acyclic network."""

    def __init__(self, n_in: int = 8, n_hidden: int = 6, n_out: int = 3) -> None:
        """Initialize enabled connection weights for a compact phenotype.

        Parameters
        ----------
        n_in
            Number of input features.
        n_hidden
            Number of hidden phenotype nodes.
        n_out
            Number of output nodes.
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.input_weights = nn.Parameter(torch.randn(n_hidden, n_in) * 0.2)
        self.hidden_weights = nn.Parameter(torch.randn(n_hidden, n_hidden) * 0.15)
        self.output_weights = nn.Parameter(torch.randn(n_out, n_hidden) * 0.2)
        self.output_skip = nn.Parameter(torch.randn(n_out, n_in) * 0.05)
        self.register_buffer("dag_mask", torch.tril(torch.ones(n_hidden, n_hidden), diagonal=-1))

    def _activate_node(self, preactivation: Tensor, node_index: int) -> Tensor:
        """Apply a fixed heterogeneous NEAT node activation.

        Parameters
        ----------
        preactivation
            Node preactivation tensor.
        node_index
            Static node index selecting the activation family.

        Returns
        -------
        Tensor
            Activated node output.
        """
        if node_index % 3 == 0:
            return torch.sigmoid(preactivation)
        if node_index % 3 == 1:
            return torch.tanh(preactivation)
        return torch.relu(preactivation)

    def forward(self, x: Tensor) -> Tensor:
        """Propagate inputs through enabled edges in topological order.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_in)``.

        Returns
        -------
        Tensor
            Output activation tensor.
        """
        hidden_states: list[Tensor] = []
        masked_hidden = self.hidden_weights * self.dag_mask
        for node_index in range(self.n_hidden):
            feedforward = x @ self.input_weights[node_index]
            if hidden_states:
                previous = torch.stack(hidden_states, dim=-1)
                recurrent = previous @ masked_hidden[node_index, :node_index]
                preactivation = feedforward + recurrent
            else:
                preactivation = feedforward
            hidden_states.append(self._activate_node(preactivation, node_index))
        hidden = torch.stack(hidden_states, dim=-1)
        return torch.tanh(hidden @ self.output_weights.T + x @ self.output_skip.T)


def build() -> nn.Module:
    """Build a compact NEAT phenotype network.

    Returns
    -------
    nn.Module
        Configured ``NEATPhenotypeNetwork`` instance.
    """
    return NEATPhenotypeNetwork()


def example_input() -> Tensor:
    """Create an example NEAT input vector.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 8)``.
    """
    return torch.randn(1, 8)


MENAGERIE_ENTRIES = [
    ("NEAT Phenotype Network (random topology forward)", "build", "example_input", "2002", "DD")
]
