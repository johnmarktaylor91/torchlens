"""Neural Relational Inference (NRI) for interacting systems.

Kipf, Fetaya, Wang, Welling, and Zemel, ICML 2018, "Neural Relational
Inference for Interacting Systems."  Source: https://github.com/ethanfetaya/NRI

NRI is a variational graph-discovery world model: an encoder observes trajectories
of multiple objects and infers a categorical latent edge type for every directed
object pair; a graph-neural decoder then rolls dynamics forward using messages
weighted by those inferred edge-type probabilities.

This compact random-init module preserves the architecture-defining pieces:

  - fully connected directed sender/receiver relation graph without self-edges,
  - trajectory encoder producing per-edge logits over relation types,
  - soft latent edge probabilities (deterministic trace-time analogue of the
    paper's Gumbel-softmax samples),
  - edge-type-specific message functions, receiver aggregation, GRU-style node
    update, and delta-state prediction.

Widths, object count, and rollout length are reduced for fast TorchLens tracing
and SVG drawing. Training losses, KL terms, and stochastic sampling are omitted
because the menagerie records inference graphs rather than objectives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _offdiag_indices(n_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sender and receiver indices for all directed non-self edges.

    Parameters
    ----------
    n_nodes:
        Number of objects in the fully connected interaction graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Sender and receiver index tensors, each with shape ``(n_edges,)``.
    """

    senders: list[int] = []
    receivers: list[int] = []
    for receiver in range(n_nodes):
        for sender in range(n_nodes):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    return torch.tensor(senders, dtype=torch.long), torch.tensor(receivers, dtype=torch.long)


class NRIEncoder(nn.Module):
    """Trajectory-to-edge encoder used by Neural Relational Inference."""

    def __init__(
        self, n_nodes: int, state_dim: int, hidden: int = 32, n_edge_types: int = 3
    ) -> None:
        """Initialize the compact MLP encoder.

        Parameters
        ----------
        n_nodes:
            Number of interacting objects.
        state_dim:
            Per-object state dimension at each time step.
        hidden:
            Hidden width for object and relation MLPs.
        n_edge_types:
            Number of categorical latent relation types.
        """

        super().__init__()
        self.n_nodes = n_nodes
        self.state_dim = state_dim
        self.obj_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.logits = nn.Linear(hidden, n_edge_types)

    def forward(
        self, trajectories: torch.Tensor, senders: torch.Tensor, receivers: torch.Tensor
    ) -> torch.Tensor:
        """Infer directed latent-relation logits from observed trajectories.

        Parameters
        ----------
        trajectories:
            Observed states with shape ``(batch, time, n_nodes, state_dim)``.
        senders:
            Sender node indices for each directed edge.
        receivers:
            Receiver node indices for each directed edge.

        Returns
        -------
        torch.Tensor
            Edge-type logits with shape ``(batch, n_edges, n_edge_types)``.
        """

        node_summary = trajectories.mean(dim=1)
        node_hidden = self.obj_mlp(node_summary)
        edge_input = torch.cat([node_hidden[:, senders], node_hidden[:, receivers]], dim=-1)
        edge_hidden = self.edge_mlp(edge_input)
        return self.logits(edge_hidden)


class NRIDecoderStep(nn.Module):
    """One NRI graph decoder step with edge-type-specific messages."""

    def __init__(self, state_dim: int, hidden: int = 32, n_edge_types: int = 3) -> None:
        """Initialize message functions, node update, and prediction head.

        Parameters
        ----------
        state_dim:
            Per-object state dimension.
        hidden:
            Hidden width for messages and recurrent node state.
        n_edge_types:
            Number of latent edge types used to weight message functions.
        """

        super().__init__()
        self.state_in = nn.Linear(state_dim, hidden)
        self.message_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden, hidden),
                    nn.ELU(),
                    nn.Linear(hidden, hidden),
                    nn.ELU(),
                )
                for _ in range(n_edge_types)
            ]
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.delta = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, state_dim)
        )

    def forward(
        self,
        state: torch.Tensor,
        hidden_state: torch.Tensor,
        edge_probs: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one dynamics step from inferred interaction probabilities.

        Parameters
        ----------
        state:
            Current object states with shape ``(batch, n_nodes, state_dim)``.
        hidden_state:
            Recurrent node states with shape ``(batch, n_nodes, hidden)``.
        edge_probs:
            Soft edge-type probabilities with shape ``(batch, n_edges, n_edge_types)``.
        senders:
            Sender node indices for each directed edge.
        receivers:
            Receiver node indices for each directed edge.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Predicted next state and updated recurrent node state.
        """

        node_hidden = hidden_state + self.state_in(state)
        edge_input = torch.cat([node_hidden[:, senders], node_hidden[:, receivers]], dim=-1)
        messages = []
        for edge_type, message_mlp in enumerate(self.message_mlps):
            messages.append(message_mlp(edge_input) * edge_probs[..., edge_type : edge_type + 1])
        edge_message = torch.stack(messages, dim=0).sum(dim=0)
        aggregated = edge_message.new_zeros(
            edge_message.shape[0], state.shape[1], edge_message.shape[-1]
        )
        index = receivers.view(1, -1, 1).expand(edge_message.shape[0], -1, edge_message.shape[-1])
        aggregated = aggregated.scatter_add(1, index, edge_message)
        flat_next = self.gru(
            aggregated.reshape(-1, aggregated.shape[-1]),
            hidden_state.reshape(-1, hidden_state.shape[-1]),
        )
        next_hidden = flat_next.reshape_as(hidden_state)
        next_state = state + self.delta(next_hidden)
        return next_state, next_hidden


class NeuralRelationalInference(nn.Module):
    """Compact NRI VAE-style relation encoder plus graph dynamics decoder."""

    def __init__(
        self,
        n_nodes: int = 5,
        state_dim: int = 4,
        hidden: int = 32,
        n_edge_types: int = 3,
        rollout_steps: int = 2,
    ) -> None:
        """Initialize a compact random NRI model.

        Parameters
        ----------
        n_nodes:
            Number of interacting objects.
        state_dim:
            Per-object state dimension.
        hidden:
            Hidden width for encoder and decoder.
        n_edge_types:
            Number of latent relation categories.
        rollout_steps:
            Number of decoder prediction steps to unroll.
        """

        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.rollout_steps = rollout_steps
        senders, receivers = _offdiag_indices(n_nodes)
        self.register_buffer("senders", senders)
        self.register_buffer("receivers", receivers)
        self.encoder = NRIEncoder(n_nodes, state_dim, hidden, n_edge_types)
        self.decoder = NRIDecoderStep(state_dim, hidden, n_edge_types)

    def forward(self, trajectories: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer latent graph probabilities and roll object dynamics forward.

        Parameters
        ----------
        trajectories:
            Observed object trajectories with shape ``(batch, time, n_nodes, state_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Predicted future states ``(batch, rollout, n_nodes, state_dim)`` and edge logits.
        """

        logits = self.encoder(trajectories, self.senders, self.receivers)
        edge_probs = torch.softmax(logits, dim=-1)
        state = trajectories[:, -1]
        hidden_state = state.new_zeros(state.shape[0], self.n_nodes, self.hidden)
        predictions = []
        for _ in range(self.rollout_steps):
            state, hidden_state = self.decoder(
                state, hidden_state, edge_probs, self.senders, self.receivers
            )
            predictions.append(state)
        return torch.stack(predictions, dim=1), logits


def build() -> nn.Module:
    """Build the compact Neural Relational Inference model.

    Returns
    -------
    nn.Module
        Random-initialized compact NRI module.
    """

    return NeuralRelationalInference()


def example_input() -> torch.Tensor:
    """Return a small trajectory tensor ``(batch, time, n_nodes, state_dim)``.

    Returns
    -------
    torch.Tensor
        Example interacting-object trajectory input.
    """

    return torch.randn(1, 4, 5, 4)


MENAGERIE_ENTRIES = [
    (
        "Neural Relational Inference (latent interaction graph VAE)",
        "build",
        "example_input",
        "2018",
        "DC",
    ),
]
