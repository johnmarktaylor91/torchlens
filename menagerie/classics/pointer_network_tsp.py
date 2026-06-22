"""Pointer Network for planar TSP-style sequence pointing.

Vinyals, Fortunato, and Jaitly (NeurIPS 2015) introduced Pointer Networks, an
encoder-decoder architecture where decoder attention scores are the output
distribution over input positions.  This compact TSP variant encodes 2-D city
coordinates with an LSTM and uses an LSTMCell decoder that points to input
cities at each output step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointerNetworkTSP(nn.Module):
    """Compact pointer network over 2-D city coordinates."""

    def __init__(self, hidden: int = 32, steps: int = 8) -> None:
        """Initialize the pointer network.

        Parameters
        ----------
        hidden:
            LSTM hidden width.
        steps:
            Number of pointer decoding steps.
        """

        super().__init__()
        self.steps = steps
        self.embed = nn.Linear(2, hidden)
        self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.decoder = nn.LSTMCell(hidden, hidden)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.start = nn.Parameter(torch.zeros(1, hidden))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Return pointer logits over input cities for each decoding step.

        Parameters
        ----------
        points:
            City coordinates of shape ``(B, N, 2)``.

        Returns
        -------
        torch.Tensor
            Pointer logits of shape ``(B, steps, N)``.
        """

        enc, (h, c) = self.encoder(self.embed(points))
        dec_in = self.start.expand(points.shape[0], -1)
        keys = self.key(enc)
        logits = []
        hidden = h.squeeze(0)
        cell = c.squeeze(0)
        for _ in range(self.steps):
            hidden, cell = self.decoder(dec_in, (hidden, cell))
            score = self.v(torch.tanh(self.query(hidden).unsqueeze(1) + keys)).squeeze(-1)
            probs = torch.softmax(score, dim=-1)
            dec_in = torch.bmm(probs.unsqueeze(1), enc).squeeze(1)
            logits.append(score)
        return torch.stack(logits, dim=1)


def build() -> nn.Module:
    """Build the compact TSP pointer network.

    Returns
    -------
    nn.Module
        Random-init pointer network in evaluation mode.
    """

    return PointerNetworkTSP().eval()


def example_input() -> torch.Tensor:
    """Return a compact set of 2-D cities.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 8, 2)``.
    """

    return torch.rand(1, 8, 2)


MENAGERIE_ENTRIES = [
    ("pointer_network_tsp", "build", "example_input", "2015", "E3"),
]
