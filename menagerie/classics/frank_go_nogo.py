"""Frank Go/NoGo dopamine basal-ganglia model, 2004, Michael Frank.

Paper: "By carrot or by stick: Cognitive reinforcement learning in parkinsonism."
Separate D1 Go and D2 NoGo pathways transform cortical state into action evidence;
dopamine-gated learning helpers are omitted from this forward-only model.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Frank Go/NoGo Dopamine BG Model", "build", "example_input", "2004", "DB")]


class FrankGoNoGo(nn.Module):
    """Dopamine-pathway action selector with Go and NoGo activations."""

    def __init__(self, n_state: int = 32, n_hidden: int = 24, n_actions: int = 6) -> None:
        """Initialize cortical, Go, and NoGo transforms.

        Parameters
        ----------
        n_state
            Cortical state-vector dimensionality.
        n_hidden
            Hidden corticostriatal feature count.
        n_actions
            Number of candidate actions.
        """
        super().__init__()
        self.cortex = nn.Linear(n_state, n_hidden)
        self.go = nn.Linear(n_hidden, n_actions)
        self.nogo = nn.Linear(n_hidden, n_actions)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute action logits and separate pathway activations.

        Parameters
        ----------
        state
            State tensor of shape ``(batch, n_state)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Action logits, Go activations, and NoGo activations.
        """
        hidden = torch.relu(self.cortex(state))
        go = torch.relu(self.go(hidden))
        nogo = torch.relu(self.nogo(hidden))
        logits = go - nogo
        return logits, go, nogo


def build() -> nn.Module:
    """Build a small Frank Go/NoGo model.

    Returns
    -------
    nn.Module
        Configured ``FrankGoNoGo`` instance.
    """
    return FrankGoNoGo()


def example_input() -> Tensor:
    """Return a cortical state example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 32)``.
    """
    return torch.randn(1, 32)
