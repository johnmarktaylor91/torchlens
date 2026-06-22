"""PonderNet adaptive-computation recurrent model.

Banino et al. (2021), "PonderNet: Learning to Ponder."  PonderNet repeatedly
updates a hidden state, predicts an output at every step, and predicts the
conditional probability of halting at that step; the final prediction is the
halting-probability-weighted mixture over step predictions.  This compact model
keeps the recurrent state, per-step prediction head, and exact halting
probability accumulation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactPonderNet(nn.Module):
    """Compact recurrent PonderNet classifier."""

    def __init__(
        self, in_features: int = 10, hidden: int = 32, classes: int = 4, steps: int = 6
    ) -> None:
        """Initialize the adaptive computation model.

        Parameters
        ----------
        in_features:
            Input feature count.
        hidden:
            Hidden state size.
        classes:
            Number of output classes.
        steps:
            Maximum ponder steps.
        """

        super().__init__()
        self.steps = steps
        self.input_proj = nn.Linear(in_features, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.pred = nn.Linear(hidden, classes)
        self.halt = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run recurrent pondering and accumulate halting probabilities.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, in_features)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Weighted logits, per-step logits, and halting probabilities.
        """

        inp = torch.tanh(self.input_proj(x))
        state = torch.zeros_like(inp)
        remaining = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        logits = []
        halt_probs = []
        for step in range(self.steps):
            state = self.cell(inp, state)
            step_logits = self.pred(state)
            conditional = torch.sigmoid(self.halt(state))
            if step == self.steps - 1:
                prob = remaining
            else:
                prob = remaining * conditional
                remaining = remaining * (1.0 - conditional)
            logits.append(step_logits)
            halt_probs.append(prob)
        stacked_logits = torch.stack(logits, dim=1)
        probs = torch.stack(halt_probs, dim=1)
        return {
            "logits": (stacked_logits * probs).sum(dim=1),
            "step_logits": stacked_logits,
            "halt_probs": probs.squeeze(-1),
        }


def build() -> nn.Module:
    """Build the compact PonderNet model.

    Returns
    -------
    nn.Module
        Random-init model in evaluation mode.
    """

    return CompactPonderNet().eval()


def example_input() -> torch.Tensor:
    """Return compact vector inputs.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(2, 10)``.
    """

    return torch.randn(2, 10)


MENAGERIE_ENTRIES = [
    ("PonderNet", "build", "example_input", "2021", "E5"),
]
