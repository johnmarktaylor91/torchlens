"""Steer with a per-position direction.

What this demonstrates
----------------------
Add a direction tensor with the same non-batch shape as the matched activation.
This mirrors per-position steering for sequence models while using a tiny
feed-forward toy.

How to run
----------
``python examples/intervention/14_per_position_steering.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class PositionModel(nn.Module):
    """Tiny model with a position axis."""

    def __init__(self) -> None:
        """Initialize projection."""

        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run projection and ReLU over batch-position-feature inputs."""

        return torch.relu(self.proj(x))


def main() -> None:
    """Apply per-position steering at the ReLU output."""

    torch.manual_seed(14)
    model = PositionModel().eval()
    x = torch.randn(2, 3, 4)
    log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
    direction = torch.zeros(3, 4)
    direction[1, :] = 0.5

    steered = log.fork("steered")
    steered.attach_hooks(tl.func("relu"), tl.steer(direction, magnitude=1.0)).replay()

    delta = (
        steered.find_sites(tl.func("relu")).first().activation
        - log.find_sites(tl.func("relu")).first().activation
    )
    assert torch.allclose(delta[:, 0, :], torch.zeros_like(delta[:, 0, :]))
    assert torch.allclose(delta[:, 1, :], torch.full_like(delta[:, 1, :], 0.5))


if __name__ == "__main__":
    main()
