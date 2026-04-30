"""Attach a linear-probe readout with hooks.

What this demonstrates
----------------------
Hooks can collect side-channel readouts through ``hook.run_ctx`` while leaving
the activation unchanged.

How to run
----------
``python examples/intervention/12_linear_probe_attachment.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.hooks import HookContext


class TinyMLP(nn.Module):
    """Small model for readout hooks."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out_proj(torch.relu(self.in_proj(x)))


def main() -> None:
    """Collect a probe score during replay."""

    torch.manual_seed(12)
    model = TinyMLP().eval()
    x = torch.randn(2, 8)
    probe = nn.Linear(8, 1, bias=False).eval()
    log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)

    def readout(activation: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Store probe scores and leave the activation unchanged."""

        hook.run_ctx.setdefault("probe_scores", []).append(probe(activation).detach())
        return activation

    edited = log.fork("probe")
    edited.attach_hooks(tl.func("relu"), readout).replay()

    scores = edited.last_run_ctx["probe_scores"]
    assert len(scores) == 1
    assert scores[0].shape == (2, 1)
    assert torch.allclose(log.layer_list[-1].activation, edited.layer_list[-1].activation)


if __name__ == "__main__":
    main()
