"""Attach an SAE-style helper with ``splice_module``.

What this demonstrates
----------------------
Use an ``nn.Module`` as a black-box activation splice. The module here mimics
an SAE decode-after-encode operation while preserving shape.

How to run
----------
``python examples/intervention/11_sae_attachment.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class TinyMLP(nn.Module):
    """Small model with one ReLU site."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out_proj(torch.relu(self.in_proj(x)))


class SAEStyleSplice(nn.Module):
    """Shape-preserving SAE-style transform."""

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        """Return a sparse-ish reconstruction."""

        return torch.relu(activation) * 0.5


def main() -> None:
    """Attach a splice module at the ReLU site."""

    torch.manual_seed(11)
    model = TinyMLP().eval()
    x = torch.randn(2, 8)
    log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)

    edited = log.fork("sae_splice")
    edited.attach_hooks(tl.func("relu"), tl.splice_module(SAEStyleSplice())).replay()

    assert edited.last_run_records()[-1].helper_name == "splice_module"
    assert not torch.allclose(log.layer_list[-1].activation, edited.layer_list[-1].activation)


if __name__ == "__main__":
    main()
