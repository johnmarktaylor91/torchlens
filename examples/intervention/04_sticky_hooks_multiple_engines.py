"""Use one sticky hook recipe with replay and rerun.

What this demonstrates
----------------------
Attach a hook once, replay over the saved graph, then rerun the original model
with the same active intervention spec.

How to run
----------
``python examples/intervention/04_sticky_hooks_multiple_engines.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class TinyMLP(nn.Module):
    """Small MLP with one ReLU intervention site."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out_proj(torch.relu(self.in_proj(x)))


def main() -> None:
    """Compare replay and rerun from one sticky hook spec."""

    torch.manual_seed(4)
    model = TinyMLP().eval()
    x = torch.randn(2, 8)
    clean = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)

    replayed = clean.fork("replay")
    replayed.attach_hooks(tl.func("relu"), tl.zero_ablate())
    replayed.replay()

    rerun = clean.fork("rerun")
    rerun.attach_hooks(tl.func("relu"), tl.zero_ablate())
    rerun.rerun(model, x)

    assert torch.allclose(replayed.layer_list[-1].activation, rerun.layer_list[-1].activation)
    assert replayed.last_run_records()[-1].engine == "replay"
    rerun_records = [
        record for layer in rerun.layer_list for record in getattr(layer, "intervention_log", ())
    ]
    assert rerun_records[-1].engine == "live"


if __name__ == "__main__":
    main()
