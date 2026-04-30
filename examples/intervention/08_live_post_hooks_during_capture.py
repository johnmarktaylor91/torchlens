"""Run post-hooks during the original capture.

What this demonstrates
----------------------
Pass ``hooks=...`` to ``log_forward_pass`` so the intervention fires live while
the model is executing.

How to run
----------
``python examples/intervention/08_live_post_hooks_during_capture.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class ReluAdd(nn.Module):
    """Tiny model where ReLU output controls the final value."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU and add one."""

        return torch.relu(x) + 1


def main() -> None:
    """Capture with a live zero-ablation hook."""

    x = torch.tensor([[-2.0, 3.0]])
    clean = tl.log_forward_pass(ReluAdd(), x, vis_opt="none", intervention_ready=True)
    live = tl.log_forward_pass(
        ReluAdd(),
        x,
        vis_opt="none",
        intervention_ready=True,
        hooks={tl.func("relu"): tl.zero_ablate()},
    )

    assert torch.allclose(live.layer_list[-1].activation, torch.ones_like(x))
    assert not torch.allclose(clean.layer_list[-1].activation, live.layer_list[-1].activation)
    assert live.last_run_records()[-1].engine == "live"


if __name__ == "__main__":
    main()
