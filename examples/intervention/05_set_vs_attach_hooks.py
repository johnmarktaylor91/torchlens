"""Choose between ``set`` and ``attach_hooks``.

What this demonstrates
----------------------
``set`` records a concrete one-shot replacement. ``attach_hooks`` records a
sticky callable/helper recipe that can be replayed or rerun repeatedly.

How to run
----------
``python examples/intervention/05_set_vs_attach_hooks.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class TinyMLP(nn.Module):
    """Small model for comparing mutation styles."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out_proj(torch.relu(self.in_proj(x)))


def main() -> None:
    """Show that static set and zero hook can produce the same result."""

    torch.manual_seed(5)
    model = TinyMLP().eval()
    x = torch.randn(2, 8)
    log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
    relu_shape = log.find_sites(tl.func("relu")).first().activation.shape

    set_log = log.fork("set")
    set_log.set(tl.func("relu"), torch.zeros(relu_shape), confirm_mutation=True).replay()

    hook_log = log.fork("hook")
    hook_log.attach_hooks(tl.func("relu"), tl.zero_ablate()).replay()

    assert torch.allclose(set_log.layer_list[-1].activation, hook_log.layer_list[-1].activation)
    assert set_log.operation_history[-2]["op"] == "set"
    assert hook_log.operation_history[-2]["op"] == "attach_hooks"


if __name__ == "__main__":
    main()
