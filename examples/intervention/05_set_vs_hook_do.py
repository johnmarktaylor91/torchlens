"""Choose between a static value and a sticky hook in ``do``.

What this demonstrates
----------------------
``do(site, tensor)`` records a concrete one-shot replacement. ``do(site, hook)``
attaches a sticky callable/helper recipe that can be replayed or rerun repeatedly.

How to run
----------
``python examples/intervention/05_set_vs_hook_do.py``

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
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    relu_shape = log.find_sites(tl.func("relu")).first().out.shape

    set_log = log.fork("set")
    set_log.do(tl.func("relu"), torch.zeros(relu_shape))

    hook_log = log.fork("hook")
    hook_log.do(tl.func("relu"), tl.zero_ablate())

    assert torch.allclose(set_log.output_ops[-1].out, hook_log.output_ops[-1].out)
    set_ops = [row["op"] for row in set_log.state_history]
    hook_ops = [row["op"] for row in hook_log.state_history]
    assert "set" in set_ops and "attach_hooks" not in set_ops
    assert "attach_hooks" in hook_ops


if __name__ == "__main__":
    main()
