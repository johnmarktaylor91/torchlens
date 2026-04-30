"""Append compatible reruns for memory-constrained evaluation.

What this demonstrates
----------------------
Capture one small batch, then append another compatible batch with
``rerun(..., append=True)``. Append requires the same graph shape and
batch-independent helpers.

How to run
----------
``python examples/intervention/06_chunked_batching.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class TinyMLP(nn.Module):
    """Small append-compatible MLP."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out_proj(torch.relu(self.in_proj(x)))


def main() -> None:
    """Append a second batch along dimension zero."""

    torch.manual_seed(6)
    model = TinyMLP().eval()
    first = torch.randn(2, 8)
    second = torch.randn(3, 8)

    log = tl.log_forward_pass(model, first, vis_opt="none", intervention_ready=True)
    log.attach_hooks(tl.func("relu"), tl.scale(1.0), confirm_mutation=True)
    log.rerun(model, first)
    log.rerun(model, second, append=True)

    output = log.layer_list[-1].activation
    assert output.shape[0] == first.shape[0] + second.shape[0]
    assert log.operation_history[-1]["op"] == "append"
    assert log.is_appended is True


if __name__ == "__main__":
    main()
