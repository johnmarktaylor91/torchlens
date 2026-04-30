"""Compare multiple ModelLogs with Bundle.

What this demonstrates
----------------------
Construct a ``Bundle``, access one node across members, compute per-member
metrics, and compare activations at a shared site.

How to run
----------
``python examples/intervention/07_bundle_comparison.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class TinyMLP(nn.Module):
    """Small model for Bundle comparison."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out_proj(torch.relu(self.in_proj(x)))


def main() -> None:
    """Build a bundle and compare one shared ReLU node."""

    torch.manual_seed(7)
    model = TinyMLP().eval()
    x = torch.randn(2, 8)
    clean = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True, name="clean")
    zero = clean.fork("zero")
    zero.attach_hooks(tl.func("relu"), tl.zero_ablate()).replay()

    bundle = tl.bundle({"clean": clean, "zero": zero}, baseline="clean")
    node = bundle.node(tl.func("relu"))
    sizes = bundle.metric(lambda member: len(member.layer_list))
    comparison = bundle.compare_at(tl.func("relu"))

    assert set(node.activations) == {"clean", "zero"}
    assert sizes["clean"] == sizes["zero"]
    assert comparison.shape == (2, 2)


if __name__ == "__main__":
    main()
