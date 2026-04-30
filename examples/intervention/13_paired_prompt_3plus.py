"""Compare three or more prompt variants in a Bundle.

What this demonstrates
----------------------
Capture three tensor "prompt" variants, package them into one ``Bundle``, and
compute both member-wise and joint metrics.

How to run
----------
``python examples/intervention/13_paired_prompt_3plus.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class TinyMLP(nn.Module):
    """Small model for 3+ input comparisons."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out_proj(torch.relu(self.in_proj(x)))


def main() -> None:
    """Build a three-member bundle and compare output norms."""

    torch.manual_seed(13)
    model = TinyMLP().eval()
    base = torch.randn(2, 8)
    logs = {
        "clean": tl.log_forward_pass(model, base, vis_opt="none", intervention_ready=True),
        "plus": tl.log_forward_pass(model, base + 0.2, vis_opt="none", intervention_ready=True),
        "minus": tl.log_forward_pass(model, base - 0.2, vis_opt="none", intervention_ready=True),
    }
    bundle = tl.bundle(logs, baseline="clean")

    norms = bundle.metric(lambda member: float(member.layer_list[-1].activation.norm()))
    count = bundle.joint_metric(lambda group: len(group.names))

    assert count == 3
    assert set(norms) == {"clean", "plus", "minus"}
    assert len(set(norms.values())) > 1


if __name__ == "__main__":
    main()
