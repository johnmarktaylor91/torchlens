"""Intervene on an exact site after discovery.

What this demonstrates
----------------------
Use ``find_sites`` to inspect labels, then convert a discovered label into an
exact ``tl.label(...)`` selector for a portable intervention target.

How to run
----------
``python examples/intervention/02_exact_site_after_discovery.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class TinyMLP(nn.Module):
    """Small MLP with one ReLU site."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out_proj(torch.relu(self.in_proj(x)))


def main() -> None:
    """Discover a site and replay an exact-label intervention."""

    torch.manual_seed(2)
    model = TinyMLP().eval()
    x = torch.randn(2, 8)
    log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)

    table = log.find_sites(tl.func("relu"))
    exact_label = table.labels()[0]
    exact_site = tl.label(exact_label)

    edited = log.fork("exact_site")
    edited.do(exact_site, tl.scale(0.0), confirm_mutation=True)

    assert edited.last_run_records()[-1].site_label == exact_label
    assert not torch.allclose(log.layer_list[-1].activation, edited.layer_list[-1].activation)


if __name__ == "__main__":
    main()
