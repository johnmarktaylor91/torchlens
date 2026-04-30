"""First five minutes with TorchLens interventions.

What this demonstrates
----------------------
Capture a tiny model, discover the ReLU site, fork the log, zero-ablate that
site, and replay the saved DAG.

How to run
----------
``python examples/intervention/01_first_five_minutes.py``

Runnable by default. No external model downloads are required.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class TinyMLP(nn.Module):
    """Small deterministic MLP used throughout the intervention examples."""

    def __init__(self) -> None:
        """Initialize two linear layers."""

        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear-ReLU-linear network.

        Parameters
        ----------
        x:
            Batch of eight-dimensional inputs.

        Returns
        -------
        torch.Tensor
            Four-dimensional model output.
        """

        return self.out_proj(torch.relu(self.in_proj(x)))


def main() -> None:
    """Run the minimal capture plus replay intervention."""

    torch.manual_seed(1)
    model = TinyMLP().eval()
    x = torch.randn(2, 8)

    log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
    relu_site = log.find_sites(tl.func("relu")).first()
    assert relu_site.func_name == "relu"

    edited = log.fork("zero_relu")
    edited.attach_hooks(tl.func("relu"), tl.zero_ablate())
    edited.replay()

    clean_out = log.layer_list[-1].activation
    edited_out = edited.layer_list[-1].activation
    assert not torch.allclose(clean_out, edited_out)
    assert edited.last_run_records()[-1].helper_name == "zero_ablate"


if __name__ == "__main__":
    main()
