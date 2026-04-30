"""Discover within a submodule before choosing a finer selector.

What this demonstrates
----------------------
Use ``tl.in_module("block")`` to restrict discovery, then compose it with
``tl.func("relu")`` for the intervention target.

How to run
----------
``python examples/intervention/09_submodule_discover_first.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class Block(nn.Module):
    """Named block containing a linear layer and ReLU."""

    def __init__(self) -> None:
        """Initialize block layers."""

        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return torch.relu(self.linear(x))


class Model(nn.Module):
    """Model with a named block and output projection."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.block = Block()
        self.out = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out(self.block(x))


def main() -> None:
    """Discover and intervene inside ``block``."""

    torch.manual_seed(9)
    model = Model().eval()
    x = torch.randn(2, 8)
    log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)

    block_sites = log.find_sites(tl.in_module("block"), max_fanout=4)
    target = tl.label(block_sites.where(lambda site: site.func_name == "relu").labels()[0])
    edited = log.fork("block_relu")
    edited.attach_hooks(target, tl.zero_ablate()).replay()

    assert any(site.func_name == "relu" for site in block_sites)
    assert edited.last_run_records()[-1].helper_name == "zero_ablate"
    assert not torch.allclose(log.layer_list[-1].activation, edited.layer_list[-1].activation)


if __name__ == "__main__":
    main()
