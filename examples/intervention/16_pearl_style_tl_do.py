"""Use the top-level ``tl.do`` shortcut.

What this demonstrates
----------------------
``tl.do(log, ...)`` forwards to ``log.do(...)`` for concise one-shot
interventions.

How to run
----------
``python examples/intervention/16_pearl_style_tl_do.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class ReluAdd(nn.Module):
    """Tiny model where ReLU affects the output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU and add one."""

        return torch.relu(x) + 1


def main() -> None:
    """Run a top-level ``tl.do`` replay intervention."""

    x = torch.tensor([[-1.0, 2.0]])
    clean = tl.log_forward_pass(ReluAdd(), x, vis_opt="none", intervention_ready=True)
    edited = clean.fork("top_level_do")

    result = tl.do(edited, tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)

    assert result is edited
    assert torch.allclose(edited.layer_list[-1].activation, torch.ones_like(x))
    assert not torch.allclose(clean.layer_list[-1].activation, edited.layer_list[-1].activation)


if __name__ == "__main__":
    main()
