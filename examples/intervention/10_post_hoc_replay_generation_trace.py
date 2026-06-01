"""Rerun a generation-style trace with a post-hoc intervention.

What this demonstrates
----------------------
Capture a tiny recurrent loop that emits several steps, then attach a
post-hoc intervention and rerun the model. Real language-model generation uses
the same intervention idea, but this example avoids external dependencies.

How to run
----------
``python examples/intervention/10_post_hoc_replay_generation_trace.py``

Runnable by default.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class TinyGenerator(nn.Module):
    """Toy generation trace with repeated ReLU operations."""

    def __init__(self) -> None:
        """Initialize the transition."""

        super().__init__()
        self.transition = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three recurrent-style update steps."""

        state = x
        outputs = []
        for _ in range(3):
            state = torch.relu(self.transition(state))
            outputs.append(state)
        return torch.stack(outputs, dim=1)


def main() -> None:
    """Capture a generation trace and rerun a ReLU intervention."""

    torch.manual_seed(10)
    model = TinyGenerator().eval()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, intervention_ready=True)
    relu_sites = log.find_sites(tl.func("relu"), max_fanout=3)

    edited = log.fork("generation_patch")
    edited.attach_hooks(tl.func("relu"), tl.zero_ablate()).rerun(model, x)

    assert edited.layer_list[-1].out.shape == (2, 3, 4)
    assert len(relu_sites) == 3
    assert edited.last_run["hooks"] == 1
    assert edited.last_run["engine"] == "rerun"
    assert not torch.allclose(log.layer_list[-1].out, edited.layer_list[-1].out)


if __name__ == "__main__":
    main()
