"""Replay over a captured generation-style trace.

What this demonstrates
----------------------
Capture a tiny recurrent loop that emits several steps, then replay a
post-hoc intervention over the saved DAG. Real language-model generation uses
the same replay idea, but this example avoids external dependencies.

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
    """Capture a generation trace and replay the first ReLU intervention."""

    torch.manual_seed(10)
    model = TinyGenerator().eval()
    x = torch.randn(2, 4)
    log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
    first_relu = log.find_sites(tl.func("relu"), max_fanout=3).labels()[0]

    edited = log.fork("generation_patch")
    edited.attach_hooks(tl.label(first_relu), tl.zero_ablate()).replay()

    assert edited.layer_list[-1].activation.shape == (2, 3, 4)
    assert edited.last_run_records()[-1].site_label == first_relu
    assert not torch.allclose(log.layer_list[-1].activation, edited.layer_list[-1].activation)


if __name__ == "__main__":
    main()
