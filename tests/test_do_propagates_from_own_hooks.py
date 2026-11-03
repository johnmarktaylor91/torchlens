"""``do()`` propagates from the hooks it attached, not from every stored rule.

A trace captured under a recipe keeps that recipe as sticky rules. A later
``fork.do(site, edit)`` must open its replay cone at ``site`` only; the older
rules still fire wherever the cone re-executes their sites, but a site that
is not downstream of the edit is not touched. Re-executing such a site from
the recipe's own target is wrong in general and visibly so for an in-place
parameter write: the replay reads the parameter's live (already written)
value and the site's recorded value changes outside the edit's cone.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _SelfWriting(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
        self.proj = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.count.add_(1.0)
        with torch.no_grad():
            self.proj.bias.add_(0.01)
        return self.proj(x) + self.count


def test_fork_do_leaves_a_recipe_target_outside_its_cone_untouched() -> None:
    torch.manual_seed(0)
    model = _SelfWriting().eval()
    x = torch.randn(3, 4)
    trace = tl.trace(
        model,
        x,
        intervene=tl.when(tl.func("add_"), tl.scale(0.5)),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    adds = [op for op in trace.ops if op.func_name == "add_"]
    assert len(adds) == 2
    (count_add,) = [op for op in adds if op.out.ndim == 0]
    (bias_add,) = [op for op in adds if op.out.ndim == 1]
    assert count_add.is_edited and bias_add.is_edited

    fork = trace.fork()
    fork.do(tl.label(count_add.label), tl.scale(0.5))

    assert torch.equal(fork[bias_add.label].out, bias_add.out), (
        "a recipe target outside the edit's cone was re-executed"
    )
    torch.testing.assert_close(fork[count_add.label].out, count_add.out * 0.5)
    torch.testing.assert_close(fork.ops[-1].out, trace.ops[-1].out - 0.5 * count_add.out)
