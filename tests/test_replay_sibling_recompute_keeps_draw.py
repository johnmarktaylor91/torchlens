"""A fork edit must not redraw a random op whose inputs replay unchanged.

The replay cone over-approximates: a hooked member's call-group siblings and
their consumers are recomputed as well. Recomputing ``exp(h)`` from the same
``h`` yields the same value, so ``randn_like(std)`` downstream of it has no
changed argument; without recorded RNG state it must keep its captured draw,
not draw fresh numbers (which would change sites outside the edit's cone and
make the fork's output unrelated to the captured one).
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _SiblingDraw(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h = torch.split(x, [2, 2], dim=-1)
        std = torch.exp(h)
        eps = torch.randn_like(std)
        return out.sum(-1, keepdim=True) + eps * std


def _single(trace: tl.Trace, func_name: str) -> tl.Op:
    (op,) = [op for op in trace.ops if op.func_name == func_name]
    return op


def test_fork_edit_on_a_call_group_member_keeps_the_sibling_fed_draw() -> None:
    model = _SiblingDraw().eval()
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    draw, std = _single(trace, "randn_like"), _single(trace, "exp")
    members = [op for op in trace.ops if op.func_name == "split"]
    assert len(members) == 2
    (hooked,) = [op for op in members if std.label not in op.children]

    fork = trace.fork()
    fork.do(tl.label(hooked.label), tl.scale(0.5))

    assert torch.equal(fork[std.label].out, std.out)
    assert torch.equal(fork[draw.label].out, draw.out), "an unseeded draw was redrawn"
    assert fork[hooked.label].is_edited
    torch.testing.assert_close(fork[hooked.label].out, hooked.out * 0.5)
    expected = trace.ops[-1].out - 0.5 * hooked.out.sum(-1, keepdim=True)
    torch.testing.assert_close(fork.ops[-1].out, expected)
