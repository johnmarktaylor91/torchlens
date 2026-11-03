"""Fixwave-2 FW2-POLISH pins for R19-3/4/5/7: module-cluster honesty.

The dashed cluster style is a semantic claim ("no input ancestor"). The b6
probes found it lying in two ways: the input-connectivity propagation broke
after the outermost module (a tautological loop guard), and clusters whose
ops were condensed away were re-emitted as empty dashed boxes.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _Inner(nn.Module):
    """Module whose ONLY edges cross its boundary (no interior op-op edge)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class _Outer(nn.Module):
    """Nested wrapper reproducing the b6 R19-4 dashed-cluster probe."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.wrap = nn.Sequential(_Inner())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wrap(self.lin(x))


def test_within_module_segment_discloses_hidden_atomic_module_calls() -> None:
    """A within-module segment names the atomic module calls it hides.

    REVIEWED rebaseline (T9, grind-p3): this pin previously asserted the
    label stayed a plain range ("spans" absent). That silence was the
    atomic undercount defect itself — the spanned-homes walk inherited the
    renderer's effective-stack atomic drop, so a segment hiding ops from
    FOUR distinct atomic module calls (Linear/ReLU children) reported zero
    homes and disclosed nothing. Hidden atomic module calls are containment
    facts like any other and must be disclosed.
    """

    from types import SimpleNamespace

    from torchlens._visualization._segment_descriptors import _make_op_segment_descriptor

    deep = nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU()))
    trace = tl.trace(deep, torch.randn(1, 4))
    context = SimpleNamespace(vis_mode="unrolled")
    labels = tuple(label for label in ("linear_1_1", "relu_1_2", "linear_2_3", "relu_2_4"))
    descriptor = _make_op_segment_descriptor(trace, context, labels, labels)
    assert "spans @0.0:1, @0.1:1, @0.2:1, +1 more" in descriptor.label, descriptor.label
