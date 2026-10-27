"""A rolled layer whose passes are both an ``if`` test and that ``if``'s body.

GCEPNet's EP step evaluates ``lamda < 0`` as an ``if`` test and again inside the
body (``if torch.any(lamda < 0): indices = torch.where(lamda < 0)``). Both ``lt``
calls share one parent and roll into one layer, so the layer's ``body`` role was
copied onto the test passes too, whose own ``conditional_branch_stack`` is empty
(conditional invariant 15; Menagerie rung 4: f-gcepnet). Each pass now holds a
``body`` role only when its own branch stack holds that arm.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

import torchlens as tl
from torchlens.validation import last_validation_failure


class _TestThenBodyRepeat(nn.Module):
    """The EP shape: the same comparison as an ``if`` test and inside its body."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lamda = x.clone()
        for _ in range(4):
            lamda = self.lin(lamda) - 0.3
            if torch.any(lamda < 0):
                indices = torch.where(lamda < 0)
                lamda[indices] = 0.5
        return lamda


def _trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(_TestThenBodyRepeat().eval(), torch.randn(3, 4))


def test_body_roles_only_on_passes_inside_the_body() -> None:
    trace = _trace()
    lt_passes = [op for op in trace.ops if op.func_name == "__lt__"]
    rolled = [op for op in lt_passes if op.num_passes > 1]
    assert rolled, "the test and body comparisons should roll into one layer"
    in_body = [op for op in rolled if op.conditional_branch_stack]
    outside = [op for op in rolled if not op.conditional_branch_stack]
    assert in_body and outside
    assert all(op.is_in_conditional_body for op in in_body)
    assert not any(op.is_in_conditional_body for op in outside)
    for op in trace.ops:
        if any(role.role == "body" for role in op.in_conditionals or []):
            assert op.conditional_branch_stack, op.label


def test_rolled_test_and_body_validates() -> None:
    torch.manual_seed(0)
    model = _TestThenBodyRepeat().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, torch.randn(3, 4), scope="forward") is True, (
            last_validation_failure()
        )
