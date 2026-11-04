"""``Op.is_in_conditional_body`` follows the glossary: true inside an arm only.

The ops that compute a branch test run before any arm is entered, so they are
outside every body; the ops an arm runs are inside it, including the ones that
feed the model output. The Op answer agrees with its Layer's.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl

# No smoke mark: the smoke tier sits at its size ceiling (tests/conftest.py).


class ThenFeedsOutput(nn.Module):
    """The then arm's Linear feeds the output; the else arm negates."""

    def __init__(self) -> None:
        super().__init__()
        self.arm = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:  # noqa: SIM108 -- the test needs an if statement with an else arm
            y = self.arm(x)
        else:
            y = -x
        return y + 1


def test_test_ops_are_outside_and_arm_ops_inside_the_body() -> None:
    trace = tl.trace(ThenFeedsOutput(), torch.ones(1, 4))
    labels = list(trace.layer_labels)
    by_prefix = {label.rsplit("_", 2)[0]: label for label in labels}
    for test_prefix in ("sum", "gt"):
        test_op = trace[by_prefix[test_prefix]]
        assert test_op.is_in_conditional_body is False, test_prefix
    arm_op = trace[by_prefix["linear"]]
    assert arm_op.has_output_descendant
    assert arm_op.is_in_conditional_body is True
    assert trace[by_prefix["add"]].is_in_conditional_body is False


def test_op_and_layer_agree_on_body_membership() -> None:
    trace = tl.trace(ThenFeedsOutput(), torch.ones(1, 4))
    for label in trace.layer_labels:
        layer = trace.layers[label]
        assert trace[label].is_in_conditional_body == layer.is_in_conditional_body, label
