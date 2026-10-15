"""A model input with a zero-length axis has no receptive-field geometry, and says so.

The receptive-field solve used to build a public axis for every input axis, and a public
axis requires a positive extent, so any trace with an empty input (an empty batch, an edge
list with no edges) raised ``ValueError`` from the always-on receptive-field metadata
invariant, which made ``validate_forward_pass`` raise instead of returning a verdict. Such
an input has no units to map: its descriptors are ``UNSUPPORTED`` with a note, and the
other inputs of the same model keep their geometry.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _engine
from torchlens.receptive_field._types import ReceptiveFieldStatus


class _TwoInputs(nn.Module):
    """A dense input and an edge list that is empty for this call."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        return self.lin(x) + edges.float().sum()


def test_empty_batch_validates() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validation.validate_forward_pass(nn.Linear(4, 2), torch.randn(0, 4)) is True


def test_empty_second_input_validates_and_keeps_the_dense_input_geometry() -> None:
    model, x, edges = _TwoInputs(), torch.randn(3, 4), torch.zeros(2, 0, dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validation.validate_forward_pass(model, [x, edges]) is True
    trace = tl.trace(model, [x, edges])
    solution = _engine.solve(trace)
    statuses = {
        descriptor.input_op_label: descriptor.status
        for descriptors in solution.per_op.values()
        for descriptor in descriptors.values()
    }
    empty = [
        label for label, status in statuses.items() if status is ReceptiveFieldStatus.UNSUPPORTED
    ]
    assert len(empty) == 1
    assert any(status is not ReceptiveFieldStatus.UNSUPPORTED for status in statuses.values())
