"""``x[idx] += y`` binds its write-back ``__setitem__`` to the base's own version.

Python runs ``x[idx] += y`` as ``v = x[idx]; v += y; x[idx] = v``. The view
mutation advances the base's label to the ``iadd`` op, so the write-back
``__setitem__(x, idx, v)`` used to record the parents
``('iadd_N', 'iadd_N')``: one label for two slots holding two different values
(the full base and the view), and argument-logging validation failed
(Menagerie rung 4: f-bsrnn-se, f-cltr). The base slot now binds to the base's
label from before the view mutation.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import last_validation_failure


class _SliceIadd(nn.Module):
    """The espnet BSRNN band-merge shape: ``m[..., a:b, :] += out`` in a loop."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.new_zeros(1, 2, 1, 6, 2)
        for i in range(2):
            out = self.lin(x).reshape(1, 2, 1, 3, 1).expand(1, 2, 1, 3, 2)
            m[..., i * 3 : (i + 1) * 3, :] += out
        return m.moveaxis(1, 2)


@pytest.mark.smoke
def test_slice_augmented_assignment_validates() -> None:
    torch.manual_seed(0)
    model = _SliceIadd().eval()
    x = torch.randn(1, 2, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="forward") is True, last_validation_failure()


def test_writeback_setitem_parents_are_the_base_and_the_view() -> None:
    torch.manual_seed(0)
    trace = tl.trace(_SliceIadd().eval(), torch.randn(1, 2, 4))
    setitems = [op for op in trace.ops if op.func_name == "__setitem__"]
    assert len(setitems) == 2
    for op in setitems:
        base_parent = op.parent_arg_positions["args"][0]
        value_parent = op.parent_arg_positions["args"][2]
        assert base_parent != value_parent
        assert value_parent.startswith("iadd")
