"""A module built inside ``forward`` has attributed parameters.

Its Parameters are created by in-place initializer writes (``kaiming_uniform_`` and
``uniform_`` on freshly allocated storage) inside the captured pass. Those writes are the
Parameters' only origin, so the live Parameter takes the initializer op's label and the
module's later read binds to that op: the initializer ops stay in the graph instead of
being pruned as orphans, the read has provenance, and replay re-derives the weights
(the initializers restore the recorded RNG stream).
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl


class _ModuleInForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(4, 4)
        self.late: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.late = nn.Linear(4, 3)
        return self.late(torch.relu(self.pre(x)))


def test_forward_created_parameters_bind_to_their_initializer_ops() -> None:
    torch.manual_seed(0)
    trace = tl.trace(_ModuleInForward(), torch.randn(2, 4))
    by_label = {op.label.split(":")[0]: op for op in trace.ops}
    late = [op for op in trace.ops if op.func_name == "linear"][-1]
    assert not late.unattributed_tensor_args
    parent_funcs = sorted(by_label[label.split(":")[0]].func_name for label in late.parents)
    assert parent_funcs == ["relu", "uniform_", "uniform_"]


@pytest.mark.smoke
def test_forward_created_module_validates() -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_ModuleInForward(), torch.randn(2, 4), scope="forward") is True
