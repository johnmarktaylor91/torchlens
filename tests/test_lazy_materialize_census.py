"""Lazy-module materialization inside a captured forward and the dispatch census (stress F3).

Torch materializes a pending ``UninitializedParameter`` in place from its lazy pre-hook:
``materialize`` reads the slot's ``.data`` for the device and the dtype before it assigns
storage, and ``reset_parameters`` then fills the new tensor. The wrapper passes calls whose
receiver is a pending slot through unlogged (they are constructor plumbing, not model
dataflow), so the ``aten.detach`` each ``.data`` read emits had no owner and the census
counted it as a silent drop. The witness now flags those reads as state-view accessor
dispatches, the same value-free category as a registered buffer's ``.data`` view, so the
validation backstop's arithmetic (dispatched = captured + orphan-pruned + state-view)
accounts for them without exempting anything a value could flow through.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch.wrappers import wrap_torch

WIDTH = 8
BATCH = 3


class LazyBuiltInForward(nn.Module):
    """``nn.LazyLinear`` created and materialized inside forward, never seen before capture."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(WIDTH, WIDTH)
        self.lazy: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lazy = nn.LazyLinear(4)
        return self.lazy(torch.relu(self.pre(x)))


class LazyAtPrep(nn.Module):
    """``nn.LazyLinear`` registered before capture and materialized by the captured forward."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(WIDTH, WIDTH)
        self.lazy = nn.LazyLinear(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lazy(torch.relu(self.pre(x)))


def _materialization_rows(trace: tl.Trace) -> list[dict[str, object]]:
    return [
        row
        for row in trace.completeness_diagnostics
        if row["reason"] == "unowned_dispatch"
        and str(row["operator"]).startswith(("aten.detach", "aten.alias"))
    ]


def test_pending_slot_data_reads_are_state_view_accessors_for_a_module_built_in_forward() -> None:
    wrap_torch(completeness_witness=True)
    torch.manual_seed(0)
    # The parameters a module built inside forward creates are still outside the
    # session inventory (no prep stamp), so the capture discloses them as
    # source-less arguments; that provenance gap is a separate, open piece of the
    # same finding and this test pins only the census accounting.
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(LazyBuiltInForward().eval(), [torch.randn(BATCH, WIDTH)])
    rows = _materialization_rows(trace)
    # weight and bias, device and dtype read each: four value-free views of pending slots
    assert len(rows) == 4
    assert all(row["state_view_accessor"] is True for row in rows)
    assert all(row["mutates"] is False for row in rows)
    unflagged = [
        row
        for row in trace.completeness_diagnostics
        if row["reason"] == "unowned_dispatch" and row["state_view_accessor"] is not True
    ]
    assert unflagged == []


def test_pending_slot_data_reads_are_state_view_accessors_for_a_lazy_module_registered_at_prep() -> (
    None
):
    wrap_torch(completeness_witness=True)
    torch.manual_seed(0)
    trace = tl.trace(LazyAtPrep().eval(), [torch.randn(BATCH, WIDTH)])
    rows = _materialization_rows(trace)
    assert rows, "the lazy pre-hook materialization emitted no unowned view dispatch"
    assert all(row["state_view_accessor"] is True for row in rows)


def test_a_detach_on_a_materialized_parameter_is_not_credited() -> None:
    """The flag covers PENDING slots only: a real tensor's unowned view stays a gap."""

    from torchlens.backends.torch._completeness_dispatch_names import (
        _is_buffer_state_view_dispatch,
    )

    param = nn.Parameter(torch.randn(2))
    pending = nn.parameter.UninitializedParameter()
    assert (
        _is_buffer_state_view_dispatch(None, torch.ops.aten.detach.default, None, False, (param,))
        is False
    )
    assert (
        _is_buffer_state_view_dispatch(None, torch.ops.aten.detach.default, None, False, (pending,))
        is True
    )
    assert (
        _is_buffer_state_view_dispatch(None, torch.ops.aten.detach.default, None, True, (pending,))
        is False
    )
