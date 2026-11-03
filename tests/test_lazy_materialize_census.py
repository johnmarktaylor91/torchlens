"""Lazy-module materialization inside a captured forward and the dispatch census (stress F3).

Torch materializes a pending ``UninitializedParameter`` in place from its lazy pre-hook:
``materialize`` reads the slot's ``.data`` for the device and the dtype before it assigns
storage, and ``reset_parameters`` then fills the new tensor. The wrapper passes calls whose
receiver is a pending slot through unlogged (they are constructor plumbing, not model
dataflow) and, under the dispatch census, runs them as the audited expected-opaque owner
``lazy_module:materialize``: the ``aten.detach`` each ``.data`` read emits is accounted to
that boundary, so the validation backstop's arithmetic (dispatched = captured + orphan-pruned
+ accounted) holds with no tolerance, count or exemption change. The pending-slot state-view
flag stays as the witness's own account of such a read should one ever reach the dispatcher
outside the wrapper; it credits pending slots only.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch._completeness_audited import LAZY_MODULE_MATERIALIZE_WRAPPER
from torchlens.backends.torch.completeness_witness import AUDITED_COMPLETENESS_BOUNDARIES
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

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


@pytest.fixture
def _armed_witness() -> Iterator[None]:
    """Arm the dispatch census for one test and restore the process-level modes after."""

    saved_escape = _state._escape_detector_mode
    saved_witness = _state._completeness_witness_mode
    unwrap_torch()
    wrap_torch(completeness_witness=True)
    yield
    unwrap_torch()
    wrap_torch(escape_detector=saved_escape, completeness_witness=saved_witness)


def _materialization_owners(trace: tl.Trace) -> list[dict[str, Any]]:
    return [
        entry
        for entry in trace.health.completeness_decompositions
        if entry["owner_wrapper"] == LAZY_MODULE_MATERIALIZE_WRAPPER
    ]


def _assert_materialization_is_audited(trace: tl.Trace) -> None:
    owners = _materialization_owners(trace)
    assert owners, "the lazy pre-hook materialization ran outside the audited owner"
    assert all(entry["scope"] == "expected_opaque" for entry in owners)
    operators = [op for entry in owners for op in entry["aten_ops"]]
    # weight and bias, device and dtype read each: four value-free views of pending slots
    assert len(operators) == 4
    assert all(op.startswith(("aten.detach", "aten.alias")) for op in operators)
    assert trace.completeness_diagnostics == []
    assert trace.provenance.verified is True


def test_the_audited_row_is_the_wrapper_wide_materialization_boundary() -> None:
    rows = [
        row
        for row in AUDITED_COMPLETENESS_BOUNDARIES
        if row.wrapper_name == LAZY_MODULE_MATERIALIZE_WRAPPER
    ]
    assert len(rows) == 1 and rows[0].operator is None and rows[0].reason


@pytest.mark.smoke
def test_materialization_of_a_module_built_in_forward_is_audited(_armed_witness: None) -> None:
    torch.manual_seed(0)
    trace = tl.trace(LazyBuiltInForward().eval(), [torch.randn(BATCH, WIDTH)])
    _assert_materialization_is_audited(trace)


def test_materialization_of_a_lazy_module_registered_at_prep_is_audited(
    _armed_witness: None,
) -> None:
    torch.manual_seed(0)
    trace = tl.trace(LazyAtPrep().eval(), [torch.randn(BATCH, WIDTH)])
    _assert_materialization_is_audited(trace)


def test_a_detach_on_a_materialized_parameter_is_not_credited() -> None:
    """The state-view flag covers PENDING slots only: a real tensor's unowned view stays a gap."""

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
