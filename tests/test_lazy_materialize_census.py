"""A lazy module materializing inside the captured forward balances the dispatch census.

``UninitializedParameter.materialize`` reads ``self.data`` (device, dtype) on the pending
slot before allocating storage. The wrapper passes a call on an uninitialized receiver
through unlogged (nothing to log: no shape, no storage), so each read is an unowned
``aten.detach`` dispatch. The completeness witness classifies it as a state-accessor
view, the same class as a registered buffer's ``.data`` getter, on strict axes: unowned,
non-mutating, a pure aliasing view, and a receiver that is still a lazy slot. These tests
pin the balanced census end to end and the predicate's narrowness: an initialized
parameter, a mutating operator or an owned dispatch never qualify.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._completeness_dispatch_names import _is_buffer_state_view_dispatch
from torchlens.backends.torch.escape_detection import ExpectedOriginalToken


class _LazyAtInit(nn.Module):
    """``nn.LazyLinear`` built at init; its first call (under capture) materializes it."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(4, 4)
        self.lazy = nn.LazyLinear(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lazy(torch.relu(self.pre(x)))


class _LazyInForward(nn.Module):
    """``nn.LazyLinear`` built and materialized inside forward."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(4, 4)
        self.lazy: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lazy = nn.LazyLinear(3)
        return self.lazy(torch.relu(self.pre(x)))


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validate(model, x, scope="forward"))


def test_lazy_module_first_called_inside_capture_validates() -> None:
    torch.manual_seed(0)
    assert _validate(_LazyAtInit(), torch.randn(2, 4)) is True


def test_lazy_module_built_inside_forward_validates() -> None:
    torch.manual_seed(0)
    assert _validate(_LazyInForward(), torch.randn(2, 4)) is True


def test_pending_lazy_slot_accessor_view_is_the_only_credited_case() -> None:
    detach = torch.ops.aten.detach.default
    pending = nn.parameter.UninitializedParameter()
    plain = nn.Parameter(torch.zeros(2))
    assert _is_buffer_state_view_dispatch(None, detach, None, False, (pending,)) is True
    # Materialized: an ordinary parameter, no longer a pending slot.
    pending.materialize((2,))
    assert _is_buffer_state_view_dispatch(None, detach, None, False, (pending,)) is False
    assert _is_buffer_state_view_dispatch(None, detach, None, False, (plain,)) is False
    # A mutating operator, an owned dispatch or a value-producing op never qualify.
    fresh = nn.parameter.UninitializedParameter()
    assert _is_buffer_state_view_dispatch(None, detach, None, True, (fresh,)) is False
    token = ExpectedOriginalToken(
        raw_id=0, original=detach, wrapper_name="x", wrapper_frame_id=0, owner_thread_id=0
    )
    assert _is_buffer_state_view_dispatch(None, detach, token, False, (fresh,)) is False
    assert (
        _is_buffer_state_view_dispatch(None, torch.ops.aten.add.Tensor, None, False, (fresh,))
        is False
    )
