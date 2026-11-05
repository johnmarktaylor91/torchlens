"""Dtype queries inside a forward are owned by a wrapper, so the completeness census holds.

``torch.promote_types``, ``torch.can_cast`` and ``torch.result_type`` are in torch's
``get_ignored_functions()``: the overridable crawl skips them, and they were not rows of
``torchlens._constants.IGNORED_FUNCS``. Each still dispatches an aten op
(``aten.promote_types``, ``aten.can_cast``, ``aten.result_type``) that returns a dtype or a
bool. Unwrapped, that dispatch had no owner, so the validation census counted it as a dropped
op and a correct model failed ``bfs_completeness`` with one more dispatched op than captured
(an ODE solver's ``torch.promote_types`` in its step setup, an interpolation helper's dtype
promotion). Wrapped, the call is owned and records no op, the same class as ``torch.equal``.
A direct ``torch.ops.aten`` call of the same operator has no wrapper and still fails.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._constants import get_orig_torch_funcs
from torchlens.validation import last_validation_failure

_DTYPE_QUERIES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "promote_types": lambda h: h.to(torch.promote_types(h.dtype, torch.float64)),
    "can_cast": lambda h: h * 2 if torch.can_cast(h.dtype, torch.float64) else h,
    "result_type_scalars": lambda h: h.to(torch.result_type(1, 2.0)),
    "result_type_tensors": lambda h: h.to(torch.result_type(h, h.double())),
}


class _DtypeQueryModel(nn.Module):
    """A linear layer whose output dtype or branch is decided by a dtype query."""

    def __init__(self, query: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.query = query

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.query(self.lin(x)).relu()


def _validate(model: nn.Module) -> bool:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.validate(model, torch.randn(2, 3), scope="forward") is True


@pytest.mark.smoke
@pytest.mark.parametrize("name", ("promote_types", "can_cast", "result_type"))
def test_dtype_query_is_decorated(name: str) -> None:
    """The dtype query is in TorchLens's decorated set."""

    assert ("torch", name) in set(get_orig_torch_funcs())


@pytest.mark.smoke
@pytest.mark.parametrize("case", tuple(_DTYPE_QUERIES))
def test_dtype_query_in_forward_validates(case: str) -> None:
    """Validation passes and the query itself records no op."""

    torch.manual_seed(0)
    model = _DtypeQueryModel(_DTYPE_QUERIES[case]).eval()
    assert _validate(model), last_validation_failure()
    trace = tl.trace(model, torch.randn(2, 3))
    try:
        names = {op.func_name for op in trace.ops}
        assert not names & {"promote_types", "can_cast", "result_type"}, names
    finally:
        trace.cleanup()


@pytest.mark.smoke
def test_direct_aten_dtype_query_still_fails_completeness() -> None:
    """The fix is ownership: a raw aten call of the same operator stays an unowned drop."""

    torch.manual_seed(0)
    model = _DtypeQueryModel(
        lambda h: h.to(torch.ops.aten.promote_types.default(h.dtype, torch.float64))
    ).eval()
    assert not _validate(model)
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == "bfs_completeness", failure
