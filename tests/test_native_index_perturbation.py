"""Perturbing an integer argument of a native extension operator stays in its domain.

Extension kernels such as torch_scatter's ``gather_csr`` read CSR index pointers and
indices without bounds checks. The generic integer perturbation broke a pointer's structure
(non-decreasing, first and last entries fixed) and the kernel wrote out of range, aborting
the process during validation. A registered slot now gets an in-domain move; an integer slot
with no proven move is recorded unverified and the kernel is never run with a guessed value.
The check itself is unchanged: a live pointer edge still validates and a dead one still fails.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import last_validation_failure
from torchlens.validation._native_index_perturbation import (
    UNCHECKED_NATIVE_INDEX_REASON,
    csr_pointer_shift_values,
    index_left_shift_values,
)

_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")
_VF = torch._C._VariableFunctions

# Module-level ledgers (capture rolls back model attributes): every pointer the operator
# bodies were called with that a bounds-unchecked kernel would have read out of range, and
# every index the unregistered operator received.
_POINTER_VIOLATIONS: list[list[int]] = []
_TAKE_ROWS_INDICES: list[list[int]] = []
_FIXED_COUNTS = torch.tensor([2, 1, 3, 1, 2, 3])


def _pointer_is_valid(indptr: torch.Tensor, num_segments: int) -> bool:
    ptr = torch._C.TensorBase.tolist(indptr)
    monotone = all(left <= right for left, right in zip(ptr[:-1], ptr[1:], strict=True))
    return len(ptr) == num_segments + 1 and ptr[0] == 0 and monotone


if _HAS_CUSTOM_OP:

    @torch.library.custom_op("tltest_native_index::gather_csr", mutates_args=())
    def _gather_csr(src: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
        # Unpatchable C calls only, like the C++ kernel it stands in for. A pointer a real
        # unchecked kernel would read out of range is recorded and refused.
        if not _pointer_is_valid(indptr, src.shape[0]):
            _POINTER_VIOLATIONS.append(torch._C.TensorBase.tolist(indptr))
            raise ValueError("index pointer out of domain")
        return _VF.repeat_interleave(src, _VF.diff(indptr), 0)

    @torch.library.custom_op("tltest_native_index_dead::gather_csr", mutates_args=())
    def _gather_csr_ignoring_pointer(src: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
        # A planted capture bug: the pointer is passed but never read.
        return _VF.repeat_interleave(src, _VF.clone(_FIXED_COUNTS), 0)

    @torch.library.custom_op("tltest_native_index::take_rows", mutates_args=())
    def _take_rows(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        _TAKE_ROWS_INDICES.append(torch._C.TensorBase.tolist(index))
        return _VF.index_select(src, 0, index)


class _CsrModel(nn.Module):
    """Linear rows gathered into variable-length segments through a CSR pointer."""

    def __init__(self, dead_pointer: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.dead_pointer = dead_pointer

    def forward(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        indptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        if self.dead_pointer:
            gathered = torch.ops.tltest_native_index_dead.gather_csr(self.fc(x), indptr)
        else:
            gathered = torch.ops.tltest_native_index.gather_csr(self.fc(x), indptr)
        return gathered * 2.0


class _TakeRowsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch.ops.tltest_native_index.take_rows(self.fc(x), index.flip(0)) + 1.0


def _validate(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validate(model, inputs, seed=0))


def test_csr_pointer_shift_keeps_ends_and_order() -> None:
    ptr = torch.tensor([0, 2, 2, 5, 6])
    down = csr_pointer_shift_values(ptr, alternate=False)
    up = csr_pointer_shift_values(ptr, alternate=True)
    assert down is not None and up is not None
    assert down.tolist() == [0, 1, 2, 4, 6]
    assert up.tolist() == [0, 2, 3, 6, 6]
    for moved in (down, up):
        assert moved[0] == ptr[0] and moved[-1] == ptr[-1]
        assert bool((moved[1:] >= moved[:-1]).all())
    assert csr_pointer_shift_values(torch.tensor([0, 0, 0]), alternate=False) is None
    assert csr_pointer_shift_values(torch.tensor([0, 4]), alternate=False) is None
    assert csr_pointer_shift_values(torch.tensor([0, 3, 1, 4]), alternate=False) is None


def test_index_left_shift_reuses_saved_values() -> None:
    index = torch.tensor([[0, 0, 1, 3], [2, 2, 2, 2]])
    shifted = index_left_shift_values(index, alternate=False)
    assert shifted is not None
    assert shifted.tolist() == [[0, 1, 3, 3], [2, 2, 2, 2]]
    assert index_left_shift_values(torch.tensor([5, 5, 5]), alternate=False) is None


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
def test_live_csr_pointer_edge_validates_in_domain() -> None:
    """The perturbed pointer stays valid, and its edge is still tested and validates."""

    _POINTER_VIOLATIONS.clear()
    counts = torch.tensor([2, 1, 3, 1, 2, 3])
    x = torch.randn(counts.numel(), 4)
    assert _validate(_CsrModel(), (x, counts)), last_validation_failure()
    assert _POINTER_VIOLATIONS == []


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
def test_dead_csr_pointer_edge_still_fails() -> None:
    """A pointer the operator never reads still fails the perturbation check."""

    counts = _FIXED_COUNTS.clone()
    x = torch.randn(counts.numel(), 4)
    assert not _validate(_CsrModel(dead_pointer=True), (x, counts))
    failure = last_validation_failure()
    assert failure is not None
    assert "insensitive" in str(failure), failure


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
def test_unregistered_native_index_slot_fails_closed_without_running() -> None:
    """No proven in-domain move: the edge is unverified and the kernel never sees a guess."""

    _TAKE_ROWS_INDICES.clear()
    index = torch.tensor([0, 2, 1, 3])
    x = torch.randn(4, 4)
    assert not _validate(_TakeRowsModel(), (x, index))
    assert UNCHECKED_NATIVE_INDEX_REASON in str(last_validation_failure())
    saved = index.flip(0).tolist()
    assert _TAKE_ROWS_INDICES and all(seen == saved for seen in _TAKE_ROWS_INDICES)


class _ScatterCsrModel(nn.Module):
    def __init__(self, reduce: str) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.reduce = reduce

    def forward(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        import torch_scatter

        indptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        gathered = torch_scatter.gather_csr(self.fc(x), indptr)
        return torch_scatter.segment_csr(gathered * 2.0, indptr, reduce=self.reduce)


@pytest.mark.parametrize("reduce", ["sum", "max"])
def test_torch_scatter_csr_ops_validate_without_crashing(reduce: str) -> None:
    """torch_scatter's unchecked CSR kernels get only valid pointers (process used to abort)."""

    pytest.importorskip("torch_scatter")
    counts = torch.tensor([3, 1, 4, 2, 5, 1, 2, 6] * 8)
    x = torch.randn(counts.numel(), 4)
    assert _validate(_ScatterCsrModel(reduce), (x, counts)), last_validation_failure()
