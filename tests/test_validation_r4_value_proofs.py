"""Posthoc value proofs for perturbation-inert edges found by Menagerie rung 4.

Each proof is keyed on one op family, one argument slot of the perturbed parent,
and a value test on the saved call (``torchlens/validation/_saved_value_proofs.py``
and the ``__setitem__`` coverage proof in ``_destination_coverage.py``). For every
proof this module holds two tests:

- the minimal trigger from the real model now validates, and its edge is settled
  by the named exemption;
- an armed negative: the same op with the proof's condition NOT met, its replay
  frozen to fake a dead edge, still FAILS ``perturbation_insensitive``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation import validate_forward_pass
from torchlens.validation._destination_coverage import _setitem_index_targets_are_unique
from torchlens.validation.core import (
    _check_whether_func_on_saved_parents_yields_saved_tensor,
)


def _capture(model: nn.Module, x: torch.Tensor) -> Any:
    """Capture a full-save trace with saved argument values."""

    torch.manual_seed(0)
    return tl.trace(
        model.eval(),
        x,
        capture=CaptureOptions(layers_to_save="all", save_arg_values=True, random_seed=0),
    )


def _validates(model: nn.Module, x: torch.Tensor) -> bool:
    """Public-path validation with provenance warnings silenced."""

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate_forward_pass(model.eval(), x))


def _op(trace: Any, func_names: set[str], *, empty: bool | None = None) -> Any:
    """Return the first op whose func_name is in ``func_names`` (optionally by emptiness)."""

    for op in trace.layer_list:
        if op.func_name not in func_names:
            continue
        if empty is None or (op.out.numel() == 0) == empty:
            return op
    raise AssertionError(f"no op among {func_names}")


def _parent_from(trace: Any, op: Any, func_name: str) -> str:
    """Return the label of ``op``'s parent produced by ``func_name``."""

    for label in op.parents:
        if trace[label].func_name == func_name:
            return label
    raise AssertionError(f"{op.label} has no {func_name} parent: {op.parents}")


def _arg_parent(op: Any, key: Any) -> str:
    """Return the parent label registered at positional ``key``."""

    return op.parent_arg_positions["args"][key]


def _decision(trace: Any, op: Any, parent: str) -> tuple[str, str]:
    """Run the perturbation check of ``parent`` -> ``op`` and return (decision, reason)."""

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[parent]
    )
    return result.decision, result.reason


def _assert_frozen_edge_fails(trace: Any, op: Any, parent: str) -> None:
    """Freeze ``op``'s replay to its saved output and assert ``parent`` fails."""

    saved = op.out.detach().clone()
    object.__setattr__(op, "func", lambda *args, **kwargs: saved.clone())
    assert _decision(trace, op, parent) == ("failed", "perturbation_insensitive")


# --------------------------------------------------------------------- division


class _Divide(nn.Module):
    def __init__(self, zero_dividend: bool) -> None:
        super().__init__()
        self.zero_dividend = zero_dividend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num = x - x if self.zero_dividend else x * 2.0
        return num / (x.abs() + 1)


def test_zero_dividend_exempts_the_divisor() -> None:
    assert _validates(_Divide(True), torch.randn(3, 4))
    trace = _capture(_Divide(True), torch.randn(3, 4))
    op = _op(trace, {"__truediv__"})
    assert _decision(trace, op, _arg_parent(op, 1)) == ("exempted", "zero_dividend_annihilator")


def test_divisor_of_nonzero_dividend_and_zero_dividend_itself_still_fail() -> None:
    trace = _capture(_Divide(False), torch.randn(3, 4))
    op = _op(trace, {"__truediv__"})
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 1))
    trace = _capture(_Divide(True), torch.randn(3, 4))
    op = _op(trace, {"__truediv__"})
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 0))


# ------------------------------------------------------------------ bitwise and


class _AndZero(nn.Module):
    def __init__(self, other_all_false: bool) -> None:
        super().__init__()
        self.other_all_false = other_all_false

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum(-1)
        a = (s > -1e9).long()
        b = ~(s > -1e9) if self.other_all_false else s > -1e9
        return x + (a & b).unsqueeze(-1)


def test_and_with_all_false_operand_exempts_the_other_operand() -> None:
    assert _validates(_AndZero(True), torch.randn(3, 4))
    trace = _capture(_AndZero(True), torch.randn(3, 4))
    op = _op(trace, {"__and__"})
    decision = _decision(trace, op, _arg_parent(op, 0))
    assert decision == ("exempted", "conjunction_zero_annihilator")


def test_and_with_true_operand_still_fails_when_frozen() -> None:
    trace = _capture(_AndZero(False), torch.randn(3, 4))
    op = _op(trace, {"__and__"})
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 0))


class _OrTrue(nn.Module):
    def __init__(self, other_all_true: bool) -> None:
        super().__init__()
        self.other_all_true = other_all_true

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum(-1)
        a = s > 1e9
        b = s > -1e9 if self.other_all_true else s > 2e9
        return x + (a | b).unsqueeze(-1)


def test_or_with_all_true_operand_exempts_the_other_operand() -> None:
    assert _validates(_OrTrue(True), torch.randn(3, 4))
    trace = _capture(_OrTrue(True), torch.randn(3, 4))
    op = _op(trace, {"__or__"})
    assert _decision(trace, op, _arg_parent(op, 0)) == ("exempted", "disjunction_saturated")


def test_or_with_false_operand_and_the_true_operand_itself_still_fail() -> None:
    trace = _capture(_OrTrue(False), torch.randn(3, 4))
    op = _op(trace, {"__or__"})
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 0))
    trace = _capture(_OrTrue(True), torch.randn(3, 4))
    op = _op(trace, {"__or__"})
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 1))


# ------------------------------------------------------------------------ einsum


class _Einsum(nn.Module):
    def __init__(self, zero_operand: bool) -> None:
        super().__init__()
        self.zero_operand = zero_operand

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x * (0.0 if self.zero_operand else 2.0)
        w = x + 1.0
        return torch.einsum("ij,ij->i", z, w)


def test_einsum_zero_operand_exempts_its_sibling() -> None:
    assert _validates(_Einsum(True), torch.randn(4, 3))
    trace = _capture(_Einsum(True), torch.randn(4, 3))
    op = _op(trace, {"einsum"})
    decision = _decision(trace, op, _parent_from(trace, op, "__add__"))
    assert decision == ("exempted", "multiplicative_zero_annihilator")


def test_einsum_zero_operand_itself_and_nonzero_sibling_still_fail() -> None:
    trace = _capture(_Einsum(False), torch.randn(4, 3))
    op = _op(trace, {"einsum"})
    _assert_frozen_edge_fails(trace, op, _parent_from(trace, op, "__add__"))
    trace = _capture(_Einsum(True), torch.randn(4, 3))
    op = _op(trace, {"einsum"})
    _assert_frozen_edge_fails(trace, op, _parent_from(trace, op, "__mul__"))


# -------------------------------------------------------------------- batch_norm


class _CenteredBatchNorm(nn.Module):
    def __init__(self, centered: bool) -> None:
        super().__init__()
        self.centered = centered
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x * (0.0 if self.centered else 2.0))


def test_eval_batch_norm_on_running_mean_exempts_running_var() -> None:
    assert _validates(_CenteredBatchNorm(True), torch.randn(3, 4))
    trace = _capture(_CenteredBatchNorm(True), torch.randn(3, 4))
    op = _op(trace, {"batch_norm"})
    decision = _decision(trace, op, _arg_parent(op, 4))
    assert decision == ("exempted", "norm_centered_input_running_var")


def test_running_var_of_uncentered_input_and_running_mean_still_fail() -> None:
    trace = _capture(_CenteredBatchNorm(False), torch.randn(3, 4))
    op = _op(trace, {"batch_norm"})
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 4))
    trace = _capture(_CenteredBatchNorm(True), torch.randn(3, 4))
    op = _op(trace, {"batch_norm"})
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 3))


# --------------------------------------------------------------------------- pow


class _Pow(nn.Module):
    def __init__(self, exponent: float) -> None:
        super().__init__()
        self.exponent = exponent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exponent = (x * 0.0).sum() + self.exponent
        return torch.pow(x + 1.0, exponent)


def test_zero_exponent_exempts_the_base() -> None:
    assert _validates(_Pow(0.0), torch.randn(3, 4))
    trace = _capture(_Pow(0.0), torch.randn(3, 4))
    op = _op(trace, {"pow", "__pow__"})
    assert _decision(trace, op, _arg_parent(op, 0)) == ("exempted", "pow_zero_exponent")


def test_base_under_nonzero_exponent_and_zero_exponent_itself_still_fail() -> None:
    trace = _capture(_Pow(2.0), torch.randn(3, 4))
    op = _op(trace, {"pow", "__pow__"})
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 0))
    trace = _capture(_Pow(0.0), torch.randn(3, 4))
    op = _op(trace, {"pow", "__pow__"})
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 1))


# --------------------------------------------------- __setitem__ index of size 1


class _SetitemSingletonIndex(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(4, 1, dtype=torch.uint8)
        lab = torch.tensor([0]).view(-1)
        z[lab, torch.arange(1)] = 1
        return x * z.float().sum()


class _SetitemWideIndex(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dest = torch.zeros(2, 3)
        dest[0, torch.arange(2)] = x[0, :2] * 2.0
        return dest


def test_index_into_size_one_dim_is_exempt() -> None:
    assert _validates(_SetitemSingletonIndex(), torch.randn(3, 4))
    trace = _capture(_SetitemSingletonIndex(), torch.randn(3, 4))
    op = _op(trace, {"__setitem__"})
    decision = _decision(trace, op, _parent_from(trace, op, "arange"))
    assert decision == ("exempted", "setitem_singleton_index_dim")


def test_index_into_wider_dim_still_fails_when_frozen() -> None:
    trace = _capture(_SetitemWideIndex(), torch.randn(3, 4))
    op = _op(trace, {"__setitem__"})
    _assert_frozen_edge_fails(trace, op, _parent_from(trace, op, "arange"))


# ---------------------------------------------------------------- scatter_reduce


class _ScatterReduce(nn.Module):
    def __init__(self, index: list[int], include_self: bool) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.index = index
        self.include_self = include_self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = self.lin(x)
        idx = torch.tensor(self.index).view(-1, 1).expand_as(src)
        dest = src.new_zeros(3, 4)
        return dest.scatter_reduce_(0, idx, src, reduce="amax", include_self=self.include_self)


_FULL_COVER = [0, 1, 2, 0, 1, 2]


def test_scatter_reduce_excluding_fully_covered_destination_is_exempt() -> None:
    assert _validates(_ScatterReduce(_FULL_COVER, False), torch.randn(6, 4))
    trace = _capture(_ScatterReduce(_FULL_COVER, False), torch.randn(6, 4))
    op = _op(trace, {"scatter_reduce_"})
    decision = _decision(trace, op, _arg_parent(op, 0))
    assert decision == ("exempted", "scatter_reduce_excluded_destination")


@pytest.mark.parametrize(
    ("index", "include_self"),
    [(_FULL_COVER, True), ([0, 1, 0, 1, 0, 1], False)],
    ids=["include_self", "partial_cover"],
)
def test_scatter_reduce_destination_still_fails_when_it_is_read(
    index: list[int], include_self: bool
) -> None:
    trace = _capture(_ScatterReduce(index, include_self), torch.randn(6, 4))
    op = _op(trace, {"scatter_reduce_"})
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 0))


# ------------------------------------------------------------------- empty split


class _SplitEmptyChunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The empty chunk stays unused, as in the real model (m16291).
        a, _, c = torch.split(self.lin(x), [3, 0, 3], dim=-1)
        return a + c


_SPLIT_NAMES = {"split", "split_with_sizes"}


def test_empty_split_chunk_is_exempt() -> None:
    assert _validates(_SplitEmptyChunk(), torch.randn(2, 4))
    trace = _capture(_SplitEmptyChunk(), torch.randn(2, 4))
    op = _op(trace, _SPLIT_NAMES, empty=True)
    assert _decision(trace, op, _arg_parent(op, 0)) == ("exempted", "empty_split_chunk")


def test_nonempty_split_chunk_still_fails_when_frozen() -> None:
    trace = _capture(_SplitEmptyChunk(), torch.randn(2, 4))
    op = _op(trace, _SPLIT_NAMES, empty=False)
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 0))


# ------------------------------------- __setitem__ full overwrite coverage proof


class _NumpyMaskOverwrite(nn.Module):
    def __init__(self, full: bool) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.full = full

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pts = self.lin(x)
        mask = np.ones(pts.shape[0], dtype=bool)
        mask[0] = self.full
        pts[mask] = 2.0 * pts[mask]
        return pts


class _PairIndexOverwrite(nn.Module):
    def __init__(self, cols: list[int]) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.cols = cols

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dest = self.lin(x)
        rows = torch.tensor([0, 0, 0, 1, 1, 1])
        dest[rows, torch.tensor(self.cols)] = x.reshape(-1) * 2.0
        return dest


class _NumpyMaskScalarFill(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pts = self.lin(x)
        pts[np.ones(pts.shape[0], dtype=bool)] = 0.5
        return pts * x


_ALL_PAIRS = [0, 1, 2, 0, 1, 2]
_REPEATED_PAIR = [0, 1, 1, 0, 1, 2]


@pytest.mark.parametrize(
    "model",
    [_NumpyMaskOverwrite(True), _PairIndexOverwrite(_ALL_PAIRS)],
    ids=["numpy_mask", "index_pairs"],
)
def test_full_overwrite_proof_reads_numpy_masks_and_index_pairs(model: nn.Module) -> None:
    x = torch.randn(2, 3)
    assert _validates(model, x)
    trace = _capture(model, x)
    op = _op(trace, {"__setitem__"})
    # A tensor right-hand side settles the destination before perturbation.
    assert _decision(trace, op, _arg_parent(op, 0)) == ("exempted", "pre_perturbation_exemption")


def test_scalar_fill_through_full_numpy_mask_is_proved_after_perturbation() -> None:
    assert _validates(_NumpyMaskScalarFill(), torch.randn(2, 3))
    trace = _capture(_NumpyMaskScalarFill(), torch.randn(2, 3))
    op = _op(trace, {"__setitem__"})
    decision = _decision(trace, op, _arg_parent(op, 0))
    assert decision == ("exempted", "scalar_destination_overwrite")


@pytest.mark.parametrize(
    "model",
    [_NumpyMaskOverwrite(False), _PairIndexOverwrite(_REPEATED_PAIR)],
    ids=["partial_numpy_mask", "repeated_index_pair"],
)
def test_partial_overwrite_destination_still_fails_when_frozen(model: nn.Module) -> None:
    trace = _capture(model, torch.randn(2, 3))
    op = _op(trace, {"__setitem__"})
    _assert_frozen_edge_fails(trace, op, _arg_parent(op, 0))


def test_index_pair_uniqueness_is_judged_per_pair() -> None:
    rows = torch.tensor([0, 0, 1, 1])
    assert _setitem_index_targets_are_unique((rows, torch.tensor([0, 1, 0, 1])))
    assert not _setitem_index_targets_are_unique((rows, torch.tensor([0, 0, 0, 1])))
    assert not _setitem_index_targets_are_unique(torch.tensor([0, 0]))
    assert _setitem_index_targets_are_unique(np.ones(3, dtype=bool))
