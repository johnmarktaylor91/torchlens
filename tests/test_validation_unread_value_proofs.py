"""Exact value proofs for writes that cannot change the output and unread arguments.

Each proof in ``torchlens.validation._unread_value_proofs`` runs only after every
perturbed replay left the output unchanged, reads saved values with no
tolerance, and exempts one parent in one slot of one op. For each, the first
test shows the dead edge is now proved, and the planted-bug test shows the
unintended case of the same shape (a replay frozen to its saved output over a
live configuration) still fails ``perturbation_insensitive``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from _validation_capture import _capture, _quiet_validate

from torchlens.validation import last_validation_failure
from torchlens.validation._core import _check_whether_func_on_saved_parents_yields_saved_tensor
from torchlens.validation._unread_value_proofs import unread_value_proof_decision


def _op(trace: Any, func_name: str, output_index: int | None = None) -> Any:
    return [
        op
        for op in trace._layer_list
        if op.func_name == func_name
        and (output_index is None or op.multi_output_index == output_index)
    ][-1]


def _perturb(trace: Any, op: Any, key: Any) -> Any:
    positions = op.call.parent_arg_positions["args"]
    return _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[positions[key]]
    )


def _freeze(op: Any, saved: Any) -> None:
    """Plant a capture bug: the replay returns the saved output whatever it is given."""

    object.__setattr__(op, "func", lambda *args, **kwargs: saved)


def _assert_exempt(result: Any, reason: str) -> None:
    assert (result.decision, result.reason) == ("exempted", reason)


def _assert_insensitive(result: Any) -> None:
    assert (result.decision, result.reason) == ("failed", "perturbation_insensitive")


class _MaskedRowWrite(nn.Module):
    """``state[row, mask] = -1`` at a traced row (the Hopfield update)."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x.clone()
        row = torch.arange(x.shape[0])[2]
        mask = x[0] > self.threshold
        state[row, mask] = -1.0
        return state


def test_setitem_empty_write_is_exempt() -> None:
    """A mask with no True entry writes nothing, so the row index is proved irrelevant."""

    trace, _ = _capture(_MaskedRowWrite(threshold=100.0), torch.randn(4, 5))
    _assert_exempt(_perturb(trace, _op(trace, "__setitem__"), (1, 0)), "setitem_empty_write")


def test_setitem_empty_write_planted_bug_fails() -> None:
    """A mask that selects a column keeps the row edge live; a frozen replay fails."""

    torch.manual_seed(0)
    x = torch.randn(4, 5)
    trace, _ = _capture(_MaskedRowWrite(threshold=-100.0), x)
    op = _op(trace, "__setitem__")
    _freeze(op, op.out.detach().clone())
    _assert_insensitive(_perturb(trace, op, (1, 0)))
    destination, row = torch.zeros(4, 5), torch.tensor(2)
    empty = torch.zeros(5, dtype=torch.bool)
    layer = SimpleNamespace(func_name="__setitem__", out=destination.clone())
    args = (destination, (row, empty), -1.0)
    assert unread_value_proof_decision(layer, {(1, 0)}, args).exempt
    assert not unread_value_proof_decision(layer, {(1, 1)}, args).exempt
    assert not unread_value_proof_decision(layer, {2}, args).exempt
    assert not unread_value_proof_decision(layer, {0}, args).exempt
    forged = SimpleNamespace(func_name="__setitem__", out=torch.ones(4, 5))
    assert not unread_value_proof_decision(forged, {(1, 0)}, args).exempt


class _ZeroMemoryWrite(nn.Module):
    """``memory[arange, argmin] = 0`` into a zero memory (the MANN write)."""

    def __init__(self, held_elsewhere: bool) -> None:
        super().__init__()
        self.held_elsewhere = held_elsewhere

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory = torch.zeros(4, 3)
        if self.held_elsewhere:
            memory[0, 2] = 5.0
        memory[torch.arange(4), x.argmin(dim=1)] = 0.0
        return memory + x


def _held_input() -> torch.Tensor:
    return torch.tensor([[0.0, 1.0, 2.0]] * 4)


def test_setitem_held_value_is_exempt() -> None:
    """Every cell the column index can reach already holds the written zero."""

    trace, _ = _capture(_ZeroMemoryWrite(held_elsewhere=False), _held_input())
    result = _perturb(trace, _op(trace, "__setitem__"), (1, 1))
    _assert_exempt(result, "setitem_destination_holds_value")


def test_setitem_held_value_planted_bug_fails() -> None:
    """One reachable cell holds another value: the edge is live and a frozen replay fails."""

    trace, _ = _capture(_ZeroMemoryWrite(held_elsewhere=True), _held_input())
    op = _op(trace, "__setitem__")
    _freeze(op, op.out.detach().clone())
    _assert_insensitive(_perturb(trace, op, (1, 1)))
    destination = torch.zeros(4, 3)
    index = (torch.arange(4), torch.zeros(4, dtype=torch.long))
    layer = SimpleNamespace(func_name="__setitem__", out=destination.clone())
    assert unread_value_proof_decision(layer, {(1, 1)}, (destination, index, 0.0)).exempt
    assert not unread_value_proof_decision(layer, {(1, 1)}, (destination, index, 1e-30)).exempt
    uneven = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert not unread_value_proof_decision(layer, {(1, 1)}, (destination, index, uneven)).exempt
    assert not unread_value_proof_decision(layer, {2}, (destination, index, 0.0)).exempt


class _Clamp(nn.Module):
    """``clamp(0.1 * x, low, high)``."""

    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x * 0.1, self.low, self.high)


def test_clamp_equal_bounds_is_exempt() -> None:
    """``clamp(x, 1, 1)`` is 1 for every input (the DeLaN call)."""

    trace, _ = _capture(_Clamp(1, 1), torch.randn(8, 1))
    _assert_exempt(_perturb(trace, _op(trace, "clamp"), 0), "clamp_equal_bounds")


def test_clamp_equal_bounds_planted_bug_fails() -> None:
    """Unequal bounds keep the input live; a frozen replay fails, and forged values refuse."""

    trace, _ = _capture(_Clamp(-1.0, 1.0), torch.randn(8, 1))
    op = _op(trace, "clamp")
    _freeze(op, op.out.detach().clone())
    _assert_insensitive(_perturb(trace, op, 0))
    x = torch.randn(3)
    layer = SimpleNamespace(func_name="clamp", out=torch.ones(3), saved_kwargs={})
    assert unread_value_proof_decision(layer, {0}, (x, 1, 1)).exempt
    assert not unread_value_proof_decision(layer, {0}, (x, 1, torch.ones(1))).exempt
    assert not unread_value_proof_decision(layer, {1}, (x, 1, 1)).exempt
    forged = SimpleNamespace(func_name="clamp", out=torch.full((3,), 2.0), saved_kwargs={})
    assert not unread_value_proof_decision(forged, {0}, (x, 1, 1)).exempt


class _SparseTimesDense(nn.Module):
    """``spmm(adjacency, 2 * x)`` with a sparse adjacency holding ``nnz`` entries."""

    def __init__(self, nnz: int) -> None:
        super().__init__()
        self.indices = torch.tensor([[0, 1, 2], [2, 0, 1]])[:, :nnz]
        self.values = torch.ones(nnz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = torch.sparse_coo_tensor(self.indices, self.values, (3, 3))
        return torch.spmm(adjacency, x * 2.0)


def test_spmm_empty_sparse_is_exempt() -> None:
    """An adjacency with no stored entry multiplies the dense operand into zeros."""

    trace, _ = _capture(_SparseTimesDense(nnz=0), torch.randn(3, 4))
    _assert_exempt(_perturb(trace, _op(trace, "spmm"), 1), "spmm_empty_sparse_operand")


def test_spmm_empty_sparse_planted_bug_fails() -> None:
    """A stored entry keeps the dense edge live; a frozen replay fails."""

    trace, _ = _capture(_SparseTimesDense(nnz=3), torch.randn(3, 4))
    op = _op(trace, "spmm")
    _freeze(op, op.out.detach().clone())
    _assert_insensitive(_perturb(trace, op, 1))
    empty = torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long), torch.zeros(0), (3, 3))
    layer = SimpleNamespace(func_name="spmm", out=torch.zeros(3, 4))
    assert unread_value_proof_decision(layer, {1}, (empty, torch.ones(3, 4))).exempt
    assert not unread_value_proof_decision(layer, {0}, (empty, torch.ones(3, 4))).exempt
    forged = SimpleNamespace(func_name="spmm", out=torch.ones(3, 4))
    assert not unread_value_proof_decision(forged, {1}, (empty, torch.ones(3, 4))).exempt


class _MaskedScatter(nn.Module):
    """``(2 * x).masked_scatter(x > threshold, x + 1)``."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 2.0).masked_scatter(x > self.threshold, x + 1.0)


def test_masked_scatter_all_true_is_exempt() -> None:
    """A mask True everywhere takes every output element from the source."""

    trace, _ = _capture(_MaskedScatter(threshold=-100.0), torch.randn(3, 4))
    result = _perturb(trace, _op(trace, "masked_scatter"), 0)
    _assert_exempt(result, "masked_scatter_all_true_mask")


def test_masked_scatter_all_true_planted_bug_fails() -> None:
    """A partial mask keeps the destination live; a frozen replay fails."""

    x = torch.tensor([[1.0, -1.0, 2.0, -2.0]])
    trace, _ = _capture(_MaskedScatter(threshold=0.0), x)
    op = _op(trace, "masked_scatter")
    _freeze(op, op.out.detach().clone())
    _assert_insensitive(_perturb(trace, op, 0))
    full = torch.ones(2, 2, dtype=torch.bool)
    source = torch.arange(4.0)
    layer = SimpleNamespace(func_name="masked_scatter", out=source.reshape(2, 2))
    assert unread_value_proof_decision(layer, {0}, (torch.ones(2, 2), full, source)).exempt
    assert not unread_value_proof_decision(layer, {2}, (torch.ones(2, 2), full, source)).exempt
    partial = full.clone()
    partial[0, 0] = False
    assert not unread_value_proof_decision(layer, {0}, (torch.ones(2, 2), partial, source)).exempt


class _Bags(nn.Module):
    """``EmbeddingBag(mode=mode)`` over two bags of token ids."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.bag = nn.EmbeddingBag(10, 3, mode=mode)
        self.register_buffer("offsets", torch.tensor([0, 2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bag(x.long(), self.offsets)


_BAG_IDS = torch.tensor([1.0, 2.0, 4.0, 5.0, 7.0])


def test_embedding_bag_sum_placeholder_is_exempt() -> None:
    """Sum mode returns bag_size and max_indices as zeros the indices never reach."""

    trace, _ = _capture(_Bags("sum"), _BAG_IDS)
    for output_index in (2, 3):
        op = _op(trace, "embedding_bag", output_index)
        assert op.call.saved_args[4] == 0 and op.out.tolist() == [0, 0]
        result = _perturb(trace, op, 1)
        _assert_exempt(result, "embedding_bag_sum_placeholder_output")


def test_embedding_bag_sum_placeholder_planted_bug_fails() -> None:
    """Mean-mode bag sizes count the indices; a frozen replay fails, and sum mode's pooled output is not proved."""

    trace, _ = _capture(_Bags("mean"), _BAG_IDS)
    siblings = sorted(
        (op for op in trace._layer_list if op.func_name == "embedding_bag"),
        key=lambda op: op.multi_output_index,
    )
    saved = tuple(op.out.detach().clone() for op in siblings)
    bag_size = _op(trace, "embedding_bag", 2)
    _freeze(bag_size, saved)
    _assert_insensitive(_perturb(trace, bag_size, 1))
    sum_trace, _ = _capture(_Bags("sum"), _BAG_IDS)
    pooled = _op(sum_trace, "embedding_bag", 0)
    assert not unread_value_proof_decision(pooled, {1}, pooled.call.saved_args).exempt
    placeholder = _op(sum_trace, "embedding_bag", 2)
    assert not unread_value_proof_decision(placeholder, {2}, placeholder.call.saved_args).exempt


class _Deform(nn.Module):
    """A deformable conv, with a computed mask or with ``mask=None``."""

    def __init__(self, with_mask: bool) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.with_mask = with_mask
        self.weight = nn.Parameter(torch.randn(4, 3, 3, 3))
        self.offset_conv = nn.Conv2d(3, 18, 3, padding=1)
        self.mask_conv = nn.Conv2d(3, 9, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ops = pytest.importorskip("torchvision.ops")
        mask = torch.sigmoid(self.mask_conv(x)) if self.with_mask else None
        return ops.deform_conv2d(x, self.offset_conv(x), self.weight, padding=(1, 1), mask=mask)


def test_deform_conv2d_unused_mask_is_exempt() -> None:
    """``mask=None`` passes a zeros mask with ``use_mask=False``; the kernel never reads it."""

    pytest.importorskip("torchvision.ops")
    trace, _ = _capture(_Deform(with_mask=False), torch.randn(1, 3, 6, 6))
    op = _op(trace, "deform_conv2d")
    assert op.call.saved_args[13] is False
    _assert_exempt(_perturb(trace, op, 3), "deform_conv2d_unused_mask")


def test_deform_conv2d_unused_mask_planted_bug_fails() -> None:
    """With ``use_mask=True`` the mask is read; a frozen replay at the mask slot fails."""

    pytest.importorskip("torchvision.ops")
    trace, _ = _capture(_Deform(with_mask=True), torch.randn(1, 3, 6, 6))
    op = _op(trace, "deform_conv2d")
    assert op.call.saved_args[13] is True
    _freeze(op, op.out.detach().clone())
    _assert_insensitive(_perturb(trace, op, 3))
    assert not unread_value_proof_decision(op, {2}, op.call.saved_args).exempt


def _assert_validates(model: nn.Module, x: torch.Tensor) -> None:
    assert _quiet_validate(model, x), (type(model).__name__, last_validation_failure())


@pytest.mark.smoke
def test_setitem_unread_writes_validate_on_the_public_path() -> None:
    """Models whose index write is proved dead validate end to end through ``validate_forward_pass``.

    The row index of an empty masked write and the column index of a write
    into cells that already hold the value change nothing under perturbation;
    without the setitem proofs both models fail as perturbation-insensitive.
    """

    torch.manual_seed(0)
    _assert_validates(_MaskedRowWrite(threshold=100.0), torch.randn(4, 5))
    _assert_validates(_ZeroMemoryWrite(held_elsewhere=False), _held_input())


@pytest.mark.smoke
def test_unread_operand_models_validate_on_the_public_path() -> None:
    """Clamp, spmm, masked_scatter and embedding_bag models with unread operands validate end to end."""

    torch.manual_seed(0)
    _assert_validates(_Clamp(1, 1), torch.randn(8, 1))
    _assert_validates(_SparseTimesDense(nnz=0), torch.randn(3, 4))
    _assert_validates(_MaskedScatter(threshold=-100.0), torch.tensor([[1.0, -1.0, 2.0, -2.0]]))
    _assert_validates(_Bags("sum"), _BAG_IDS)
