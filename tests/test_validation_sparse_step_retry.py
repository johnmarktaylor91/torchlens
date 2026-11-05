"""Sparse parents on the perturbation retry ladder.

The first perturbation of a sparse parent goes through its values component; the
deterministic retry strategies (minimal and unit steps, the lift) used to build
dense tensors with ``full_like`` and ``nextafter`` on the sparse parent itself,
which the sparse backends do not implement, so the raise escaped ``tl.validate``.
These tests pin that every retry strategy keeps the sparse layout and indices,
that a dead sparse edge reaches a perturbation verdict instead of an exception,
and that a live sparse edge still validates.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.validation import last_validation_failure
from torchlens.validation._perturb import perturb_through_values
from torchlens.validation._shift_probe import VALUE_STEP_STRATEGIES
from torchlens.validation.core import _directional_step_perturb

_RETRY_STRATEGIES = [*VALUE_STEP_STRATEGIES, "unit_step_up:100", "negate_values", "zero_values"]


def _sparse_parent() -> torch.Tensor:
    indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
    values = torch.tensor([0.5, -1.25, 3.0])
    return torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()


@pytest.mark.parametrize("strategy", _RETRY_STRATEGIES)
def test_retry_step_keeps_sparse_layout_and_indices(strategy: str) -> None:
    parent = _sparse_parent()

    stepped = perturb_through_values(parent, lambda v: _directional_step_perturb(v, strategy))

    assert stepped.layout is torch.sparse_coo
    assert stepped.shape == parent.shape
    assert torch.equal(stepped.coalesce().indices(), parent.indices())
    assert not torch.equal(stepped.coalesce().values(), parent.values())


def test_retry_step_on_a_strided_parent_is_the_dense_step() -> None:
    parent = torch.tensor([1.0, -2.0, 0.0])

    stepped = perturb_through_values(parent, lambda v: _directional_step_perturb(v, "step_up"))

    assert torch.equal(stepped, _directional_step_perturb(parent, "step_up"))


class _SparseProduct(nn.Module):
    """A sparse adjacency built from the input, multiplied into a dense operand."""

    def __init__(self, dense_fill: float) -> None:
        super().__init__()
        self.dense_fill = dense_fill

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
        adjacency = torch.sparse_coo_tensor(indices, x, (3, 3))
        dense = torch.full((3, 4), self.dense_fill)
        return torch.spmm(adjacency, dense)


def test_dead_sparse_edge_reaches_a_perturbation_verdict() -> None:
    # The dense operand is all zero, so the adjacency values cannot reach the
    # output: the first draw and every retry leave it unchanged. The verdict must
    # be a perturbation failure, not a sparse-backend exception.
    model = _SparseProduct(dense_fill=0.0)

    with pytest.warns(tl.errors.TorchLensWarning, match="tl.validate FAILED"):
        verdict = tl.validate(model, torch.tensor([1.0, 2.0, 3.0]), scope="forward", seed=0)

    failure = last_validation_failure()
    assert verdict is False
    assert failure is not None
    assert failure.check_name == "perturbation"
    assert failure.func_name == "spmm"


def test_live_sparse_edge_validates() -> None:
    model = _SparseProduct(dense_fill=1.5)

    assert tl.validate(model, torch.tensor([1.0, 2.0, 3.0]), scope="forward", seed=0) is True
