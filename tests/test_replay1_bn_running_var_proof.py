"""Eval-mode batch_norm whose input sits exactly at running_mean: running_var cannot matter.

Mirrors the 32 Menagerie replay gaps: ``torch.batch_norm(x, weight, bias, running_mean,
running_var, False, 0.1, 1e-05, True)`` with ``x`` all zero (a dead ReLU or a silent spiking
layer) and a fresh ``running_mean`` of zero, so the perturbed ``running_var`` buffer at
slot 4 cannot change the output. The proof must still refuse a running_var edge whose
input is off the mean by any amount, a running_mean or input edge, and a training call.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation import validate_forward_pass
from torchlens.validation.core import _check_whether_func_on_saved_parents_yields_saved_tensor


def _capture(model: nn.Module, x: torch.Tensor) -> Any:
    torch.manual_seed(0)
    return tl.trace(
        model, x, capture=CaptureOptions(layers_to_save="all", save_arg_values=True, random_seed=0)
    )


def _bn_op(trace: Any) -> Any:
    return next(op for op in trace.layer_list if op.func_name == "batch_norm")


def _edge(trace: Any, op: Any, slot: int) -> Any:
    parent = op.parent_arg_positions["args"][slot]
    return _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[parent]
    )


def _freeze(op: Any) -> None:
    """Simulate a dead edge: the replay returns the saved out whatever its arguments."""

    saved = op.out.detach().clone()
    object.__setattr__(op, "func", lambda *args, **kwargs: saved.clone())


CASES = [
    pytest.param(lambda: nn.BatchNorm2d(4), (2, 4, 3, 3), id="bn2d-4d"),
    pytest.param(lambda: nn.BatchNorm1d(6), (5, 6), id="bn1d-2d"),
]


@pytest.mark.parametrize(("make", "shape"), CASES)
def test_eval_bn_with_input_at_running_mean_validates(make, shape) -> None:
    model = make().eval()
    x = torch.zeros(shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert validate_forward_pass(model, x, random_seed=0) is True


@pytest.mark.parametrize(("make", "shape"), CASES)
def test_running_var_edge_is_exempted_by_the_value_proof(make, shape) -> None:
    trace = _capture(make().eval(), torch.zeros(shape))
    op = _bn_op(trace)
    assert op.saved_args[5] is False
    result = _edge(trace, op, 4)
    assert result.decision == "exempted", (result.decision, result.reason)


@pytest.mark.parametrize(("make", "shape"), CASES)
def test_dead_running_var_edge_off_the_mean_still_fails(make, shape) -> None:
    """A plain-capture bug of the same shape: input one ulp-scale step off the mean."""

    x = torch.zeros(shape)
    x.view(-1)[0] = 1e-6
    trace = _capture(make().eval(), x)
    op = _bn_op(trace)
    _freeze(op)
    result = _edge(trace, op, 4)
    assert result.decision == "failed", (result.decision, result.reason)
    assert result.reason == "perturbation_insensitive"


@pytest.mark.parametrize("slot", [0, 3])
def test_dead_input_or_running_mean_edge_at_the_mean_still_fails(slot) -> None:
    trace = _capture(nn.BatchNorm2d(4).eval(), torch.zeros(2, 4, 3, 3))
    op = _bn_op(trace)
    _freeze(op)
    result = _edge(trace, op, slot)
    assert result.decision == "failed", (slot, result.decision, result.reason)


def test_dead_running_var_edge_with_random_input_still_fails() -> None:
    trace = _capture(nn.BatchNorm2d(4).eval(), torch.randn(2, 4, 3, 3))
    op = _bn_op(trace)
    _freeze(op)
    result = _edge(trace, op, 4)
    assert result.decision == "failed", (result.decision, result.reason)
    assert result.reason == "perturbation_insensitive"
