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
from torchlens.validation.core import (
    _check_whether_func_on_saved_parents_yields_saved_tensor,
    _execute_func_with_restored_state,
    _prepare_input_args_for_validating_layer,
)

#: The strategies validation applies to a float running_var parent of a float batch_norm.
STRATEGIES = [
    "default",
    "step_up",
    "step_down",
    "unit_step_up",
    "unit_step_down",
    "lift_min_entry",
]


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


class _RawBatchNorm(nn.Module):
    """``torch.batch_norm`` with explicit buffers and eps, as the module path calls it."""

    def __init__(self, channels: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.batch_norm(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            False,
            0.1,
            self.eps,
            False,
        )


def _bn2d(channels: int = 4, *, affine: bool = True, **stats: torch.Tensor) -> nn.Module:
    model = nn.BatchNorm2d(channels, affine=affine).eval()
    with torch.no_grad():
        for name, value in stats.items():
            getattr(model, name).copy_(value)
    return model


def _at(mean: torch.Tensor, shape: tuple[int, ...] = (2, 4, 3, 3)) -> torch.Tensor:
    return mean.view(1, -1, *([1] * (len(shape) - 2))).expand(*shape).clone()


def _live_edge_changes(trace: Any, op: Any, label: str) -> list[str]:
    """Strategies under which the real op, fed validation's perturbation, changes its output."""

    torch.manual_seed(0)
    changed = []
    for strategy in STRATEGIES:
        args, _ = _prepare_input_args_for_validating_layer(
            trace, op, [label], perturb_strategy=strategy
        )
        if args is None:
            continue
        out = _execute_func_with_restored_state(op, args, [label], op.label, False)
        same = (out == op.out) | (torch.isnan(out) & torch.isnan(op.out))
        if out.shape != op.out.shape or not bool(same.all()):
            changed.append(strategy)
    return changed


def _hole_case_var_below_unit_step() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(1)
    return _bn2d(running_var=torch.full((4,), 0.5), bias=torch.randn(4)), torch.zeros(2, 4, 3, 3)


def _hole_case_large_output() -> tuple[nn.Module, torch.Tensor]:
    return _bn2d(bias=torch.full((4,), 30.0)), torch.zeros(2, 4, 3, 3)


def _hole_case_zero_weight_channel_low_var() -> tuple[nn.Module, torch.Tensor]:
    weight = torch.tensor([1.0, 1.0, 0.0, 1.0])
    var = torch.tensor([1.0, 1.0, 0.3, 1.0])
    x = torch.zeros(2, 4, 3, 3)
    x[:, 2] = 0.7
    return _bn2d(weight=weight, running_var=var), x


def _hole_case_negative_eps() -> tuple[nn.Module, torch.Tensor]:
    return _RawBatchNorm(4, eps=-0.5).eval(), torch.zeros(2, 4, 3, 3)


def _hole_case_nonzero_mean() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(2)
    mean = torch.randn(64) * 3
    stats = {
        "running_var": torch.rand(64) + 1.5,
        "weight": torch.randn(64),
        "bias": torch.randn(64),
    }
    return _bn2d(64, running_mean=mean, **stats), _at(mean, (2, 64, 3, 3))


def _hole_case_nonzero_mean_affine_false() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(2)
    mean = torch.randn(64) * 3
    model = _bn2d(64, affine=False, running_mean=mean, running_var=torch.rand(64) + 1.5)
    return model, _at(mean, (2, 64, 3, 3))


def _hole_case_nonzero_mean_channels_last() -> tuple[nn.Module, torch.Tensor]:
    model, x = _hole_case_nonzero_mean()
    return model, x.contiguous(memory_format=torch.channels_last)


def _hole_case_scale_overflows_after_unit_step() -> tuple[nn.Module, torch.Tensor]:
    return _bn2d(weight=torch.full((4,), 1e37)), torch.zeros(2, 4, 3, 3)


HOLE_CASES = [
    pytest.param(_hole_case_var_below_unit_step, id="var-below-unit-step"),
    pytest.param(_hole_case_large_output, id="widening-by-output-scale"),
    pytest.param(_hole_case_zero_weight_channel_low_var, id="zero-weight-channel-low-var"),
    pytest.param(_hole_case_negative_eps, id="negative-eps"),
    pytest.param(_hole_case_nonzero_mean, id="nonzero-mean"),
    pytest.param(_hole_case_nonzero_mean_affine_false, id="nonzero-mean-affine-false"),
    pytest.param(_hole_case_nonzero_mean_channels_last, id="nonzero-mean-channels-last"),
    pytest.param(_hole_case_scale_overflows_after_unit_step, id="scale-overflow"),
]


@pytest.mark.parametrize("make", HOLE_CASES)
def test_dead_running_var_edge_a_live_edge_would_expose_still_fails(make) -> None:
    """Adversarial review holes: validation's own running_var perturbation moves a live output.

    A NaN from ``running_var + eps <= 0`` or an overflowing scale, or a few ulps from the CPU
    kernel's folded statistics at a non-zero mean, changes a live edge's output; a dead edge
    there must still fail, never be exempted.
    """

    model, x = make()
    trace = _capture(model, x)
    op = _bn_op(trace)
    label = op.parent_arg_positions["args"][4]
    assert _live_edge_changes(trace, op, label), "case no longer exposes a live edge"
    _freeze(op)
    result = _edge(trace, op, 4)
    assert result.decision == "failed", (result.decision, result.reason)


def _sound_case_zero_weight_channel_off_mean() -> tuple[nn.Module, torch.Tensor]:
    x = torch.zeros(2, 4, 3, 3)
    x[:, 2] = 0.7
    return _bn2d(weight=torch.tensor([1.0, 1.0, 0.0, 1.0])), x


SOUND_CASES = [
    pytest.param(lambda: (nn.BatchNorm2d(4).eval(), torch.zeros(2, 4, 3, 3)), id="bn2d"),
    pytest.param(lambda: (nn.BatchNorm1d(6).eval(), torch.zeros(5, 6)), id="bn1d"),
    pytest.param(lambda: (nn.BatchNorm2d(4).eval(), -torch.zeros(2, 4, 3, 3)), id="negative-zero"),
    pytest.param(_sound_case_zero_weight_channel_off_mean, id="zero-weight-channel-off-mean"),
    pytest.param(lambda: (_RawBatchNorm(4, eps=1e-3).eval(), torch.zeros(2, 4, 3, 3)), id="eps"),
]


@pytest.mark.parametrize("make", SOUND_CASES)
def test_every_exempted_case_is_unchanged_under_each_validation_strategy(make) -> None:
    """The proof fires only where no perturbation validation applies can move a live output."""

    model, x = make()
    trace = _capture(model, x)
    op = _bn_op(trace)
    label = op.parent_arg_positions["args"][4]
    assert _edge(trace, op, 4).decision == "exempted"
    assert _live_edge_changes(trace, op, label) == []
