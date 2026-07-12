"""Gradient receptive-field oracle tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _rules
from torchlens.receptive_field._errors import ReceptiveFieldUnavailableError
from torchlens.receptive_field._gradient import gradient_for_unit
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult


@pytest.fixture(autouse=True)
def isolated_rule_registry() -> Iterator[None]:
    """Restore the process-global RF rule registry after every test."""

    saved_rules = dict(_rules._RF_RULES)
    saved_epoch = _rules._RF_RULES_EPOCH
    yield
    _rules._RF_RULES.clear()
    _rules._RF_RULES.update(saved_rules)
    _rules._RF_RULES_EPOCH = saved_epoch


def _register_conv_rule() -> None:
    """Register the exact convolution geometry needed by focused gradient tests."""

    @_rules.register_rf_rule("conv2d", replace="conv2d" in _rules._RF_RULES)
    def convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit the captured two-dimensional convolution recurrence."""

        kernel = context.cfg("kernel_size")
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        assert isinstance(kernel, tuple)
        return context.window(
            kernel=kernel,
            stride=context.cfg("stride", (1, 1)),
            padding=context.cfg("padding", (0, 0)),
            dilation=context.cfg("dilation", (1, 1)),
        )


def _op(trace: object, func_name: str) -> object:
    """Return the last captured operation with the requested function name."""

    matches = [op for op in trace.layer_list if op.func_name == func_name]  # type: ignore[attr-defined]
    assert matches
    return matches[-1]


def test_conv_support_is_exact_and_matches_geometric_hull() -> None:
    """Seed one output element and recover exactly its 3x3 convolution support."""

    _register_conv_rule()
    model = nn.Conv2d(1, 2, 3, padding=1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    inputs = torch.randn(2, 1, 7, 7, requires_grad=True)
    capture = tl.options.CaptureOptions(backward_ready=True)
    trace = tl.trace(model, inputs, capture=capture, save_mode="reference")
    target = _op(trace, "conv2d")
    input_op = next(op for op in trace.layer_list if op.is_input)

    result = gradient_for_unit(target, (0, 0, 3, 3), input=input_op)

    assert result.grad.shape == inputs.shape
    assert result.support_ranges == ((0, 1), (0, 1), (2, 5), (2, 5))
    assert result.support_mask.sum().item() == 9
    assert torch.equal(result.grad[result.support_mask], torch.ones(9))
    assert result.spatial_support_mask is not None
    assert result.spatial_support_mask.shape == (7, 7)
    assert result.spatial_support_mask[2:5, 2:5].all()
    assert result.batch_support == (0,)
    assert not result.cross_batch_influence

    descriptor = next(iter(_rules_mapping(trace, target).values()))
    assert descriptor.size == (3, 3)
    assert tuple(stop - start for start, stop in result.support_ranges[-2:]) == descriptor.size


def _rules_mapping(trace: object, target: object) -> object:
    """Return geometric descriptors without importing the engine at module import time."""

    from torchlens.receptive_field import _engine

    return _engine.lookup(trace, target)  # type: ignore[arg-type]


def test_effective_mask_and_indices_use_deterministic_flat_ties() -> None:
    """Select magnitude mass deterministically and expose the exact influence set."""

    _register_conv_rule()
    model = nn.Conv2d(1, 1, 3, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    inputs = torch.randn(1, 1, 5, 5, requires_grad=True)
    capture = tl.options.CaptureOptions(backward_ready=True)
    trace = tl.trace(model, inputs, capture=capture, save_mode="reference")
    target = _op(trace, "conv2d")
    input_op = next(op for op in trace.layer_list if op.is_input)
    result = gradient_for_unit(target, (0, 0, 1, 1), input=input_op)

    effective = result.effective(0.5)
    expected = torch.zeros_like(result.support_mask)
    expected[0, 0, 1, 1:4] = True
    expected[0, 0, 2, 1:3] = True
    assert torch.equal(effective, expected)
    assert result.effective_ranges(0.5) == ((0, 1), (0, 1), (1, 3), (1, 4))
    indices = result.indices()
    assert len(indices) == inputs.ndim
    assert all(index.numel() == 9 for index in indices)


def test_training_batchnorm_reports_cross_batch_influence() -> None:
    """Retain full gradients and detect training-statistics sample coupling."""

    _register_conv_rule()

    @_rules.register_rf_rule("batch_norm", replace="batch_norm" in _rules._RF_RULES)
    def batch_norm(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Declare training BatchNorm coupling over batch and spatial axes."""

        return context.full(axes=(0, 2, 3))

    model = nn.Sequential(
        nn.Conv2d(1, 1, 1, bias=False),
        nn.BatchNorm2d(1, affine=False, track_running_stats=False),
    ).train()
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    inputs = torch.tensor(
        [[[[0.0, 1.0], [2.0, 4.0]]], [[[5.0, 7.0], [8.0, 11.0]]]],
        requires_grad=True,
    )
    capture = tl.options.CaptureOptions(backward_ready=True)
    trace = tl.trace(model, inputs, capture=capture, save_mode="reference")
    target = _op(trace, "batch_norm")
    input_op = next(op for op in trace.layer_list if op.is_input)

    result = gradient_for_unit(target, (0, 0, 0, 0), input=input_op)

    assert result.grad.shape == inputs.shape
    assert result.batch_support == (0, 1)
    assert result.cross_batch_influence
    assert result.support_mask[1].any()


class _DetachedPath(nn.Module):
    """Model with a structurally captured but autograd-detached input path."""

    def __init__(self) -> None:
        """Create a differentiable scalar bias for the detached output."""

        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return an output differentiable only through the parameter."""

        return inputs.detach() + self.bias


def test_reachable_but_none_gradient_raises_typed_error() -> None:
    """Reject a detached reachable path instead of reporting silent empty support."""

    inputs = torch.randn(1, 3, requires_grad=True)
    capture = tl.options.CaptureOptions(backward_ready=True)
    trace = tl.trace(_DetachedPath(), inputs, capture=capture, save_mode="reference")
    target = next(op for op in reversed(trace.layer_list) if not op.is_output)
    input_op = next(op for op in trace.layer_list if op.is_input)

    with pytest.raises(ReceptiveFieldUnavailableError, match="reachable.*autograd returned no"):
        gradient_for_unit(target, (0, 0), input=input_op)
