"""Tightness oracle for EXACT receptive-field claims.

The check()/cross_validate tripwire is one-sided: it proves gradient support is
INSIDE the geometric box, so an oversized-EXACT rule passes containment, and
the adjoint corner cross-check consults the same rule registry through the
opposite-direction engine, so a symmetric overclaim passes that too. Gradient
support is only a lower bound on influence (dead units, argmax pooling), so a
blanket in-check FAIL gate on slack would be unsound for user models — the
sound tightness oracle is this battery: under SATURATING models (all-ones
weights, uniform pooling, everything influential) the gradient support of an
honest EXACT box must reach its per-axis endpoints at interior (unclipped)
units, i.e. ``slack_per_axis`` must be all zeros. An oversized-EXACT built-in
rule turns this battery red even though containment stays green.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from support.rf_isolation import preserved_rf_registry
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _rules
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult
from torchlens.receptive_field._types import (
    ReceptiveFieldStatus,
    ReceptiveFieldValidationStatus,
)


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the built-in RF rules while preserving registry isolation."""

    with preserved_rf_registry(install_builtin=True):
        yield


def _saturated(module: nn.Module) -> nn.Module:
    """Fill every parameter with ones so all claimed influence paths carry gradient."""

    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(1.0)
    return module


def _check_at(model: nn.Module, inputs: torch.Tensor, name: str, unit: tuple[int, ...]) -> object:
    """Trace a saturating model and check one interior unit against gradients."""

    capture = tl.options.CaptureOptions(backward_ready=True)
    trace = tl.trace(model, inputs.requires_grad_(True), capture=capture, save_mode="reference")
    target = [op for op in trace.layer_list if op.func_name == name][-1]
    input_op = next(op for op in trace.layer_list if op.is_input)
    return target.receptive_field.check(unit, input=input_op)


# Interior units only: clipping at the input extent legitimately leaves the
# claimed (unclipped) endpoints unreachable. Max pooling is deliberately
# absent: its gradient support is argmax-sparse, so nonzero slack is honest.
@pytest.mark.smoke_cells(
    "test_exact_builtin_rules_are_tight_under_saturation[model4-input_shape4-avg_pool1d-unit4]"
)
@pytest.mark.parametrize(
    ("model", "input_shape", "name", "unit"),
    [
        (
            _saturated(nn.Conv2d(1, 1, 3, padding=1, bias=False)),
            (1, 1, 9, 9),
            "conv2d",
            (0, 0, 4, 4),
        ),
        (
            _saturated(nn.Conv1d(1, 1, 5, dilation=2, padding=4, bias=False)),
            (1, 1, 15),
            "conv1d",
            (0, 0, 7),
        ),
        (
            _saturated(nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False)),
            (1, 1, 11, 11),
            "conv2d",
            (0, 0, 3, 3),
        ),
        (nn.AvgPool2d(3, stride=2), (1, 1, 11, 11), "avg_pool2d", (0, 0, 2, 2)),
        (nn.AvgPool1d(2), (1, 1, 12), "avg_pool1d", (0, 0, 3)),
    ],
)
def test_exact_builtin_rules_are_tight_under_saturation(
    model: nn.Module, input_shape: tuple[int, ...], name: str, unit: tuple[int, ...]
) -> None:
    """Require every claimed-EXACT endpoint to be attained by gradient support."""

    result = _check_at(model, torch.ones(input_shape), name, unit)
    assert result.status is ReceptiveFieldValidationStatus.PASS
    assert result.slack_per_axis == (0,) * len(input_shape)


def test_oversized_exact_claim_is_caught_by_the_slack_oracle() -> None:
    """Turn the battery red for a symmetric overclaim that containment passes.

    A deliberately oversized exact rule (9x9 claimed for a true 5x5 conv) still
    PASSes one-sided gradient containment, and the adjoint corner cross-check
    consults the same lying registry in both directions, so it stays blind too
    — that PASS is the documented gap this oracle exists to cover, and a future
    strengthening that turns it into a FAIL should update this control. The
    slack oracle exposes the overclaim today.
    """

    @_rules.register_rf_rule("conv2d", replace=True)
    def oversized_convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Deliberately claim a nine-by-nine exact field for a five-by-five kernel."""

        return context.window(kernel=(9, 9), stride=(1, 1), padding=(4, 4), dilation=(1, 1))

    model = _saturated(nn.Conv2d(1, 1, 5, padding=2, bias=False))
    result = _check_at(model, torch.ones(1, 1, 11, 11), "conv2d", (0, 0, 5, 5))
    assert result.status is ReceptiveFieldValidationStatus.PASS
    box = next(iter(result.geometric.values()))
    assert box.exact
    assert result.slack_per_axis is not None
    assert result.slack_per_axis != (0, 0, 0, 0)
    assert result.slack_per_axis[-2:] == (4, 4)


@pytest.mark.smoke
def test_undersized_claim_still_fails_containment() -> None:
    """Keep the containment tripwire armed alongside the tightness oracle."""

    @_rules.register_rf_rule("conv2d", replace=True)
    def undersized_convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Deliberately claim a one-by-one exact field for a five-by-five kernel."""

        return context.window(kernel=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1))

    model = _saturated(nn.Conv2d(1, 1, 5, padding=2, bias=False))
    result = _check_at(model, torch.ones(1, 1, 11, 11), "conv2d", (0, 0, 5, 5))
    assert result.status is ReceptiveFieldValidationStatus.FAIL


@pytest.mark.smoke
def test_upper_bound_claims_are_outside_the_tightness_contract() -> None:
    """Keep honest UPPER_BOUND envelopes (adaptive pool) exempt from slack demands.

    The adaptive-pool DESCRIPTOR is an UPPER_BOUND envelope (no tightness
    demanded of it), while the per-unit box routes through the exact bin
    callback and is genuinely exact — so the per-unit check may demand
    tightness and gets it, with zero slack under uniform average pooling.
    """

    capture = tl.options.CaptureOptions(backward_ready=True)
    trace = tl.trace(
        nn.AdaptiveAvgPool1d(3),
        torch.ones(1, 1, 15, requires_grad=True),
        capture=capture,
        save_mode="reference",
    )
    target = [op for op in trace.layer_list if op.func_name == "adaptive_avg_pool1d"][-1]
    input_op = next(op for op in trace.layer_list if op.is_input)
    assert target.receptive_field.status is ReceptiveFieldStatus.UPPER_BOUND
    result = target.receptive_field.check((0, 0, 1), input=input_op)
    assert result.status is ReceptiveFieldValidationStatus.PASS
    box = next(iter(result.geometric.values()))
    assert box.exact
    assert result.slack_per_axis == (0, 0, 0)
