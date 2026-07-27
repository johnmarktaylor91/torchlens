"""Generalized and folded receptive-field validation tests."""

from __future__ import annotations

from collections.abc import Iterator
import importlib
from unittest import mock

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.op import Op
from torchlens.receptive_field import _rules
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult
from torchlens.receptive_field._types import (
    ReceptiveFieldDirection,
    ReceptiveFieldValidationStatus,
)
from torchlens.receptive_field._validation import cross_validate
from torchlens.validation.invariants import (
    METADATA_INVARIANT_CONTRACTS,
    check_metadata_invariants,
)


_PACK: dict[str, object] | None = None


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the built-in RF rules while preserving registry isolation."""

    global _PACK
    original = dict(_rules._RF_RULES)
    original_epoch = _rules._RF_RULES_EPOCH
    _rules._RF_RULES.clear()
    if _PACK is None:
        module = importlib.import_module("torchlens.receptive_field.rules")
        if not _rules._RF_RULES:
            for name in module.__all__:
                importlib.reload(getattr(module, name))
        _PACK = dict(_rules._RF_RULES)
    else:
        _rules._RF_RULES.update(_PACK)
    try:
        yield
    finally:
        _rules._RF_RULES.clear()
        _rules._RF_RULES.update(original)
        _rules._RF_RULES_EPOCH = original_epoch


def _trace(model: nn.Module, inputs: object) -> object:
    """Capture a graph-connected model suitable for repeated RF probes."""

    return tl.trace(model, inputs, backward_ready=True, save_mode="reference")


def _op(trace: object, name: str) -> Op:
    """Return the last captured operation with a raw function name."""

    matches = [item for item in trace.layer_list if item.func_name == name]
    assert matches
    return matches[-1]


def _input_ops(trace: object) -> tuple[Op, ...]:
    """Return graph-native input operations in capture order."""

    return tuple(item for item in trace.layer_list if item.is_input)


class _Chain(nn.Module):
    """Two-stage spatial chain for layer-to-layer validation."""

    def __init__(self) -> None:
        """Initialize deterministic convolution and pooling stages."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the two spatial stages."""

        return self.pool(self.conv(value))


def test_generalized_cross_validation_covers_all_four_endpoint_corners() -> None:
    """Pass receptive/projective checks at model and internal endpoints."""

    trace = _trace(_Chain(), torch.ones(1, 1, 7, 7, requires_grad=True))
    source = _input_ops(trace)[0]
    conv = _op(trace, "conv2d")
    pool = _op(trace, "avg_pool2d")
    checks = [
        cross_validate(trace, ops=[pool], inputs=source),
        cross_validate(trace, ops=[pool], source=conv),
        cross_validate(trace, ops=[source], direction="projective", target=conv),
        cross_validate(trace, ops=[conv], direction="projective", target=pool),
    ]

    assert all(group[0].status is ReceptiveFieldValidationStatus.PASS for group in checks)
    assert checks[0][0].direction is ReceptiveFieldDirection.RECEPTIVE
    assert checks[-1][0].direction is ReceptiveFieldDirection.PROJECTIVE


@pytest.mark.parametrize("taint", ["unknown", "data_dependent"])
def test_tainted_geometry_is_gracefully_indeterminate(taint: str) -> None:
    """Classify unavailable grid truth as indeterminate without a failure."""

    @_rules.register_rf_rule("softmax", replace=True)
    def tainted_softmax(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Return the selected deliberately tainted rule result."""

        if taint == "unknown":
            return context.unknown("mask unavailable")
        return context.data_dependent("mask depends on values")

    trace = _trace(nn.Softmax(dim=-1), torch.randn(1, 2, 5, requires_grad=True))
    result = cross_validate(trace, ops=[_op(trace, "softmax")])[0]

    assert result.status is ReceptiveFieldValidationStatus.INDETERMINATE
    assert result.n_violations == 0


def test_unavailable_gradient_is_indeterminate() -> None:
    """Keep valid geometry tri-state when the trace lacks a saved autograd graph."""

    trace = tl.trace(nn.Conv2d(1, 1, 3, padding=1), torch.ones(1, 1, 5, 5))
    result = cross_validate(trace, ops=[_op(trace, "conv2d")])[0]

    assert result.status is ReceptiveFieldValidationStatus.INDETERMINATE
    assert result.n_violations == 0


def test_plural_endpoint_keys_cover_multiple_inputs_and_outputs() -> None:
    """Expose honest aggregate endpoint keys for fan-in and fan-out validation."""

    class FanInOut(nn.Module):
        """Merge two inputs and return two dependent outputs."""

        def forward(
            self, left: torch.Tensor, right: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return a merged tensor and a scaled sibling output."""

            merged = left + right
            return merged, merged * 2

    inputs = (
        torch.ones(1, 1, 5, requires_grad=True),
        torch.full((1, 1, 5), 2.0, requires_grad=True),
    )
    trace = _trace(FanInOut(), inputs)
    merged = _op(trace, "__add__")
    source = _input_ops(trace)[0]
    receptive = cross_validate(trace, ops=[merged])[0]
    projective = cross_validate(trace, ops=[source], direction="projective")[0]

    assert len(receptive.source_keys) == 2
    assert receptive.target_keys == (merged.label,)
    assert projective.source_keys == (source.label,)
    assert len(projective.target_keys) == 2


def test_deliberately_undersized_rule_fails_in_both_directions() -> None:
    """Keep the containment tripwire strict in receptive and projective directions."""

    @_rules.register_rf_rule("conv2d", replace=True)
    def undersized_convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Lie that a real 3x3 convolution has 1x1 support."""

        return context.window(kernel=(1, 1), stride=1, padding=0, dilation=1)

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    trace = _trace(model, torch.ones(1, 1, 5, 5, requires_grad=True))
    source = _input_ops(trace)[0]
    conv = _op(trace, "conv2d")
    receptive = cross_validate(trace, ops=[conv], inputs=source)[0]
    projective = cross_validate(trace, ops=[source], direction="projective", target=conv)[0]

    assert receptive.status is ReceptiveFieldValidationStatus.FAIL
    assert projective.status is ReceptiveFieldValidationStatus.FAIL
    assert receptive.n_violations > 0
    assert projective.n_violations > 0


def test_metadata_contract_is_always_registered_and_runs_without_autograd() -> None:
    """Register RF geometry in the default metadata invariant sequence."""

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    trace = tl.trace(model, torch.ones(1, 1, 5, 5))

    assert "receptive_field_metadata" in {
        contract.name for contract in METADATA_INVARIANT_CONTRACTS
    }
    with mock.patch("torch.autograd.grad", side_effect=AssertionError("autograd invoked")):
        assert check_metadata_invariants(trace)


def test_validate_receptive_field_scope_samples_both_directions() -> None:
    """Fold sampled gradient containment into the consolidated validator."""

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    results = tl.validate(model, torch.ones(1, 1, 5, 5), scope="receptive_field")

    assert isinstance(results, list)
    assert {result.direction for result in results} == {
        ReceptiveFieldDirection.RECEPTIVE,
        ReceptiveFieldDirection.PROJECTIVE,
    }
    assert all(result.status is ReceptiveFieldValidationStatus.PASS for result in results)


def test_folded_scope_catches_undersized_untraced_support() -> None:
    """Catch capture/rule gaps through the folded zero-exemption tripwire."""

    @_rules.register_rf_rule("conv2d", replace=True)
    def undersized_convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Deliberately omit the convolution's off-center support."""

        return context.window(kernel=(1, 1), stride=1, padding=0, dilation=1)

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    results = tl.validate(model, torch.ones(1, 1, 5, 5), scope="receptive_field")

    failures = [
        result for result in results if result.status is ReceptiveFieldValidationStatus.FAIL
    ]
    assert {result.direction for result in failures} == {
        ReceptiveFieldDirection.RECEPTIVE,
        ReceptiveFieldDirection.PROJECTIVE,
    }
