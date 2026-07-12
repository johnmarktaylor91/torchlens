"""Status-honesty goldens for receptive-field rule remediation."""

from __future__ import annotations

from collections.abc import Iterator
import importlib

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field._types import (
    ReceptiveFieldStatus,
    ReceptiveFieldValidationStatus,
)


_rf_package = importlib.import_module("torchlens.receptive_field")
_rules = importlib.import_module("torchlens.receptive_field._rules")
setattr(_rf_package, "_rules", _rules)
cross_validate = importlib.import_module("torchlens.receptive_field._validation").cross_validate
_PACK: dict[str, object] | None = None


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install built-in RF rules while preserving registry isolation."""

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


def _op(trace: object, name: str) -> object:
    """Return the last captured operation with the requested function name."""

    return next(
        item
        for item in reversed(trace.layer_list)  # type: ignore[union-attr]
        if item.func_name == name
    )


def _input_op(trace: object) -> object:
    """Return the canonical input operation from a one-input trace."""

    return next(item for item in trace.layer_list if item.is_input)  # type: ignore[union-attr]


def _trace(model: nn.Module, inputs: torch.Tensor) -> object:
    """Capture a model with autograd history retained for RF cross-validation."""

    return tl.trace(
        model,
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )


def test_depthwise_convolution_uses_containing_upper_bound() -> None:
    """Never label a channel-global depthwise descriptor exact."""

    model = nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    trace = _trace(model, torch.ones(2, 4, 5, 5, requires_grad=True))
    target = _op(trace, "conv2d")
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.UPPER_BOUND
    assert all(axis.kind == "full" and not axis.exact for axis in descriptor.axes)
    result = cross_validate(
        trace,
        ops=[target],
        units=(0, 0, 2, 2),
        inputs=_input_op(trace),
    )[0]
    assert result.status is ReceptiveFieldValidationStatus.PASS


class _Reduction(nn.Module):
    """Middle-axis reduction fixture."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Reduce channels while preserving batch and spatial identities."""

        return value.mean(dim=1)


def test_reduction_preserves_surviving_axis_coordinates() -> None:
    """Keep batch and spatial coordinate languages after a removed middle axis."""

    trace = _trace(_Reduction(), torch.ones(2, 4, 3, 5, requires_grad=True))
    target = _op(trace, "mean")
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.WHOLE_INPUT
    assert tuple(axis.kind for axis in descriptor.axes) == (
        "pointwise",
        "full",
        "pointwise",
        "pointwise",
    )
    assert tuple(axis.output_axis for axis in descriptor.axes) == (0, None, 1, 2)
    result = cross_validate(
        trace,
        ops=[target],
        units=(1, 2, 4),
        inputs=_input_op(trace),
    )[0]
    assert result.status is ReceptiveFieldValidationStatus.PASS


class _Cumulative(nn.Module):
    """Dimensioned cumulative-operation fixture."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Accumulate over height rather than the trailing axis."""

        return value.cumsum(dim=2)


def test_cumulative_reads_dimension_and_bounds_that_axis() -> None:
    """Use a containing bound on the selected cumulative axis only."""

    trace = _trace(_Cumulative(), torch.ones(2, 4, 3, 5, requires_grad=True))
    target = _op(trace, "cumsum")
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.UPPER_BOUND
    assert tuple(axis.kind for axis in descriptor.axes) == (
        "pointwise",
        "pointwise",
        "full",
        "pointwise",
    )
    result = cross_validate(
        trace,
        ops=[target],
        units=(1, 3, 2, 4),
        inputs=_input_op(trace),
    )[0]
    assert result.status is ReceptiveFieldValidationStatus.PASS


def test_bare_training_batch_norm_declares_batch_coupling() -> None:
    """Recognize positional-None BatchNorm captures as batch-coupled whole input."""

    model = nn.BatchNorm2d(2, affine=False, track_running_stats=False).train()
    trace = _trace(model, torch.randn(3, 2, 4, 4, requires_grad=True))
    target = _op(trace, "batch_norm")
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.WHOLE_INPUT
    assert descriptor.batch_coupled is True
    result = cross_validate(
        trace,
        ops=[target],
        units=(0, 0, 2, 2),
        inputs=_input_op(trace),
    )[0]
    assert result.status is ReceptiveFieldValidationStatus.PASS
    assert result.cross_batch == "geometric"
