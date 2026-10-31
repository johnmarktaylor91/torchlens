"""Every False backward verdict carries a structured failure record.

``validate_backward_pass`` used to warn on some of its refusals and return a bare
False on the rest (a module-output gradient mismatch, a parameter census that
differs from stock, a parameter gradient outside tolerance), so
``last_validation_failure()`` was None after a False backward verdict and a
harness reading it saw "no failure record". Each False now records one
``ValidationFailure`` under ``CHECK_BACKWARD`` naming its reason, and a True
verdict leaves no stale record behind.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import (
    _layer_grad_report as layer_grad_report,
    backward as backward_validation,
    last_validation_failure,
)
from torchlens.validation.diagnostics import (
    CHECK_BACKWARD,
    ValidationFailure,
    record_validation_failure,
)


class _WithChild(nn.Module):
    """One child module, so the module-output gradient check has evidence."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x)).sum(dim=-1)


def _validate() -> bool:
    torch.manual_seed(0)
    return bool(tl.validate(_WithChild(), torch.randn(4, 3), scope="backward", seed=0))


def _recorded() -> ValidationFailure:
    failure = last_validation_failure()
    assert failure is not None, "a False backward verdict left no failure record"
    assert failure.check_name == CHECK_BACKWARD
    return failure


def _tamper_observed_param_grads(monkeypatch: pytest.MonkeyPatch, edit: Any) -> None:
    """Apply ``edit`` to the captured pass's parameter gradients (the second census)."""

    real = backward_validation._param_grads
    calls: list[int] = []

    def tampered(model: nn.Module) -> dict[str, torch.Tensor]:
        grads = real(model)
        calls.append(1)
        return edit(grads) if len(calls) == 2 else grads

    monkeypatch.setattr(backward_validation, "_param_grads", tampered)


def test_passing_backward_validation_clears_a_stale_record() -> None:
    record_validation_failure(None, ValidationFailure(check_name="stale", message="an earlier run"))
    assert _validate() is True
    assert last_validation_failure() is None


def test_parameter_gradient_mismatch_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    def double_weight(grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: grad * 2 if name == "lin.weight" else grad for name, grad in grads.items()}

    _tamper_observed_param_grads(monkeypatch, double_weight)
    assert _validate() is False
    failure = _recorded()
    assert failure.extra["reason"] == "parameter_grad_mismatch"
    assert failure.op_label == "lin.weight"
    assert failure.max_abs_diff is not None and failure.max_abs_diff > 0
    assert failure.saved_shape == (2, 3) and failure.saved_dtype == "float32"
    assert failure.summary().startswith("backward mismatch at lin.weight")


def test_parameter_census_mismatch_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    def drop_bias(grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: grad for name, grad in grads.items() if name != "lin.bias"}

    _tamper_observed_param_grads(monkeypatch, drop_bias)
    assert _validate() is False
    failure = _recorded()
    assert failure.extra["reason"] == "parameter_census_mismatch"
    assert failure.op_label == "lin.bias"
    assert "missing ['lin.bias']" in failure.message


def test_module_output_gradient_mismatch_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    real = layer_grad_report._compare_module_output_grads
    seen: list[str] = []

    def tampered(*args: Any, **kwargs: Any) -> Any:
        report = real(*args, **kwargs)
        label = next(name for name, bucket in report.coverage.items() if bucket == "covered")
        seen.append(label)
        coverage = dict(report.coverage)
        coverage[label] = "mismatched"
        return dataclasses.replace(
            report,
            overall_passed=False,
            coverage=coverage,
            mismatched_count=1,
            mismatched_labels=(label,),
            max_abs_diffs={**report.max_abs_diffs, label: 0.25},
        )

    monkeypatch.setattr(layer_grad_report, "_compare_module_output_grads", tampered)
    assert _validate() is False
    failure = _recorded()
    assert failure.extra["reason"] == "module_output_grads"
    assert failure.op_label == seen[0]
    assert failure.max_abs_diff == 0.25
    assert "1 mismatched" in failure.message
