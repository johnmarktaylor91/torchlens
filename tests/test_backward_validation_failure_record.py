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

import ast
import dataclasses
from pathlib import Path
from textwrap import dedent
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
    return bool(tl.validate(_WithChild(), torch.randn(4, 3), scope="backward", random_seed=0))


def _recorded() -> ValidationFailure:
    failure = last_validation_failure()
    assert failure is not None, "a False backward verdict left no failure record"
    assert failure.check == CHECK_BACKWARD
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
    record_validation_failure(None, ValidationFailure(check="stale", message="an earlier run"))
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


# ---------------------------------------------------------------------------
# Structural tripwire: no bare False return may be added to validate_backward_pass.
# ---------------------------------------------------------------------------

_RECORDERS = frozenset({"_record_backward_failure", "_record_module_output_grad_failure"})
_FALSE_RETURNING_HELPERS = frozenset({"_warn_and_skip_appended_trace_validation"})


def _is_false_return(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Return):
        return False
    value = statement.value
    if isinstance(value, ast.Constant) and value.value is False:
        return True
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in _FALSE_RETURNING_HELPERS
    )


def _records_a_failure(statement: ast.stmt) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _RECORDERS
        for node in ast.walk(statement)
    )


def _unrecorded_false_returns(function: ast.FunctionDef) -> list[int]:
    """Line numbers of False returns with no recorder call earlier in their own block."""

    bare: list[int] = []
    for node in ast.walk(function):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for index, statement in enumerate(block):
                if _is_false_return(statement) and not any(
                    _records_a_failure(earlier) for earlier in block[:index]
                ):
                    bare.append(statement.lineno)
    return sorted(bare)


def _function_named(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_every_false_return_in_validate_backward_pass_records_a_failure() -> None:
    """Each ``return False`` in the verdict function follows a recorder call in its block.

    The per-path tests above prove the recorded reasons; this check keeps a future
    False path from shipping without one.
    """

    source = Path(backward_validation.__file__).read_text(encoding="utf-8")
    function = _function_named(ast.parse(source), "validate_backward_pass")
    false_returns = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.stmt) and _is_false_return(node)
    ]
    assert len(false_returns) >= 6, false_returns
    assert _unrecorded_false_returns(function) == []


def test_bare_false_return_tripwire_fires_on_a_planted_path() -> None:
    planted = dedent(
        """
        def validate_backward_pass(model):
            if model is None:
                _record_backward_failure(None, reason="x", message="y")
                return False
            if not model:
                return False
            return True
        """
    )
    function = _function_named(ast.parse(planted), "validate_backward_pass")
    assert _unrecorded_false_returns(function) == [7]
