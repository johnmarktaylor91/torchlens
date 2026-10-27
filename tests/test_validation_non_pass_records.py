"""Every non-passing forward-validation status carries a recorded failure.

A run whose status is ``failed`` or ``unverified`` returns ``False`` from
``_validate_forward_pass_torch``; before ``_non_pass_record`` some of those runs
left ``get_validation_failure(trace)`` empty, so a caller could not tell why.
The structural test pins the single exit; the behavioral tests drive a deciding
site that records nothing and check the record names the check and the op.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens.validation.core as core
from torchlens.user_funcs import _run_model_and_save_specified_outs
from torchlens.validation import get_validation_failure
from torchlens.validation.core import ValidationCheckResult, validate_saved_outs
from torchlens.validation.diagnostics import (
    CHECK_PERTURBATION,
    CHECK_REPLAY,
    CHECK_REPLAY_UNVERIFIED,
)

_VALIDATION_DIR = Path(core.__file__).parent


def _capture(model: nn.Module, x: torch.Tensor) -> Any:
    """Capture a full save_arg_values trace the way forward validation does."""

    return _run_model_and_save_specified_outs(
        model=model,
        input_args=(x,),
        input_kwargs={},
        layers_to_save="all",
        activation_transform=None,
        mark_layer_depths=False,
        detach_saved_activations=False,
        save_grads=False,
        save_arg_values=True,
        random_seed=0,
        save_rng_states=True,
    )


def _validate_with_patched_check(monkeypatch: pytest.MonkeyPatch, fake: Any) -> tuple[Any, Any]:
    """Validate a small model with the per-op check replaced by ``fake``."""

    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    x = torch.randn(2, 4)
    ground_truth = model(x).detach().clone()
    trace = _capture(model, x)
    monkeypatch.setattr(core, "_check_whether_func_on_saved_parents_yields_saved_tensor", fake)
    try:
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        return status, get_validation_failure(trace)
    finally:
        trace.cleanup()


@pytest.mark.smoke
def test_every_status_exit_of_validate_saved_outs_goes_through_the_recorder() -> None:
    """Structural: no status is built or returned outside ``finish_validation_status``."""

    for path in sorted(_VALIDATION_DIR.glob("*.py")):
        if path.name == "_non_pass_record.py":
            continue
        tree = ast.parse(path.read_text())
        builders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "as_status"
        ]
        assert not builders, f"{path.name} builds a status directly at lines {builders}"
    tree = ast.parse((_VALIDATION_DIR / "core.py").read_text())
    entry = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_saved_outs"
    )
    returns = [node for node in ast.walk(entry) if isinstance(node, ast.Return)]
    assert returns
    for node in returns:
        value = node.value
        assert (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_finish_validation_status"
        ), f"validate_saved_outs line {node.lineno} returns without the failure recorder"


def test_unrecorded_replay_failure_is_recorded_with_its_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replay site that fails without recording still yields a typed failure."""

    def fake(trace: Any, label: str, perturb: bool = False, **_: Any) -> ValidationCheckResult:
        return ValidationCheckResult.failed_result("synthetic_unrecorded")

    status, failure = _validate_with_patched_check(monkeypatch, fake)
    assert status.state == "failed"
    assert failure is not None
    assert failure.check == CHECK_REPLAY
    assert failure.op_label is not None
    assert failure.extra["decision_reason"] == "synthetic_unrecorded"


def test_unrecorded_perturbation_failure_names_the_perturbation_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A perturbation-phase failed decision maps to the perturbation check."""

    real = core._check_whether_func_on_saved_parents_yields_saved_tensor

    def fake(trace: Any, label: str, perturb: bool = False, **kwargs: Any) -> ValidationCheckResult:
        if perturb:
            return ValidationCheckResult.failed_result("synthetic_unrecorded")
        return real(trace, label, perturb=perturb, **kwargs)

    status, failure = _validate_with_patched_check(monkeypatch, fake)
    assert status.state == "failed"
    assert failure is not None and failure.check == CHECK_PERTURBATION
    assert failure.op_label is not None


def test_unverified_status_is_recorded_as_replay_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unverified run is not a pass, so it records which op replay could not verify."""

    def fake(trace: Any, label: str, perturb: bool = False, **_: Any) -> ValidationCheckResult:
        return ValidationCheckResult.unverified("synthetic_unverifiable")

    status, failure = _validate_with_patched_check(monkeypatch, fake)
    assert status.state == "unverified" and not status.passed
    assert failure is not None
    assert failure.check == CHECK_REPLAY_UNVERIFIED
    assert failure.op_label is not None
    assert failure.extra["decision_reason"] == "synthetic_unverifiable"


def test_a_site_record_is_kept_over_the_status_record() -> None:
    """The recorder never overwrites the first failure a deciding site recorded."""

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    x = torch.randn(2, 4)
    trace = _capture(model, x)
    try:
        status = validate_saved_outs(trace, [torch.zeros(2, 3) + 7.0], validate_metadata=False)
        failure = get_validation_failure(trace)
        assert status.state == "failed"
        assert failure is not None and "decision_reason" not in failure.extra
    finally:
        trace.cleanup()
