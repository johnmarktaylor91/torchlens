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
from torchlens._user_funcs import _run_model_and_save_specified_outs
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
        save_args=True,
        seed=0,
        save_rng=True,
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
    functions = _module_functions("core.py") | _module_functions("_saved_outs_steps.py")
    offenders = _status_exits_outside_recorder("validate_saved_outs", functions, set())
    assert not offenders, f"status exits outside the failure recorder: {offenders}"


def _module_functions(name: str) -> dict[str, ast.FunctionDef]:
    """Return the module-level functions of ``validation/<name>`` by name."""

    tree = ast.parse((_VALIDATION_DIR / name).read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _status_exits_outside_recorder(
    name: str, functions: dict[str, ast.FunctionDef], seen: set[str]
) -> list[str]:
    """Return the ``<function>:<line>`` exits of ``name`` that bypass the recorder.

    Every ``return`` must be ``None`` (a helper's "no failure" branch), a call to
    the recorder (``_finish_run_status`` / ``finish_validation_status``), a call to
    another function of the two modules whose exits are checked in turn, or a name
    bound in the function to such a call (``failed_status = _check_...(...)``).
    """

    if name in seen or name not in functions:
        return []
    seen.add(name)
    entry = functions[name]
    bound: dict[str, str] = {}
    for node in ast.walk(entry):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = node.value.func
            if isinstance(callee, ast.Name):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bound[target.id] = callee.id
    offenders: list[str] = []
    returns = [node for node in ast.walk(entry) if isinstance(node, ast.Return)]
    assert returns, name
    for node in returns:
        value = node.value
        if isinstance(value, ast.Constant) and value.value is None:
            continue
        callee = None
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            callee = value.func.id
        elif isinstance(value, ast.Name):
            callee = bound.get(value.id)
        if callee in {"_finish_run_status", "finish_validation_status"}:
            continue
        if callee in functions:
            offenders.extend(_status_exits_outside_recorder(callee, functions, seen))
            continue
        offenders.append(f"{name}:{node.lineno}")
    return offenders


def test_unrecorded_replay_failure_is_recorded_with_its_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replay site that fails without recording still yields a typed failure."""

    def fake(trace: Any, label: str, perturb: bool = False, **_: Any) -> ValidationCheckResult:
        return ValidationCheckResult.failed_result("synthetic_unrecorded")

    status, failure = _validate_with_patched_check(monkeypatch, fake)
    assert status.state == "failed"
    assert failure is not None
    assert failure.check_name == CHECK_REPLAY
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
    assert failure is not None and failure.check_name == CHECK_PERTURBATION
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
    assert failure.check_name == CHECK_REPLAY_UNVERIFIED
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


def test_perturbation_exception_text_reaches_the_unverified_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A perturbed replay that raises leaves its exception text in the record."""

    real = core.execute_replay_func

    def raising(layer: Any, input_args: Any, layers_to_perturb: list[str]) -> Any:
        if layers_to_perturb:
            raise ValueError("synthetic perturbed-replay error")
        return real(layer, input_args, layers_to_perturb)

    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    x = torch.randn(2, 4)
    ground_truth = model(x).detach().clone()
    trace = _capture(model, x)
    monkeypatch.setattr(core, "execute_replay_func", raising)
    try:
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        failure = get_validation_failure(trace)
    finally:
        trace.cleanup()
    assert status.state == "unverified"
    assert failure is not None and failure.check_name == CHECK_REPLAY_UNVERIFIED
    assert failure.extra["decision_reason"] == "perturbation_execution_exception"
    assert failure.extra["replay_exceptions"]
    assert "synthetic perturbed-replay error" in failure.message
