"""Validation's save budget charges nothing for a parameter alias on every output_device.

Forward validation's own capture keeps a parameter argument as an alias of its
model copy's parameter storage (see ``torchlens._validation_argument_references``).
Such an alias adds no storage, so the save budget charges it nothing, with
``output_device="same"`` as with an offloading ``output_device``. These tests
pin, on the CPU:

* the ``"same"`` guard charges 0 bytes for a proven alias at admission and at
  commit, and a capture whose only large saved arguments are parameters
  validates under a budget below the weights;
* a tight budget still refuses non-alias saved arguments, and a cloned
  parameter (its own storage) is still charged in full;
* the empty-owner guard of the clone-path rerun and of a live-model run, and a
  user's own ``save_arg_values`` trace, still charge parameters as copies;
* a corrupted saved argument and a changed live parameter still fail.
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest import mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torchlens as tl
import torchlens.user_funcs as user_funcs
from torchlens import _user_public_impls, _validation_argument_references as arg_references
from torchlens._save_budget import SaveBudget, SaveBudgetExceededError
from torchlens.backends.torch import _ops_retention
from torchlens.options import CaptureOptions
from torchlens.validation.diagnostics import get_validation_diagnostics, get_validation_failure

_TIGHT = 64  # bytes: below every tensor these unit tests charge
_BELOW_WEIGHTS = 1 << 16  # 64 KiB: far above every real copy, far below a 256x256 weight


class _BudgetTrace:
    """The one attribute the saved-args budget sites read from a trace."""

    def __init__(self, budget_bytes: int) -> None:
        self._save_budget_accountant = SaveBudget.from_option(budget_bytes)


def _same_guard(model: nn.Module) -> Any:
    """The guard of validation's first attempt on its copy under ``output_device="same"``."""

    references = arg_references.ParameterArgumentReferences(arg_references._parameter_owners(model))
    assert not references.offloads
    return references


def _admit(references: Any, *t_args: Any) -> Any:
    token = arg_references._ACTIVE.set(references)
    try:
        return _ops_retention._admit_saved_args_budget(
            _BudgetTrace(_TIGHT), {"_label_raw": "op"}, t_args, {}
        )
    finally:
        arg_references._ACTIVE.reset(token)


def _admit_and_commit(references: Any, saved: torch.Tensor) -> None:
    trace = _BudgetTrace(_TIGHT)
    fields: dict[str, Any] = {"_label_raw": "op"}
    token = arg_references._ACTIVE.set(references)
    try:
        reservations = _ops_retention._admit_saved_args_budget(trace, fields, (), {})
        fields["saved_args"], fields["saved_kwargs"] = [saved], {}
        _ops_retention._commit_saved_args_budget(trace, fields, reservations)
    finally:
        arg_references._ACTIVE.reset(token)


class _ClonedWeight(nn.Module):
    """Uses a clone of its weight, so the saved argument has its own storage."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.lin.weight.clone(), self.lin.bias)


class _Offsets(nn.Module):
    """Linears, a relu and a buffer argument."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 5)
        self.out = nn.Linear(5, 3)
        self.register_buffer("scale", torch.full((3,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.lin(x))) * self.scale


def _validate(model: nn.Module, x: torch.Tensor, **options: Any) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validate(model, x, scope="forward", seed=0, **options))


# ---------------------------------------------------------------------------
# The budget sites under the "same" guard
# ---------------------------------------------------------------------------


def test_same_guard_admission_charges_nothing_for_a_parameter_alias() -> None:
    model = nn.Linear(16, 16)
    same = _same_guard(model)
    alias = model.weight.detach()
    assert same.shares_parameter_storage(alias)
    assert _admit(same, alias) == []
    assert _admit(same, model.weight, model.bias) == []


def test_same_guard_still_charges_non_alias_arguments_in_full() -> None:
    model = nn.Linear(16, 16)
    same = _same_guard(model)
    for value in (torch.zeros_like(model.weight), model.weight.detach().clone(), torch.ones(17)):
        assert not same.shares_parameter_storage(value)
        with pytest.raises(SaveBudgetExceededError):
            _admit(same, value)
    with pytest.raises(SaveBudgetExceededError):
        _admit(same, model.weight, torch.ones(17))


def test_same_guard_commit_charges_a_saved_alias_nothing_and_a_clone_in_full() -> None:
    model = nn.Linear(16, 16)
    same = _same_guard(model)
    _admit_and_commit(same, model.weight.detach())
    with pytest.raises(SaveBudgetExceededError):
        _admit_and_commit(same, model.weight.detach().clone())


def test_a_rebound_parameter_storage_is_charged_again_under_same() -> None:
    model = nn.Linear(16, 16)
    same = _same_guard(model)
    stale = model.weight.detach()
    model.weight.data = torch.zeros(16, 16)
    assert not same.shares_parameter_storage(stale)
    with pytest.raises(SaveBudgetExceededError):
        _admit(same, stale)


def _admit_under_the_active_guard(model: nn.Module) -> Any:
    return _ops_retention._admit_saved_args_budget(
        _BudgetTrace(_TIGHT), {"_label_raw": "op"}, (model.weight,), {}
    )


@pytest.mark.parametrize("output_device", ["same", "cpu"])
def test_clone_path_rerun_and_live_model_guards_charge_parameters(output_device: str) -> None:
    model = nn.Linear(16, 16)
    sessions = [
        (arg_references._ReferenceSession(fallback_reason="changed"), True),
        (arg_references._ReferenceSession(), False),
    ]
    for session, model_is_copy in sessions:
        token = arg_references._SESSION.set(session)
        try:
            with arg_references.parameter_argument_references(model, model_is_copy, output_device):
                assert _ops_retention._uncharged_parameter_aliases((model.weight,)) == set()
                with pytest.raises(SaveBudgetExceededError):
                    _admit_under_the_active_guard(model)
        finally:
            arg_references._SESSION.reset(token)


def test_outside_validation_parameters_are_charged() -> None:
    model = nn.Linear(16, 16)
    assert arg_references.active_argument_references() is None
    assert _ops_retention._uncharged_parameter_aliases((model.weight,)) == set()
    with pytest.raises(SaveBudgetExceededError):
        _admit_under_the_active_guard(model)


# ---------------------------------------------------------------------------
# End to end on the CPU (output_device="same")
# ---------------------------------------------------------------------------


def test_parameter_aliases_alone_do_not_trip_the_budget_on_same() -> None:
    torch.manual_seed(0)
    model, x = nn.Linear(256, 256), torch.randn(2, 256)
    assert _validate(model, x, output_device="same", save_budget=_BELOW_WEIGHTS)


def test_a_tight_budget_still_refuses_real_copies_on_same() -> None:
    torch.manual_seed(0)
    model, x = _Offsets(), torch.randn(64, 6)
    with pytest.raises(SaveBudgetExceededError):
        _validate(model, x, output_device="same", save_budget=256)


def test_a_cloned_parameter_is_still_charged_on_same() -> None:
    torch.manual_seed(0)
    model, x = _ClonedWeight(), torch.randn(2, 256)
    with pytest.raises(SaveBudgetExceededError):
        _validate(model, x, output_device="same", save_budget=_BELOW_WEIGHTS)
    assert _validate(model, x, output_device="same")


def test_a_user_trace_still_charges_parameter_copies() -> None:
    torch.manual_seed(0)
    model, x = nn.Linear(256, 256), torch.randn(2, 256)
    capture = CaptureOptions(layers_to_save="all", save_args=True, save_budget=_BELOW_WEIGHTS)
    with warnings.catch_warnings(), pytest.raises(SaveBudgetExceededError):
        warnings.simplefilter("ignore")
        tl.trace(model, x, capture=capture)


# ---------------------------------------------------------------------------
# Negative controls on "same"
# ---------------------------------------------------------------------------


def _corrupt_relu_input(trace: Any) -> None:
    relu = next(op for op in trace.layer_list if op.func_name == "relu")
    relu.saved_args[0].add_(1.0)


def _corrupt_saved_weight(trace: Any) -> None:
    linear = next(op for op in trace.layer_list if op.func_name == "linear")
    with torch.no_grad():
        linear.saved_args[1].add_(1.0)


def _validate_corrupted(corrupt: Any) -> tuple[bool, list[str]]:
    """Validate ``_Offsets`` on "same", corrupting validation's own capture before replay."""

    real_runner = user_funcs._run_model_and_save_specified_outs
    checks: list[str] = []

    def runner(*args: Any, **kwargs: Any) -> Any:
        trace = real_runner(*args, **kwargs)
        if kwargs["save_args"]:
            corrupt(trace)
        return trace

    def observe(trace: Any) -> None:
        failure = get_validation_failure(trace) if trace is not None else None
        if failure is not None:
            checks.append(failure.check)
        checks.extend(item.check for item in get_validation_diagnostics(trace) or ())

    torch.manual_seed(0)
    model, x = _Offsets(), torch.randn(4, 6)
    with (
        mock.patch.object(user_funcs, "_run_model_and_save_specified_outs", runner),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        verdict = _user_public_impls._validate_forward_pass_torch(
            model, x, seed=0, output_device="same", _trace_observer=observe
        )
    return bool(verdict), checks


def test_a_corrupted_saved_argument_still_fails_on_same() -> None:
    verdict, _ = _validate_corrupted(_corrupt_relu_input)
    assert verdict is False


def test_a_changed_live_parameter_still_fails_on_same() -> None:
    verdict, checks = _validate_corrupted(_corrupt_saved_weight)
    assert verdict is False
    assert arg_references.FALLBACK_CHECK in checks
