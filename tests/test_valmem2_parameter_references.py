"""Forward validation keeps its own parameter arguments by reference, under a version guard.

Validation captures with ``save_args=True`` on a deep copy of the model.
During that one capture a parameter argument is kept by reference rather than
copied, and its version counter is recorded. These tests pin:

* the reference is taken only inside validation's capture, on the deep copy;
  a user's ``save_args=True`` trace still saves copies, and the context
  is cleared on every exit, an exception included;
* a model that writes a parameter in place during forward makes validation
  rerun with every argument copied (the guard's fallback, recorded as a
  diagnostic) and reach the same verdict as the copy-only path;
* a model that does not write a parameter never reruns.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _user_public_impls, _validation_argument_references as arg_references
from torchlens.validation.diagnostics import get_validation_diagnostics


def _validate(model: nn.Module, x: torch.Tensor) -> tuple[bool, list[Any], int]:
    """Validate ``model``; return the verdict, the replayed trace's diagnostics, the forwards run."""

    forwards = [0]
    diagnostics: list[Any] = []

    def count(module: nn.Module, inputs: Any) -> None:
        forwards[0] += 1

    def observe(trace: Any) -> None:
        diagnostics.extend(get_validation_diagnostics(trace))

    handle = model.register_forward_pre_hook(count)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            verdict = _user_public_impls._validate_forward_pass_torch(
                model, x, seed=0, _trace_observer=observe
            )
    finally:
        handle.remove()
    return bool(verdict), diagnostics, forwards[0]


def _copy_only_verdict(model: nn.Module, x: torch.Tensor) -> bool:
    """The verdict with every argument copied (the path before references existed)."""

    unwrapped: Callable[..., bool] = _user_public_impls._validate_forward_pass_torch.__wrapped__  # type: ignore[attr-defined]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(unwrapped(model, x, seed=0))


def _fallbacks(diagnostics: list[Any]) -> list[Any]:
    return [item for item in diagnostics if item.check_name == arg_references.FALLBACK_CHECK]


class _Plain(nn.Module):
    """No parameter is written: two linears, a weight view, a frozen scale."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 4)
        self.out = nn.Linear(4, 3)
        self.scale = nn.Parameter(torch.full((3,), 0.5), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.lin(x))
        tied = hidden @ self.lin.weight[:, :4].t()
        return self.out(tied) * self.scale


class _BatchNormTraining(nn.Module):
    """Buffers change every forward (train-mode batch norm); parameters do not."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 5)
        self.bn = nn.BatchNorm1d(5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.lin(x))


_WRITES = (
    "no_grad_add_zero",
    "no_grad_add_one",
    "frozen_add_zero_op",
    "data_mul_one",
    "data_mul_two",
)


class _ParameterWriter(nn.Module):
    """Writes a parameter in place between two uses of it, one way per ``mode``."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.lin = nn.Linear(4, 4)
        self.frozen = nn.Parameter(torch.ones(4), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.lin(x) * self.frozen
        if self.mode == "no_grad_add_zero":
            with torch.no_grad():
                self.lin.weight.add_(0.0)
        elif self.mode == "no_grad_add_one":
            with torch.no_grad():
                self.lin.weight.add_(1.0)
        elif self.mode == "frozen_add_zero_op":
            self.frozen.add_(0.0)
        elif self.mode == "data_mul_one":
            self.lin.weight.data.mul_(1.0)
        elif self.mode == "data_mul_two":
            self.lin.weight.data.mul_(2.0)
        return self.lin(torch.tanh(hidden)) * self.frozen


@pytest.mark.parametrize("factory", [_Plain, _BatchNormTraining], ids=["plain", "batchnorm-train"])
def test_a_model_that_writes_no_parameter_never_takes_the_fallback(
    factory: Callable[[], nn.Module],
) -> None:
    torch.manual_seed(0)
    model = factory()
    model.train()
    verdict, diagnostics, forwards = _validate(model, torch.randn(8, 6))
    assert verdict is True
    assert _fallbacks(diagnostics) == []
    # Ground truth, capture, re-trace: a rerun would add a second set.
    assert forwards == 3


@pytest.mark.parametrize("mode", _WRITES)
def test_a_parameter_written_in_forward_reruns_on_the_copy_path_with_the_same_verdict(
    mode: str,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    copy_only = _copy_only_verdict(_ParameterWriter(mode), x)
    torch.manual_seed(0)
    model = _ParameterWriter(mode)
    weight_before = model.lin.weight.detach().clone()
    verdict, diagnostics, forwards = _validate(model, x)
    assert verdict is copy_only
    fallbacks = _fallbacks(diagnostics)
    assert len(fallbacks) == 1, diagnostics
    assert "written in place" in fallbacks[0].extra["reason"]
    assert forwards > 3
    assert torch.equal(model.lin.weight.detach(), weight_before)


def test_validation_capture_saves_an_alias_of_its_copys_parameters() -> None:
    torch.manual_seed(0)
    model = _Plain()
    forward_storages: list[int] = []
    saved_weights: list[torch.Tensor] = []

    def record(module: nn.Module, inputs: Any) -> None:
        forward_storages.append(module.lin.weight.untyped_storage().data_ptr())

    def observe(trace: Any) -> None:
        for op in trace.layer_list:
            if op.func_name == "linear" and op.saved_args:
                saved_weights.append(op.saved_args[1])

    handle = model.register_forward_pre_hook(record)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _user_public_impls._validate_forward_pass_torch(
            model, torch.randn(2, 6), _trace_observer=observe
        )
    handle.remove()
    # Forwards: ground truth (its own copy), capture and re-trace (the replay copy).
    assert len(forward_storages) == 3
    live = {p.untyped_storage().data_ptr() for p in model.parameters()}
    capture_weight = saved_weights[0]
    assert len(saved_weights) == 2
    assert type(capture_weight) is torch.Tensor
    assert capture_weight.requires_grad
    assert capture_weight.untyped_storage().data_ptr() == forward_storages[1]
    assert forward_storages[1] not in live


def test_a_user_save_arg_values_trace_still_saves_copies_after_validation() -> None:
    torch.manual_seed(0)
    model = _Plain()
    x = torch.randn(2, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="forward")
    assert arg_references.active_argument_references() is None
    assert arg_references._SESSION.get() is None
    trace = tl.trace(
        model, x, capture=tl.options.CaptureOptions(layers_to_save="all", save_args=True)
    )
    linears = [op for op in trace.layer_list if op.func_name == "linear"]
    assert len(linears) == 2
    for op, parameter in zip(linears, (model.lin.weight, model.out.weight), strict=True):
        saved = op.saved_args[1]
        assert not isinstance(saved, nn.Parameter)
        assert saved.untyped_storage().data_ptr() != parameter.untyped_storage().data_ptr()
        assert torch.equal(saved, parameter.detach())


def test_the_reference_context_is_cleared_when_the_capture_raises() -> None:
    seen: list[Any] = []

    def raise_on_capture(module: nn.Module, inputs: Any) -> None:
        # A hook closure is shared by validation's deep copies (a list
        # attribute would be copied with the model).
        seen.append(arg_references.active_argument_references())
        if len(seen) == 2:
            raise ValueError("raised inside the validation capture")

    model = nn.Linear(4, 4)
    model.register_forward_pre_hook(raise_on_capture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _user_public_impls._validate_forward_pass_torch(model, torch.randn(2, 4))
        except ValueError as exc:
            assert "inside the validation capture" in str(exc)
    # The ground truth runs outside the context; the raising forward ran inside it.
    assert seen[0] is None
    assert isinstance(seen[1], arg_references.ParameterArgumentReferences)
    assert arg_references.active_argument_references() is None
    assert arg_references._SESSION.get() is None


def test_the_context_is_cleared_when_its_block_raises() -> None:
    inside: list[Any] = []

    def capture_that_raises(model: nn.Module) -> bool:
        with arg_references.parameter_argument_references(model, True):
            inside.append(arg_references.active_argument_references())
            raise ValueError("capture failed")

    wrapped = arg_references.with_parameter_reference_fallback(capture_that_raises)
    with pytest.raises(ValueError, match="capture failed"):
        wrapped(nn.Linear(2, 2))
    assert isinstance(inside[0], arg_references.ParameterArgumentReferences)
    assert arg_references.active_argument_references() is None
    assert arg_references._SESSION.get() is None


def test_the_guard_flags_a_write_through_a_view_or_through_data() -> None:
    model = nn.Linear(3, 2)
    owners = arg_references._parameter_owners(model)
    view_guard = arg_references.ParameterArgumentReferences(owners)
    assert view_guard.keep(model.weight.t())
    assert not view_guard.keep(torch.ones(2, 3))
    assert view_guard.first_change() is None
    with torch.no_grad():
        model.weight[0].add_(0.0)
    assert "'weight'" in (view_guard.first_change() or "")

    data_guard = arg_references.ParameterArgumentReferences(owners)
    detached = model.weight.data
    assert data_guard.keep(detached)
    detached.mul_(1.0)
    assert data_guard.first_change() is not None
    with pytest.raises(arg_references._ParameterArgumentChanged):
        data_guard.raise_if_changed()


def test_inference_tensor_parameters_are_copied_not_referenced() -> None:
    with torch.inference_mode():
        model = nn.Linear(3, 2)
    assert model.weight.is_inference()
    guard = arg_references.ParameterArgumentReferences(arg_references._parameter_owners(model))
    assert not guard.keep(model.weight)
    assert guard.first_change() is None
