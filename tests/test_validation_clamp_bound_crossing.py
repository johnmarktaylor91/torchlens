"""Perturbation evidence for tensor ``clamp`` bounds that never bind.

A ``clamp`` whose ``min``/``max`` is a tensor (a ``0`` floor under positive values,
an ``inf`` ceiling, a ``full_like`` epsilon) is a live edge: a bound above the input
changes the output. The generic draw samples near the bound's own values and stays
inactive, so the edge looked dead. The bound is now moved one step past the value
it bounds. Each live case must validate; a replay that ignores its recorded bound
(a planted capture bug) and a bound that the other bound overrides everywhere (a
genuinely dead edge) must still fail the perturbation check.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens.validation import last_validation_failure
from torchlens.validation._crossing_perturbations import clamp_bound_crossing_values


class _TensorFloor(nn.Module):
    """Positive values clamped below by tensor bounds that never bind."""

    def __init__(self, form: str) -> None:
        super().__init__()
        self.form = form
        self.register_buffer("eps", torch.tensor([1e-3, 2e-3, 3e-3]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.exp() + 2.0  # above the generic draw's reach around a near-constant bound
        if self.form == "ode_step":  # a 0 floor and an inf ceiling, both 0-d tensors
            return y.clamp(torch.as_tensor(0.0), torch.as_tensor(float("inf")))
        if self.form == "full_like":
            return torch.clamp(y, min=torch.full_like(y, 1e-6))
        if self.form == "broadcast":
            return torch.clamp(y, min=self.eps * 1.0)
        if self.form == "clamp_max":
            return torch.clamp_max(y, torch.as_tensor(1e6))
        return y.clip(max=torch.full_like(y, 1e6))


_FORMS = ["ode_step", "full_like", "broadcast", "clamp_max", "clip_max"]
_FUNC_NAMES = {"clamp_max": "clamp_max", "clip_max": "clip"}


def _input() -> torch.Tensor:
    return torch.tensor([[0.1, -0.4, 0.9], [1.2, 0.3, -1.1]])


@pytest.mark.parametrize("form", _FORMS)
def test_inactive_tensor_bound_validates(form: str) -> None:
    assert tl.validate(_TensorFloor(form).eval(), _input(), scope="forward", seed=0) is True


def _freeze_bounds(func: Any, frozen_args: dict[int, Any], frozen_kwargs: dict[str, Any]) -> Any:
    """Wrap ``func`` into a capture bug that replays with its recorded bounds only."""

    def planted(*args: Any, **kwargs: Any) -> Any:
        args = tuple(frozen_args.get(i, a) for i, a in enumerate(args))
        kwargs = {k: frozen_kwargs.get(k, v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return planted


def _plant_frozen_bounds(monkeypatch: pytest.MonkeyPatch, func_name: str) -> list[str]:
    """Make every ``func_name`` op ignore the bound values it is replayed with."""

    planted_labels: list[str] = []
    original = Trace.validate_forward_pass

    def validate_with_plant(trace: Trace, *args: Any, **kwargs: Any) -> Any:
        for op in trace._layer_list:
            if op.func_name != func_name:
                continue
            saved_args = list(op.saved_args or ())
            frozen_args = {
                i: a.clone() for i, a in enumerate(saved_args) if i and torch.is_tensor(a)
            }
            frozen_kwargs = {
                k: v.clone() for k, v in (op.saved_kwargs or {}).items() if torch.is_tensor(v)
            }
            op.func = _freeze_bounds(op.func, frozen_args, frozen_kwargs)
            planted_labels.append(op.label)
        return original(trace, *args, **kwargs)

    monkeypatch.setattr(Trace, "validate_forward_pass", validate_with_plant)
    return planted_labels


@pytest.mark.parametrize("form", _FORMS)
def test_replay_ignoring_its_bound_fails(monkeypatch: pytest.MonkeyPatch, form: str) -> None:
    func_name = _FUNC_NAMES.get(form, "clamp")
    planted_labels = _plant_frozen_bounds(monkeypatch, func_name)

    with pytest.warns(tl.errors.TorchLensWarning, match="tl.validate FAILED"):
        verdict = tl.validate(_TensorFloor(form).eval(), _input(), scope="forward", seed=0)

    failure = last_validation_failure()
    assert planted_labels, "no op was planted; the test proves nothing"
    assert verdict is False
    assert failure is not None and failure.check_name == "perturbation"
    assert failure.func_name == func_name


class _OverriddenFloor(nn.Module):
    """A tensor floor under a ceiling the input exceeds everywhere: the floor is dead."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.exp() + 10.0
        return y.clamp(min=torch.as_tensor(0.0), max=torch.as_tensor(2.0))


def test_bound_overridden_by_the_other_bound_still_fails() -> None:
    with pytest.warns(tl.errors.TorchLensWarning, match="tl.validate FAILED"):
        verdict = tl.validate(_OverriddenFloor().eval(), _input(), scope="forward", seed=0)

    failure = last_validation_failure()
    assert verdict is False
    assert failure is not None and failure.check_name == "perturbation"
    assert failure.func_name == "clamp"


def _layer(func_name: str, args: tuple[Any, ...], slots: dict[Any, str], **kwargs: Any) -> Any:
    return SimpleNamespace(
        func_name=func_name,
        saved_args=args,
        saved_kwargs=kwargs,
        parent_arg_positions={
            "args": {k: v for k, v in slots.items() if isinstance(k, int)},
            "kwargs": {k: v for k, v in slots.items() if isinstance(k, str)},
        },
    )


def test_broadcast_lower_bound_binds_every_element() -> None:
    x = torch.tensor([[0.5, 2.0, -1.0], [3.0, 0.1, 4.0]])
    lo = torch.tensor([0.0, 1.0, -5.0])
    layer = _layer("clamp", (x, lo), {1: "lo"})

    moved = clamp_bound_crossing_values(layer, "lo", lo)

    assert moved is not None and moved.shape == lo.shape
    assert bool((torch.clamp(x, min=moved) != torch.clamp(x, min=lo)).all())


def test_upper_bound_keyword_binds_every_element() -> None:
    x = torch.tensor([0.5, 2.0, -1.0])
    lo = torch.tensor(0.0)
    hi = torch.full((3,), float("inf"))
    layer = _layer("clamp", (x,), {"max": "hi"}, min=lo, max=hi)

    moved = clamp_bound_crossing_values(layer, "hi", hi)

    assert moved is not None and bool(torch.isfinite(moved).all())
    assert bool((torch.clamp(x, min=lo, max=moved) != torch.clamp(x, min=lo, max=hi)).all())


def test_integer_bound_at_the_dtype_edge_is_not_moved() -> None:
    x = torch.tensor([torch.iinfo(torch.int64).max, 3])
    lo = torch.tensor([0, 0])
    layer = _layer("clamp", (x, lo), {1: "lo"})

    assert clamp_bound_crossing_values(layer, "lo", lo) is None


def test_parent_feeding_the_input_and_a_bound_is_not_moved() -> None:
    x = torch.tensor([0.5, 2.0])
    layer = _layer("clamp", (x, x), {0: "x", 1: "x"})

    assert clamp_bound_crossing_values(layer, "x", x) is None
