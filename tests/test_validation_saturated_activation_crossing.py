"""Perturbation evidence for fully saturated smooth activations.

``sigmoid`` of an input above about 17 is exactly 1.0 in float32 (``tanh`` above
about 9, ``erf`` above about 4 are exactly 1.0), so a perturbation drawn from the
saved range, or a small step, stays saturated and a live edge looked dead. A fully
saturated output now gets a zero input, inside every responsive region. Each live
edge must validate; a replay that ignores its recorded input (a planted capture
bug) must still fail the perturbation check.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens.validation import last_validation_failure
from torchlens.validation._domain_perturbations import _perturb_domain_sensitive_parent_values

_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "erf": torch.erf,
}


class _SaturatedGate(nn.Module):
    """A gate whose pre-activation is large everywhere, so the activation is pinned."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = x.abs() * 50.0 + 40.0
        return x * _ACTIVATIONS[self.name](pre)


def _input() -> torch.Tensor:
    return torch.tensor([[0.2, -1.3, 0.7], [2.1, -0.4, 0.05]])


@pytest.mark.parametrize("name", sorted(_ACTIVATIONS))
def test_saturated_activation_validates(name: str) -> None:
    assert tl.validate(_SaturatedGate(name).eval(), _input(), scope="forward", seed=0) is True


def _frozen_input(func: Callable[..., Any], frozen: torch.Tensor) -> Callable[..., Any]:
    """Wrap ``func`` into a capture bug that replays its recorded input only."""

    def planted(*args: Any, **kwargs: Any) -> Any:
        return func(frozen, *args[1:], **kwargs)

    return planted


def _plant(monkeypatch: pytest.MonkeyPatch, func_name: str) -> list[str]:
    """Make every ``func_name`` op ignore the input it is replayed with."""

    planted_labels: list[str] = []
    original = Trace.validate_forward_pass

    def validate_with_plant(trace: Trace, *args: Any, **kwargs: Any) -> Any:
        for op in trace._layer_list:
            if op.func_name == func_name:
                op.func = _frozen_input(op.func, op.call.saved_args[0].clone())
                planted_labels.append(op.label)
        return original(trace, *args, **kwargs)

    monkeypatch.setattr(Trace, "validate_forward_pass", validate_with_plant)
    return planted_labels


@pytest.mark.parametrize("name", sorted(_ACTIVATIONS))
def test_replay_ignoring_its_input_fails(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    planted_labels = _plant(monkeypatch, name)

    with pytest.warns(tl.errors.TorchLensWarning, match="tl.validate FAILED"):
        verdict = tl.validate(_SaturatedGate(name).eval(), _input(), scope="forward", seed=0)

    failure = last_validation_failure()
    assert planted_labels, "no op was planted; the test proves nothing"
    assert verdict is False
    assert failure is not None and failure.check_name == "perturbation"
    assert failure.func_name == name


def _layer(func_name: str, parent: torch.Tensor) -> Any:
    return SimpleNamespace(
        func_name=func_name,
        saved_args=(parent,),
        saved_kwargs={},
        parent_arg_positions={"args": {0: "p"}, "kwargs": {}},
        out=_ACTIVATIONS[func_name](parent),
    )


def test_saturated_sigmoid_gets_a_zero_input() -> None:
    parent = torch.tensor([19.0, 400.0, 33914.0])

    values = _perturb_domain_sensitive_parent_values(_layer("sigmoid", parent), "p", parent)

    assert values is not None and torch.equal(values, torch.zeros(3))


def test_unsaturated_activation_keeps_the_generic_draw() -> None:
    parent = torch.tensor([19.0, 0.5, 33914.0])

    assert _perturb_domain_sensitive_parent_values(_layer("sigmoid", parent), "p", parent) is None
