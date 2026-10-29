"""A degenerate forward fails validation under its own ``degenerate_forward`` check.

When a perturbation check finds an op's output unchanged, the failure is recorded as
``degenerate_forward`` only when the saved value proves no perturbation could move it:
every saved output element is non-finite, or an in-forward ``autograd.grad`` gradient
is identically zero. The run still fails in every case. A partly non-finite output and
a finite forward with a mis-recorded parent keep the plain ``perturbation`` check.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

import torchlens as tl
import torchlens.validation.core as validation_core
from torchlens.validation import last_validation_failure
from torchlens.validation._degenerate_forward import (
    degenerate_forward_kind,
    perturbation_insensitive_failure,
)
from torchlens.validation.diagnostics import CHECK_DEGENERATE_FORWARD, CHECK_PERTURBATION


def _validate(model: nn.Module, *inputs: torch.Tensor) -> bool:
    """Validate forward, silencing the capture's own warnings."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validate(model, inputs if len(inputs) > 1 else inputs[0], scope="forward"))


@contextmanager
def _tampered_validation(tamper: Callable[[Any], None]) -> Iterator[None]:
    """Run ``tamper`` on the validator's own capture before its checks run."""

    original = validation_core.validate_saved_outs

    def tampering(trace: Any, *args: Any, **kwargs: Any) -> Any:
        tamper(trace)
        return original(trace, *args, **kwargs)

    validation_core.validate_saved_outs = tampering
    try:
        yield
    finally:
        validation_core.validate_saved_outs = original


class _LinearBatchNorm(nn.Module):
    """Eval-mode Linear then BatchNorm: an all-NaN input leaves every output NaN."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.lin(x))


class _ZeroHeadEnergy(nn.Module):
    """Energy from a zero-weight head, differentiated against both inputs.

    The energy is the head's bias alone, so the force and the dipole are identically
    zero whatever the inputs (the m16570 shape).
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(3, 3)
        self.head = nn.Linear(3, 1)
        nn.init.zeros_(self.head.weight)

    def forward(self, cart: torch.Tensor, ef: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cart.requires_grad = True
        ef.requires_grad = True
        energy = self.head(torch.tanh(self.body(cart)) * ef).sum()
        force, dipole = torch.autograd.grad(energy, (cart, ef), create_graph=True)
        return energy, -force, dipole


class _TanhMul(nn.Module):
    """A finite forward with one two-parent ``mul`` op to mis-record."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x)) * x


def _ignore_parents_of_mul(trace: Any) -> None:
    """Make the ``mul`` op's func return its saved output, ignoring both parents."""

    for op in trace.ops:
        if op.func_name in {"__mul__", "mul"}:
            saved = op.out.detach().clone()

            def fake_mul(*_args: Any, _saved: torch.Tensor = saved, **_kwargs: Any) -> Any:
                return _saved.clone()

            fake_mul.__name__ = getattr(op.func, "__name__", "__mul__")
            op.func = fake_mul


def test_all_nan_forward_fails_as_degenerate_forward() -> None:
    """An all-NaN forward fails, recorded as degenerate at the all-NaN op."""

    torch.manual_seed(0)
    model = _LinearBatchNorm()
    assert _validate(model, torch.full((2, 4), float("nan"))) is False
    failure = last_validation_failure()
    assert failure is not None, "a degenerate forward must still record its failure"
    assert failure.check == CHECK_DEGENERATE_FORWARD, failure.summary()
    assert failure.extra["degenerate_kind"] == "non_finite_output"
    assert failure.op_label is not None and failure.op_label.startswith("batchnorm")


def test_zero_gradient_autograd_grad_forward_fails_as_degenerate_forward() -> None:
    """Identically zero in-forward gradients fail, recorded as degenerate."""

    torch.manual_seed(0)
    assert _validate(_ZeroHeadEnergy(), torch.randn(2, 3), torch.randn(2, 3)) is False
    failure = last_validation_failure()
    assert failure is not None
    assert failure.check == CHECK_DEGENERATE_FORWARD, failure.summary()
    assert failure.extra["degenerate_kind"] == "zero_gradient"
    assert failure.op_label is not None and failure.op_label.startswith("autogradgrad")


def test_mis_recorded_parent_in_finite_forward_stays_a_perturbation_failure() -> None:
    """ARMED NEGATIVE: a finite op ignoring its recorded parents is not degenerate."""

    torch.manual_seed(0)
    with _tampered_validation(_ignore_parents_of_mul):
        assert _validate(_TanhMul(), torch.randn(2, 4)) is False
    failure = last_validation_failure()
    assert failure is not None
    assert failure.check == CHECK_PERTURBATION, failure.summary()
    assert "degenerate_kind" not in failure.extra
    assert failure.op_label is not None and failure.op_label.startswith("mul")


def test_mis_recorded_parent_with_partly_nan_output_stays_a_perturbation_failure() -> None:
    """ARMED NEGATIVE: a partly NaN output keeps the plain perturbation check."""

    torch.manual_seed(0)
    x = torch.randn(2, 4)
    x[0, 0] = float("nan")
    with _tampered_validation(_ignore_parents_of_mul):
        assert _validate(_TanhMul(), x) is False
    failure = last_validation_failure()
    assert failure is not None
    assert failure.check == CHECK_PERTURBATION, failure.summary()


def _op(out: Any, func: Any = None) -> Any:
    return SimpleNamespace(out=out, func=func, func_name="op")


def test_degenerate_kind_needs_every_element_non_finite() -> None:
    """Only a wholly non-finite payload is degenerate; empty and mixed ones are not."""

    nan = float("nan")
    inf = float("inf")
    assert degenerate_forward_kind(_op(torch.tensor([nan, inf, -inf]))) == "non_finite_output"
    assert degenerate_forward_kind(_op((torch.tensor([nan]), torch.tensor([inf])))) == (
        "non_finite_output"
    )
    assert degenerate_forward_kind(_op(torch.tensor([nan, 1.0]))) is None
    assert degenerate_forward_kind(_op((torch.tensor([nan]), torch.tensor([0.5])))) is None
    assert degenerate_forward_kind(_op(torch.empty(0))) is None
    assert degenerate_forward_kind(_op(None)) is None


def test_zero_output_is_degenerate_only_for_autograd_grad() -> None:
    """An all-zero output of an ordinary op stays a plain perturbation failure."""

    zero = _op(torch.zeros(3))
    assert degenerate_forward_kind(zero) is None
    failure = perturbation_insensitive_failure(zero, "relu_1_1", ["input_1"])
    assert failure.check == CHECK_PERTURBATION
    assert failure.extra == {"perturbed_parents": ["input_1"]}
