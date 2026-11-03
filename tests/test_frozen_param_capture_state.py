"""Capture leaves frozen parameters frozen, and their provenance does not depend on autograd.

Capture used to force ``requires_grad=True`` on every floating parameter outside
``backward_ready`` so intermediate tensors carried ``grad_fn`` metadata. That changed
the program torch ran (attention picks a different kernel for gradient-tracking
weights), so the captured forward differed from eager. Capture now never flips the
flag: parameter reads are resolved by identity at the op boundary, so parameter
edges, ``Param`` fields, saved traces, backward capture and validation hold for
frozen weights exactly as for trainable ones. These tests use a partly frozen model.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import torchlens as tl

FROZEN = ("frozen.weight", "frozen.bias", "scale")
TRAINABLE = ("head.weight", "head.bias")


class _PartlyFrozen(nn.Module):
    """A frozen linear and a frozen scale feeding a trainable head."""

    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(4, 5)
        self.scale = nn.Parameter(torch.full((5,), 0.5), requires_grad=False)
        self.head = nn.Linear(5, 3)
        self.frozen.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.frozen(x)) * self.scale
        return self.head(hidden)


def _model_and_input(seed: int = 0) -> tuple[_PartlyFrozen, torch.Tensor]:
    torch.manual_seed(seed)
    return _PartlyFrozen(), torch.randn(2, 4)


def _flags_inside_forward(model: nn.Module, run: Any) -> dict[str, bool]:
    """Run ``run()`` and return each parameter's ``requires_grad`` as the forward saw it."""

    seen: dict[str, bool] = {}

    def record(module: nn.Module, args: Any) -> None:
        seen.update({name: bool(p.requires_grad) for name, p in module.named_parameters()})

    handle = model.register_forward_pre_hook(record)
    try:
        run()
    finally:
        handle.remove()
    return seen


def _param_state(model: nn.Module) -> dict[str, tuple[Any, ...]]:
    return {
        name: (
            bool(param.requires_grad),
            param.is_leaf,
            param._version,
            None if param.grad is None else param.grad.clone(),
        )
        for name, param in model.named_parameters()
    }


def _assert_same_state(before: dict[str, tuple[Any, ...]], after: dict[str, tuple[Any, ...]]):
    assert before.keys() == after.keys()
    for name, (flag, leaf, version, grad) in before.items():
        a_flag, a_leaf, a_version, a_grad = after[name]
        assert (a_flag, a_leaf, a_version) == (flag, leaf, version), name
        if grad is None:
            assert a_grad is None, name
        else:
            assert a_grad is not None and torch.equal(a_grad, grad), name


def _ops_reading(trace: tl.Trace, address: str) -> list[Any]:
    return [op for op in trace.ops if address in [p.address for p in op.params]]


def test_gradient_analysis_scope_builds_a_path_through_frozen_weights() -> None:
    """Only an analysis that backpropagates through its own capture flips frozen flags."""

    from torchlens.backends.torch.frozen_parameter_gradients import (
        frozen_parameter_gradients,
    )
    from torchlens.mechinterp._eap import _capture

    model, x = _model_and_input()
    model.requires_grad_(False)
    before = _param_state(model)

    plain = tl.trace(model, x)
    assert _ops_reading(plain, "head.weight")[0].autograd.grad_fn_class_name is None

    def scoped_capture() -> None:
        with frozen_parameter_gradients():
            captured.append(tl.trace(model, x))

    captured: list[tl.Trace] = []
    assert all(_flags_inside_forward(model, scoped_capture).values())
    scoped = captured[0]
    assert _ops_reading(scoped, "frozen.weight")[0].autograd.grad_fn_class_name is not None
    assert all(scoped.params[name].is_trainable is False for name in (*FROZEN, *TRAINABLE))
    _assert_same_state(before, _param_state(model))

    analysis = _capture(model, x)
    assert _ops_reading(analysis, "head.weight")[0].autograd.grad_fn_class_name is not None
    _assert_same_state(before, _param_state(model))

    plain_flags = _flags_inside_forward(model, lambda: tl.trace(model, x))
    assert not any(plain_flags.values())
