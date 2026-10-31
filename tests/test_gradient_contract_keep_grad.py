"""``keep_grad(edit)`` and ``InterventionDecision(keep_grad=True)``: the connectivity check.

Both spellings run one check: an edit that writes values carrying no autograd
graph raises ``edit_detached_from_graph`` with the user's site label and a
spelling that keeps the graph, and that spelling runs. ``keep_grad=False``
(the decision default) checks nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import add, replace_with, scale, zero_ablate, zero_grad
from torchlens._errors import InvalidArgumentError
from torchlens.intervention import (
    compose,
    keep_grad,
)
from torchlens.intervention.types import InterventionDecision

_SITE = "linear_1_1"
_X = torch.randn(2, 4, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
_VALUE = torch.full((2, 4), 0.25, dtype=torch.float64)


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(4, 4).double()
        self.lin2 = nn.Linear(4, 3).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.lin1(x))


def _capture(edit: Any, model: nn.Module | None = None) -> tuple[torch.Tensor, nn.Module]:
    model = _Net() if model is None else model
    trace = tl.trace(model, _X, intervene=tl.when(tl.module("lin1"), edit))
    return trace[trace.output_layers[0]].out, model


def _detached_error(call: Any) -> InvalidArgumentError:
    with pytest.raises(InvalidArgumentError) as caught:
        call()
    assert caught.value.fields["code"] == "edit_detached_from_graph"
    return caught.value


def test_keep_grad_on_a_transparent_edit_changes_nothing() -> None:
    plain, plain_model = _capture(scale(2.0))
    kept, kept_model = _capture(keep_grad(scale(2.0)))
    assert torch.equal(plain, kept)
    plain.sum().backward()
    kept.sum().backward()
    assert torch.equal(plain_model.lin1.weight.grad, kept_model.lin1.weight.grad)


def test_keep_grad_zero_ablate_raises_and_scale_zero_keeps_the_graph() -> None:
    error = _detached_error(lambda: _capture(keep_grad(zero_ablate())))
    assert error.fields["site"] == "lin1:1"  # the capture-time label of the selected module
    assert error.fields["reason"] == "edit_detached"
    assert "keep_grad(tl.scale(0.0))" in error.fields["remedy"]
    out, model = _capture(keep_grad(scale(0.0)))
    plain, _ = _capture(zero_ablate())
    assert torch.equal(out, plain)
    out.sum().backward()
    assert torch.equal(model.lin1.weight.grad, torch.zeros_like(model.lin1.weight))


def test_keep_grad_replace_with_raises_and_the_fixed_value_spelling_runs() -> None:
    error = _detached_error(lambda: _capture(keep_grad(replace_with(_VALUE))))
    assert "tl.intervention.compose(tl.scale(0.0), tl.add(value))" in error.fields["remedy"]
    out, model = _capture(keep_grad(compose(scale(0.0), add(_VALUE))))
    plain, _ = _capture(replace_with(_VALUE))
    assert torch.equal(out, plain)
    out.sum().backward()
    assert model.lin1.weight.grad is not None


def test_decision_keep_grad_true_runs_the_same_check() -> None:
    decision = InterventionDecision(action="add_hook", hook=zero_ablate(), keep_grad=True)
    error = _detached_error(lambda: _capture(decision))
    assert error.fields["reason"] == "edit_detached"
    kept = InterventionDecision(action="add_hook", hook=scale(0.0), keep_grad=True)
    out, model = _capture(kept)
    out.sum().backward()
    assert model.lin1.weight.grad is not None


def test_decision_keep_grad_false_default_is_a_no_op() -> None:
    decision = InterventionDecision(action="add_hook", hook=zero_ablate())
    assert decision.keep_grad is False
    out, model = _capture(decision)
    plain, _ = _capture(zero_ablate())
    assert torch.equal(out, plain)
    out.sum().backward()
    assert model.lin1.weight.grad is None
    assert model.lin2.weight.grad is not None


def test_keep_grad_refuses_when_gradients_are_disabled_and_enabled_runs() -> None:
    with torch.no_grad():
        error = _detached_error(lambda: _capture(keep_grad(scale(2.0))))
    assert error.fields["reason"] == "grad_disabled"
    out, _ = _capture(keep_grad(scale(2.0)))
    assert out.requires_grad


def test_keep_grad_refuses_a_site_with_no_graph_and_trainable_params_run() -> None:
    """A capture-free bound call on frozen parameters has no graph at the site."""

    frozen = _Net().requires_grad_(False)
    bound = tl.when(tl.module("lin1"), keep_grad(scale(2.0))).bind(frozen)
    error = _detached_error(lambda: bound(_X))
    assert error.fields["reason"] == "site_has_no_graph"
    trainable = tl.when(tl.module("lin1"), keep_grad(scale(2.0))).bind(frozen.requires_grad_(True))
    assert trainable(_X).requires_grad


def test_keep_grad_wraps_a_hook_callable() -> None:
    def detaching(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        del hook
        return out.detach() * 2.0

    _detached_error(lambda: _capture(keep_grad(detaching)))
    out, _ = _capture(keep_grad(lambda out, *, hook: out * 2.0))
    assert out.requires_grad


def test_keep_grad_is_idempotent_and_marks_the_edit() -> None:
    once = keep_grad(scale(2.0))
    assert keep_grad(once) is once
    assert dict(once.metadata)["keep_grad"] is True
    assert once.portability == "opaque_audit"


@pytest.mark.parametrize("bad", [zero_grad(), object()], ids=["backward_edit", "not_an_edit"])
def test_keep_grad_refuses_what_is_not_a_forward_edit(bad: Any) -> None:
    with pytest.raises(InvalidArgumentError) as caught:
        keep_grad(bad)
    assert caught.value.fields["code"] == "keep_grad_edit_invalid"
    keep_grad(scale(2.0))


def test_keep_grad_inside_a_subspace_checks_the_coordinates() -> None:
    """The component an interchange writes is the donor's (detached): the check fires."""

    model = _Net()
    options = tl.options.CaptureOptions(replayable=True)
    target = tl.trace(model, _X, capture=options)
    source = tl.trace(model, _X * 2.0, capture=options)
    basis = 0.5 * torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    selection = tl.subspace(_SITE, basis, origin="rows")
    with pytest.raises(InvalidArgumentError) as caught:
        target.fork().do(selection, keep_grad(tl.patch_from(source)))
    assert caught.value.fields["code"] == "edit_detached_from_graph"
    fork = target.fork()
    fork.do(selection, keep_grad(scale(0.5)))
    assert fork[fork.output_layers[0]].out.requires_grad
