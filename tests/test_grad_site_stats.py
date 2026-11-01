"""Gradient sites in the sweep and stats doors: ``tl.sweep(loss=)`` and selector keys.

``tl.stats.aggregate`` has no ``target=``: a metric key names its tensor, a
plain key the activation and ``tl.grad(site)`` the gradient. ``tl.sweep`` takes
``loss=`` (a callable of each member) to run every member's backward. The
first block runs on any branch; the second needs ``tl.grad`` and the
output-gradient mount.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


_X = torch.randn(3, 4, generator=torch.Generator().manual_seed(5))
_Y = torch.tensor([0, 1, 0])


def _relu_label(model: nn.Module) -> str:
    trace = tl.trace(model, _X)
    return next(str(op.layer_label) for op in trace._layer_list if op.func_name == "relu")


@dataclass(frozen=True)
class _GradKey:
    """A hashable stand-in for a ``tl.grad`` metric key (kind ``"grad"``)."""

    label: str
    selector_kind: str = "grad"


def _grad_key(label: str) -> _GradKey:
    return _GradKey(label)


def test_aggregate_has_no_target_parameter() -> None:
    assert "target" not in inspect.signature(tl.stats.aggregate).parameters


def test_aggregate_gradient_key_needs_loss_fn() -> None:
    model = _Net()
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.stats.aggregate(
            model,
            [_X],
            {_relu_label(model): tl.stats.Mean(), _grad_key("relu_1_2"): tl.stats.Mean()},
        )
    assert excinfo.value.fields["code"] == "aggregate_loss_fn_required"


def test_aggregate_string_keys_keep_their_names() -> None:
    model = _Net()
    relu = _relu_label(model)
    results = tl.stats.aggregate(model, [_X, _X], {relu: tl.stats.Mean()})
    assert list(results) == [relu]


def test_sweep_refuses_a_tensor_loss() -> None:
    model = _Net()
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.sweep(
            model,
            _X,
            site=_relu_label(model),
            values=[0.0],
            loss=torch.tensor(1.0),  # type: ignore[arg-type]
        )
    assert excinfo.value.fields["code"] == "do_loss_not_callable"


def test_sweep_loss_needs_a_gradient_site() -> None:
    model = _Net()
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.sweep(model, _X, site=_relu_label(model), values=[0.0], loss=lambda t: t)
    assert excinfo.value.fields["code"] == "do_loss_without_gradient_site"


def _loss(member: Any) -> torch.Tensor:
    return member[member.output_layers[0]].out.sum()


def test_sweep_gradient_site_with_loss_runs_every_backward() -> None:  # A32
    model = _Net()
    bundle = tl.sweep(
        model,
        _X,
        site=tl.grad(tl.func("relu")),
        values=[tl.scale(0.0), tl.scale(0.5)],
        names=["zero", "half"],
        include_baseline=True,
        loss=_loss,
        capture=tl.options.CaptureOptions(save_grads="all"),
    )
    members = {name: bundle[name] for name in ("baseline", "zero", "half")}
    for member in members.values():
        assert member.num_backward_passes == 1
    fc1 = next(
        str(op.layer_label) for op in members["baseline"]._layer_list if op.func_name == "linear"
    )
    base = members["baseline"][fc1].grad
    assert torch.count_nonzero(members["zero"][fc1].grad) == 0
    assert torch.allclose(members["half"][fc1].grad, 0.5 * base)


def test_sweep_gradient_site_without_loss_stays_armed() -> None:  # A32
    model = _Net()
    bundle = tl.sweep(
        model,
        _X,
        site=tl.grad(tl.func("relu")),
        values=[tl.scale(0.0)],
        names=["zero"],
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    envelopes = [
        row
        for row in bundle["zero"].state_history
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]
    assert envelopes[-1]["status"] == "armed"


def test_aggregate_measures_activation_and_gradient_of_one_site() -> None:  # A32
    model = _Net()
    relu = _relu_label(model)

    def loss_fn(output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(output, y)

    results = tl.stats.aggregate(
        model,
        [(_X, _Y)],
        {relu: tl.stats.Mean(), tl.grad(relu): tl.stats.Mean()},
        loss_fn=loss_fn,
    )
    assert set(results) == {relu, repr(tl.grad(relu))}
    trace = tl.trace(
        model, _X, capture=tl.options.CaptureOptions(backward_ready=True, save_grads="all")
    )
    trace.backward(loss_fn(trace[trace.output_layers[0]].out, _Y))
    expected_out = trace[relu].out.mean()
    expected_grad = trace[relu].grad.mean()
    assert torch.allclose(torch.as_tensor(results[relu]), expected_out)
    assert torch.allclose(torch.as_tensor(results[repr(tl.grad(relu))]), expected_grad)
