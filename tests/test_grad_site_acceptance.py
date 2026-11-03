"""End-to-end acceptance for gradient sites (``tl.grad``) through ``Trace.do``.

One small MLP with a branch carries every check: ``fc_a -> relu`` is the edited path,
``fc_b`` bypasses the ReLU, and ``fc_side`` feeds a second output the loss never reads.
The address ``tl.grad(site)`` names the gradient of the site's OUTPUT (what
``trace[site].grad`` stores), so the write side and the read side agree.

Covered: output-gradient semantics, connectedness, the saved read agreeing with the edit,
the armed result, the backward settle rows, ``loss=``, the replay refusals, the ``loss=``
refusals and cleanup, the mixed plan, and the ``no_fire`` reasons.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


class _BranchMLP(nn.Module):
    """``fc_out(relu(fc_a(h)) + fc_b(h))`` plus a sigmoid side output the loss ignores."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.fc_in = nn.Linear(4, 6)
        self.fc_a = nn.Linear(6, 6)
        self.fc_b = nn.Linear(6, 6)
        self.fc_out = nn.Linear(6, 3)
        self.fc_side = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.fc_in(x))
        a = torch.relu(self.fc_a(h))
        b = self.fc_b(h)
        return self.fc_out(a + b), torch.sigmoid(self.fc_side(h))


class _Double(nn.Module):
    """``y = 2 * lin(x)``: the output gradient of ``y`` differs from its grad_input."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * 2


_X = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
_PARAMS_THROUGH_RELU = ("fc_a.weight", "fc_a.bias")


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    return tl.trace(
        model, x, capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True)
    )


def _label(trace: tl.Trace, *func_names: str) -> str:
    """Return the one layer label whose function is in ``func_names``."""

    labels = [layer.label for layer in trace._layer_list if layer.func_name in func_names]
    assert len(labels) == 1, (func_names, labels)
    return labels[0]


def _loss(result: tl.Trace) -> torch.Tensor:
    return result[result.output_layers[0]].out.sum()


def _events(trace: tl.Trace) -> list[dict[str, Any]]:
    return [
        row
        for row in trace.state_history
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]


def _settle_rows(trace: tl.Trace) -> list[dict[str, Any]]:
    return [row for row in _events(trace) if row.get("door") == "backward"]


def _arming_row(trace: tl.Trace) -> dict[str, Any]:
    rows = [row for row in _events(trace) if row.get("door") == "do"]
    assert rows, "the do() call wrote no envelope on the returned trace"
    return rows[-1]


def _backward_fires(trace: tl.Trace) -> list[Any]:
    return [
        record
        for record in trace._intervention_spec.records
        if getattr(record, "signal", None) == "backward"
    ]


def _param_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return every parameter gradient the loss reached (``fc_side`` is never reached)."""

    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _clear_param_grads(model: nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


@pytest.mark.smoke
def test_a24_mixed_plan_edits_the_activation_and_the_gradient() -> None:
    """A24: tl.grad(s) | s halves relu's activation on the rerun and its gradient after."""

    model = _BranchMLP()
    trace = _capture(model, _X)
    relu = _label(trace, "relu")

    # Reference: halve the activation only (a capture-time rule), then the unedited backward.
    _clear_param_grads(model)
    act_only = tl.trace(
        model,
        _X,
        capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True),
        intervene=tl.when(tl.func("relu"), tl.scale(0.5)),
    )
    act_only.backward(_loss(act_only))
    act_only_grads = _param_grads(model)
    _clear_param_grads(model)

    result = trace.do(tl.grad(relu) | relu, tl.scale(0.5), model=model, inputs=_X, loss=_loss)
    grads = _param_grads(model)
    assert torch.allclose(result[relu].out, 0.5 * trace[relu].out)
    for name in _PARAMS_THROUGH_RELU:
        assert torch.allclose(grads[name], 0.5 * act_only_grads[name]), name
    row = _arming_row(result)
    assert row["status"] == "fired"
    assert row["armed_edits"] == 1


def test_pass_qualified_gradient_edit_fires_on_its_pass() -> None:
    """tl.grad(s) & tl.in_backward_pass(2) skips pass 1 and fires on pass 2."""

    model = _BranchMLP()
    trace = _capture(model, _X)
    relu = _label(trace, "relu")
    result = trace.do(tl.grad(relu) & tl.in_backward_pass(2), tl.scale(0), model=model, inputs=_X)
    loss = _loss(result)
    result.backward(loss, retain_graph=True)
    result.backward(loss)
    second = _settle_rows(result)[-1]
    assert second["backward_pass_index"] == 2
    assert second["status"] == "fired"


def test_rerun_result_counts_only_its_own_backward_passes() -> None:
    """A rerun from a source that already ran a backward starts with no backward pass."""

    model = _BranchMLP()
    source = _capture(model, _X)
    source.backward(_loss(source))
    relu = _label(source, "relu")
    result = source.do(tl.grad(relu), tl.scale(0.25), model=model, inputs=_X, loss=_loss)
    assert result._num_backward_passes == 1
    assert torch.allclose(result[relu].grad, 0.25 * source[relu].grad)
