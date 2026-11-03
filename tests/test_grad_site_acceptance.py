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

import re
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError


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
_PARAMS_DOWNSTREAM = ("fc_out.weight", "fc_out.bias")


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


def _unedited(model: nn.Module) -> tuple[tl.Trace, dict[str, torch.Tensor]]:
    """Capture, run the unedited backward, and return the trace and parameter grads."""

    _clear_param_grads(model)
    trace = _capture(model, _X)
    trace.backward(_loss(trace))
    grads = _param_grads(model)
    _clear_param_grads(model)
    return trace, grads


@pytest.mark.smoke
def test_a9_edit_applies_to_the_output_gradient() -> None:
    """A9: clamp on tl.grad(y) clamps dL/dy (1.0 -> 0.5), so dL/dh = 2 * 0.5 = 1.0.

    The grad_input reading (clamp the 2.0 flowing into ``h``) would give 0.5.
    """

    model = _Double()
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(2))
    trace = _capture(model, x)
    y_site = _label(trace, "mul", "__mul__")
    h_site = _label(trace, "linear")
    result = trace.do(
        tl.grad(y_site),
        tl.clamp(low=-0.5, high=0.5),
        model=model,
        inputs=x,
        loss=lambda t: _loss(t),
    )
    assert torch.allclose(result[h_site].grad, torch.ones_like(result[h_site].grad))
    fires = _backward_fires(result)
    assert fires and all(record.grad_kind == "grad_output" for record in fires)
    # Records name the forward layer (no pass suffix on this single-pass op).
    assert {record.site_label for record in fires} == {y_site.split(":")[0]}


@pytest.mark.smoke
def test_a10_zero_gradient_disconnects_only_the_relu_path() -> None:
    """A10: scale(0) at relu zeroes fc_a's grads, keeps fc_out's, leaves forward values."""

    model = _BranchMLP()
    baseline, base_grads = _unedited(model)
    relu = _label(baseline, "relu")
    result = baseline.do(tl.grad(relu), tl.scale(0), model=model, inputs=_X, loss=_loss)
    grads = _param_grads(model)
    for name in _PARAMS_THROUGH_RELU:
        assert torch.count_nonzero(grads[name]) == 0, name
    for name in _PARAMS_DOWNSTREAM:
        assert torch.allclose(grads[name], base_grads[name]), name
    assert "fc_side.weight" not in grads
    # fc_in still learns through the fc_b path, so its gradient changes but survives.
    assert torch.count_nonzero(grads["fc_in.weight"]) > 0
    for label in (_label(baseline, "tanh"), relu, baseline.output_layers[0]):
        assert torch.equal(result[label].out, baseline[label].out), label


@pytest.mark.smoke
def test_a11_saved_grad_is_the_edited_gradient() -> None:
    """A11: after scale(0.25) fires, result[relu].grad is 0.25 times the unedited one."""

    model = _BranchMLP()
    baseline, _grads = _unedited(model)
    relu = _label(baseline, "relu")
    source = _capture(model, _X)
    result = source.do(tl.grad(relu), tl.scale(0.25), model=model, inputs=_X, loss=_loss)
    assert result._num_backward_passes == 1
    assert torch.allclose(result[relu].grad, 0.25 * baseline[relu].grad)


@pytest.mark.smoke
def test_a19_do_returns_an_armed_trace_and_leaves_the_source_alone() -> None:
    """A19: without a backward the result is ARMED, not a completed intervention."""

    model = _BranchMLP()
    trace = _capture(model, _X)
    relu = _label(trace, "relu")
    recipe_before = trace.recipe
    result = trace.do(tl.grad(relu), tl.scale(0), model=model, inputs=_X)
    assert result is not trace
    row = _arming_row(result)
    assert row["status"] == "armed"
    assert row["fire_count"] == 0
    assert row["awaiting"] == "backward"
    assert row["armed_edits"] == 1
    assert len(result.armed_edits) == 1
    assert result.armed_edits[0].signal == "backward"
    assert not _backward_fires(result)
    assert not result[relu].has_grad
    assert trace.recipe == recipe_before
    assert re.search(r"\b1 gradient edits? armed; call trace\.backward\(loss\)", repr(result))


@pytest.mark.smoke
def test_a20_each_backward_writes_its_own_settle_row() -> None:
    """A20: one settle row per pass, parented on the arming row; pass 1 is never rewritten."""

    model = _BranchMLP()
    trace = _capture(model, _X)
    relu = _label(trace, "relu")
    result = trace.do(tl.grad(relu), tl.scale(0.5), model=model, inputs=_X)
    arming = _arming_row(result)
    loss = _loss(result)
    result.backward(loss, retain_graph=True)
    first = _settle_rows(result)
    assert len(first) == 1
    settle = first[0]
    assert settle["door"] == "backward"
    assert settle["lane"] == "live_hook"
    assert settle["parent_event_id"] == arming["event_id"]
    assert settle["arming_event_id"] == arming["event_id"]
    assert settle["status"] == "fired"
    assert settle["fire_count"] >= 1
    assert settle["backward_pass_index"] == 1
    snapshot = dict(settle)
    result.backward(loss)
    rows = _settle_rows(result)
    assert len(rows) == 2
    assert rows[0] == snapshot
    assert rows[1]["backward_pass_index"] == 2
    assert rows[1]["status"] == "fired"


@pytest.mark.smoke
def test_a21_loss_callable_runs_the_backward_inside_do() -> None:
    """A21: loss= performs the backward in the same call and settles it."""

    model = _BranchMLP()
    trace = _capture(model, _X)
    relu = _label(trace, "relu")
    result = trace.do(tl.grad(relu), tl.scale(0), model=model, inputs=_X, loss=_loss)
    assert result._num_backward_passes == 1
    rows = _settle_rows(result)
    assert len(rows) == 1 and rows[0]["status"] == "fired"
    assert torch.count_nonzero(result[relu].grad) == 0


@pytest.mark.parametrize("engine", [None, "replay"])
def test_a22_gradient_edit_without_a_rerun_refuses_before_mutation(engine: str | None) -> None:
    """A22: no model/x (or engine='replay') refuses typed and changes nothing."""

    model = _BranchMLP()
    trace = _capture(model, _X)
    relu = _label(trace, "relu")
    recipe_before = trace.recipe
    audit_before = list(trace.intervention_audit)
    history_before = len(trace.state_history)
    options = None if engine is None else tl.options.InterventionOptions(engine=engine)
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.do(tl.grad(relu), tl.scale(0), intervention=options)
    assert excinfo.value.fields["code"] == "replay_gradient_edit_unsupported"
    message = str(excinfo.value)
    assert "model= and x=" in message
    assert "engine='set_only'" in message
    assert trace.recipe == recipe_before
    assert trace.intervention_audit == audit_before
    assert len(trace.state_history) == history_before


def test_push_replay_hooks_refuse_a_gradient_edit() -> None:
    """SPEC 3.6: a gradient edit passed as push replay hooks refuses; nothing changes."""

    model = _BranchMLP()
    trace = tl.trace(
        model,
        _X,
        capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True, replayable=True),
    )
    relu = _label(trace, "relu")
    out_before = trace[relu].out.clone()
    audit_before = list(trace.intervention_audit)
    replay = tl.options.ReplayOptions(hooks={tl.grad(relu): tl.scale(0)})
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.push(replay=replay)
    assert excinfo.value.fields["code"] == "replay_gradient_edit_unsupported"
    assert "Replay re-executes forward ops only" in str(excinfo.value)
    assert torch.equal(trace[relu].out, out_before)
    assert trace.intervention_audit == audit_before


def test_a23_loss_refusals() -> None:
    """A23: a tensor loss and a loss with no gradient site refuse typed."""

    model = _BranchMLP()
    trace = _capture(model, _X)
    relu = _label(trace, "relu")
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.do(tl.grad(relu), tl.scale(0), model=model, inputs=_X, loss=_loss(trace))
    assert excinfo.value.fields["code"] == "do_loss_not_callable"
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.do(relu, tl.scale(0), model=model, inputs=_X, loss=_loss)
    assert excinfo.value.fields["code"] == "do_loss_without_gradient_site"


def test_a23_raising_loss_writes_error_and_cleans_up() -> None:
    """A23: a loss callable that raises leaves an error envelope and re-raises it."""

    model = _BranchMLP()
    trace = _capture(model, _X)
    relu = _label(trace, "relu")
    recipe_before = trace.recipe
    boom = RuntimeError("loss exploded")
    seen: list[tl.Trace] = []

    def bad_loss(result: tl.Trace) -> torch.Tensor:
        seen.append(result)
        raise boom

    with pytest.raises(RuntimeError) as excinfo:
        trace.do(tl.grad(relu), tl.scale(0), model=model, inputs=_X, loss=bad_loss)
    assert excinfo.value is boom
    assert _events(trace)[-1]["status"] == "error"
    assert trace.recipe == recipe_before
    (result,) = seen
    assert result.armed_edits == ()
    assert _events(result)[-1]["status"] == "error"


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


@pytest.mark.smoke
def test_a27_unreached_gradient_settles_no_fire_without_a_zero_tensor() -> None:
    """A27: a matched site the loss never reaches settles no_fire; no zero grad appears."""

    model = _BranchMLP()
    trace = _capture(model, _X)
    side = _label(trace, "sigmoid")
    result = trace.do(tl.grad(tl.func("sigmoid")), tl.scale(0), model=model, inputs=_X, loss=_loss)
    rows = _settle_rows(result)
    assert len(rows) == 1
    assert rows[0]["status"] == "no_fire"
    assert rows[0]["no_fire_reason"] == "gradient_not_reached"
    assert result[side].has_grad is False


@pytest.mark.smoke
def test_a27_pass_qualifier_settles_pass_filtered() -> None:
    """A27: tl.grad(s) & tl.in_backward_pass(2) on pass 1 settles no_fire, pass_filtered."""

    model = _BranchMLP()
    trace = _capture(model, _X)
    relu = _label(trace, "relu")
    result = trace.do(tl.grad(relu) & tl.in_backward_pass(2), tl.scale(0), model=model, inputs=_X)
    loss = _loss(result)
    result.backward(loss, retain_graph=True)
    first = _settle_rows(result)
    assert len(first) == 1
    assert first[0]["status"] == "no_fire"
    assert first[0]["no_fire_reason"] == "pass_filtered"


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
