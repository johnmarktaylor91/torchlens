"""Gradient sites at the intervention doors: arming, settling, refusals, persistence.

A gradient is addressed as a site (``tl.grad(site)``); no door takes a pass
keyword. ``trace.do(tl.grad(s), edit, model=m, inputs=x)`` reruns with the edit armed
(audit status ``"armed"``), the next backward settles it (``door="backward"``
rows), and ``loss=`` runs that backward inside the call.

The first block runs on any branch: it drives the door plumbing with autograd
node sites and frozen ``grad`` targets. The second block needs ``tl.grad`` and
the output-gradient mount and is skipped until both are present.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._trace_grad_sites import settle_armed_gradient_edits
from torchlens._errors import InvalidArgumentError
from torchlens.intervention.types import TargetSpec

_X = torch.randn(3, 4, generator=torch.Generator().manual_seed(7))


class _BranchNet(nn.Module):
    """``fc2(relu(fc1(x)))`` plus a side branch the loss never uses."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 2)
        self.side = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fc2(torch.relu(self.fc1(x))), torch.sigmoid(self.side(x))


def _trace(model: nn.Module, **capture: Any) -> Any:
    options = {"replayable": True, **capture}
    return tl.trace(model, _X, capture=tl.options.CaptureOptions(**options))


def _relu_label(trace: Any) -> str:
    return next(str(op.layer_label) for op in trace._layer_list if op.func_name == "relu")


def _events(trace: Any, **match: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in trace.state_history
        if isinstance(row, dict)
        and row.get("op") == "intervention_event"
        and all(row.get(key) == value for key, value in match.items())
    ]


# ---------------------------------------------------------------------------
# Doors without a pass keyword (A28)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Site signal derivation (Rule.signal, narrate signal, plan split)
# ---------------------------------------------------------------------------


def _grad_target(label: str) -> TargetSpec:
    """A frozen ``tl.grad`` target as the saved recipe spells it."""

    return TargetSpec(selector_kind="grad", selector_value=TargetSpec("label", label))


# ---------------------------------------------------------------------------
# Refusals before any mutation (A22, A23)
# ---------------------------------------------------------------------------


def _snapshot(trace: Any) -> tuple[int, int, int]:
    spec = trace._intervention_spec
    hooks = len(spec.hook_specs) if spec is not None else 0
    return hooks, len(trace.intervention_audit), len(trace.state_history)


def test_loss_needs_a_gradient_site() -> None:
    model = _BranchNet()
    trace = _trace(model)
    before = _snapshot(trace)
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.do(
            _relu_label(trace),
            tl.scale(0.5),
            model=model,
            inputs=_X,
            loss=lambda result: result._layer_list[-1].out.sum(),
        )
    assert excinfo.value.fields["code"] == "do_loss_without_gradient_site"
    assert _snapshot(trace) == before


@pytest.mark.parametrize("direction", ["backward", "both", "sideways"])
def test_set_refuses_a_gradient_or_unknown_direction(direction: str) -> None:
    """``Trace.set`` keeps only its activation path; gradients are sites (tl.grad)."""

    trace = _trace(_BranchNet()).fork()
    label = _relu_label(trace)
    recipe_before = trace.recipe
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.set(label, torch.zeros_like(trace[label].out), direction=direction)
    assert excinfo.value.fields["code"] == "intervention_direction_invalid"
    assert trace.recipe == recipe_before


# ---------------------------------------------------------------------------
# Settle rows (A20, A27) driven directly
# ---------------------------------------------------------------------------


class _Fire:
    """A backward FireRecord stand-in carrying the fields settle reads."""

    def __init__(self, helper: Any, *, pass_index: int, label: str) -> None:
        self.signal = "backward"
        self.backward_pass_number = pass_index
        self.edit = helper
        self.edit_name = getattr(helper, "name", None)
        self.site_label = label
        self.target_label = label


def _armed_trace(target: Any, edit: Any, *, event_id: str | None = "lineage:1") -> Any:
    trace = _trace(_BranchNet()).fork()
    trace._ensure_intervention_spec().add_hook(
        target,
        edit,
        helper=edit,
        handle="hook-grad",
        metadata={"arming_group": "arm-1", "arming_event_id": event_id},
    )
    trace._mark_intervention_spec_mutated(outs_stale=False)
    return trace


def test_settle_attributes_fires_to_their_own_arming_transaction() -> None:
    first_edit = tl.scale(0.0)
    second_edit = tl.scale(0.0)
    trace = _armed_trace(_grad_target("relu_1_2"), first_edit, event_id="lineage:1")
    trace._intervention_spec.add_hook(
        _grad_target("relu_1_2"),
        second_edit,
        helper=second_edit,
        handle="hook-grad-2",
        metadata={"arming_group": "arm-2", "arming_event_id": "lineage:2"},
    )
    trace._intervention_spec.records.append(_Fire(second_edit, pass_index=1, label="relu_1_2"))
    settle_armed_gradient_edits(trace, pass_index=1)
    by_parent = {row["parent_event_id"]: row for row in _events(trace, door="backward")}
    assert by_parent["lineage:1"]["status"] == "no_fire"
    assert by_parent["lineage:2"]["status"] == "fired"


def test_armed_edits_and_summary_line() -> None:
    edit = tl.scale(0.0)
    trace = _armed_trace(tl.grad_fn("ReluBackward0").to_target_spec(), edit)
    (rule,) = trace.armed_edits
    assert rule.edit is edit
    assert rule.signal == "backward"
    assert "1 gradient edit armed; call trace.backward(loss)" in repr(trace)
    trace._intervention_spec.records.append(_Fire(edit, pass_index=1, label="relu_1_2"))
    settle_armed_gradient_edits(trace, pass_index=1)
    assert "gradient edit armed" not in repr(trace)


# ---------------------------------------------------------------------------
# End to end (needs tl.grad and the output-gradient mount)
# ---------------------------------------------------------------------------


def _loss(result: Any) -> torch.Tensor:
    return result[result.output_layers[0]].out.sum()


#: A rerun matches sites during its forward, before final labels exist, so the
#: live doors address the relu structurally; reads use the final label.
_RELU = tl.func("relu")


def test_failing_loss_writes_error_and_detaches_this_dos_hooks() -> None:  # A23
    model = _BranchNet()
    source = _trace(model, backward_ready=True)
    seen: list[Any] = []

    def _bad_loss(result: Any) -> torch.Tensor:
        seen.append(result)
        raise RuntimeError("loss exploded")

    with pytest.raises(RuntimeError, match="loss exploded"):
        source.do(tl.grad(_RELU), tl.scale(0), model=model, inputs=_X, loss=_bad_loss)
    (result,) = seen
    assert result.armed_edits == ()
    assert _events(result, status="error")
    assert _events(source, status="error")


def test_mixed_plan_edits_the_activation_then_the_gradient() -> None:  # A24
    model = _BranchNet()
    source = _trace(model, backward_ready=True, save_grads="all")
    relu = _relu_label(source)
    plain = source.fork()
    plain.backward(_loss(plain))
    result = source.do(tl.grad(_RELU) | _RELU, tl.scale(0.5), model=model, inputs=_X, loss=_loss)
    assert torch.allclose(result[relu].out, 0.5 * source[relu].out)
    (arming,) = _events(result, door="do")
    assert arming["status"] == "fired"
    assert arming["armed_edits"] == 1
    assert _events(result, door="backward", status="fired")


def test_set_only_arms_in_place_and_fires_on_this_trace() -> None:  # A25
    model = _BranchNet()
    trace = _trace(model, backward_ready=True)
    layers_before = trace._layer_list
    returned = trace.do(
        tl.grad(_RELU),
        tl.scale(0),
        intervention=tl.options.InterventionOptions(engine="set_only", confirm_mutation=True),
    )
    assert returned is trace
    assert trace._layer_list is layers_before
    assert trace.intervention_audit[-1]["status"] == "armed"
    assert len(trace.armed_edits) == 1
    trace.backward(_loss(trace))
    assert _events(trace, door="backward", status="fired")


def test_saved_armed_result_keeps_its_armed_row(tmp_path: Any) -> None:  # A30
    model = _BranchNet()
    source = _trace(model, backward_ready=True)
    result = source.do(tl.grad(_RELU), tl.scale(0), model=model, inputs=_X)
    assert result.annotations["armed_gradient_edits"] == 1
    path = tmp_path / "armed"
    result.save(path)
    loaded = tl.load(path)
    assert loaded.annotations["armed_gradient_edits"] == 1
    assert loaded.intervention_audit[-1]["status"] == "armed"
    with pytest.raises(Exception, match="bundle-loaded"):
        loaded.backward(torch.tensor(1.0, requires_grad=True) * 2)


def test_attach_hooks_on_a_gradient_site_registers_only() -> None:
    model = _BranchNet()
    trace = _trace(model, backward_ready=True)
    events_before = len(_events(trace))
    handle = trace.attach_hooks(tl.grad(_RELU), tl.scale(0), confirm_mutation=True)
    assert len(_events(trace)) == events_before
    assert len(trace.armed_edits) == 1
    handle.remove()
    assert trace.armed_edits == ()
