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

import inspect
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._trace_grad_sites import settle_armed_gradient_edits
from torchlens._errors import InvalidArgumentError
from torchlens.intervention.spec import (
    addresses_autograd_nodes,
    is_gradient_site,
    site_signal,
    split_site_by_signal,
)
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


@pytest.mark.parametrize(
    "door",
    [
        lambda: tl.Trace.do,
        lambda: tl.Trace.clear_edits,
        lambda: tl.when,
        lambda: tl.observers.tap,
        lambda: tl.stats.narrate,
    ],
    ids=["do", "clear_edits", "when", "tap", "narrate"],
)
def test_site_doors_take_no_direction_keyword(door: Any) -> None:
    assert "direction" not in inspect.signature(door()).parameters


def test_passing_direction_to_a_site_door_is_a_type_error() -> None:
    trace = _trace(_BranchNet())
    with pytest.raises(TypeError):
        trace.do(_relu_label(trace), tl.scale(0.5), direction="forward")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        tl.when(tl.func("relu"), tl.scale(0.5), direction="backward")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        tl.observers.tap(tl.func("relu"), direction="backward")  # type: ignore[call-arg]


def test_do_signature_carries_loss_keyword() -> None:
    parameters = inspect.signature(tl.Trace.do).parameters
    assert parameters["loss"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["loss"].default is None


# ---------------------------------------------------------------------------
# Site signal derivation (Rule.signal, narrate signal, plan split)
# ---------------------------------------------------------------------------


def _grad_target(label: str) -> TargetSpec:
    """A frozen ``tl.grad`` target as the saved recipe spells it."""

    return TargetSpec(selector_kind="grad", selector_value=TargetSpec("label", label))


def test_site_signal_reads_the_site() -> None:
    assert site_signal("relu_1_2") == "forward"
    assert site_signal(tl.func("relu")) == "forward"
    assert site_signal(tl.grad_fn("ReluBackward0")) == "backward"
    assert site_signal(tl.in_backward_pass(2)) == "backward"
    assert site_signal(_grad_target("relu_1_2")) == "backward"
    assert site_signal(tl.label("relu_1_2") | tl.grad_fn("ReluBackward0")) == "both"


def test_split_site_by_signal_keeps_both_halves_of_a_union() -> None:
    node = tl.grad_fn("ReluBackward0")
    activation = tl.label("relu_1_2")
    forward, backward = split_site_by_signal(activation | node)
    assert forward == activation
    assert backward == node
    assert split_site_by_signal(activation) == (activation, None)
    assert split_site_by_signal(node) == (None, node)


def test_gradient_and_node_sites_are_told_apart() -> None:
    assert is_gradient_site(_grad_target("relu_1_2"))
    assert not is_gradient_site(tl.grad_fn("ReluBackward0"))
    assert addresses_autograd_nodes(tl.grad_fn("ReluBackward0"))
    assert not addresses_autograd_nodes(_grad_target("relu_1_2"))
    assert not addresses_autograd_nodes(tl.func("relu"))


def test_rule_signal_is_derived_and_rule_has_no_direction() -> None:
    forward_rule = tl.when(tl.func("relu"), tl.scale(0.5)).rules[0]
    node_rule = tl.when(tl.grad_fn("ReluBackward0"), lambda g, *, ctx: g).rules[0]
    mixed_rule = tl.when(tl.label("relu_1_2") | tl.grad_fn("ReluBackward0"), tl.scale(0.5)).rules[0]
    assert forward_rule.signal == "forward"
    assert node_rule.signal == "backward"
    assert mixed_rule.signal == "both"
    # Walk names.5 (JMT 2026-11-01): Rule.direction is removed; the site decides.
    assert not hasattr(forward_rule, "direction")
    assert not hasattr(mixed_rule, "direction")
    assert "signal='forward'" in repr(forward_rule)


# ---------------------------------------------------------------------------
# Refusals before any mutation (A22, A23)
# ---------------------------------------------------------------------------


def _snapshot(trace: Any) -> tuple[int, int, int]:
    spec = trace._intervention_spec
    hooks = len(spec.hook_specs) if spec is not None else 0
    return hooks, len(trace.intervention_audit), len(trace.state_history)


def test_gradient_plan_without_model_refuses_replay_before_any_mutation() -> None:
    trace = _trace(_BranchNet())
    before = _snapshot(trace)
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.do(tl.grad_fn("ReluBackward0"), lambda g, *, ctx: g)
    assert excinfo.value.fields["code"] == "replay_gradient_edit_unsupported"
    message = str(excinfo.value)
    assert "model= and inputs=" in message
    assert "engine='set_only'" in message
    assert _snapshot(trace) == before


def test_gradient_plan_with_replay_engine_refuses_even_with_model() -> None:
    model = _BranchNet()
    trace = _trace(model)
    before = _snapshot(trace)
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.do(
            tl.grad_fn("ReluBackward0"),
            lambda g, *, ctx: g,
            model=model,
            inputs=_X,
            intervention=tl.options.InterventionOptions(engine="replay"),
        )
    assert excinfo.value.fields["code"] == "replay_gradient_edit_unsupported"
    assert _snapshot(trace) == before


def test_loss_must_be_a_callable_of_the_returned_trace() -> None:
    model = _BranchNet()
    trace = _trace(model)
    before = _snapshot(trace)
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.do(
            tl.grad_fn("ReluBackward0"),
            lambda g, *, ctx: g,
            model=model,
            inputs=_X,
            loss=torch.tensor(1.0),  # type: ignore[arg-type]
        )
    assert excinfo.value.fields["code"] == "do_loss_not_callable"
    assert _snapshot(trace) == before


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


def test_replay_do_refuses_a_recipe_holding_a_gradient_rule() -> None:
    trace = _trace(_BranchNet()).fork()
    trace._ensure_intervention_spec().add_hook(
        _grad_target(_relu_label(trace)),
        tl.scale(0.0),
        handle="hook-grad",
        metadata={"arming_group": "arm-1", "arming_event_id": None},
    )
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.do(_relu_label(trace), tl.scale(1.0))
    assert excinfo.value.fields["code"] == "replay_gradient_edit_unsupported"
    assert "Replay re-executes forward ops only" in str(excinfo.value)


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


def test_settle_writes_one_fired_row_per_pass_and_keeps_earlier_rows() -> None:
    edit = tl.scale(0.0)
    label = "relu_1_2"
    trace = _armed_trace(_grad_target(label), edit)
    spec = trace._intervention_spec
    assert trace.annotations["armed_gradient_edits"] == 1
    spec.records.append(_Fire(edit, pass_index=1, label=label))
    settle_armed_gradient_edits(trace, pass_index=1)
    first = _events(trace, door="backward")
    assert len(first) == 1
    row = first[0]
    assert row["lane"] == "live_hook"
    assert row["status"] == "fired"
    assert row["fire_count"] == 1
    assert row["parent_event_id"] == "lineage:1"
    assert row["arming_event_id"] == "lineage:1"
    assert row["backward_pass_number"] == 1
    assert trace.intervention_audit[-1]["kind"] == "EVENT"
    assert trace.intervention_audit[-1]["status"] == "fired"
    assert trace.annotations["armed_gradient_edits"] == 0
    snapshot = dict(row)

    spec.records.append(_Fire(edit, pass_index=2, label=label))
    settle_armed_gradient_edits(trace, pass_index=2)
    rows = _events(trace, door="backward")
    assert [r["backward_pass_number"] for r in rows] == [1, 2]
    assert rows[0] == snapshot


def test_settle_no_fire_reasons() -> None:
    edit = tl.scale(0.0)
    trace = _armed_trace(_grad_target("relu_1_2"), edit)
    settle_armed_gradient_edits(trace, pass_index=1)
    (row,) = _events(trace, door="backward")
    assert row["status"] == "no_fire"
    assert row["fire_count"] == 0
    assert row["no_fire_reason"] == "gradient_not_reached"

    qualified = TargetSpec(
        selector_kind="and",
        selector_value=(_grad_target("relu_1_2"), TargetSpec("backward_pass", 2)),
    )
    filtered = _armed_trace(qualified, tl.scale(0.0))
    settle_armed_gradient_edits(filtered, pass_index=1)
    (row,) = _events(filtered, door="backward")
    assert row["no_fire_reason"] == "pass_filtered"


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


def test_settle_is_a_no_op_without_gradient_rules() -> None:
    trace = _trace(_BranchNet()).fork()
    before = len(trace.state_history)
    settle_armed_gradient_edits(trace, pass_index=1)
    assert len(trace.state_history) == before


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


def test_arming_a_gradient_rule_keeps_the_outs_current() -> None:
    trace = _trace(_BranchNet()).fork()
    state = trace.state
    assert trace._recipe_is_clean()
    trace._ensure_intervention_spec().add_hook(
        _grad_target("relu_1_2"), tl.scale(0.0), metadata={"arming_group": "arm-1"}
    )
    trace._mark_intervention_spec_mutated(outs_stale=False)
    assert trace._recipe_is_clean()
    assert trace.state == state


# ---------------------------------------------------------------------------
# End to end (needs tl.grad and the output-gradient mount)
# ---------------------------------------------------------------------------


def _loss(result: Any) -> torch.Tensor:
    return result[result.output_layers[0]].out.sum()


#: A rerun matches sites during its forward, before final labels exist, so the
#: live doors address the relu structurally; reads use the final label.
_RELU = tl.func("relu")


def test_do_gradient_site_returns_an_armed_rerun() -> None:  # A19
    model = _BranchNet()
    source = _trace(model, backward_ready=True)
    relu = _relu_label(source)
    recipe_before = repr(source.recipe)
    result = source.do(tl.grad(relu), tl.scale(0), model=model, inputs=_X)
    assert result is not source
    assert result.intervention_audit[-1]["status"] == "armed"
    assert result.intervention_audit[-1]["fire_count"] == 0
    (arming,) = _events(result, door="do", status="armed")
    assert arming["awaiting"] == "backward"
    assert arming["armed_edits"] == 1
    (rule,) = result.armed_edits
    assert rule.signal == "backward"
    assert not [r for r in result._intervention_spec.records if r.signal == "backward"]
    assert repr(source.recipe) == recipe_before


def test_backward_settles_the_armed_row_per_pass() -> None:  # A20
    model = _BranchNet()
    source = _trace(model, backward_ready=True)
    result = source.do(tl.grad(_RELU), tl.scale(0), model=model, inputs=_X)
    (arming,) = _events(result, door="do", status="armed")
    loss = _loss(result)
    result.backward(loss, retain_graph=True)
    (first,) = _events(result, door="backward")
    assert first["lane"] == "live_hook"
    assert first["parent_event_id"] == arming["event_id"]
    assert first["status"] == "fired"
    assert first["fire_count"] >= 1
    assert first["backward_pass_number"] == 1
    snapshot = dict(first)
    result.backward(loss)
    rows = _events(result, door="backward")
    assert [row["backward_pass_number"] for row in rows] == [1, 2]
    assert rows[0] == snapshot


def test_loss_runs_the_backward_inside_do() -> None:  # A21
    model = _BranchNet()
    source = _trace(model, backward_ready=True, save_grads="all")
    relu = _relu_label(source)
    result = source.do(tl.grad(_RELU), tl.scale(0), model=model, inputs=_X, loss=_loss)
    assert result._num_backward_passes == 1
    assert _events(result, door="backward", status="fired")
    assert torch.count_nonzero(result[relu].grad) == 0
    assert result.intervention_audit[-1]["status"] == "fired"


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


def test_rerun_forces_backward_readiness_and_discloses_it() -> None:  # A26
    model = _BranchNet()
    unready = tl.trace(model, _X, capture=tl.options.CaptureOptions(replayable=True))
    result = unready.do(tl.grad(_RELU), tl.scale(0), model=model, inputs=_X, loss=_loss)
    assert result.last_run["backward_ready_forced"] is True
    assert _events(result, door="backward", status="fired")
    ready = _trace(model, backward_ready=True)
    armed = ready.do(tl.grad(_RELU), tl.scale(0), model=model, inputs=_X)
    assert armed.last_run.get("backward_ready_forced", False) is False


def test_unreached_and_pass_filtered_gradient_sites_settle_no_fire() -> None:  # A27
    model = _BranchNet()
    source = _trace(model, backward_ready=True, save_grads="all")
    side = next(str(op.layer_label) for op in source._layer_list if op.func_name == "sigmoid")
    result = source.do(tl.grad(tl.func("sigmoid")), tl.scale(0), model=model, inputs=_X)
    result.backward(result[result.output_layers[0]].out.sum())
    (row,) = _events(result, door="backward")
    assert row["status"] == "no_fire"
    assert row["no_fire_reason"] == "gradient_not_reached"
    assert not result[side].has_grad

    filtered = source.do(
        tl.grad(_RELU) & tl.in_backward_pass(2), tl.scale(0), model=model, inputs=_X, loss=_loss
    )
    (row,) = _events(filtered, door="backward")
    assert row["no_fire_reason"] == "pass_filtered"


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


def test_staged_do_on_a_gradient_site_arms_only_and_clear_edits_disarms() -> None:
    model = _BranchNet()
    trace = _trace(model, backward_ready=True)
    backward_events_before = len(_events(trace, door="backward"))
    trace.do(
        tl.grad(_RELU),
        tl.scale(0),
        intervention=tl.options.InterventionOptions(engine="set_only", confirm_mutation=True),
    )
    assert len(_events(trace, door="backward")) == backward_events_before
    assert len(trace.armed_edits) == 1
    trace.clear_edits(tl.grad(_RELU), confirm_mutation=True)
    assert trace.armed_edits == ()


def test_saved_recipe_round_trips_the_grad_target() -> None:  # A31
    rule = tl.when(tl.grad("relu_1_2"), tl.scale(0)).rules[0]
    assert rule.signal == "backward"
    assert tl.when(tl.label("relu_1_2"), tl.scale(0)).rules[0].signal == "forward"
    assert tl.when(tl.grad("relu_1_2") | "relu_1_2", tl.scale(0)).rules[0].signal == "both"
    from torchlens._ir.selector_eval import normalize_selector_like

    rebuilt = normalize_selector_like(rule.target.to_target_spec(), lifecycle="site")
    assert rebuilt == rule.target
