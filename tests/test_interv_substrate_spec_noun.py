"""C03: the ONE immutable public InterventionSpec accepted at every door.

Surgery memo 3.1: ``tl.when(...)`` returns the immutable multi-clause spec;
the exact same object is accepted by ``fork.do``, ``fork.attach_hooks``,
``intervene=``, ``sweep``, and the save/load path. Lane options never enter
the spec, no lane silently drops a rule, and per-rule IDs distinguish rules
differing in ANY action argument.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import ArgumentTypeError, InvalidArgumentError
from torchlens.intervention import InterventionSpec
from torchlens.intervention.spec import classify_where


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _ready_trace(model: nn.Module) -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )


# ---------------------------------------------------------------------------
# The noun itself
# ---------------------------------------------------------------------------


def test_when_returns_immutable_public_spec() -> None:
    spec = tl.when(tl.func("relu"), tl.scale(0.0))
    assert isinstance(spec, InterventionSpec)
    assert len(spec.rules) == 1
    assert spec.rules[0].rule_id.startswith("r-")
    with pytest.raises((AttributeError, TypeError)):
        spec.rules = ()  # type: ignore[misc]


def test_rule_ids_distinguish_action_arguments() -> None:
    """noise(std=0.1) vs noise(std=0.9) carry DIFFERENT rule identities.

    The ledger memo's distinguishability oracle (D1d): two rules differing
    in any edit parameter must produce different canonical identity.
    """

    low = tl.when(tl.func("relu"), tl.noise(std=0.1))
    high = tl.when(tl.func("relu"), tl.noise(std=0.9))
    assert low.rules[0].rule_id != high.rules[0].rule_id
    assert low.spec_digest != high.spec_digest


def test_merge_preserves_rule_ids_and_refuses_duplicates() -> None:
    a = tl.when(tl.func("relu"), tl.scale(0.0))
    b = tl.when(tl.func("linear"), tl.noise(std=0.1))
    merged = a.merge(b)
    assert [rule.rule_id for rule in merged.rules] == [
        a.rules[0].rule_id,
        b.rules[0].rule_id,
    ]
    with pytest.raises(InvalidArgumentError) as excinfo:
        a.merge(a)
    assert excinfo.value.fields["code"] == "spec_rules_duplicate"


def test_single_rule_compat_surface() -> None:
    """Single-clause specs keep the historical .selector/.decision reads."""

    selector = tl.func("relu")
    spec = tl.when(selector, tl.scale(0.5))
    assert spec.selector is selector
    assert spec.decision is not None
    multi = spec.merge(tl.when(tl.func("linear"), tl.scale(0.5)))
    assert multi.selector is None
    assert multi.decision is None


def test_where_term_must_be_callable() -> None:
    with pytest.raises(ArgumentTypeError) as excinfo:
        tl.when("relu_1_2", tl.scale(0.0))  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "intervention_where_invalid"


def test_spec_clauses_must_be_intervention_rules() -> None:
    with pytest.raises(ArgumentTypeError) as excinfo:
        InterventionSpec(rules=("not a rule",))  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "intervention_rule_type_invalid"


def test_merge_operands_must_be_specs() -> None:
    spec = tl.when(tl.func("relu"), tl.scale(0.0))
    with pytest.raises(ArgumentTypeError) as excinfo:
        spec.merge("not a spec")  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "intervention_spec_type_invalid"


def test_address_law_classification() -> None:
    assert classify_where(tl.func("relu")) == "structural"
    assert classify_where(tl.in_module("fc1")) == "structural"
    assert classify_where(tl.label("relu_1_2")) == "recorded"
    assert classify_where(lambda ctx: True) == "value_dependent"
    # composite: worst class wins
    assert classify_where(tl.func("relu") & tl.label("relu_1_2")) == "recorded"


# ---------------------------------------------------------------------------
# Doors
# ---------------------------------------------------------------------------


def test_capture_door_accepts_spec_unchanged() -> None:
    model = _TinyModel()
    spec = tl.when(tl.func("relu"), tl.scale(0.0))
    log = tl.trace(model, torch.randn(2, 4), intervene=spec)
    assert torch.count_nonzero(log["relu_1_2"].out) == 0


def test_do_door_accepts_spec_unchanged() -> None:
    model = _TinyModel()
    log = _ready_trace(model)
    fork = log.fork()
    fork.do(tl.when(tl.func("relu"), tl.scale(0.0)))
    assert torch.count_nonzero(fork["relu_1_2"].out) == 0


def test_attach_hooks_door_accepts_spec_unchanged() -> None:
    model = _TinyModel()
    log = _ready_trace(model)
    fork = log.fork()
    handle = fork.attach_hooks(tl.when(tl.func("relu"), tl.scale(0.0)), confirm_mutation=True)
    assert len(handle.handle_ids) == 1


def test_spec_door_refuses_extra_arguments() -> None:
    model = _TinyModel()
    log = _ready_trace(model)
    spec = tl.when(tl.func("relu"), tl.scale(0.0))
    with pytest.raises(InvalidArgumentError) as excinfo:
        log.fork().do(spec, tl.scale(0.5))
    assert excinfo.value.fields["code"] == "spec_door_extra_arguments"
    with pytest.raises(InvalidArgumentError) as excinfo:
        log.fork().attach_hooks(spec, direction="forward", confirm_mutation=True)
    assert excinfo.value.fields["code"] == "spec_door_extra_arguments"


def test_replay_door_refuses_value_dependent_rule_by_name() -> None:
    """No lane silently drops a rule: runtime-only predicates refuse typed."""

    model = _TinyModel()
    log = _ready_trace(model)
    spec = tl.when(tl.func("relu"), tl.scale(0.0)).merge(
        tl.when(lambda ctx: bool(ctx.tensor.mean() > 0), tl.scale(0.5))
    )
    fork = log.fork()
    with pytest.raises(InvalidArgumentError) as excinfo:
        fork.do(spec)
    assert excinfo.value.fields["code"] == "spec_rule_unreplayable"
    # the refused rule is named
    assert spec.rules[1].rule_id in str(excinfo.value)
    # all-or-nothing: the structural rule attached NO hooks either
    staged = getattr(fork, "_intervention_spec", None)
    assert staged is None or not staged.hook_specs


def test_multi_clause_spec_applies_every_rule() -> None:
    model = _TinyModel()
    spec = tl.when(tl.in_module("fc1"), tl.scale(0.0)).merge(tl.when(tl.func("relu"), tl.add(1.0)))
    log = tl.trace(model, torch.randn(2, 4), intervene=spec)
    # fc1 output zeroed, then relu(0)=0 gets +1: relu out is all ones
    assert torch.allclose(log["relu_1_2"].out, torch.ones(2, 4))


def test_overlapping_rules_refuse_at_fire_time() -> None:
    model = _TinyModel()
    spec = tl.when(tl.func("relu"), tl.scale(0.0)).merge(tl.when(tl.func("relu"), tl.scale(0.5)))
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(model, torch.randn(2, 4), intervene=spec)
    assert excinfo.value.fields["code"] == "spec_rules_overlap"


def test_sweep_door_accepts_specs_unchanged() -> None:
    from torchlens.intervention.sweep import sweep

    model = _TinyModel()
    bundle = sweep(
        model,
        torch.randn(2, 4),
        values=[
            tl.when(tl.func("relu"), tl.scale(0.0)),
            tl.when(tl.func("relu"), tl.scale(0.5)),
        ],
    )
    assert len(bundle) == 2


def test_sweep_spec_values_conflicts_refuse_typed() -> None:
    from torchlens.intervention.sweep import sweep

    model = _TinyModel()
    spec = tl.when(tl.func("relu"), tl.scale(0.0))
    with pytest.raises(InvalidArgumentError) as excinfo:
        sweep(model, torch.randn(2, 4), values=[spec, 0.5])
    assert excinfo.value.fields["code"] == "sweep_spec_values_mixed"
    with pytest.raises(InvalidArgumentError) as excinfo:
        sweep(model, torch.randn(2, 4), at="relu", values=[spec])
    assert excinfo.value.fields["code"] == "sweep_spec_at_conflict"


def test_sweep_values_ride_typed_helper_specs() -> None:
    """Ledger item 1: swept values are typed replace helpers, not closures."""

    from torchlens.intervention.sweep import sweep

    model = _TinyModel()
    bundle = sweep(model, torch.randn(2, 4), at="relu", values=[0.0, 1.5])
    member_values = [float(trace["relu_1_2"].out.flatten()[0]) for trace in bundle.members.values()]
    assert member_values == [0.0, 1.5]


def test_backward_rules_of_multi_clause_spec_not_dropped() -> None:
    """A multi-clause spec's backward rule builds its sticky backward hook."""

    model = _TinyModel()
    spec = tl.when(tl.grad_fn(type="relu"), tl.grad_zero()).merge(
        tl.when(tl.in_module("fc1"), tl.scale(1.0))
    )
    x = torch.randn(2, 4, requires_grad=True)
    log = tl.trace(
        model,
        x,
        intervene=spec,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    staged = getattr(log, "_intervention_spec", None)
    assert staged is not None
    assert any(
        dict(hook_spec.metadata).get("created_by") == "intervene_backward_selector"
        for hook_spec in staged.hook_specs
    )
