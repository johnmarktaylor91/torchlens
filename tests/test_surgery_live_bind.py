"""Lane F01 bind checkpoint: ``spec.bind(model)`` capture-free executor.

Covers the amended lane brief's bind-side surface (foldA D12 / surgery memo
3.3 items 4, 7, 8, 9 and the bind-side half of 13): the serial non-reentrant
capture-free callable, transparency, real HF ``generate`` (KV cache on/off),
``.last_report``, read-only surface, atomic install/removal on success and
exception, FOLD-A3 zero-fire fail-closed settlement, the two-edit misfire
disclosure naming the unfired rule, the OP2 model-door funnel (BOTH arms
behind one switch), the closed engine-set ``execution_effect`` vocabulary,
and the bind-side turnkey steer wrapper.

Toy-model tests are smoke-tier; the real-model gate rows (distilgpt2 + one
small Llama family) are heavy + real_model and skip when the HF snapshot is
not cached.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention import BindReport, BoundInterventionExecutor
from torchlens.intervention.binding import bind_spec_to_model
from torchlens.intervention.errors import BindingPreflightError, BindingRuntimeError
from torchlens.intervention.model_door import (
    MODEL_DOOR_INVENTORY,
    door_policy,
    model_door_policy,
    resolve_model_operand,
)
from torchlens.intervention.steering import SteerResult, steer_generate

HF_HUB_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


def _require_hf_snapshot(repo_dirname: str) -> None:
    """Skip when the named HF snapshot is not in the local offline cache."""

    if not (HF_HUB_CACHE / repo_dirname).exists():
        pytest.skip(
            f"GATE F01 bind-x-generate: HF snapshot {repo_dirname} not cached;"
            " fetch it once online, then rerun offline."
        )


class _Toy(nn.Module):
    """Two-linear toy with one relu: the smallest op + boundary target."""

    def __init__(self) -> None:
        """Build fc1 -> relu -> fc2 with fixed shapes."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two-layer forward."""

        return self.fc2(torch.relu(self.fc1(x)))


class _TwoPass(nn.Module):
    """Calls the SAME submodule twice: the pass-qualified boundary target."""

    def __init__(self) -> None:
        """Build one shared linear block."""

        super().__init__()
        self.block = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared block twice (pass 1 then pass 2)."""

        return self.block(self.block(x))


class _Boom(nn.Module):
    """Raises mid-forward AFTER one interceptable op (cleanup-path probe)."""

    def __init__(self) -> None:
        """Build the single linear the pre-raise relu consumes."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fire one relu, then raise."""

        torch.relu(self.fc(x))
        raise RuntimeError("boom")


class _ToyGen(nn.Module):
    """Toy with a ``generate`` method: the steer-wrapper composition target."""

    def __init__(self) -> None:
        """Build one linear step function."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One relu step."""

        return torch.relu(self.fc(x))

    def generate(self, x: torch.Tensor, steps: int = 3) -> torch.Tensor:
        """Loop the forward ``steps`` times (a stand-in generation loop)."""

        for _ in range(steps):
            x = self(x)
        return x


@pytest.fixture()
def toy() -> tuple[_Toy, torch.Tensor]:
    """A seeded eval-mode toy model plus one fixed input batch."""

    torch.manual_seed(0)
    model = _Toy().eval()
    x = torch.randn(3, 4)
    return model, x


# ---------------------------------------------------------------------------
# mechanism: op-level and boundary edits, transparency, report contents
# ---------------------------------------------------------------------------


def test_bind_op_level_edit_correct_and_transparent(toy) -> None:
    """An op-level zero-ablate edits the live value; the model stays untouched."""

    model, x = toy
    base = model(x)
    bound = tl.when(tl.func("relu"), tl.zero_ablate()).bind(model)
    out = bound(x)
    assert torch.allclose(out, model.fc2(torch.zeros(3, 4)))
    assert not torch.allclose(out, base)
    # transparency: the base model is bit-identical afterwards, hook-free
    assert torch.allclose(model(x), base)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_bind_boundary_rule_module_output(toy) -> None:
    """A plain tl.module rule substitutes the module's OUTPUT boundary value."""

    model, x = toy
    bound = tl.when(tl.module("fc1"), tl.scale(0.0)).bind(model)
    out = bound(x)
    assert torch.allclose(out, model.fc2(torch.relu(torch.zeros(3, 4))))
    assert bound.last_report.fire_count == 1
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_bind_report_contents(toy) -> None:
    """The out-of-band ledger carries stamps, counts, effects, and cleanup."""

    model, x = toy
    spec = tl.when(tl.func("relu"), tl.scale(2.0))
    bound = spec.bind(model)
    assert bound.last_report is None
    bound(x)
    report = bound.last_report
    assert isinstance(report, BindReport)
    assert report.schema == "bind_report_v1"
    assert report.lane == "bind"
    assert report.door == "call"
    assert report.status == "fired"
    assert report.trace is None
    assert report.spec_digest == spec.spec_digest
    assert report.model_class == "_Toy"
    assert report.fire_count == 1
    assert report.rule_fire_counts == {spec.rules[0].rule_id: 1}
    assert report.zero_fire_rule_ids == ()
    assert report.cleanup == "removed"
    assert report.duration_s is not None and report.duration_s >= 0
    (fire,) = report.fires
    assert fire["execution_effect"] == "values_replaced_after_execution"
    assert fire["rule_id"] == spec.rules[0].rule_id
    assert fire["site_key"], "every fire carries a live-minted structural site key"
    (record,) = report.fire_records
    assert record.replaced is True
    assert report.resolved_static_targets == {spec.rules[0].rule_id: ("op-level",)}


@pytest.mark.smoke
def test_bind_pass_qualified_boundary_target() -> None:
    """A pass-qualified boundary target (``block:2``) fires only on that pass."""

    torch.manual_seed(0)
    model = _TwoPass().eval()
    x = torch.randn(2, 4)
    bound = tl.when(tl.module("block:2"), tl.scale(0.0)).bind(model)
    out = bound(x)
    assert torch.allclose(out, torch.zeros(2, 4))
    report = bound.last_report
    assert report.fire_count == 1
    (fire,) = report.fires
    assert fire["pass_index"] == 2
    assert fire["target"] == "block:2"


# ---------------------------------------------------------------------------
# zero-fire settlement (FOLD-A3) + the two-edit misfire disclosure
# ---------------------------------------------------------------------------


def test_bind_zero_fire_fail_closed_default(toy) -> None:
    """A never-firing rule raises AFTER the call; the report is retained."""

    model, x = toy
    bound = tl.when(tl.func("sigmoid"), tl.zero_ablate()).bind(model)
    with pytest.raises(BindingRuntimeError) as excinfo:
        bound(x)
    assert excinfo.value.fields["code"] == "bind_zero_fire"
    report = bound.last_report
    assert report is not None
    assert report.status == "no_fire"
    assert report.zero_fire_rule_ids == (bound.spec.rules[0].rule_id,)


def test_bind_zero_fire_disclose_policy(toy) -> None:
    """``on_zero_fire='disclose'`` records zero-fire rules without raising."""

    model, x = toy
    base = model(x)
    bound = tl.when(tl.func("sigmoid"), tl.zero_ablate()).bind(model, on_zero_fire="disclose")
    out = bound(x)
    assert torch.allclose(out, base)
    assert bound.last_report.status == "no_fire"
    assert len(bound.last_report.zero_fire_rule_ids) == 1


@pytest.mark.smoke
def test_bind_two_edit_misfire_names_only_the_unfired_rule(toy) -> None:
    """GATE row: two edits, one misfires -- the refusal names the unfired rule."""

    model, x = toy
    fired_rule = tl.when(tl.func("relu"), tl.scale(1.1))
    unfired_rule = tl.when(tl.func("hardshrink"), tl.zero_ablate())
    spec = fired_rule & unfired_rule
    bound = spec.bind(model)
    with pytest.raises(BindingRuntimeError) as excinfo:
        bound(x)
    assert excinfo.value.fields["code"] == "bind_zero_fire"
    report = bound.last_report
    unfired = [rid for rid, count in report.rule_fire_counts.items() if count == 0]
    fired = [rid for rid, count in report.rule_fire_counts.items() if count > 0]
    assert len(unfired) == 1 and len(fired) == 1
    message = str(excinfo.value)
    assert unfired[0] in message
    assert fired[0] not in message
    assert report.rule_fire_counts[fired[0]] == 1


# ---------------------------------------------------------------------------
# preflight refusals (bind time, before ANY forward)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_bind_preflight_refusals(toy) -> None:
    """Empty specs, bad policies, bad models, stale anchors refuse typed."""

    model, _x = toy
    from torchlens.intervention.spec import InterventionSpec

    with pytest.raises(BindingPreflightError) as excinfo:
        InterventionSpec(rules=()).bind(model)
    assert excinfo.value.fields["code"] == "bind_spec_empty"

    spec = tl.when(tl.func("relu"), tl.zero_ablate())
    with pytest.raises(BindingPreflightError) as excinfo:
        spec.bind(model, on_zero_fire="warn")
    assert excinfo.value.fields["code"] == "bind_zero_fire_policy_invalid"

    with pytest.raises(BindingPreflightError) as excinfo:
        spec.bind(lambda x: x)
    assert excinfo.value.fields["code"] == "bind_model_invalid"

    with pytest.raises(BindingPreflightError) as excinfo:
        spec.bind(model.forward)
    assert excinfo.value.fields["code"] == "bind_model_invalid"
    assert "F41" in str(excinfo.value)  # bound-method roots are F41's contract

    with pytest.raises(BindingPreflightError) as excinfo:
        bind_spec_to_model("not a spec", model)
    assert excinfo.value.fields["code"] == "bind_spec_invalid"

    with pytest.raises(BindingPreflightError) as excinfo:
        tl.when(tl.module("nope.blocks.7"), tl.scale(0.5)).bind(model)
    assert excinfo.value.fields["code"] == "bind_static_anchor_unresolved"
    assert "nope.blocks.7" in str(excinfo.value)


def test_bind_rule_unsupported_composite_boundary(toy) -> None:
    """A module OUTPUT-boundary term composed with op-level terms refuses."""

    model, _x = toy
    composite = tl.module("fc1") & tl.func("relu")
    with pytest.raises(BindingPreflightError) as excinfo:
        tl.when(composite, tl.zero_ablate()).bind(model)
    assert excinfo.value.fields["code"] == "bind_rule_unsupported"
    assert "in_module" in excinfo.value.fields["remedy"]  # the teaching alternative


# ---------------------------------------------------------------------------
# the binding is NOT an nn.Module: read-only surface + refusals (items 7, 9)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_binding_is_not_a_module_and_teaches(toy) -> None:
    """Training-surface access refuses with the wrapper-module teaching."""

    model, _x = toy
    bound = tl.when(tl.func("relu"), tl.zero_ablate()).bind(model)
    assert not isinstance(bound, nn.Module)
    for name in ("parameters", "state_dict", "to", "train", "eval", "forward"):
        with pytest.raises(BindingRuntimeError) as excinfo:
            getattr(bound, name)
        assert excinfo.value.fields["code"] == "binding_training_surface"
    # non-module unknown attributes stay ordinary AttributeError
    with pytest.raises(AttributeError):
        _ = bound.definitely_not_an_attr


@pytest.mark.smoke
def test_binding_read_only_surface_and_serialization_refusal(toy) -> None:
    """``.spec``/``.base_model`` are read-only; pickle refuses typed."""

    model, _x = toy
    spec = tl.when(tl.func("relu"), tl.zero_ablate())
    bound = spec.bind(model)
    assert bound.spec is spec
    assert bound.base_model is model
    with pytest.raises(AttributeError):
        bound.spec = spec
    with pytest.raises(AttributeError):
        bound.base_model = model
    with pytest.raises(BindingRuntimeError) as excinfo:
        pickle.dumps(bound)
    assert excinfo.value.fields["code"] == "binding_serialization_unsupported"
    assert "read-only" in repr(bound)


def test_binding_reentrant_call_refuses(toy) -> None:
    """A hook that re-enters the binding hits the serial/non-reentrant wall."""

    model, x = toy
    holder: dict[str, BoundInterventionExecutor] = {}

    def _evil(out: torch.Tensor, *, hook) -> torch.Tensor:
        """Re-enter the executing binding (forbidden)."""

        return holder["bound"](x)

    holder["bound"] = tl.when(tl.func("relu"), _evil).bind(model, on_zero_fire="disclose")
    with pytest.raises(BindingRuntimeError) as excinfo:
        holder["bound"](x)
    assert excinfo.value.fields["code"] == "binding_reentrant_call"


def test_bind_exception_cleanup_is_atomic() -> None:
    """A mid-forward user exception propagates; runtime state is fully removed."""

    torch.manual_seed(0)
    model = _Boom()
    bound = tl.when(tl.func("relu"), tl.scale(2.0)).bind(model)
    with pytest.raises(RuntimeError, match="boom"):
        bound(torch.randn(2, 4))
    report = bound.last_report
    assert report.status == "error"
    assert report.error == "RuntimeError: boom"
    assert report.cleanup == "removed"
    assert report.fire_count == 1  # the relu fired before the raise
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    # and no torch-function mode is left installed: plain ops run clean
    assert torch.relu(torch.tensor([-1.0, 1.0])).tolist() == [0.0, 1.0]


# ---------------------------------------------------------------------------
# OP2: the ONE audited model-door funnel, BOTH arms behind one switch
# ---------------------------------------------------------------------------


def test_model_door_refuse_arm_is_default(toy) -> None:
    """Arm (a): doors refuse a binding with the canonical spelling printed."""

    model, x = toy
    bound = tl.when(tl.func("relu"), tl.zero_ablate()).bind(model)
    assert model_door_policy() == "refuse"
    with pytest.raises(Exception) as excinfo:
        tl.trace(bound, x)
    assert excinfo.value.fields["code"] == "model_door_binding_refused"
    assert "binding.base_model" in str(excinfo.value)
    assert "intervene=binding.spec" in str(excinfo.value)
    with pytest.raises(Exception) as excinfo:
        tl.release_model(bound)
    assert excinfo.value.fields["code"] == "model_door_binding_refused"


def test_model_door_normalize_arm_matches_canonical_spelling(toy) -> None:
    """Arm (b): the funnel normalizes to base model + intervene=spec."""

    model, x = toy
    spec = tl.when(tl.func("relu"), tl.zero_ablate())
    bound = spec.bind(model)
    with door_policy("normalize"):
        via_door = tl.trace(bound, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    canonical = tl.trace(
        model, x, intervene=spec, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    door_rows = [r for r in via_door.intervention_audit if r.get("kind") != "event"]
    canon_rows = [r for r in canonical.intervention_audit if r.get("kind") != "event"]
    assert len(door_rows) == len(canon_rows) >= 1
    assert torch.allclose(via_door["relu_1_2"].out, canonical["relu_1_2"].out)


def test_model_door_normalize_arm_intervene_conflict(toy) -> None:
    """Arm (b) never merges the binding's spec with an explicit intervene=."""

    model, x = toy
    bound = tl.when(tl.func("relu"), tl.zero_ablate()).bind(model)
    other = tl.when(tl.func("linear"), tl.scale(0.5))
    with door_policy("normalize"), pytest.raises(Exception) as excinfo:
        tl.trace(bound, x, intervene=other)
    assert excinfo.value.fields["code"] == "model_door_intervene_conflict"


def test_model_door_policy_switch_is_closed_and_scoped() -> None:
    """The OP2 switch takes exactly two arms and restores itself on exit."""

    with pytest.raises(Exception) as excinfo, door_policy("maybe"):
        pass
    assert excinfo.value.fields["code"] == "model_door_policy_invalid"
    assert model_door_policy() == "refuse"
    with door_policy("normalize"):
        assert model_door_policy() == "normalize"
    assert model_door_policy() == "refuse"


@pytest.mark.smoke
def test_model_door_funnel_coverage_inventory(toy) -> None:
    """Every door the inventory marks wired actually routes through the funnel."""

    model, _x = toy
    bound = tl.when(tl.func("relu"), tl.zero_ablate()).bind(model)
    wired = {door for door, is_wired in MODEL_DOOR_INVENTORY.items() if is_wired}
    assert wired == {"trace", "release_model", "bind"}
    # each wired door classifies a binding (typed refusal under the default arm)
    for door in ("trace", "release_model"):
        with pytest.raises(Exception) as excinfo:
            resolve_model_operand(bound, door=door)
        assert excinfo.value.fields["code"] == "model_door_binding_refused"
    # bind's door: bindings never nest, regardless of arm -- arm (a) refuses
    # at the funnel, arm (b) refuses re-binding explicitly
    with pytest.raises(Exception) as excinfo:
        bound.spec.bind(bound)
    assert excinfo.value.fields["code"] == "model_door_binding_refused"
    with door_policy("normalize"), pytest.raises(BindingPreflightError) as excinfo:
        bound.spec.bind(bound)
    assert excinfo.value.fields["code"] == "bind_model_invalid"
    # a plain model passes through untouched under both arms
    for policy in ("refuse", "normalize"):
        with door_policy(policy):
            resolution = resolve_model_operand(model, door="trace")
        assert resolution.model is model and not resolution.normalized


# ---------------------------------------------------------------------------
# execution_effect: the closed engine-set vocabulary (foldA D17)
# ---------------------------------------------------------------------------


def test_execution_effect_vocabulary_is_closed() -> None:
    """The audit chokepoint refuses effects outside the closed two-value set."""

    from torchlens.intervention.audit import EXECUTION_EFFECTS

    assert {
        "values_replaced_after_execution",
        "exits_substituted_interior_not_replayed",
    } == EXECUTION_EFFECTS
    for banned in ("skipped", "deleted", "removed"):
        assert banned not in EXECUTION_EFFECTS


def test_execution_effect_chokepoint_refuses_invalid(toy) -> None:
    """``record_intervention_event`` refuses a non-vocabulary effect typed."""

    model, x = toy
    from torchlens.intervention.audit import record_intervention_event

    log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    common: dict = {
        "lane": "bind",
        "door": "call",
        "edit_names": ("scale",),
        "selection_repr": "tl.func('relu')",
        "status": "fired",
        "fire_count": 1,
        "append_audit_row": False,
    }
    with pytest.raises(Exception) as excinfo:
        record_intervention_event(log, execution_effect="skipped", **common)
    assert excinfo.value.fields["code"] == "execution_effect_invalid"
    row = record_intervention_event(
        log, execution_effect="values_replaced_after_execution", **common
    )
    assert row["execution_effect"] == "values_replaced_after_execution"


# ---------------------------------------------------------------------------
# generate + the bind-side turnkey steer wrapper (item 13, bind side)
# ---------------------------------------------------------------------------


def test_bind_generate_unavailable_refuses(toy) -> None:
    """``generate()`` on a generate-less base model refuses typed."""

    model, _x = toy
    bound = tl.when(tl.func("relu"), tl.zero_ablate()).bind(model)
    with pytest.raises(BindingRuntimeError) as excinfo:
        bound.generate(torch.randn(1, 4))
    assert excinfo.value.fields["code"] == "bind_generate_unavailable"


@pytest.mark.smoke
def test_steer_generate_toy_composition() -> None:
    """The steer wrapper returns model outputs + the ledger, never a Trace."""

    torch.manual_seed(0)
    model = _ToyGen().eval()
    x = torch.randn(2, 4)
    result = steer_generate(model, x, tl.when(tl.func("relu"), tl.scale(0.5)), steps=3)
    assert isinstance(result, SteerResult)
    assert isinstance(result.outputs, torch.Tensor)
    assert isinstance(result.report, BindReport)
    assert result.report.door == "generate"
    assert result.report.fire_count == 3  # one relu per generation step
    assert result.report.trace is None
    from torchlens import Trace

    assert not isinstance(result.outputs, Trace)


# ---------------------------------------------------------------------------
# GATE rows: bind x real HF generate (distilgpt2 + one small Llama family)
# ---------------------------------------------------------------------------


def _real_generate_gate(model_name: str, steer_site: str, ids: torch.Tensor) -> None:
    """The shared real-model gate body: call, generate, cache on/off, misfire."""

    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name).eval()
    torch.manual_seed(0)
    base_out = model.generate(ids, max_new_tokens=6, do_sample=False)

    spec = tl.when(tl.module(steer_site), tl.scale(4.0))
    bound = spec.bind(model)
    torch.manual_seed(0)
    out_cache_on = bound.generate(ids, max_new_tokens=6, do_sample=False, use_cache=True)
    report_on = bound.last_report
    assert report_on.status == "fired" and report_on.door == "generate"
    assert report_on.fire_count >= 6  # at least one boundary fire per step
    torch.manual_seed(0)
    out_cache_off = bound.generate(ids, max_new_tokens=6, do_sample=False, use_cache=False)
    assert bound.last_report.fire_count >= 6
    assert torch.equal(out_cache_on, out_cache_off), "KV cache on/off must agree"
    assert not torch.equal(out_cache_on, base_out), "the steer must change generation"

    # transparency: unbound model reproduces the base generation, hook-free
    torch.manual_seed(0)
    assert torch.equal(model.generate(ids, max_new_tokens=6, do_sample=False), base_out)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())

    # the plain-call door works on the same binding
    out_call = bound(ids)
    assert bound.last_report.door == "call" and bound.last_report.fire_count >= 1
    assert out_call.logits.shape[1] == ids.shape[1]

    # zero fires during a real generate fail closed and retain the report
    zero_bound = tl.when(tl.func("hardshrink"), tl.zero_ablate()).bind(model)
    with pytest.raises(BindingRuntimeError) as excinfo:
        zero_bound.generate(ids, max_new_tokens=2, do_sample=False)
    assert excinfo.value.fields["code"] == "bind_zero_fire"
    assert zero_bound.last_report.status == "no_fire"

    # GATE: two-edit misfire -- the refusal names exactly the unfired rule
    two = tl.when(tl.module(steer_site), tl.scale(1.1)) & tl.when(
        tl.func("hardshrink"), tl.zero_ablate()
    )
    two_bound = two.bind(model)
    with pytest.raises(BindingRuntimeError) as excinfo:
        two_bound.generate(ids, max_new_tokens=2, do_sample=False)
    counts = two_bound.last_report.rule_fire_counts
    unfired = [rid for rid, count in counts.items() if count == 0]
    fired = [rid for rid, count in counts.items() if count > 0]
    assert len(unfired) == 1 and len(fired) == 1
    assert unfired[0] in str(excinfo.value)
    assert fired[0] not in str(excinfo.value)


@pytest.mark.heavy
@pytest.mark.real_model
def test_bind_real_generate_distilgpt2() -> None:
    """GATE row: bind x real HF generate on distilgpt2."""

    _require_hf_snapshot("models--distilgpt2")
    _real_generate_gate(
        "distilgpt2", "transformer.h.2.mlp", torch.tensor([[464, 3139, 286, 4881, 318]])
    )


@pytest.mark.heavy
@pytest.mark.real_model
def test_bind_real_generate_llama_68m() -> None:
    """GATE row: bind x real HF generate on one small Llama-family model."""

    _require_hf_snapshot("models--JackFram--llama-68m")
    _real_generate_gate(
        "JackFram/llama-68m", "model.layers.1.mlp", torch.tensor([[1, 450, 7483, 310, 3444, 338]])
    )


@pytest.mark.smoke
def test_bind_rule_runtime_error_is_typed(toy) -> None:
    """A WHERE predicate that raises at a live op refuses typed, chained."""

    model, x = toy

    def _exploding_where(ctx) -> bool:
        """Raise on every candidate op."""

        raise ValueError("predicate boom")

    bound = tl.when(_exploding_where, tl.zero_ablate()).bind(model)
    with pytest.raises(BindingRuntimeError) as excinfo:
        bound(x)
    assert excinfo.value.fields["code"] == "bind_rule_runtime_error"
    assert isinstance(excinfo.value.__cause__, ValueError)


@pytest.mark.smoke
def test_bind_cleanup_failure_is_typed(toy) -> None:
    """A teardown that cannot remove runtime state raises bind_cleanup_failed."""

    from torchlens.intervention import binding as binding_module

    model, _x = toy
    bound = tl.when(tl.func("relu"), tl.zero_ablate()).bind(model)

    class _StuckHandle:
        """A hook handle whose removal fails (the leftover-hooks hazard)."""

        def remove(self) -> None:
            """Refuse removal."""

            raise RuntimeError("handle stuck")

    session = binding_module._BindSession(bound, "call")
    runtime = binding_module._ArmedRuntime(bound, session)
    with pytest.raises(BindingRuntimeError) as excinfo, runtime:
        runtime._handles.append(_StuckHandle())
    assert excinfo.value.fields["code"] == "bind_cleanup_failed"
    assert bound._cleanup_verdict.startswith("failed")
    # the REAL hooks installed before the stuck one were still removed
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
