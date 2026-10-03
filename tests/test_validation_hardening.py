"""Round-26 W3 validation-tripwire hardening tests.

The round-26 meta-audit proved the validation tripwire BLESSED three real
capture-bug classes (false negatives) and wrongly FAILED two correct-model
spellings (false positives). Each false-negative test here is a mutation
proof: it constructs the exact bug class on a real capture, asserts the
baseline (unmutated) trace passes, and asserts the corrupted trace now FAILS
-- where before this hardening it passed. Each false-positive test asserts
the correct model now validates while the tripwire provably stays armed.

Covered findings:

* W3-1: a dropped parent edge in a diamond (parent keeps another child, child
  keeps another parent) passed replay AND invariants -- the r22 argpos bug
  class with no regression net. Closed by the inverse orphan-arg sweep.
* W3-2: a ``func=None`` / ``func_name="intervention_replacement"`` op in a
  PLAIN capture passed because the exemptions trusted per-op attributes the
  placeholder synthesizer itself writes (2026-06-02 lesson not armed). Closed
  by the trace-level replacement-event ledger.
* W3-3: three of four graph_topology "flag checks" were property-vs-property
  tautologies that could never fail. Replaced by a real independent
  arg-map/graph cross-check; the stored has_children check stays.
* W3-4: a silently dropped ``torch.ops.aten.*`` op yielded a disconnected
  graph that ``Trace.validate_forward_pass`` blessed. Closed by the hardened
  ``graph_connectivity`` invariant (a demonstrated tensor consumer cannot be
  an internal source) plus census-outcome recording.
* W3-5 / F2: zero-WEIGHTED terms (``x + 0.0 * x.sum()``) and in-place zero
  masking (``view.mul_(0.0)``) wrongly failed ``perturbation_insensitive``
  while the zero-TENSOR twin passed. Closed by extending the multiplicative
  zero-annihilator proof to every multiplication spelling.
"""

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation import validate_forward_pass
from torchlens.validation.core import validate_saved_outs
from torchlens.validation.exemptions import _multiplicative_zero_annihilator_decision
from torchlens.validation.invariants import (
    MetadataInvariantError,
    check_metadata_invariants,
)


def _capture(model: nn.Module, x: torch.Tensor, seed: int = 0):
    """Capture a trace the way validate_forward_pass does, plus ground truth."""

    torch.manual_seed(seed)
    ground_truth = model(x)
    torch.manual_seed(seed)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save="all", save_arg_values=True, random_seed=seed),
    )
    return trace, ground_truth


def _plant_forged_intervention_edit(trace, event):
    """Splice a forged intervention edit into a FINISHED trace's journal.

    The ordinary writer (``append_intervention``) now refuses on a sealed
    journal (finding B1-15), which is a real boundary defense and not the
    property these tests pin: the point is that a forged edit stays INERT even
    when it reaches the lane. So the forger bypasses the single writer and also
    fakes its bookkeeping -- a writer-plausible ``seq`` plus the matching
    counter bump -- leaving the journal's seq invariants (stamped, monotone,
    unique, counter-bounded, dense) satisfied. The forgery must therefore be
    refused for the RIGHT reason: no causal binding.

    Parameters
    ----------
    trace:
        Finished trace whose journal is being tampered with.
    event:
        Forged ``InterventionAppliedEvent``.
    """

    events = trace._capture_events
    object.__setattr__(event, "seq", events.event_seq + 1)
    events.event_seq += 1
    events.intervention_events.append(event)


def _quiet_validate(model: nn.Module, x) -> bool:
    """Public-path validation with provenance warnings silenced."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate_forward_pass(model, x))


# ---------------------------------------------------------------------------
# W3-1: dropped parent edge in a diamond (inverse orphan-arg sweep)
# ---------------------------------------------------------------------------


class _Diamond(nn.Module):
    """p has two children; c1 has two parents -- a single edge can hide."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x * 2.0
        c1 = p + x
        c2 = torch.relu(p)
        return c1 + c2


def _drop_diamond_edge(trace) -> None:
    """Consistently drop the p->c1 edge exactly as a capture bug would."""

    p_op = [op for op in trace.layer_list if op.func_name == "__mul__"][0]
    c1_op = [op for op in trace.layer_list if op.func_name == "__add__"][0]
    p_labels = {p_op.label, p_op.layer_label}
    c1_labels = {c1_op.label, c1_op.layer_label}
    assert p_op.layer_label in c1_op.parents

    c1_op.parents = [v for v in c1_op.parents if v not in p_labels]
    p_op.children = [v for v in p_op.children if v not in c1_labels]
    for domain in ("args", "kwargs"):
        positions = c1_op.parent_arg_positions.get(domain, {})
        for key in [k for k, v in positions.items() if v in p_labels]:
            del positions[key]
    for holder in (c1_op, p_op):
        records = getattr(holder, "_edge_uses", None)
        if records:
            holder._edge_uses = [
                record
                for record in records
                if not (
                    getattr(record, "parent_label", None) in p_labels
                    and getattr(record, "child_label", None) in c1_labels
                )
            ]
    for key in [k for k in list(p_op.out_versions_by_child) if k in c1_labels]:
        del p_op.out_versions_by_child[key]
    p_op.has_children = len(p_op.children) > 0


def test_w31_diamond_baseline_passes() -> None:
    """Mutation-proof baseline: the unmutated diamond validates cleanly."""

    trace, ground_truth = _capture(_Diamond(), torch.randn(3, 4))
    status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
    assert status.state == "passed"
    assert check_metadata_invariants(trace) is True


def test_w31_dropped_parent_edge_now_fails_replay() -> None:
    """FAIL-AFTER-WHERE-PASSED-BEFORE: the dropped diamond edge is caught.

    Before this hardening the mutated trace reported ``state=passed`` (the
    round-26 executed repro), because replay validates VALUES from saved args
    and never inspects the provenance of an unattributed slot. The inverse
    orphan-arg sweep now identifies the dropped parent by value.
    """

    trace, ground_truth = _capture(_Diamond(), torch.randn(3, 4))
    _drop_diamond_edge(trace)
    status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
    assert status.state == "failed"
    assert any(decision.get("reason") == "unattributed_tensor_arg" for decision in status.decisions)


# ---------------------------------------------------------------------------
# W3-2: plain-capture placeholder must fail without ledger evidence
# ---------------------------------------------------------------------------


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x)) + 1.0


@pytest.mark.smoke
def test_w32_forged_placeholder_in_plain_capture_fails() -> None:
    """FAIL-AFTER-WHERE-PASSED-BEFORE: the 2026-06-02 lesson is armed.

    An op stamped with exactly the attributes the placeholder synthesizer
    writes (``func=None``, ``func_name='intervention_replacement'``,
    ``intervention_replaced=True``) passed replay AND invariants on a PLAIN
    capture before this hardening. With no replacement event in the
    trace-level ledger, both nets must now fail it.
    """

    trace, ground_truth = _capture(_Tiny(), torch.randn(3, 4))
    relu_op = [op for op in trace.layer_list if op.func_name == "relu"][0]
    object.__setattr__(relu_op, "func", None)
    object.__setattr__(relu_op, "func_name", "intervention_replacement")
    object.__setattr__(relu_op, "intervention_replaced", True)
    # No journal edit record exists for a plain capture: the forged per-op
    # stamps below have no corroborating InterventionAppliedEvent.
    assert not getattr(getattr(trace, "_capture_events", None), "intervention_events", ())

    status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
    assert status.state == "failed"
    with pytest.raises(MetadataInvariantError) as exc_info:
        check_metadata_invariants(trace)
    assert exc_info.value.check_name == "op_log_fields"


def test_w32_genuine_raw_hook_replacement_still_validates() -> None:
    """Control (no new FP): a genuine raw output-replacement hook passes.

    The replacement boundary is minted into the ledger by ``wrapped_hook``
    (the frame that directly observed the user hook returning a new object),
    so the narrow exemption still applies to genuine user interventions.
    """

    def raw_hook(module, inputs, output):  # type: ignore[no-untyped-def]
        return torch.ops.aten.mul.Tensor(output, torch.tensor(0.5))

    model = _Tiny()
    model.lin.register_forward_hook(raw_hook)
    assert _quiet_validate(model, torch.randn(3, 4)) is True


def test_w32_genuine_intervene_capture_keeps_exemption() -> None:
    """Control (no new FP): a live-fire intervention keeps its exemption.

    Func-site live-fire replacements are corroborated by the push/rerun
    fallback (a hook-minted ``replaced=True`` FireRecord on the op PLUS the
    trace-level intervention spec) -- evidence a plain-capture placeholder
    can never carry together.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    x = torch.randn(3, 4)
    trace = tl.trace(
        model,
        x,
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    replaced_ops = [op for op in trace.layer_list if getattr(op, "intervention_replaced", False)]
    assert replaced_ops, "zero_ablate must stamp its site"
    assert getattr(trace, "_intervention_spec", None) is not None
    assert any(
        getattr(record, "replaced", False)
        for op in replaced_ops
        for record in (op.interventions or [])
    )
    assert check_metadata_invariants(trace) is True
    ground_truth = [trace[label].out for label in trace.output_layers]
    status = validate_saved_outs(trace, ground_truth, validate_metadata=False)
    assert status.state == "passed"


def test_w32_stale_fire_leak_in_plain_capture_stays_refused() -> None:
    """Armed-proof: fallback evidence requires BOTH signals, not either.

    An op carrying a leaked ``replaced=True`` FireRecord on a PLAIN capture
    (no trace-level intervention spec) must stay refused -- the cross-trace
    ``_tl_live_fire_results`` leak cannot launder a placeholder.
    """

    from torchlens.intervention.types import FireRecord
    from torchlens.validation.invariants import (
        _intervention_spec_is_armed,
        op_has_genuine_replacement_evidence,
    )

    trace, _ground_truth = _capture(_Tiny(), torch.randn(3, 4))
    relu_op = [op for op in trace.layer_list if op.func_name == "relu"][0]
    relu_op.interventions.append(FireRecord(target_label=relu_op.label, replaced=True))
    # A plain capture may own an EMPTY InterventionSpec object; that must not
    # count as an armed spec for the fallback.
    assert _intervention_spec_is_armed(getattr(trace, "_intervention_spec", None)) is False
    assert op_has_genuine_replacement_evidence(relu_op, trace) is False


def test_raw_hook_recontainer_does_not_mint_replacement_evidence() -> None:
    """Returning an already-traced tensor only rewires the module boundary."""
    from torchlens.validation.core import _is_intentional_intervention_replacement

    model = nn.Linear(4, 4)
    handle = model.register_forward_hook(lambda _module, _args, out: (out,))
    try:
        trace = tl.trace(
            model,
            torch.randn(3, 4),
            capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
        )
    finally:
        handle.remove()

    linear_op = next(op for op in trace.layer_list if op.func_name == "linear")
    assert linear_op.intervention_replaced is False
    assert _is_intentional_intervention_replacement(linear_op) is False


def test_callable_op_needs_own_live_fire_to_use_replacement_exemption() -> None:
    """A raw-hook ledger stamp alone cannot exempt a replayable callable op."""
    from torchlens.backends.torch.model_prep import _note_replacement_event
    from torchlens.validation.core import _is_intentional_intervention_replacement

    trace, _ground_truth = _capture(_Tiny(), torch.randn(3, 4))
    relu_op = next(op for op in trace.layer_list if op.func_name == "relu")
    relu_op.intervention_replaced = True
    _note_replacement_event(trace, relu_op._label_raw)

    assert relu_op.func is not None
    assert _is_intentional_intervention_replacement(relu_op) is False


# ---------------------------------------------------------------------------
# W3-3: graph_topology checks must be real (no security theater)
# ---------------------------------------------------------------------------


def test_w33_arg_map_naming_non_parent_now_fails() -> None:
    """The new independent cross-check catches real topology corruption.

    Before this hardening the three property-vs-property flag comparisons
    could never fail, and an arg-map entry naming a non-parent op passed every
    invariant (only label RESOLUTION was checked). This is a genuine
    two-independent-structures inconsistency and must fail graph_topology.
    """

    trace, _ = _capture(_Diamond(), torch.randn(3, 4))
    relu_op = [op for op in trace.layer_list if op.func_name == "relu"][0]
    add_ops = [op for op in trace.layer_list if op.func_name == "__add__"]
    foreign = add_ops[-1].layer_label
    assert foreign not in relu_op.parents
    relu_op.parent_arg_positions["args"][1] = foreign

    with pytest.raises(MetadataInvariantError) as exc_info:
        check_metadata_invariants(trace)
    assert exc_info.value.check_name == "graph_topology"


def test_w33_has_children_corruption_still_caught() -> None:
    """The stored has_children flag check (the one real flag check) survives."""

    trace, _ = _capture(_Diamond(), torch.randn(3, 4))
    p_op = [op for op in trace.layer_list if op.func_name == "__mul__"][0]
    assert p_op.children
    p_op.has_children = False

    with pytest.raises(MetadataInvariantError) as exc_info:
        check_metadata_invariants(trace)
    assert exc_info.value.check_name == "graph_topology"


def test_w33_derived_flag_properties_are_read_only_truths() -> None:
    """Documented ground truth: the removed comparisons were tautologies."""

    trace, _ = _capture(_Diamond(), torch.randn(3, 4))
    op_type = type(trace.layer_list[1])
    for name in ("has_parents", "has_siblings", "has_co_parents"):
        assert isinstance(getattr(op_type, name), property)
    assert not isinstance(getattr(op_type, "has_children", None), property)


# ---------------------------------------------------------------------------
# B3R7-R05: stored edge labels must be canonical PRE-resolution
# ---------------------------------------------------------------------------


def _op_by_layer_label(trace, layer_label: str):
    """Return the raw op record (stored labels, no canonicalizing view)."""

    for record in trace.layer_list:
        if record.layer_label == layer_label:
            return record
    raise KeyError(layer_label)


def test_fuzzy_resolvable_corrupt_parent_label_fails_graph_topology() -> None:
    """FAIL-AFTER-WHERE-PASSED-BEFORE: a truncated stored parent label is caught.

    Before this hardening ``_check_graph_topology`` resolved every stored edge
    label through ``Trace.__getitem__``'s intelligent lookup, so a corrupted
    non-canonical spelling that stays uniquely substring-resolvable
    (``'input'`` for ``'input_1'``) was repaired in-flight and BLESSED -- while
    the stored (portable) edge table kept the spelling exact-match consumers
    cannot resolve. The membership test must fire before any resolution.
    """

    trace, _ = _capture(_Diamond(), torch.randn(3, 4))
    mul_op = _op_by_layer_label(trace, "mul_1_1")
    assert "input_1" in mul_op.parents
    mul_op.parents = tuple("input" if label == "input_1" else label for label in mul_op.parents)

    with pytest.raises(MetadataInvariantError) as exc_info:
        check_metadata_invariants(trace)
    assert exc_info.value.check_name == "graph_topology"
    assert "'input'" in str(exc_info.value)


def test_dangling_edge_label_reports_owning_contract() -> None:
    """A dangling stored child label fails as ``MetadataInvariantError``.

    Before this hardening the bare ``ml[c]`` lookup at the top of the loop
    leaked the user-facing ``InvalidArgumentError`` ("Did you mean ...?")
    instead of the owning contract's error: consolidated status aggregation
    branching on ``MetadataInvariantError`` misclassified real metadata
    corruption as an internal validation crash.
    """

    trace, _ = _capture(_Diamond(), torch.randn(3, 4))
    input_op = _op_by_layer_label(trace, "input_1")
    input_op.children = ("zz_nonexistent_9_9",) + tuple(input_op.children)

    with pytest.raises(MetadataInvariantError) as exc_info:
        check_metadata_invariants(trace)
    assert exc_info.value.check_name == "graph_topology"
    assert "zz_nonexistent_9_9" in str(exc_info.value)


class _Nested(nn.Module):
    """Sequential inside an attribute plus a top-level op between modules."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.l2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.sigmoid(self.a(x)))


def test_module_call_stack_is_single_fact_containment() -> None:
    """FAIL-AFTER-WHERE-PASSED-BEFORE: the persisted tri-fact field is gone.

    ``module_call_stack`` used to be three unrelated facts in one persisted
    slot: the FED call's address stack for module-input ops (``input_1``
    reported ``('a', 'a.0')`` while running in no module at all), its own
    containment ADDRESSES for module-output ops, and empty otherwise -- with
    ``module_call_depth`` computed from a different field. It is now the one
    glossary fact for every op: root-first ModuleCall labels active for this
    op, equal to ``modules``, with ``module_call_depth`` its depth.
    """

    trace, _ = _capture(_Nested(), torch.randn(2, 4))
    by_label = {op.layer_label: op for op in trace.layer_list}

    # Top-level ops (the model input, an op between modules) run in NO module
    # call: the stack must be empty, not the fed call's stack.
    assert tuple(by_label["input_1"].module_call_stack) == ()
    assert tuple(by_label["sigmoid_1_3"].module_call_stack) == ()
    # The fed-call fact lives on input_to_module_calls, un-conflated.
    assert tuple(by_label["input_1"].input_to_module_calls) == ("a:1", "a.0:1")
    assert tuple(by_label["sigmoid_1_3"].input_to_module_calls) == ("l2:1",)

    # Interior and module-output ops carry root-first ModuleCall LABELS
    # (pass-qualified), not bare addresses.
    assert tuple(by_label["linear_1_1"].module_call_stack) == ("a:1", "a.0:1")
    assert tuple(by_label["relu_1_2"].module_call_stack) == ("a:1", "a.1:1")

    # The three facts are tied on every op: stack == modules, depth == len.
    for op in trace.layer_list:
        assert tuple(op.module_call_stack) == tuple(op.modules)
        assert op.module_call_depth == len(op.module_call_stack)


def test_module_call_stack_corruption_fails_invariants() -> None:
    """The tri-fact tie is armed: a fed-call-style stack fails validation.

    Nothing used to check ``module_call_stack`` against ``modules`` or
    ``module_call_depth``, so the shipped inconsistency passed
    ``check_metadata_invariants`` on every plain capture.
    """

    trace, _ = _capture(_Nested(), torch.randn(2, 4))
    input_op = _op_by_layer_label(trace, "input_1")
    assert tuple(input_op.modules) == ()
    input_op.module_call_stack = ("a", "a.0")  # the retired fed-call fact

    with pytest.raises(MetadataInvariantError) as exc_info:
        check_metadata_invariants(trace)
    assert "module_call_stack" in str(exc_info.value)


def test_parent_arg_positions_foreign_domain_fails() -> None:
    """FAIL-AFTER-WHERE-PASSED-BEFORE: the arg map's top-level domain is closed.

    The per-entry checks iterate only the ``"args"``/``"kwargs"`` buckets, so a
    foreign top-level key (an op label used as a domain -- the B3R7-R05-2 plant
    shape) was never examined by ANY invariant and passed silently.
    """

    trace, _ = _capture(_Diamond(), torch.randn(3, 4))
    relu_op = [op for op in trace.layer_list if op.func_name == "relu"][0]
    relu_op.parent_arg_positions["nosuchop_9_9"] = {"args": [0]}

    with pytest.raises(MetadataInvariantError) as exc_info:
        check_metadata_invariants(trace)
    assert exc_info.value.check_name == "graph_topology"
    assert "nosuchop_9_9" in str(exc_info.value)


# ---------------------------------------------------------------------------
# W3-4: silently dropped aten op must not be blessed
# ---------------------------------------------------------------------------


class _RawAten(nn.Module):
    """A directly-dispatched aten op TorchLens silently fails to capture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.ops.aten.mul.Tensor(x, 2.0)
        return torch.relu(h)


def test_w34_dropped_aten_op_trace_method_now_fails() -> None:
    """FAIL-AFTER-WHERE-PASSED-BEFORE: the Trace-method entrypoint is armed.

    Before this hardening ``Trace.validate_forward_pass`` returned True for
    this capture (the round-26 executed repro): the input layer was
    disconnected, the consumer of the dropped op was stamped
    ``is_internal_source`` despite its callable func, and the census result
    was discarded. The hardened graph_connectivity invariant now refuses the
    demonstrated tensor consumer posing as a source.
    """

    model = _RawAten()
    x = torch.randn(3, 4)
    ground_truth = model(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            model,
            x,
            capture=CaptureOptions(layers_to_save="all", save_arg_values=True),
        )
    assert "relu_1_1" in trace.layer_labels and len(trace.layer_labels) == 3

    with pytest.raises(MetadataInvariantError) as exc_info:
        trace.validate_forward_pass([ground_truth])
    assert exc_info.value.check_name == "graph_connectivity"

    with pytest.raises(MetadataInvariantError):
        check_metadata_invariants(trace)


def test_w34_public_path_census_still_fails_dropped_aten_op() -> None:
    """The public census backstop stays armed (it already caught this)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert validate_forward_pass(_RawAten(), torch.randn(3, 4)) is False


def test_w34_census_unverified_outcome_is_recorded_not_discarded() -> None:
    """An unverified census outcome now leaves an auditable diagnostic."""

    from torchlens.validation.diagnostics import get_validation_diagnostics

    # Bind the model: replay reads parameters through TorchLens' weak model
    # reference, so a temporary model can be collected mid-test and turn the
    # replay into PostTraceParamUnavailable.
    model = _Tiny()
    trace, ground_truth = _capture(model, torch.randn(3, 4))
    status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
    assert status.state == "passed"
    diagnostics = get_validation_diagnostics(trace)
    assert any(
        diagnostic.check == "completeness_census_unverified"
        and diagnostic.extra.get("reason") == "dispatch_op_count_not_collected"
        for diagnostic in diagnostics
    )


def test_w34_census_validated_outcome_recorded_in_decisions() -> None:
    """A matched census is positive verdict evidence, not silence."""

    # Bind the model for the duration of the replay (see the sibling test).
    model = _Tiny()
    trace, ground_truth = _capture(model, torch.randn(3, 4))
    # Simulate the public path's collected census: counts agree.
    trace._validation_dispatch_op_count = 3
    trace._validation_captured_dispatchable_op_count = 3
    status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
    assert status.state == "passed"
    assert any(
        decision.get("reason") == "dispatch_op_count_matched" for decision in status.decisions
    )


def test_w34_factory_internal_sources_keep_exemption() -> None:
    """Control (no new FP): genuine func-bearing sources stay exempt.

    Factory ops (``torch.ones``, ``torch.randn``) are parentless with callable
    funcs; the narrowed connectivity rule keys on DEMONSTRATED tensor
    consumption (the capture witness), which factories never trigger.
    """

    class _Factories(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2 + torch.ones(x.shape[-1]) + torch.randn(1).abs() * 0

    assert _quiet_validate(_Factories(), torch.randn(3, 4)) is True


def test_w34_outside_tensor_consumer_with_traced_parent_still_validates() -> None:
    """Control (no new FP): the locked global-payload pattern stays green.

    A model consuming a genuinely outside tensor alongside a traced input has
    known partial provenance; it warns but validates (locked behavior).
    """

    outside = torch.randn(3, 4)

    class _GlobalTensor(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + outside

    assert _quiet_validate(_GlobalTensor(), torch.randn(3, 4)) is True


def test_w34_unused_input_model_still_validates() -> None:
    """Control (no new FP): a genuinely unused input is legal, not a drop."""

    class _UnusedInput(nn.Module):
        def forward(self, x: torch.Tensor, unused: torch.Tensor) -> torch.Tensor:
            return torch.relu(x * 3)

    assert _quiet_validate(_UnusedInput(), [torch.randn(3, 4), torch.randn(2)]) is True


# ---------------------------------------------------------------------------
# W3-5 / F2: zero-annihilator spellings (false-positive fixes, net stays armed)
# ---------------------------------------------------------------------------


def test_w35_literal_zero_coefficient_validates() -> None:
    """``x + 0.0 * x.sum()`` (disabled-loss spelling) must pass validation."""

    class _LiteralZeroCoef(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 0.0 * x.sum()

    assert _quiet_validate(_LiteralZeroCoef(), torch.randn(3, 4)) is True


def test_w35_zero_tensor_coefficient_still_validates() -> None:
    """The zero-TENSOR twin keeps passing (consistency, not a new class)."""

    class _ZeroTensorCoef(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + torch.zeros(()) * x.sum()

    assert _quiet_validate(_ZeroTensorCoef(), torch.randn(3, 4)) is True


def test_f2_inplace_zero_mask_validates() -> None:
    """``view.mul_(0.0)`` (in-place masking spelling) must pass validation."""

    class _InplaceZeroMask(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = x * 2.0
            view = y[0]
            view.mul_(0.0)
            return y + 1.0

    assert _quiet_validate(_InplaceZeroMask(), torch.randn(3, 4)) is True


def test_w35_nonzero_literal_never_exempted() -> None:
    """Armed-proof: the annihilator proof requires an exactly-zero co-arg.

    The exemption extension is spelling-only; a non-zero literal co-arg (a
    REAL sensitivity the perturbation net must demand) is never exempted, for
    any multiplication spelling.
    """

    class _FakeOp:
        saved_args = None

        def __init__(self, func_name: str, args_map: dict) -> None:
            self.func_name = func_name
            self.parent_arg_positions = {"args": args_map, "kwargs": {}}

    tensor_operand = torch.randn(3)
    for func_name in ("__mul__", "__rmul__", "mul", "mul_", "__imul__", "multiply"):
        layer = _FakeOp(func_name, {0: "parent_1"})
        decision = _multiplicative_zero_annihilator_decision(
            layer, ["parent_1"], (tensor_operand, 0.5)
        )
        assert decision.exempt is False
        zero_decision = _multiplicative_zero_annihilator_decision(
            layer, ["parent_1"], (tensor_operand, 0.0)
        )
        assert zero_decision.exempt is True


def test_f3_tuple_out_destination_keys_accepted() -> None:
    """F3: nested ``("out", idx)`` tuple-out keys are write-only destinations.

    ``torch.sort(x, out=(values, indices))`` records each tuple member at a
    nested ``("out", index)`` arg-map key (capture-side r26 fix). The out=
    destination exemption's own contract -- a perturbed parent whose ONLY
    positions are the write-only ``out=`` destination -- must accept those
    keys identically to the scalar ``"out"`` key, or tuple-out models
    false-alarm at perturbation. A parent that ALSO feeds a positional slot
    stays strict.
    """

    from torchlens.validation.core import _perturbed_parents_only_occupy_out_kwarg

    class _FakeOp:
        def __init__(self, args_map: dict, kwargs_map: dict) -> None:
            self.parent_arg_positions = {"args": args_map, "kwargs": kwargs_map}

    scalar_out = _FakeOp({0: "src_1"}, {"out": "dest_1"})
    assert _perturbed_parents_only_occupy_out_kwarg(scalar_out, ["dest_1"]) is True

    tuple_out = _FakeOp({0: "src_1"}, {("out", 0): "values_1", ("out", 1): "indices_1"})
    assert _perturbed_parents_only_occupy_out_kwarg(tuple_out, ["values_1"]) is True
    assert _perturbed_parents_only_occupy_out_kwarg(tuple_out, ["indices_1"]) is True
    assert _perturbed_parents_only_occupy_out_kwarg(tuple_out, ["values_1", "indices_1"]) is True

    # Armed: a parent that also feeds a positional (data) slot is never exempt.
    mixed = _FakeOp({0: "dest_1"}, {("out", 0): "dest_1"})
    assert _perturbed_parents_only_occupy_out_kwarg(mixed, ["dest_1"]) is False
    # Armed: non-out keyword slots are never exempt.
    other_kw = _FakeOp({}, {("mask", 0): "dest_1"})
    assert _perturbed_parents_only_occupy_out_kwarg(other_kw, ["dest_1"]) is False
    # Armed: the source operand of the out-op stays strict.
    assert _perturbed_parents_only_occupy_out_kwarg(tuple_out, ["src_1"]) is False


def test_f3_tuple_out_model_validates() -> None:
    """F3 end-to-end: a ``sort(x, out=(v, i))`` model validates cleanly."""

    class _TupleOut(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            values = torch.empty_like(x)
            indices = torch.empty(x.shape, dtype=torch.long)
            torch.sort(x, dim=-1, out=(values, indices))
            return values + 1.0

    assert _quiet_validate(_TupleOut(), torch.randn(3, 4)) is True


def test_w35_perturbation_net_still_fails_true_insensitivity() -> None:
    """Armed-proof: a genuinely wrong sensitivity claim still fails.

    Freeze an op's replay callable to return its saved out regardless of
    inputs (the recorded parent provably does not influence the output, and no
    zero-annihilator co-arg exists to prove why): the perturbation net must
    still fail it with ``perturbation_insensitive``.
    """

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    class _Mul(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x * 2.0)

    trace, _ground_truth = _capture(_Mul(), torch.randn(3, 4))
    relu_op = [op for op in trace.layer_list if op.func_name == "relu"][0]
    saved = relu_op.out.detach().clone()
    object.__setattr__(relu_op, "func", lambda *args, **kwargs: saved.clone())
    parent_label = relu_op.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, relu_op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "failed"
    assert result.reason == "perturbation_insensitive"


# ---------------------------------------------------------------------------
# r32-abc Fix B (round-34 Finding B): value-discretizing dead zone false-FAIL
# ---------------------------------------------------------------------------


class _ZeroConstIntCast(nn.Module):
    """``(x * 0).long().float() + x`` -- correct capture that used to false-FAIL.

    The all-zero float parent of ``.long()`` calibrated the perturbation draw
    to ``[-1, 1]`` and the ULP step retries to denormals, all inside integer
    truncation's dead zone, so the REAL mul -> long edge read as
    ``perturbation_insensitive`` with zero diagnostic output.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 0).long().float() + x


def test_zero_const_integer_cast_chain_validates_true() -> None:
    """PIN (round-34 Finding B): the correct capture must validate True.

    The unit-step retry crosses an integer boundary, proving the edge is real.
    """

    torch.manual_seed(0)
    assert _quiet_validate(_ZeroConstIntCast(), torch.randn(4, 5)) is True


def test_zero_index_gather_validates_true() -> None:
    """PIN (round-34 Finding B sibling): all-zero gather indices via a cast."""

    class _ZeroGather(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.gather(x, 1, (x.abs() * 0).long())

    torch.manual_seed(0)
    assert _quiet_validate(_ZeroGather(), torch.randn(4, 5)) is True


def test_unit_step_retry_still_fails_true_insensitivity() -> None:
    """Armed-proof: the SAME shape with a genuinely dead edge still fails.

    Freeze the ``long`` op's replay callable to return its saved out
    regardless of inputs: no draw -- wide, ULP step, or unit step -- can
    change the output, so the tripwire must still fire
    ``perturbation_insensitive``. This proves Fix B narrowed the false-FAIL
    without weakening real-bug detection.
    """

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    trace, _ground_truth = _capture(_ZeroConstIntCast(), torch.randn(4, 5))
    long_op = [op for op in trace.layer_list if op.func_name == "long"][0]
    saved = long_op.out.detach().clone()
    object.__setattr__(long_op, "func", lambda *args, **kwargs: saved.clone())
    parent_label = long_op.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, long_op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "failed"
    assert result.reason == "perturbation_insensitive"


# ---------------------------------------------------------------------------
# r33 R1/R2 (round-35): dead zones WIDER than one unit (geometric ladder)
# ---------------------------------------------------------------------------


class _WideBucketize(nn.Module):
    """``bucketize(x*0, [100])`` -- a 100-wide dead zone the unit step missed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(x * 0, torch.tensor([100.0])).float() + x


class _NegativeDecimalsRound(nn.Module):
    """``round(x*0, decimals=-2)`` quantizes to the nearest 100."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x * 0, decimals=-2) + x


def test_wide_bucketize_dead_zone_validates_true() -> None:
    """PIN (round-35 R2): a bin wider than one unit must still validate True.

    The all-zero parent sits 100 away from the only ``bucketize`` boundary,
    so the +-1.0 unit-step retry stayed inside the bin and the provably-real
    edge false-FAILed. The geometric magnitude ladder crosses the boundary.
    """

    torch.manual_seed(0)
    assert _quiet_validate(_WideBucketize(), torch.randn(4, 5)) is True


def test_negative_decimals_round_dead_zone_validates_true() -> None:
    """PIN (round-35 R2): ``round(decimals=-2)`` has a +-50 dead zone."""

    torch.manual_seed(0)
    assert _quiet_validate(_NegativeDecimalsRound(), torch.randn(4, 5)) is True


def test_geometric_ladder_still_fails_spurious_discretizing_edge() -> None:
    """Armed-proof: the ladder cannot bless a genuinely dead discretizing edge.

    Freeze the wide-bin ``bucketize`` op's replay callable to return its saved
    out regardless of inputs (the recorded parent provably does not influence
    the output): no rung of the geometric ladder -- +-1 through +-1e9 -- can
    change the output, so the tripwire must still fire
    ``perturbation_insensitive``. This proves the round-35 R2 extension only
    ADDED influence-detection attempts and introduced no false-VERIFIED path.
    """

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    trace, _ground_truth = _capture(_WideBucketize(), torch.randn(4, 5))
    bucketize_op = [op for op in trace.layer_list if op.func_name == "bucketize"][0]
    saved = bucketize_op.out.detach().clone()
    object.__setattr__(bucketize_op, "func", lambda *args, **kwargs: saved.clone())
    parent_label = bucketize_op.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, bucketize_op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "failed"
    assert result.reason == "perturbation_insensitive"


def test_geometric_ladder_scoped_to_value_discretizing_children() -> None:
    """PIN (round-35 R2 refinement): the ladder must not defeat fp swamping.

    The geometric magnitudes exist to cross TRUNCATION dead zones (finite
    quantization steps). An fp-SWAMPED child -- ``x + 1e8`` in fp32 -- is
    numerically inert at realistic step sizes, and escalating to +-1e3 would
    falsely "confirm" influence the actual forward never transmits. So the
    ladder runs only for value-discretizing children; everything else keeps
    the plain minimal/unit steps and the ``ulp_swamped_perturbation``
    exemption.
    """

    from torchlens.validation.core import _perturbation_retry_strategies

    bucketize_trace, _ = _capture(_WideBucketize(), torch.randn(4, 5))
    bucketize_op = [op for op in bucketize_trace.layer_list if op.func_name == "bucketize"][0]
    ladder = _perturbation_retry_strategies(bucketize_op)
    assert any(strategy.startswith("unit_step_up:") for strategy in ladder)

    class _SwampedAdd(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + torch.full_like(x, 1.0e8)

    swamped_trace, _ = _capture(
        _SwampedAdd(), torch.tensor([10000.0, 10001.0], dtype=torch.float32)
    )
    add_op = [op for op in swamped_trace.layer_list if op.func_name == "__add__"][0]
    assert _perturbation_retry_strategies(add_op) == [
        "step_up",
        "step_down",
        "unit_step_up",
        "unit_step_down",
    ]


def test_genuine_raw_hook_replacement_mints_journal_edit_and_validates() -> None:
    """A genuine untraceable replacement is a journal EDIT and validates.

    The user's raw ``register_forward_hook`` returns a tensor TorchLens never
    traced (built outside the forward), so capture synthesizes a functionless
    boundary op. The observing site appends an ``InterventionAppliedEvent``
    to the journal — the ONLY authority the functionless-op carve-out
    accepts — so metadata invariants pass for the genuine case while the
    forged plain-capture case (previous test) still fails.
    """

    from torchlens.ir.events import InterventionAppliedEvent
    from torchlens.validation.invariants import check_metadata_invariants

    model = _Tiny()
    injected = torch.ones(3, 4) * 7.0

    def raw_hook(
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        del module, args, output
        return injected

    handle = model.lin.register_forward_hook(raw_hook)
    try:
        trace, _ = _capture(model, torch.randn(3, 4))
    finally:
        handle.remove()

    boundary_layers = [
        layer for layer in trace.layer_list if layer.func_name == "intervention_replacement"
    ]
    assert boundary_layers, "the untraceable replacement must synthesize a boundary op"
    edits = [
        event
        for event in getattr(trace._capture_events, "intervention_events", ())
        if isinstance(event, InterventionAppliedEvent)
    ]
    assert edits, "the observing site must append a journal edit record"
    assert {event.kind for event in edits} == {"replaced"}
    assert all(event.seq >= 1 for event in edits)
    edited_labels = {event.label_raw for event in edits}
    assert {layer._label_raw for layer in boundary_layers} & edited_labels
    # The journal edit is the corroboration the carve-out requires: the
    # genuine case validates; no side ledger exists anymore. Instance-level
    # hasattr covers both the removed instance ledger and any class-level
    # reintroduction (the old type-level check missed instance attributes).
    assert not hasattr(trace, "_replacement_event_labels")
    check_metadata_invariants(trace)


@pytest.mark.smoke
def test_forged_journal_edit_does_not_bless_plain_placeholder() -> None:
    """FAIL-AFTER-WHERE-PASSED-BEFORE: a forged edit record stays inert.

    Review probe regression (extends the round-26 pin to the edit-record
    surface): appending a bare ``InterventionAppliedEvent`` through the
    ordinary journal writer used to bless the exact plain-capture placeholder
    the 2026-06-02 lesson exists to fail, because corroboration matched on
    the label string alone. An edit with no causal binding (no run token, no
    target event) must be refused by both nets.
    """

    import dataclasses

    from torchlens.ir.capture_events import SealedJournalAppendError
    from torchlens.ir.events import InterventionAppliedEvent

    trace, ground_truth = _capture(_Tiny(), torch.randn(3, 4))
    relu_op = [op for op in trace.layer_list if op.func_name == "relu"][0]
    object.__setattr__(relu_op, "func", None)
    object.__setattr__(relu_op, "func_name", "intervention_replacement")
    object.__setattr__(relu_op, "intervention_replaced", True)
    bare_edit = InterventionAppliedEvent(
        label_raw=relu_op._label_raw,
        kind="replaced",
        origin="raw_forward_hook",
        timestamp=2.0,
    )
    # B1-15: the ordinary writer no longer accepts this at all -- the journal is
    # sealed once the capture finished. The inertness pin below therefore uses
    # the harder attack that bypasses the writer entirely.
    with pytest.raises(SealedJournalAppendError):
        trace._capture_events.append_intervention(bare_edit)
    _plant_forged_intervention_edit(trace, bare_edit)

    status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
    assert status.state == "failed"
    with pytest.raises(MetadataInvariantError):
        check_metadata_invariants(trace)

    # A forged binding built from stale facts (a plausible-looking target seq
    # that names no real op event in this journal) is refused the same way.
    trace2, ground_truth2 = _capture(_Tiny(), torch.randn(3, 4))
    relu_op2 = [op for op in trace2.layer_list if op.func_name == "relu"][0]
    object.__setattr__(relu_op2, "func", None)
    object.__setattr__(relu_op2, "func_name", "intervention_replacement")
    object.__setattr__(relu_op2, "intervention_replaced", True)
    _plant_forged_intervention_edit(
        trace2,
        dataclasses.replace(
            InterventionAppliedEvent(
                label_raw=relu_op2._label_raw,
                kind="replaced",
                origin="raw_forward_hook",
                timestamp=2.0,
            ),
            run_token=trace2._capture_events.run_nonce,
            target_seq=10_000,
        ),
    )
    status2 = validate_saved_outs(trace2, [ground_truth2], validate_metadata=False)
    assert status2.state == "failed"
    with pytest.raises(MetadataInvariantError):
        check_metadata_invariants(trace2)


def test_cross_run_replayed_edit_does_not_bless_placeholder() -> None:
    """A GENUINE edit record replayed into another run's journal stays inert.

    The edit is causally bound to its own run (run token + target event
    instance); replaying it against a different capture -- even relabelled to
    name the victim op -- must not corroborate a placeholder there.
    """

    import dataclasses

    from torchlens.ir.events import InterventionAppliedEvent

    # Run A: a genuine raw-hook replacement mints a BOUND edit record.
    model_a = _Tiny()
    injected = torch.ones(3, 4) * 7.0
    handle = model_a.lin.register_forward_hook(lambda _m, _a, _o: injected)
    try:
        trace_a, _ = _capture(model_a, torch.randn(3, 4))
    finally:
        handle.remove()
    genuine_edits = [
        event
        for event in trace_a._capture_events.intervention_events
        if isinstance(event, InterventionAppliedEvent)
    ]
    assert genuine_edits, "run A must mint a genuine journal edit"
    assert all(
        event.run_token == trace_a._capture_events.run_nonce and event.target_seq
        for event in genuine_edits
    ), "genuine edits must be causally bound at the observation site"

    # Run B: plain capture; forge the placeholder and replay run A's edit.
    trace_b, ground_truth_b = _capture(_Tiny(), torch.randn(3, 4))
    relu_op = [op for op in trace_b.layer_list if op.func_name == "relu"][0]
    object.__setattr__(relu_op, "func", None)
    object.__setattr__(relu_op, "func_name", "intervention_replacement")
    object.__setattr__(relu_op, "intervention_replaced", True)
    _plant_forged_intervention_edit(
        trace_b, dataclasses.replace(genuine_edits[0], label_raw=relu_op._label_raw)
    )

    status = validate_saved_outs(trace_b, [ground_truth_b], validate_metadata=False)
    assert status.state == "failed"
    with pytest.raises(MetadataInvariantError):
        check_metadata_invariants(trace_b)


# ---------------------------------------------------------------------------
# R08 (b1-opus round-5): the discrete-bool blanket posthoc exemption
# ---------------------------------------------------------------------------


class _FarThresholdGate(nn.Module):
    """``(x > 1e30).float()`` -- no feasible perturbation crosses the threshold."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 1.0e30).float()


def test_bool_comparison_spurious_edge_now_fails() -> None:
    """Armed-proof (R08): a dead edge on a COMPARISON op is no longer blessed.

    Freeze the ``gt`` op's replay callable to return its saved bool output
    regardless of inputs -- the recorded parent provably does not influence
    the output. Pre-fix, ``dtype == torch.bool`` was a blanket posthoc pass,
    so this exact spurious-edge class was unfalsifiable by perturbation
    (red-capable: pre-fix this returns ``exempted``/``discrete_bool_output``).
    The threshold-straddle probe now proves no value influence and the
    tripwire fires.
    """

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    trace, _ground_truth = _capture(_FarThresholdGate(), torch.randn(3, 4))
    gt_op = [op for op in trace.layer_list if op.func_name == "__gt__"][0]
    saved = gt_op.out.detach().clone()
    object.__setattr__(gt_op, "func", lambda *args, **kwargs: saved.clone())
    parent_label = gt_op.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, gt_op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "failed"
    assert result.reason == "perturbation_insensitive"


def test_far_threshold_comparison_keeps_evidence_backed_exemption() -> None:
    """A REAL comparison edge no feasible perturbation can flip stays exempt,
    and the exemption now carries straddle-probe evidence instead of being a
    blanket dtype pass (red-capable: pre-fix the justification was None)."""

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    trace, _ground_truth = _capture(_FarThresholdGate(), torch.randn(3, 4))
    gt_op = [op for op in trace.layer_list if op.func_name == "__gt__"][0]
    parent_label = gt_op.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, gt_op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "exempted"
    assert result.reason == "discrete_bool_output"
    assert result.justification and "straddle" in result.justification


def test_isnan_edge_validates_via_nan_probe() -> None:
    """The NaN retry rung upgrades a real ``isnan`` edge from exempt to
    VALIDATED (red-capable: pre-fix the finite draws never flipped the output
    and the blanket bool exemption absorbed it)."""

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    class _NanGate(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.isnan(x).float()

    trace, _ground_truth = _capture(_NanGate(), torch.randn(3, 4))
    isnan_op = [op for op in trace.layer_list if op.func_name == "isnan"][0]
    parent_label = isnan_op.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, isnan_op.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "validated"
    assert result.reason == "perturbation_changed"


def test_bool_output_models_still_validate_true_end_to_end() -> None:
    """No false-fail regression: correct captures with bool ops validate True."""

    class _BoolMix(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate = (x > 0.5).float()
            nan_gate = torch.isnan(x).float()
            eq_gate = torch.eq(x, x.detach().clone() + 3.0).float()
            return gate + nan_gate + eq_gate

    torch.manual_seed(0)
    assert _quiet_validate(_BoolMix(), torch.randn(3, 4)) is True
    torch.manual_seed(0)
    assert _quiet_validate(_FarThresholdGate(), torch.randn(3, 4)) is True


class _PredicateZoo(nn.Module):
    """Bool-output NON-comparison ops: the R08-2 residual family.

    ``isnan``/``isfinite``/``logical_and``/``any``/``bitwise_not`` each get a
    perturbable float (or derived-bool) parent so the spurious-edge freeze
    harness and the honest controls run on real captured edges.
    """

    def __init__(self, func_name: str) -> None:
        super().__init__()
        self.func_name = func_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        if self.func_name == "isnan":
            return torch.isnan(h).float()
        if self.func_name == "isfinite":
            return torch.isfinite(h).float()
        if self.func_name == "logical_and":
            return torch.logical_and(h > 0, h < 1.0e30).float()
        if self.func_name == "any":
            return torch.any(h > 0).float()
        if self.func_name == "bitwise_not":
            return torch.bitwise_not(h > 0).float()
        raise AssertionError(self.func_name)


@pytest.mark.parametrize("func_name", ["isnan", "isfinite", "logical_and", "any", "bitwise_not"])
def test_bool_predicate_spurious_edge_now_fails(func_name: str) -> None:
    """Armed-proof (r7 R08-2): a dead edge on a NON-comparison bool op fails.

    Round-6 residual of the R08 comparison fix: freezing the replay callable
    of ``logical_and``/``isnan``/``isfinite``/``any``/``bitwise_not`` settled
    ``exempted``/``discrete_bool_output`` with an empty justification -- the
    ``probe is None`` fallback was still a blanket pass, so a capture bug
    that drops or invents a parent edge on the predicate family was
    unfalsifiable by perturbation. The substitution-battery probe now proves
    no value influence and the tripwire fires (red-capable: pre-fix every
    parametrization settles ``exempted``).
    """

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    trace, _ground_truth = _capture(_PredicateZoo(func_name), torch.randn(3, 4))
    target = [op for op in trace.layer_list if op.func_name == func_name][0]
    saved = target.out.detach().clone()
    object.__setattr__(target, "func", lambda *args, **kwargs: saved.clone())
    parent_label = target.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, target.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision == "failed", (func_name, result.decision, result.reason)
    assert result.reason == "perturbation_insensitive"


@pytest.mark.parametrize("func_name", ["isnan", "isfinite", "logical_and", "any", "bitwise_not"])
def test_bool_predicate_honest_edge_stays_green(func_name: str) -> None:
    """Honest control: the real edge validates (rungs) or exempts WITH
    battery evidence -- never the evidence-free blanket, never a failure."""

    from torchlens.validation.core import (
        _check_whether_func_on_saved_parents_yields_saved_tensor,
    )

    trace, _ground_truth = _capture(_PredicateZoo(func_name), torch.randn(3, 4))
    target = [op for op in trace.layer_list if op.func_name == func_name][0]
    parent_label = target.parents[0]

    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, target.label, perturb=True, layers_to_perturb=[parent_label]
    )
    assert result.decision in ("validated", "exempted"), (
        func_name,
        result.decision,
        result.reason,
    )
    if result.decision == "exempted" and result.reason == "discrete_bool_output":
        assert result.justification, (
            f"{func_name}: honest edge exempted through the evidence-free blanket"
        )


def test_validate_forward_pass_accepts_a_bare_ground_truth_tensor() -> None:
    """r7 b1-opus R08-3: ``validate_forward_pass(model(x))`` must not FAIL.

    A bare tensor was iterated as rows, so the arity check counted the BATCH
    as expected outputs and a byte-correct capture reported
    ``ground_truth_missing_out`` ("1 logged vs N expected") -- a validation
    FAILURE for a caller-arity slip. The bare tensor now normalizes to
    ``[tensor]`` (red-capable: pre-fix this asserts False).
    """

    model = _FarThresholdGate()
    x = torch.randn(4, 3)
    trace, ground_truth = _capture(model, x)
    assert trace.validate_forward_pass(ground_truth) is True
    assert trace.validate_forward_pass([ground_truth]) is True


class _InplaceMaskGate(nn.Module):
    """In-place bool op whose probe used to corrupt the retained operand."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x > 0
        gate = x < 1.0e30
        mask.logical_and_(gate)
        return mask.float()


def test_bool_predicate_probe_never_mutates_retained_saved_args() -> None:
    """r8 R08 (fable MH): the battery must execute against CLONES.

    The bool universe includes in-place ops (``logical_and_``); the probe
    executed the captured func against the record's retained ``saved_args``
    with only the probed slot substituted, so one probe run MUTATED the
    capture evidence in place and returned a verdict computed on the
    corrupted operand (red-capable: pre-fix the byte comparison fails).
    """

    from torchlens.validation.exemptions import _bool_predicate_influence_probe

    trace, _ground_truth = _capture(_InplaceMaskGate(), torch.randn(3, 4))
    target = [op for op in trace.layer_list if op.func_name == "logical_and_"][0]
    saved_args = list(target.saved_args or ())
    assert saved_args and isinstance(saved_args[0], torch.Tensor)
    snapshots = [
        arg.detach().clone() if isinstance(arg, torch.Tensor) else arg for arg in saved_args
    ]
    arg_positions = (getattr(target, "parent_arg_positions", None) or {}).get("args", {})
    assert arg_positions, "expected positional parent metadata on the in-place op"
    # Probe through each single-parent slot the metadata knows about.
    for position, parent_label in arg_positions.items():
        _bool_predicate_influence_probe(target, [parent_label])
        _ = position
    for index, snapshot in enumerate(snapshots):
        if isinstance(snapshot, torch.Tensor):
            assert torch.equal(saved_args[index], snapshot), (
                f"probe mutated retained saved_args[{index}] in place"
            )


def test_bool_predicate_probe_catch_is_typed() -> None:
    """r8 R08 (sol fault-injection): unexpected probe crashes PROPAGATE.

    The old blanket ``except Exception`` swallowed genuine capture-bug
    crashes into the evidence-free heuristic pass; benign substitute
    refusals (TypeError/ValueError/RuntimeError) still settle "cannot run".
    """

    from torchlens.validation.exemptions import _bool_predicate_influence_probe

    trace, _ground_truth = _capture(_PredicateZoo("isnan"), torch.randn(3, 4))
    target = [op for op in trace.layer_list if op.func_name == "isnan"][0]
    parent_label = target.parents[0]

    object.__setattr__(
        target, "func", lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("capture bug"))
    )
    with pytest.raises(KeyError, match="capture bug"):
        _bool_predicate_influence_probe(target, [parent_label])

    def _benign_refusal(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - test shim
        raise RuntimeError("substitute rejected")

    object.__setattr__(target, "func", _benign_refusal)
    assert _bool_predicate_influence_probe(target, [parent_label]) is None
