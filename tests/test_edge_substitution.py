"""L6 stage 3: edge substitution — storage-fork honesty, tripwires, save boundary.

Pins: replay-engine-only scoping (typed refusals), edge-provenance gating,
the three-tier storage decision (tier-(ii) store + stamps; tier-(iii)
capture-surface parity), convenience-field coherence (producer truth vs
consumer view), node-level flag untouched, the 4.3 positive invariant +
re-execution acceptance (verdict ``edge_intervention_boundary``, never
"exempted"), the skip-shaped-acceptance meta-test, the v7 persistence
boundary (level-exhaustive refusal, two-conjunct key, precedence over
``artifact_save_level_unsupported``, switch-on round-trip with the
pre-release marker), ``trace.edges``, EDGE algebra rows, and the OBSERVE
masked-read kwarg.
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.selection import SelectionError, edge_address_of
from torchlens.validation.core import _check_edge_intervention_boundary


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))) + 1.0)


def _capture():
    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(1, 1, 12, 12)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    return model, x, trace


@pytest.fixture(scope="module")
def capture():
    model, x, trace = _capture()
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _edge(trace, parent="relu_1_2"):
    return next(e for e in trace.edges if e.parent_label == parent)


def test_trace_edges_family_and_provenance_gate(capture):
    model, x, trace = capture
    labels = [(e.parent_label, e.child_label) for e in trace.edges]
    assert ("relu_1_2", "conv2d_2_3") in labels
    # identity-stable rows: same record objects across reads
    assert trace.edges[0] is trace.edges[0]

    torch.manual_seed(0)
    plain = tl.trace(_Net(), torch.randn(1, 1, 12, 12))
    with pytest.raises(SelectionError) as excinfo:
        _ = plain.edges
    assert excinfo.value.fields["code"] == "edge_provenance_unavailable"


def test_edge_algebra_rows(capture):
    model, x, trace = capture
    a = _edge(trace).__selection__()
    b = _edge(trace, parent="conv2d_2_3").__selection__()
    union = (a | b).resolve(trace)
    assert union.kind == "EDGE" and len(union) == 2
    # complement within the trace's dataflow edge family (well-defined universe)
    complement = ~union
    assert len(complement) == len(trace.edges) - 2
    # ACT x EDGE refuses via the closed matrix
    with pytest.raises(SelectionError) as excinfo:
        a | tl.units("relu_1_2", [(0, 0, 0, 0)])
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"


def _selection_for_kind_state(trace: tl.Trace, kind: str, state: str) -> Any:
    """Build one query selection for a closure-matrix kind/state cell."""

    if kind == "ACT":
        first = tl.units("relu_1_2", [(0, 0, 0, 0)])
        second = trace["conv2d_1_1"].__selection__()
    elif kind == "PARAM":
        first = tl.params("c1.weight")
        second = tl.params("c2.weight")
    else:
        first = _edge(trace).__selection__()
        second = _edge(trace, parent="conv2d_2_3").__selection__()
    if state == "nonempty":
        return first
    if state == "element_empty":
        return first - first
    if state == "no_sites":
        return first & second
    raise AssertionError(f"unknown closure-matrix state {state!r}")


def _selection_family(selection: Any) -> frozenset[Any]:
    """Return the touched-site family of one resolved selection."""

    return frozenset(entry.site_key for entry in selection)


@pytest.mark.parametrize("operator", ["or", "and", "sub"])
@pytest.mark.parametrize("left_state", ["no_sites", "element_empty", "nonempty"])
@pytest.mark.parametrize("right_state", ["no_sites", "element_empty", "nonempty"])
@pytest.mark.parametrize("level", ["query", "resolved"])
def test_closure_matrix_edge_totality_cells(
    capture: tuple[nn.Module, torch.Tensor, tl.Trace],
    operator: str,
    left_state: str,
    right_state: str,
    level: str,
) -> None:
    """Pin every EDGE same-kind operator/emptiness cell at both levels."""

    _model, _x, trace = capture
    left = _selection_for_kind_state(trace, "EDGE", left_state)
    right = _selection_for_kind_state(trace, "EDGE", right_state)
    if level == "resolved":
        left = left.resolve(trace)
        right = right.resolve(trace)
    composed = {
        "or": lambda: left | right,
        "and": lambda: left & right,
        "sub": lambda: left - right,
    }[operator]()
    resolved = composed.resolve(trace) if level == "query" else composed
    resolved_left = left.resolve(trace) if level == "query" else left
    resolved_right = right.resolve(trace) if level == "query" else right
    expected_family = {
        "or": _selection_family(resolved_left) | _selection_family(resolved_right),
        "and": _selection_family(resolved_left) & _selection_family(resolved_right),
        "sub": _selection_family(resolved_left),
    }[operator]
    assert resolved.kind == "EDGE"
    assert _selection_family(resolved) == expected_family


@pytest.mark.parametrize(
    "left_kind,right_kind",
    [
        ("ACT", "PARAM"),
        ("PARAM", "ACT"),
        ("ACT", "EDGE"),
        ("EDGE", "ACT"),
        ("PARAM", "EDGE"),
        ("EDGE", "PARAM"),
    ],
)
@pytest.mark.parametrize("operator", ["or", "and", "sub"])
@pytest.mark.parametrize("state", ["element_empty", "nonempty"])
@pytest.mark.parametrize("level", ["query", "resolved"])
def test_closure_matrix_all_mixed_kind_cells_refuse(
    capture: tuple[nn.Module, torch.Tensor, tl.Trace],
    left_kind: str,
    right_kind: str,
    operator: str,
    state: str,
    level: str,
) -> None:
    """Pin every ordered mixed-kind/operator/level/emptiness refusal cell."""

    _model, _x, trace = capture
    left = _selection_for_kind_state(trace, left_kind, state)
    right = _selection_for_kind_state(trace, right_kind, state)
    if level == "resolved":
        left = left.resolve(trace)
        right = right.resolve(trace)
    with pytest.raises(SelectionError) as excinfo:
        {
            "or": lambda: left | right,
            "and": lambda: left & right,
            "sub": lambda: left - right,
        }[operator]()
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"
    assert excinfo.value.fields["left_kind"] == left_kind
    assert excinfo.value.fields["right_kind"] == right_kind


def test_edge_substitution_storage_fork_honesty(capture):
    model, x, trace = capture
    edge = _edge(trace)
    fork = trace.fork()
    baseline_parent = trace["relu_1_2"].out.clone()
    baseline_child_args = [
        a.clone() if isinstance(a, torch.Tensor) else a
        for a in trace["conv2d_2_3"].ops[0].saved_args
    ]
    fork.do(edge.__selection__(), tl.zero_ablate())

    child = fork["conv2d_2_3"].ops[0]
    # only the child's consumption changed; producer truth intact
    assert torch.equal(fork["relu_1_2"].out, baseline_parent)
    assert torch.allclose(child.out, model.c2(torch.zeros_like(baseline_parent)))
    assert not torch.equal(fork["output_1"].out, trace["output_1"].out)

    # tier (ii): consumer view lives in the occurrence-granular store
    store_key = ("positional", (0,))
    assert store_key in child.edge_substitutions
    assert bool((child.edge_substitutions[store_key]["value"] == 0).all())
    assert child.edge_replacement_stamps[store_key]["verdict"] is True

    # tier (iii) capture-surface parity pin: no post-edit value reaches a
    # persisted capture field on the supported engine
    for saved, baseline in zip(child.saved_args, baseline_child_args, strict=True):
        if isinstance(saved, torch.Tensor):
            assert torch.equal(saved, baseline)
    parent_op = fork["relu_1_2"].ops[0]
    assert not (parent_op.out_versions_by_child or {})

    # per-edge intervened marker; node-level key correctly never fires
    edge_records = [r for r in child.interventions if r.edge_address]
    assert edge_records and edge_records[0].edge_address == edge_address_of(edge)
    assert not child.intervention_replaced

    # audit disclosure
    audit = fork.intervention_audit[-1]
    assert audit["kind"] == "EDGE" and audit["edges"][0]["parent"] == "relu_1_2"


def test_engine_scoping_refusals(capture):
    model, x, trace = capture
    selection = _edge(trace).__selection__()
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(
            selection,
            tl.zero_ablate(),
            model=model,
            x=x,
            intervention=tl.options.InterventionOptions(engine="rerun"),
        )
    assert excinfo.value.fields["code"] == "edge_intervention_engine_unsupported"
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(
            selection,
            tl.zero_ablate(),
            intervention=tl.options.InterventionOptions(engine="set_only"),
        )
    assert excinfo.value.fields["code"] == "edge_intervention_engine_unsupported"
    # auto with model+x resolves to rerun -> same refusal
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(selection, tl.zero_ablate(), model=model, x=x)
    assert excinfo.value.fields["code"] == "edge_intervention_engine_unsupported"


def _identity_edge_fork(trace):
    """Edge-substitute the edge with its OWN consumed value (identity)."""

    edge = _edge(trace)
    consumed = trace["relu_1_2"].out.clone()
    fork = trace.fork()
    fork.do(edge.__selection__(), consumed)
    return fork


def test_validation_boundary_accepts_corroborated_identity(capture):
    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    child = fork["conv2d_2_3"].ops[0]
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None
    assert verdict.decision == "edge_intervention_boundary"  # DISTINCT term, not "exempted"
    assert not verdict.failed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fork.validate_forward_pass(model(x)) is True


def test_validation_positive_invariant_strip_fire_record(capture):
    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    child = fork["conv2d_2_3"].ops[0]
    child._internal_set("interventions", [r for r in child.interventions if not r.edge_address])
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None and verdict.failed
    assert verdict.reason == "edge_substitution_uncorroborated"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fork.validate_forward_pass(model(x)) is False


def test_validation_positive_invariant_forge_without_stamp(capture):
    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    child = fork["conv2d_2_3"].ops[0]
    child._internal_set("edge_replacement_stamps", {})
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None and verdict.failed
    assert verdict.reason == "edge_substitution_uncorroborated"


def test_validation_captured_native_via_capture_surface_fails(capture: Any) -> None:
    """A substituted value forged into capture truth cannot validate green."""

    model, x, trace = capture
    fork = trace.fork()
    edge = _edge(trace)
    fork.do(edge.__selection__(), tl.zero_ablate())
    child = fork["conv2d_2_3"].ops[0]
    parent = fork["relu_1_2"].ops[0]
    child._internal_set("edge_substitutions", {})
    child._internal_set("edge_replacement_stamps", {})
    child._internal_set(
        "interventions", [record for record in child.interventions if not record.edge_address]
    )
    # Present the substituted value as capture-native in ONE persisted
    # consumed-value surface. The untouched saved_args twin contradicts it.
    parent._internal_set("out_versions_by_child", {child.label: torch.zeros_like(parent.out)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fork.validate_forward_pass(model(x)) is False


def test_validation_skip_shaped_acceptance_meta_test(capture):
    """A WRONG stored child output under a corroborated entry FAILS: the
    boundary is a DIFFERENT check, never NO check."""

    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    child = fork["conv2d_2_3"].ops[0]
    child._internal_set("out", child.out + 5.0)
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None and verdict.failed
    assert verdict.reason == "edge_boundary_reexecution_mismatch"


def test_edge_save_refusal_refires_if_schema_regresses(capture, tmp_path, monkeypatch):
    """The erasure-prevention invariant survives the bump as a tripwire.

    tlspec v8 persists the edge carriers, so ordinary saves proceed; the
    guard predicate keys on the ACTIVE policy, and a schema regression that
    re-drops Op.edge_substitutions must re-fire the typed refusal on every
    public level rather than silently erasing edge provenance.
    """

    from dataclasses import replace

    from torchlens._io import FieldPolicy
    from torchlens.data_classes.op import Op

    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    entry = Op.FIELD_POLICY["edge_substitutions"]
    monkeypatch.setitem(
        Op.FIELD_POLICY, "edge_substitutions", replace(entry, portable_policy=FieldPolicy.DROP)
    )
    for index, level in enumerate(("audit", "executable_with_callables", "portable", "runnable")):
        with pytest.raises(Exception) as excinfo:
            tl.save(fork, tmp_path / f"edge_{index}.tlspec", level=level)
        assert getattr(excinfo.value, "fields", {}).get("code") == (
            "edge_intervention_save_unsupported"
        ), level
    # non-edge saves are untouched even under the regressed schema
    tl.save(trace, tmp_path / "clean.tlspec", level="audit")


def test_edge_carriers_persist_on_plain_v8_round_trip(capture, tmp_path):
    """tlspec v8: the occurrence carriers persist on a PLAIN save/load, with
    the store's payload materializing back into REAL tensors -- presence
    alone is a skip-shaped acceptance (prebump lane finding: rehydration
    once handed back dead BlobRefs here). No pre-release marker rides the
    artifact."""

    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    path = tmp_path / "edge_plain.tlspec"
    tl.save(fork, path, level="portable")
    loaded = tl.load(path)
    child = loaded["conv2d_2_3"].ops[0]
    assert child.edge_substitutions and child.edge_replacement_stamps
    live_child = fork["conv2d_2_3"].ops[0]
    for key, entry in child.edge_substitutions.items():
        assert isinstance(entry["value"], torch.Tensor)
        assert torch.equal(entry["value"], live_child.edge_substitutions[key]["value"])


def test_forced_bundle_without_edge_corroboration_fails_validation(
    capture: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypassing the save guard cannot manufacture a validation-green bundle.

    The guard matters in the REGRESSED-schema world (tlspec v8 persists the
    edge carriers, so an un-bypassed ordinary save now simply keeps the
    provenance): force the carrier policy back to DROP AND bypass the guard,
    then prove the resulting provenance-free artifact still cannot pass the
    divergence oracle.
    """

    from dataclasses import replace

    from torchlens._io import FieldPolicy
    from torchlens.data_classes.op import Op

    model, x, trace = capture
    fork = trace.fork()
    fork.do(_edge(trace).__selection__(), tl.zero_ablate())
    bundle_module = importlib.import_module("torchlens._io.bundle")
    monkeypatch.setattr(bundle_module, "_refuse_edge_intervened_save", lambda _trace: None)
    for field_name in ("edge_substitutions", "edge_replacement_stamps"):
        monkeypatch.setitem(
            Op.FIELD_POLICY,
            field_name,
            replace(Op.FIELD_POLICY[field_name], portable_policy=FieldPolicy.DROP),
        )
    monkeypatch.setitem(Op.PORTABLE_STATE_SPEC, "edge_substitutions", FieldPolicy.DROP)
    monkeypatch.setitem(Op.PORTABLE_STATE_SPEC, "edge_replacement_stamps", FieldPolicy.DROP)
    path = tmp_path / "forced_edge.tlspec"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tl.save(fork, path, level="executable_with_callables")
    loaded = tl.load(path)
    loaded_child = loaded["conv2d_2_3"].ops[0]
    assert not loaded_child.edge_substitutions
    assert not loaded_child.edge_replacement_stamps

    # Trace func callables are DROP at this schema level. Reattach only the
    # source callables so the forced artifact reaches the divergence oracle.
    for loaded_op in loaded.layer_list:
        source_op = fork[loaded_op.layer_label].ops[0]
        loaded_op._internal_set("func", source_op.func)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert loaded.validate_forward_pass(model(x)) is False


def test_tap_masked_values(capture):
    model, x, trace = capture
    selection = tl.units("relu_1_2", [(0, 0, 1, 1), (0, 1, 2, 2)]).resolve(trace)
    observer = tl.tap(selection)
    fork = trace.fork()
    fork.attach_hooks("relu_1_2", observer, confirm_mutation=True)
    fork.push()
    full = observer.values()
    assert full and tuple(full[0].shape) == tuple(trace["relu_1_2"].out.shape)
    masked = observer.values(masked=True)
    assert masked[0].numel() == 2  # the two selected elements
    # stored mask immutability discipline: fresh copies each read
    assert observer.values(masked=True)[0] is not masked[0]
