"""Tripwires for the postprocess dependency derivation (design-ppdag-v3).

The pipeline order is derived from declared contracts oriented by the frozen
``LEGACY_STEP_RANK`` and semantically pinned by the reason-bearing
``PINNED_ORDER_PAIRS`` corpus. These tests freeze the goldens, pin the day-1
findings by name, re-derive the DAG independently (spec-drives-code), and
pin both halves of the coordinated-reversal counterexample.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.postprocess import (
    LEGACY_STEP_RANK,
    PINNED_ORDER_PAIRS,
    POSTPROCESS_STEP_CONTRACTS,
    REGISTRY_ORDER,
    _executor,
    execution_order,
)
from torchlens.postprocess._contracts import (
    CAPTURE_BASELINE_COLUMNS,
    iter_corpus_violations,
)

pytestmark = pytest.mark.smoke


class _TinyModel(nn.Module):
    """Minimal linear+relu model for dynamic audit tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


# ---------------------------------------------------------------------------
# Frozen goldens
# ---------------------------------------------------------------------------

#: The frozen rank (key 1 of the direction authority). Editing it is a
#: reviewed semantic diff; this golden makes the edit loud.
RANK_GOLDEN = {
    "1": 10,
    "2": 20,
    "3": 30,
    "4": 40,
    "5": 50,
    "6": 60,
    "7": 70,
    "8": 80,
    "9": 90,
    "10": 100,
    "11": 110,
    "11.5": 115,
    "11.75": 118,
    "12": 120,
    "13": 130,
    "14": 140,
    "15": 150,
    "15.5": 155,
    "16": 160,
    "16.5": 165,
    "17": 170,
    "17.5": 175,
    "18": 180,
    "19": 190,
    "20": 200,
}

#: Multi-writer columns with their writers in rank order — the post-repair
#: golden (design-ppdag-v3 §4). Independently re-derived by both design
#: reviewers for the pre-repair rows; the repairs added: the step-11.75
#: save_activation family, steps 18/19 streaming columns, var_names (11.5),
#: equivalent_ops gaining step 3 (orphan-removal scrub, conditional), and
#: step 6's buffer-merge write family (B2). Step 3 deliberately does NOT
#: declare parents/children scrub writes: the flood-closure argument alone
#: is insufficient (the func-call-group expansion at graph_traversal.py
#: adds siblings without flooding, so a survivor CAN hold edges to
#: orphans — opus impl-review F3), but the trigger needs a multi-output
#: call sharing no parent with any flooded member, which standard torch
#: ops cannot produce, and the write audit fails loud if one ever does.
#: recurrent_ops cannot fire at 3: groups are not built until step 7.
MULTI_WRITER_GOLDEN = {
    "_edge_uses": ("1", "3", "6", "9"),
    "_layer_label_raw": ("1", "7"),
    "_param_logs": ("1", "15", "16"),
    "activation_memory": ("1", "11.75"),
    # fix/distributed-r3 R18: step 1 resets the synthetic output node's
    # clone-inherited annotations namespace (a returned collective boundary
    # result otherwise duplicated its portable payload + correlation key).
    "annotations": ("1", "11.75"),
    "args_template": ("3", "6", "9"),
    "atomic_module_call": ("1", "9"),
    "children": ("1", "6", "9"),
    "conditional_arm_children": ("3", "5", "6", "9"),
    "conditional_elif_children": ("3", "5", "6", "9"),
    "conditional_else_children": ("3", "5", "6", "9"),
    "conditional_entry_children": ("3", "5", "6", "9"),
    "conditional_then_children": ("3", "5", "6", "9"),
    "dtype": ("1", "11.75"),
    "equivalence_class": ("1", "7"),
    "equivalent_ops": ("1", "3", "6", "9"),
    "func": ("1", "6"),
    "func_name": ("1", "6"),
    "has_children": ("1", "6"),
    "has_input_ancestor": ("4", "6"),
    # r3settle e12aa996: the step-6 duplicate-buffer merge unions the removed
    # duplicate's child-direction reach into the survivor and re-derives the
    # ancestor cone's output reach (step 2 computed it from PRE-MERGE edges).
    "has_output_descendant": ("1", "2", "4", "6"),
    "input_ancestors": ("4", "6", "9"),
    "input_to_module_calls": ("1", "11"),
    "internal_source_ancestors": ("6", "9"),
    # SF-01 made step 1 a real ancestry writer (ingest-derived parents);
    # L4's D1 buffer-merge hardening added buffer_source's step-6 write.
    "buffer_source": ("6", "9"),
    "internal_source_parents": ("1", "6", "9"),
    "interventions": ("1", "3", "6", "9"),
    "is_buffer": ("1", "9"),
    "is_input": ("1", "9"),
    "is_output": ("1", "9"),
    "is_terminal_bool": ("3", "5"),
    "kwargs_template": ("3", "6", "9"),
    "module": ("1", "11"),
    # B3R7-R05-1: module_call_stack is the same containment fact as modules
    # (seeded from the modules facet at ingest, relabeled at step 11); step 1
    # writes the empty stack on synthetic output nodes.
    "module_call_stack": ("1", "11"),
    "modules": ("1", "11"),
    "num_passes": ("1", "7"),
    "out": ("1", "11.75", "19"),
    # Step 6 joined at r3settle e12aa996 (buffer-merge reach repair, above).
    "output_descendants": ("1", "2", "6", "9"),
    "output_of_module_calls": ("1", "11"),
    "parent_arg_positions": ("1", "6", "9"),
    "parent_params": ("1", "15"),
    "parents": ("1", "6", "9"),
    "pass_index": ("1", "7"),
    "recurrent_ops": ("1", "7", "9"),
    # b9 R71-1: step 1's output-node minting re-derives root_ancestors
    # (input_ancestors | internal_source_ancestors over the ONE parent) so the
    # non-exempt synthetic output row conforms to the armed ancestry closure
    # even when its direct parent is a parentless factory source.
    "root_ancestors": ("1", "6", "9"),
    "saved_args": ("1", "11.75"),
    "saved_kwargs": ("1", "11.75"),
    "shape": ("1", "11.75"),
    "step_index": ("8", "9"),
    "transformed_activation_memory": ("1", "11.75"),
    "transformed_out": ("1", "11.75", "19"),
    "transformed_out_dtype": ("1", "11.75"),
    "transformed_out_shape": ("1", "11.75"),
    "var_names": ("1", "11.5"),
}

#: The TOTAL placeholder-probe set (review note N6): the one laundering
#: channel neither anti-laundering guard touches is silently DECLARING a
#: probe — so the full set is a golden and growing it is a reviewed diff.
PROBES_GOLDEN = {
    "1": frozenset(("label", "layer_label", "out_ref")),
    "3": frozenset(
        (
            "conditional_arm_children",
            "conditional_elif_children",
            "conditional_else_children",
            "conditional_entry_children",
            "conditional_then_children",
            "layer_label",
            "out_ref",
            "recurrent_ops",
        )
    ),
    "6": frozenset(("layer_label", "recurrent_ops")),
    # Step 11 out_ref (F20 D-17 merge reconcile): the saved-summary refresh
    # reads ``out`` per saved op, and the accessor's lazy-materialization
    # guard checks ``out_ref`` whenever ``out`` is empty (lookback windows,
    # disk-only routes); in-pipeline refs attach only at step 18, so the
    # read always observes the placeholder and falls through — the same
    # reviewed shape as step 12's undecorate probe.
    "11": frozenset(("out_ref",)),
    "12": frozenset(("out_ref",)),
    # Step 16/18 grad-family rows (manifest-swap 2026-08-13): the provisional
    # baseline over-included the backward-phase grad channel; the projected
    # baseline excludes it, and the in-pipeline reads (which always observe
    # the step-0 constant seed — grads are written post-backward, outside
    # the pipeline) are reviewed observe-and-fall-through probes.
    "16": frozenset(("_grad_records",)),
    # Step 18 L6 tier-(ii) rows (tlspec v8 coordinated bump): the bump
    # retired the S3 pre-release registrations and made edge_substitutions
    # (BLOB_RECURSIVE) / edge_replacement_stamps (KEEP) permanent persisted
    # schema rows, so the whole-row portable scrub now sweeps them. Same
    # reviewed shape as the grad channel: in-pipeline both always hold the
    # step-0 empty-dict seed (tier-(ii) entries are written only by
    # session-time fork.do() on a FINISHED trace, outside the pipeline),
    # no postprocess step writes either column, and the derived order is
    # unchanged. The tlspec v9 entry-dark trio (episode_step /
    # injection_provenance / tl_authored_root) is the same reviewed shape:
    # their Phase-3 writers (F-EPISODE / F01 / F41) run outside this
    # pipeline, so the whole-row scrub always observes the step-0 None seed.
    "18": frozenset(
        (
            "_facets_cache",
            "_grad_records",
            "_pending_grad_blob_id",
            "_pending_transformed_grad_blob_id",
            "_projective_field_cache",
            "_receptive_field_cache",
            "edge_replacement_stamps",
            "edge_substitutions",
            "episode_step",
            "grad",
            "grad_dtype",
            "grad_fn",
            "grad_shape",
            "gradient_memory",
            "has_grad",
            "injection_provenance",
            "tl_authored_root",
            "transformed_grad",
            "transformed_grad_dtype",
            "transformed_grad_shape",
            "transformed_gradient_memory",
        )
    ),
}

#: Day-1 category-(c) findings, PINNED BY NAME (design-ppdag-v3 §2.4),
#: classified with guard 2's ledger (PINNED_NOOP_WRITERS) so a pinned
#: no-op writer can never discharge a read. Reported for root-cause;
#: NEVER silenced by widening the baseline. Any NEW finding fails this
#: test and is root-caused the same way.
PINNED_FINDINGS = {
    # THE day-1 finding, undiluted (opus impl-review §4 split the formerly
    # co-pinned layer_label off as the _label_for_reference_removal
    # fallback probe it actually is): orphan_records stores op.label as
    # DATA, but label's only writer is step 8 — every record ships the
    # None placeholder into a serialized public field
    # (FieldPolicy.BLOB_RECURSIVE, round-tripped by the tlspec suite).
    # Root-cause fix (record _label_raw instead) changes a persisted
    # artifact's bytes: outside this lane's byte-identity mandate,
    # deferred to JMT by name. NOT a probe — a data read into a persisted
    # field is never an observe-not-set.
    ("3", "label"),
}

#: Declared-but-never-observed writes with their named config-gated
#: exemptions (the Opus-4 phantom-declaration guard): each entry names WHY
#: the recording matrix cannot observe it. An exemption without a reason is
#: a laundering channel; removing the code path must remove the row.
PHANTOM_WRITE_EXEMPTIONS = {
    ("9", "args_template"): (
        "intervention-ready replay-template rename; observed on the "
        "closure-review intervention/observer suites"
    ),
    ("9", "kwargs_template"): "same as args_template",
    ("18", "grad_ref"): (
        "written only by the post-backward OUT-OF-PIPELINE deferred-grad "
        "streaming re-run of the same function body"
    ),
    ("6", "address"): (
        "the None-address recovery/anonymous fallback is code-real but no "
        "known capture path materializes a buffer row without a display "
        "address (every buffer source-logging call site records one, and "
        "step 0 resolves it through the registered pool, the "
        "equivalence-class recovery, and the recorded I/O address in turn; "
        "bounded empirical sweep: registered read, journal write, dynamic "
        "register_buffer, cooked recording, top-level read). Retires "
        "loudly the day a path produces one"
    ),
}

#: Declared-but-never-observed READS with their named exemptions — the
#: read-side mirror of the table above, same discipline: each entry names
#: WHY the recording matrix cannot observe it, and removing the code path
#: must remove the row. Asserted against the live matrix union as
#: EXPECTED_PHANTOM_READS in test_postprocess_enforcement.py.
PHANTOM_READ_EXEMPTIONS = {
    ("6", "equivalence_class"): (
        "read only on the None-address recovery branch (the buffer_ prefix "
        "strip), guarded by the same unreachability as the ('6', 'address') "
        "phantom write — the two rows retire together"
    ),
}


#: Seed-era machine-templated corpus reasons, grandfathered (opus
#: impl-review F4). The design sanctioned tooling for the one-time corpus
#: SEED, not for the shipped steady state: "editing an entry is THE
#: reviewed act that blesses a reorder" is hollow where the reason merely
#: restates the derivation. Named follow-up: rewrite these as prose. The
#: lint below refuses NEW templated reasons, so the set can only shrink —
#: remove an entry here when its pair gets real prose.
_TEMPLATED_REASON = re.compile(
    r"^step [\d.]+ (consumes/refines .* after step [\d.]+ writes"
    r"|depends on token:.* produced by step [\d.]+)$"
)
LEGACY_TEMPLATED_PAIRS = frozenset(
    (
        ("1", "2"),
        ("1", "4"),
        ("1", "5"),
        ("1", "8"),
        ("1", "9"),
        ("1", "10"),
        ("1", "11"),
        ("1", "11.5"),
        ("1", "11.75"),
        ("1", "12"),
        ("1", "15"),
        ("1", "15.5"),
        ("1", "16"),
        ("1", "16.5"),
        ("1", "17.5"),
        ("1", "18"),
        ("1", "19"),
        ("2", "3"),
        ("2", "4"),
        ("2", "5"),
        ("2", "9"),
        ("2", "15.5"),
        ("2", "18"),
        ("3", "4"),
        ("3", "7"),
        ("3", "8"),
        ("3", "9"),
        ("3", "10"),
        ("3", "11.5"),
        ("3", "11.75"),
        ("3", "12"),
        ("3", "15.5"),
        ("3", "16"),
        ("3", "17.5"),
        ("3", "18"),
        ("4", "5"),
        ("4", "6"),
        ("4", "9"),
        ("4", "15.5"),
        ("4", "18"),
        ("5", "9"),
        ("5", "15.5"),
        ("5", "18"),
        ("6", "8"),
        ("6", "10"),
        ("6", "11"),
        ("6", "11.5"),
        ("6", "17.5"),
        ("7", "9"),
        ("7", "11"),
        ("7", "11.75"),
        ("7", "15.5"),
        ("7", "16.5"),
        ("7", "18"),
        ("8", "10"),
        ("8", "11"),
        ("8", "11.75"),
        ("8", "12"),
        ("8", "15"),
        ("8", "15.5"),
        ("8", "16"),
        ("8", "16.5"),
        ("8", "18"),
        ("9", "10"),
        ("9", "11.5"),
        ("9", "11.75"),
        ("9", "15.5"),
        ("9", "16"),
        ("9", "16.5"),
        ("9", "17.5"),
        ("9", "18"),
        ("10", "16.5"),
        ("11", "15.5"),
        ("11", "16"),
        ("11", "16.5"),
        ("11", "18"),
        ("11.5", "18"),
        ("11.75", "12"),
        ("11.75", "13"),
        ("11.75", "15.5"),
        ("11.75", "16"),
        ("11.75", "18"),
        ("11.75", "19"),
        ("12", "13"),
        ("12", "18"),
        ("15", "15.5"),
        ("15", "16"),
        ("15", "18"),
        ("15", "20"),
        ("15.5", "18"),
        ("16", "18"),
        ("16", "20"),
        ("16.5", "18"),
        ("17", "18"),
        ("18", "19"),
    )
)


# ---------------------------------------------------------------------------
# Structural checks and goldens
# ---------------------------------------------------------------------------


def test_import_checks_pass_on_real_artifacts() -> None:
    """The 7.1 family and K1 hold on the shipped artifacts."""

    assert list(_executor._iter_structural_violations()) == []
    assert iter_corpus_violations(dict(LEGACY_STEP_RANK)) == []


def test_rank_golden() -> None:
    """LEGACY_STEP_RANK is frozen; editing it is a reviewed semantic diff."""

    assert dict(LEGACY_STEP_RANK) == RANK_GOLDEN


def test_derived_order_reproduces_registry() -> None:
    """R2: rank-keyed Kahn reproduces the hand order exactly (day-1 identity)."""

    assert execution_order() == REGISTRY_ORDER


def test_registry_contract_bijection() -> None:
    """Every registry step has a contract and vice versa (step 0 fenced)."""

    assert set(REGISTRY_ORDER) == set(POSTPROCESS_STEP_CONTRACTS) - {"0"}
    for contract in POSTPROCESS_STEP_CONTRACTS.values():
        assert contract.writes is not None
        assert contract.reads is not None


def test_multi_writer_golden() -> None:
    """The multi-writer column table is frozen (reviewed-diff to change)."""

    writers: dict[str, list[str]] = {}
    for step, contract in POSTPROCESS_STEP_CONTRACTS.items():
        if step == "0":
            continue
        for column in contract.writes:
            writers.setdefault(column, []).append(step)
    derived = {
        column: tuple(sorted(steps, key=LEGACY_STEP_RANK.__getitem__))
        for column, steps in writers.items()
        if len(steps) > 1
    }
    assert derived == MULTI_WRITER_GOLDEN


def test_probes_golden() -> None:
    """The total placeholder-probe set is frozen (review note N6)."""

    derived = {
        step: contract.placeholder_probes
        for step, contract in POSTPROCESS_STEP_CONTRACTS.items()
        if contract.placeholder_probes
    }
    assert derived == PROBES_GOLDEN


def test_read_findings_pinned_by_name() -> None:
    """Category-(c) findings are exactly the pinned set; a NEW one fails.

    Classification runs with guard 2's ledger (the reviewed no-op-writer
    table): a writer that is never content-effective on any matrix axis
    cannot discharge a read, so the historical laundering path — a pinned
    no-op self-write or no-op step-1 write silently blessing a
    read-before-write — stays closed (opus impl-review B1).
    """

    from test_postprocess_enforcement import PINNED_NOOP_WRITERS

    findings = {
        key
        for key, category in _executor.classify_declared_reads(PINNED_NOOP_WRITERS).items()
        if category == "finding"
    }
    assert findings == PINNED_FINDINGS


def test_noop_writer_cannot_discharge_reads() -> None:
    """B1 regression: un-probing a laundered read resurfaces it as a finding.

    Step 3 reads recurrent_ops; its only lower-rank writer is step 1, whose
    recurrent_ops write is a pinned permanent no-op (output-row placeholder
    init). The read is probe-blessed today; if the probe is ever dropped,
    the ledger-wired classifier must classify FINDING — never discharge
    through the no-op writer (the pre-fix classifier said earlier_writer
    here). The same holds for step 3's no-op self-writes of the
    conditional child views.
    """

    from test_postprocess_enforcement import PINNED_NOOP_WRITERS

    from torchlens.postprocess import PostprocessStepContract

    original = POSTPROCESS_STEP_CONTRACTS["3"]
    unblessed = PostprocessStepContract(
        original.step,
        original.name,
        original.contract,
        writes=original.writes,
        reads=original.reads,
        placeholder_probes=frozenset(("out_ref",)),
        row_effects=original.row_effects,
        trace_state=original.trace_state,
    )
    contracts = dict(POSTPROCESS_STEP_CONTRACTS)
    contracts["3"] = unblessed
    import unittest.mock as mock

    with mock.patch.object(
        _executor,
        "_registry_contracts",
        lambda: {s: c for s, c in contracts.items() if s != "0"},
    ):
        with_ledger = _executor.classify_declared_reads(PINNED_NOOP_WRITERS)
        without_ledger = _executor.classify_declared_reads()
    for key in (
        ("3", "recurrent_ops"),
        ("3", "conditional_arm_children"),
        ("3", "conditional_elif_children"),
        ("3", "conditional_else_children"),
        ("3", "conditional_entry_children"),
        ("3", "conditional_then_children"),
    ):
        assert with_ledger[key] == "finding", key
    # The structural (ledger-less) classifier is the lenient pre-fix view:
    # it still discharges the recurrent_ops read through step 1 — which is
    # exactly why the pinned-findings test always passes the ledger.
    assert without_ledger[("3", "recurrent_ops")] == "earlier_writer"


def test_phantom_write_exemptions_are_exact() -> None:
    """Every phantom-exemption row names a real declared write (or read).

    The full observed-vs-declared diff runs over the recording matrix (the
    env-gated enforcement leg); this fast check keeps the exemption tables
    from referencing declarations that no longer exist.
    """

    for (step, column), reason in PHANTOM_WRITE_EXEMPTIONS.items():
        assert column in POSTPROCESS_STEP_CONTRACTS[step].writes, (step, column)
        assert reason
    for (step, column), reason in PHANTOM_READ_EXEMPTIONS.items():
        assert column in POSTPROCESS_STEP_CONTRACTS[step].reads, (step, column)
        assert reason


def test_guard2_ledger_static_mirror() -> None:
    """Smoke-tier half of the guard-2 mirror (closure-review nit).

    The full runtime no-op/phantom reports run only in the heavy-marked
    matrix union test, so between heavy runs the static tables could drift
    against the contracts without anything going red in the smoke tier.
    This pins the static consistency: every guard-2 ledger row names a real
    declared write of an existing step, the phantom tables in the two test
    files mirror exactly, and a column cannot be both declared-never-
    observed (phantom) and observed-never-effective (no-op ledger).
    """

    from test_postprocess_enforcement import (
        EXPECTED_PHANTOM_READS,
        EXPECTED_PHANTOM_WRITES,
        PINNED_NOOP_WRITERS,
    )

    for step, columns in PINNED_NOOP_WRITERS.items():
        assert step in POSTPROCESS_STEP_CONTRACTS, step
        undeclared = columns - POSTPROCESS_STEP_CONTRACTS[step].writes
        assert not undeclared, (
            f"guard-2 ledger names step-{step} writes no longer declared: {sorted(undeclared)}"
        )
    assert set(PHANTOM_WRITE_EXEMPTIONS) == EXPECTED_PHANTOM_WRITES
    assert set(PHANTOM_READ_EXEMPTIONS) == EXPECTED_PHANTOM_READS
    for step, column in EXPECTED_PHANTOM_WRITES:
        assert column not in PINNED_NOOP_WRITERS.get(step, frozenset()), (
            step,
            column,
        )


@pytest.mark.requires_assertions
def test_guard2_ledger_holds_on_default_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-tier runtime tether for the guard-2 ledger (closure-review nit).

    PINNED_NOOP_WRITERS claims its writers are never content-effective on
    ANY matrix axis, which makes every single capture a valid probe of the
    claim's dangerous drift direction: a pinned entry that has become
    effective would let the heavy union test alone catch a now-stale ledger
    still discharging... nothing — but the smoke tier would keep passing the
    stale ledger to the pinned-findings classifier. One default capture in
    record mode asserts no pinned row is content-effective, so that drift
    goes red per-step without paying the full matrix.
    """

    from test_postprocess_enforcement import PINNED_NOOP_WRITERS

    import torchlens.postprocess as pp

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_WRITE_AUDIT", "record")
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "record")
    for sink in (
        pp.RECORDED_STEP_WRITES,
        pp.RECORDED_STEP_READS,
        pp.RECORDED_STEP_CLONE_READS,
        pp.RECORDED_STEP_EFFECTIVE_WRITES,
    ):
        sink.clear()
    try:
        trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
        trace.cleanup()
        for step, pinned in PINNED_NOOP_WRITERS.items():
            effective = pp.RECORDED_STEP_EFFECTIVE_WRITES.get(step, set())
            leaked = pinned & effective
            assert not leaked, (
                f"pinned step-{step} no-op writers were content-effective "
                f"on a default capture — the guard-2 ledger is stale: "
                f"{sorted(leaked)}"
            )
    finally:
        for sink in (
            pp.RECORDED_STEP_WRITES,
            pp.RECORDED_STEP_READS,
            pp.RECORDED_STEP_CLONE_READS,
            pp.RECORDED_STEP_EFFECTIVE_WRITES,
        ):
            sink.clear()


# ---------------------------------------------------------------------------
# The two-key direction authority (Sol 1's counterexample, both halves)
# ---------------------------------------------------------------------------


def test_registry_swap_refused_by_r1_and_r2() -> None:
    """Permuting the registry alone refuses at R1 AND R2, by name."""

    order = list(REGISTRY_ORDER)
    i7, i8 = order.index("7"), order.index("8")
    order[i7], order[i8] = order[i8], order[i7]
    violations = list(_executor._iter_structural_violations(tuple(order)))
    assert any("(R1)" in violation for violation in violations)
    assert any("(R2)" in violation for violation in violations)


def test_coordinated_reversal_refused_by_corpus() -> None:
    """Rank AND registry swapped together: R1/R2 pass, K1 refuses.

    Sol round-2 finding 1, pinned: a coordinated 7/8 reversal re-orients
    every derived edge to the new rank, so the drift detectors are blind —
    step 1 co-writes pass_index/recurrent_ops, so step 8's reads keep an
    earlier writer and no finding fires either. The ONLY mechanical catch
    is the reason-bearing corpus entry (key 2).
    """

    order = list(REGISTRY_ORDER)
    i7, i8 = order.index("7"), order.index("8")
    order[i7], order[i8] = order[i8], order[i7]
    rank = dict(LEGACY_STEP_RANK)
    rank["7"], rank["8"] = rank["8"], rank["7"]
    # Both drift detectors are silent under the coordinated edit...
    assert list(_executor._iter_structural_violations(tuple(order), rank)) == []
    # ...and the corpus refuses by name, carrying the reviewed reason.
    violations = iter_corpus_violations(rank)
    assert violations, "K1 must catch the coordinated reversal"
    assert any("('7', '8')" in violation for violation in violations)


def test_derived_pairs_pinned_in_corpus() -> None:
    """K2: every derived RAW/WW pair is pinned with its exact carriers.

    A coordinated WAR reversal manufactures a NEW RAW pair under the new
    rank, which lands here unpinned and turns the suite red until a
    reason-bearing corpus entry is reviewed in.
    """

    pairs = _executor.derived_pinnable_pairs()
    for pair, carriers in pairs.items():
        assert pair in PINNED_ORDER_PAIRS, f"unpinned derived pair {pair}"
        missing = set(carriers) - PINNED_ORDER_PAIRS[pair].carriers
        assert not missing, f"pair {pair} missing carriers {sorted(missing)}"


def test_no_new_templated_corpus_reasons() -> None:
    """F4 lint: a NEW corpus entry must carry real reviewed prose.

    The reason field is the corpus's entire value — design §2.1 makes
    editing it THE reviewed act that blesses a reorder, which is hollow
    when the text just restates the derivation. Seed-era templated
    entries are grandfathered in LEGACY_TEMPLATED_PAIRS (named follow-up:
    rewrite as prose); the set can only shrink.
    """

    templated = {
        pair for pair, entry in PINNED_ORDER_PAIRS.items() if _TEMPLATED_REASON.match(entry.reason)
    }
    new_templated = templated - LEGACY_TEMPLATED_PAIRS
    assert not new_templated, (
        "new corpus entries must carry real reviewed prose reasons, not "
        f"machine-templated derivation restatements: {sorted(new_templated)}"
    )
    stale_grandfathers = LEGACY_TEMPLATED_PAIRS - set(PINNED_ORDER_PAIRS)
    assert not stale_grandfathers, (
        "LEGACY_TEMPLATED_PAIRS names pairs no longer in the corpus; "
        f"remove them: {sorted(stale_grandfathers)}"
    )
    # The shrink-only guarantee's other half (closure-review nit): a
    # grandfathered pair whose reason gained real prose must LEAVE the set,
    # or "the set can only shrink" is unenforced — the entry could silently
    # flip back to a templated reason later without this lint noticing.
    retired_grandfathers = LEGACY_TEMPLATED_PAIRS - templated
    assert not retired_grandfathers, (
        "these grandfathered pairs now carry prose reasons; remove them "
        "from LEGACY_TEMPLATED_PAIRS so the set provably only shrinks: "
        f"{sorted(retired_grandfathers)}"
    )


def test_docstring_invariants_carried_by_corpus() -> None:
    """The absorbed prose invariants exist as reason-bearing entries (§7.3)."""

    for pair in (
        ("7", "8"),
        ("8", "9"),
        ("9", "11"),
        ("10", "11"),
        ("15.5", "16"),
        ("17.5", "18"),
        ("17", "17.5"),
        ("18", "20"),
        ("19", "20"),
    ):
        assert pair in PINNED_ORDER_PAIRS, pair
        assert PINNED_ORDER_PAIRS[pair].reason


# ---------------------------------------------------------------------------
# Independent re-derivation (spec-drives-code cross-check)
# ---------------------------------------------------------------------------


def test_independent_edge_rederivation() -> None:
    """A test-side reimplementation of the edge rules matches the module."""

    contracts = {
        step: contract for step, contract in POSTPROCESS_STEP_CONTRACTS.items() if step != "0"
    }
    rank = LEGACY_STEP_RANK
    expected: set[tuple[str, str, str, str]] = set()

    writers: dict[str, set[str]] = {}
    readers: dict[str, set[str]] = {}
    for step, contract in contracts.items():
        for column in contract.writes:
            writers.setdefault(column, set()).add(step)
        for column in contract.reads | contract.placeholder_probes:
            readers.setdefault(column, set()).add(step)
    for column, its_writers in writers.items():
        ordered = sorted(its_writers, key=rank.__getitem__)
        for i, low in enumerate(ordered):
            for high in ordered[i + 1 :]:
                expected.add((low, high, "ww", column))
        for reader in readers.get(column, ()):
            for writer in its_writers:
                if writer == reader:
                    continue
                if rank[writer] < rank[reader]:
                    expected.add((writer, reader, "raw", column))
                else:
                    expected.add((reader, writer, "war", column))

    token_writers: dict[str, set[str]] = {}
    token_readers: dict[str, set[str]] = {}
    for step, contract in contracts.items():
        for entry in contract.trace_state:
            prefix, _, token = entry.partition(":")
            bucket = token_writers if prefix == "w" else token_readers
            bucket.setdefault(token, set()).add(step)
    for token, its_writers in token_writers.items():
        ordered = sorted(its_writers, key=rank.__getitem__)
        for i, low in enumerate(ordered):
            for high in ordered[i + 1 :]:
                expected.add((low, high, "token_ww", token))
        for reader in token_readers.get(token, ()):
            for writer in its_writers:
                if writer == reader:
                    continue
                if rank[writer] < rank[reader]:
                    expected.add((writer, reader, "token_raw", token))
                else:
                    expected.add((reader, writer, "token_war", token))

    def op_touching(contract: object) -> bool:
        return bool(
            contract.writes or contract.reads or contract.placeholder_probes or contract.row_effects
        )

    row_steps = [s for s, c in contracts.items() if c.row_effects]
    op_steps = [s for s, c in contracts.items() if op_touching(c)]
    for row_step in row_steps:
        for other in op_steps:
            if other == row_step:
                continue
            low, high = sorted((row_step, other), key=rank.__getitem__)
            expected.add((low, high, "row_barrier", ""))
    for barrier_step in (s for s, c in contracts.items() if c.barrier):
        for other in contracts:
            if other == barrier_step:
                continue
            low, high = sorted((barrier_step, other), key=rank.__getitem__)
            expected.add((low, high, "barrier", ""))

    derived = {(edge.src, edge.dst, edge.kind, edge.carrier) for edge in _executor.derive_edges()}
    assert derived == expected


def test_all_edges_rank_ascending() -> None:
    """Acyclicity by construction: every derived edge ascends in rank."""

    for edge in _executor.derive_edges():
        assert LEGACY_STEP_RANK[edge.src] < LEGACY_STEP_RANK[edge.dst], edge


# ---------------------------------------------------------------------------
# Non-vacuity: the tripwires can actually fire
# ---------------------------------------------------------------------------


def test_classifier_non_vacuity_synthetic_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    """A read of a non-baseline/no-writer/non-probe column IS a finding."""

    from torchlens.postprocess import PostprocessStepContract

    original = POSTPROCESS_STEP_CONTRACTS["10"]
    assert "_facets_cache" not in CAPTURE_BASELINE_COLUMNS
    synthetic = PostprocessStepContract(
        original.step,
        original.name,
        original.contract,
        writes=original.writes,
        reads=original.reads | {"_facets_cache"},
        placeholder_probes=original.placeholder_probes,
        row_effects=original.row_effects,
        trace_state=original.trace_state,
    )
    monkeypatch.setitem(POSTPROCESS_STEP_CONTRACTS, "10", synthetic)
    findings = {
        key
        for key, category in _executor.classify_declared_reads().items()
        if category == "finding"
    }
    assert ("10", "_facets_cache") in findings


def test_token_read_before_write_check_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    """7.1-5: an r:token with no earlier writer and no baseline is refused."""

    from torchlens.postprocess import PostprocessStepContract, tokens

    original = POSTPROCESS_STEP_CONTRACTS["2"]
    broken = PostprocessStepContract(
        original.step,
        original.name,
        original.contract,
        writes=original.writes,
        reads=original.reads,
        trace_state=original.trace_state | tokens("r:module_logs"),
    )
    monkeypatch.setitem(POSTPROCESS_STEP_CONTRACTS, "2", broken)
    violations = list(_executor._iter_structural_violations())
    assert any("r:module_logs" in violation for violation in violations)


# ---------------------------------------------------------------------------
# Dynamic: the combined audit through a real capture
# ---------------------------------------------------------------------------


@pytest.mark.requires_assertions
def test_clone_scope_tags_step1_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Op.copy's whole-schema loop lands as category (d), not findings.

    Step 1 clones the output node; naive recording would report ~180
    category-(c) reads. The row-clone scope routes them into the clone
    channel, and legality rides the step's 'creates' row effect.
    """

    import torchlens.postprocess as pp

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "record")
    pp.RECORDED_STEP_READS.clear()
    pp.RECORDED_STEP_CLONE_READS.clear()
    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        clone_reads = pp.RECORDED_STEP_CLONE_READS.get("1", set())
        assert len(clone_reads) > 100, "clone loop must land in the clone channel"
        plain_reads = pp.RECORDED_STEP_READS.get("1", set())
        assert len(plain_reads) < 30, "clone reads must NOT land as plain reads"
        assert "creates" in POSTPROCESS_STEP_CONTRACTS["1"].row_effects
    finally:
        pp.RECORDED_STEP_READS.clear()
        pp.RECORDED_STEP_CLONE_READS.clear()
        pp.RECORDED_STEP_EFFECTIVE_WRITES.clear()
        trace.cleanup()


@pytest.mark.requires_assertions
def test_read_enforcement_green_on_default_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observed reads stay a subset of declared reads+probes (enforce mode)."""

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "enforce")
    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    trace.cleanup()


@pytest.mark.requires_assertions
def test_read_enforcement_trips_on_undeclared_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undeclared read on a covered axis fails the day it is introduced."""

    import torchlens.postprocess as pp
    from torchlens.postprocess import PostprocessStepContract

    original = POSTPROCESS_STEP_CONTRACTS["2"]
    real_step2 = pp._find_output_ancestors

    def snooping_step2(trace: object) -> None:
        real_step2(trace)
        first_op = next(iter(trace._raw_graph_ws.raw_layer_dict.values()))
        _ = first_op.func_rng_states  # undeclared read on step 2

    narrowed = PostprocessStepContract(
        original.step,
        original.name,
        original.contract,
        writes=original.writes,
        reads=original.reads - {"func_rng_states"},
        trace_state=original.trace_state,
    )
    monkeypatch.setitem(POSTPROCESS_STEP_CONTRACTS, "2", narrowed)
    monkeypatch.setattr(pp, "_find_output_ancestors", snooping_step2)
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "enforce")
    with pytest.raises(AssertionError, match="read undeclared op-store columns"):
        tl.trace(_TinyModel().eval(), torch.randn(2, 3))


def test_row_clone_scope_free_when_unarmed() -> None:
    """Op.copy works outside any audit (pickle/fork/preview paths)."""

    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        op = trace.layer_list[0]
        copied = op.copy()
        assert copied is not op
        from torchlens._trace_core.op_store import (
            _AUDIT_CLONE_READS,
            _AUDIT_READS,
            _CLONE_SCOPE_DEPTH,
        )

        assert not _AUDIT_READS
        assert not _AUDIT_CLONE_READS
        assert not _CLONE_SCOPE_DEPTH
    finally:
        trace.cleanup()


@pytest.mark.requires_assertions
def test_executor_seam_patched_step_executes_and_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A monkeypatched step function still executes AND still trips the audit.

    Design §5.2 tripwire integrity: the executor resolves step callables
    through the module namespace at CALL time; a registry of imported
    references would silently break every monkeypatch seam.
    """

    import torchlens.postprocess as pp

    calls: list[str] = []
    real_step2 = pp._find_output_ancestors

    def patched_step2(trace: object) -> None:
        calls.append("ran")
        real_step2(trace)
        first_op = next(iter(trace._raw_graph_ws.raw_layer_dict.values()))
        first_op.annotations["seam_smuggle"] = 1  # undeclared in-place write

    monkeypatch.setattr(pp, "_find_output_ancestors", patched_step2)
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    with pytest.raises(AssertionError, match=r"Step 2 .*annotations"):
        tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    assert calls == ["ran"], "the patched step body must have executed"


def test_executor_should_run_called_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor invariant: should_run evaluates once per step, in order.

    Step 18's predicate is deliberately context-writing (THE streaming
    snapshot point); a re-evaluated step-19 predicate after 18 cleared
    _out_writer would always be false and streamed outs would never be
    evicted (design §5.4).
    """

    from torchlens.postprocess import _executor as ex

    counts: dict[str, int] = {}
    seen_order: list[str] = []
    original_registry = ex.STEP_REGISTRY

    def counting(spec: ex.StepSpec) -> ex.StepSpec:
        inner = spec.should_run

        def counted(ctx: ex.StepContext) -> bool:
            counts[spec.step] = counts.get(spec.step, 0) + 1
            seen_order.append(spec.step)
            return inner(ctx)

        return ex.StepSpec(spec.step, spec.run, counted, spec.assert_when_skipped)

    monkeypatch.setattr(ex, "STEP_REGISTRY", tuple(counting(spec) for spec in original_registry))
    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        assert counts == {spec.step: 1 for spec in original_registry}
        assert seen_order == list(REGISTRY_ORDER)
    finally:
        trace.cleanup()


@pytest.mark.requires_assertions
def test_executor_failing_step_propagates_and_cleans_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising step propagates its exception and leaves no armed window.

    The finally-cleanup half of the window protocol: historically a raising
    step left the store class-swapped with a live collector.
    """

    import torchlens.postprocess as pp
    from torchlens._trace_core.op_store import (
        _AUDIT_COLLECTORS,
        _AUDIT_FINGERPRINTS,
        _AUDIT_READS,
    )

    class _StepBoom(RuntimeError):
        pass

    def exploding_step9(trace: object) -> None:
        raise _StepBoom("step 9 exploded")

    monkeypatch.setattr(pp, "_log_final_info_for_layers", exploding_step9)
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    with pytest.raises(_StepBoom):
        tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    assert not _AUDIT_COLLECTORS, "no collector may survive a raising step"
    assert not _AUDIT_FINGERPRINTS
    assert not _AUDIT_READS


@pytest.mark.requires_assertions
def test_no_window_open_past_step_20(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review note N11: the freeze seam runs unaudited by construction."""

    from torchlens._trace_core.op_store import _AUDIT_COLLECTORS

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        assert not _AUDIT_COLLECTORS
    finally:
        trace.cleanup()


def test_phase_timing_bucket_names_default_capture() -> None:
    """The _vtimed bucket-name set for a default capture is frozen (§5.5).

    The surface oracle normalizes timing fields, so a drift in WHICH
    buckets exist would pass the byte gate silently; this pins the set
    (step 13 and 17.5 are deliberately unwrapped; conditional steps absent
    when skipped).
    """

    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        buckets = {name for name in trace._phase_timings if name.startswith("postprocess:Step")}
        assert buckets == {
            "postprocess:Step 0: Materialize capture events",
            "postprocess:Step 1: Add output layers",
            "postprocess:Step 2: Trace output ancestors",
            "postprocess:Step 3: Remove orphan nodes",
            "postprocess:Step 4: Input/output distances",
            "postprocess:Step 5: Mark conditional branches",
            "postprocess:Step 6: Fix buffer layers",
            "postprocess:Step 7: Loop detection (full)",
            "postprocess:Step 8: Map labels",
            "postprocess:Step 9: Log final info",
            "postprocess:Step 10: Rename labels",
            "postprocess:Step 11: Build lookup keys",
            "postprocess:Step 11.5: Populate source var names",
            "postprocess:Step 11.75: Resolve deferred retention",
            "postprocess:Step 12: Undecorate tensors",
            "postprocess:Step 14: Log timing",
            "postprocess:Step 15: Finalize params",
            "postprocess:Step 15.5: Build layer logs",
            "postprocess:Step 16: Build module logs",
            "postprocess:Step 16.5: Graph shape hash",
            "postprocess:Step 17: Mark pass finished",
            "postprocess:Step 20: Release param refs",
        }
    finally:
        trace.cleanup()


def test_step17_5_drops_capture_phase_workspaces() -> None:
    """Step 17.5 is the contracted terminal consume of the per-phase workspaces.

    Mutation-margin arming (W2): a whole-function ``return None`` disarm of
    ``_run_step_17_5`` survived because nothing asserted its effect directly
    (its contract declares no op-store writes, so the write-audit matrix
    test cannot see it). ``_raw_graph_ws`` and ``_wrapper_runtime_ws`` are
    set unconditionally at capture start (``data_classes/trace.py``), so
    their absence here proves the terminal consume ran.
    """

    trace = tl.trace(_TinyModel().eval(), torch.randn(2, 3))
    try:
        for field_name in (
            "_raw_graph_ws",
            "_module_capture_ws",
            "_wrapper_runtime_ws",
            "capture_events",
            "_output_container_specs_by_raw_label",
        ):
            assert field_name not in trace.__dict__, (
                f"{field_name} survived step 17.5's terminal consume"
            )
    finally:
        trace.cleanup()
