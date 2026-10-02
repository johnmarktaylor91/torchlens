"""Read/write enforcement over the postprocess axes matrix (design §8.2).

Enforcement ships THIS phase: every axis of the recording matrix runs with
the combined audit in ENFORCE mode — observed writes must be a subset of
declared writes, observed reads a subset of declared reads + probes, and
row-clone reads only on row-creating steps. An undeclared read or write on
a covered axis fails CI the day it is introduced. The honest residual is a
new configuration-gated path on an axis NOT in the matrix
(``tests/support/postprocess_axes.py``); adding the config to the matrix is
part of adding the config.

The union-level reports (phantom declarations and permanent no-op writers)
run the whole matrix in one process and are ``heavy``-marked; the per-axis
enforcement runs are sub-second and live in the fast tier.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from support.postprocess_axes import iter_axes

_AXES = iter_axes()
_AXIS_IDS = [name for name, _ in _AXES]

#: Declared-but-never-observed writes with named config-gated exemptions —
#: mirror of PHANTOM_WRITE_EXEMPTIONS in test_postprocess_dag.py, asserted
#: here against the LIVE matrix union (the Opus-4 anti-laundering guard: a
#: fabricated declaration is red the day it lands).
EXPECTED_PHANTOM_WRITES = {
    ("9", "args_template"),
    ("9", "kwargs_template"),
    ("18", "grad_ref"),
    ("6", "address"),
    # Reviewed widening (sol finding 6 in-place audit, see _contracts.py):
    # steps 15 and 16 mutate the referenced Param rows' OWN fields
    # (used_by_ops, co_parent_params, ...) rather than reassigning the Op's
    # _param_logs column, so the column-reassignment write audit never
    # observes either declared write. Mirrored in PHANTOM_WRITE_EXEMPTIONS,
    # test_postprocess_dag.py.
    ("15", "_param_logs"),
    ("16", "_param_logs"),
}

#: Declared-but-never-observed READS, the read-side mirror of the table
#: above (B2 residual closure): step 6's equivalence_class read sits on the
#: None-address recovery branch, guarded by the same unreachability as the
#: ("6", "address") phantom write — no capture path materializes a buffer
#: row without a display address. Reason-bearing mirror:
#: PHANTOM_READ_EXEMPTIONS in test_postprocess_dag.py. Growing this set is
#: reviewed, never silent.
EXPECTED_PHANTOM_READS = {
    ("6", "equivalence_class"),
}

#: Writers whose intercepted writes are never content-effective on ANY
#: matrix axis (design §2.4 guard 2, pinned-findings discipline): these are
#: placeholder-equal rewrites (output-row init, equal-content scrub
#: rebinds, absent-feature configs) plus the FINDING-favoring ambiguity
#: default — a rich-object cell (e.g. a tensor) rewritten with a distinct
#: same-class value is deliberately classified no-op, so effectiveness can
#: never launder a read discharge through an unprovable rewrite (the
#: step-1 payload/memory columns land here by that rule). A permanent
#: no-op writer cannot discharge a read-before-write finding — this table
#: is guard 2's ledger, passed to classify_declared_reads by the pinned-
#: findings test. A NEW entry here is reviewed, never silently accepted;
#: a matrix-gap entry is closed by adding the axis that makes the writer
#: effective (the conditional_elif_else and var_names axes retired the
#: 5/9 elif-else and 11.5 var_names rows exactly that way, and the
#: depths-off buffer_from_input axis retired ("6", "has_input_ancestor")).
PINNED_NOOP_WRITERS = {
    "1": frozenset(
        (
            "activation_memory",
            "bytes_delta_at_call",
            "bytes_peak_at_call",
            "container_path",
            "container_spec",
            "dropped_edge_tensor_args",
            "dtype",
            "func_duration",
            "has_out_variations",
            "input_to_module_calls",
            "is_buffer",
            "is_input",
            # The factory_source_output axis (b9 R71-1) retired the
            # ("1", "is_internal_source"/"non_tensor_kwargs"/"num_kwargs")
            # rows: cloning a parentless factory source into the synthetic
            # output node genuinely flips all three.
            "is_transform",
            "num_params_frozen",
            "num_passes",
            "out",
            "out_versions_by_child",
            "param_memory",
            "pass_index",
            "recurrent_ops",
            "shape",
            "transform_chain",
            "transform_fn_name",
            "transform_fn_qualname",
            "transform_fn_source",
            "transform_kind",
            "transformed_activation_memory",
            "transformed_out",
            "transformed_out_dtype",
            "transformed_out_shape",
            "unattributed_tensor_args",
            "var_names",
        )
    ),
    "3": frozenset(
        (
            "_edge_uses",
            "args_template",
            "conditional_arm_children",
            "conditional_elif_children",
            "conditional_else_children",
            "conditional_entry_children",
            "conditional_then_children",
            "interventions",
            "kwargs_template",
        )
    ),
    "4": frozenset(("has_output_descendant",)),
    # The elif/else axis retired the ("5","is_terminal_bool") phantom row:
    # the write is now OBSERVED (host-escape witness classification runs)
    # but rewrites the False placeholder on these models, so it lands here
    # instead — still unable to discharge a read.
    "5": frozenset(("is_terminal_bool",)),
    # has_input_ancestor left this row when the buffer_from_input axis
    # (layer depths OFF, so step 4 does not pre-propagate ancestry) made
    # step 6's buffer-source ancestry fallback content-effective.
    "6": frozenset(
        (
            "args_template",
            # L4's D1 buffer-merge hardening declares survivor ancestry-flag
            # propagation; on the recorded axes the flag is already true, so
            # the write is contract-share-honest but observed no-op (reviewed
            # 2026-08-14; cannot discharge reads, per this test's doctrine).
            "has_internal_source_ancestor",
            "conditional_arm_children",
            "conditional_elif_children",
            "conditional_else_children",
            "conditional_entry_children",
            "conditional_then_children",
            "has_children",
            # The ("6","output_descendants") and ("6","has_output_descendant")
            # rows were RETIRED by the buffer_divergent_reach axis (r3
            # b1-opus R04-F1): a dead-ending survivor merged with an
            # output-reaching duplicate makes e12aa996's child-direction
            # reach repair content-effective, exactly the retirement pattern
            # this ledger describes for conditional_elif_else and var_names.
            "interventions",
            "kwargs_template",
        )
    ),
    "7": frozenset(("equivalence_class",)),
    "9": frozenset(("is_buffer", "is_input", "is_output")),
    "11.75": frozenset(
        (
            "activation_memory",
            "dtype",
            "shape",
            "transformed_activation_memory",
            "transformed_out",
            "transformed_out_dtype",
            "transformed_out_shape",
        )
    ),
}


@pytest.mark.requires_assertions
@pytest.mark.parametrize(("axis_name", "axis_fn"), _AXES, ids=_AXIS_IDS)
def test_axis_passes_read_and_write_enforcement(
    axis_name: str,
    axis_fn: Callable[[], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One matrix axis captures green under full contract enforcement."""

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "enforce")
    trace = axis_fn()
    if trace is not None:
        trace.cleanup()


def test_buffer_duplicate_axis_actually_merges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-vacuity: the buffer_duplicate axis really fires the merge.

    Step 6's merge write family is enforced against this axis; if a model
    or capture change ever stops the merge from triggering, the enforcement
    leg would stay green while proving nothing — this pins the trigger, and
    the scalar ``buffer_source`` repoint with it.
    """

    from support.postprocess_axes import _axis_buffer_duplicate

    import torchlens.postprocess.control_flow as cf

    merges: list[tuple[str, str]] = []
    real_merge = cf._merge_buffer_entries

    def spying_merge(
        trace: Any,
        survivor: Any,
        removed: Any,
        *,
        deferred_removals: Any = None,
        survivor_edge_shadows: Any = None,
    ) -> None:
        merges.append((survivor._label_raw, removed._label_raw))
        real_merge(
            trace,
            survivor,
            removed,
            deferred_removals=deferred_removals,
            survivor_edge_shadows=survivor_edge_shadows,
        )

    monkeypatch.setattr(cf, "_merge_buffer_entries", spying_merge)
    trace = _axis_buffer_duplicate()
    try:
        assert merges, "the buffer_duplicate axis must reach _merge_buffer_entries"
        survivor_label, removed_label = merges[0]
        # buffer_source now rides the raw->final label rename (L4's
        # _SCALAR_LABEL_FIELDS_TO_RENAME fix), so resolve the survivor's FINAL
        # label for the comparison; the removed buffer never gets a final label.
        survivor_final = next(
            op.label for op in trace.layer_list if op._label_raw == survivor_label
        )
        repointed = [
            op for op in trace.layer_list if op.is_buffer and op.buffer_source == survivor_final
        ]
        assert repointed, "the scalar buffer_source repoint must have fired"
        assert all(op.buffer_source not in (removed_label,) for op in trace.layer_list), (
            "no surviving op may still reference the merged-away buffer"
        )
    finally:
        trace.cleanup()


@pytest.mark.requires_assertions
def test_buffer_from_input_axis_makes_ancestry_writes_effective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-vacuity: the buffer_from_input axis fires step 6's ancestry fallback.

    The ("6", "has_input_ancestor") permanent no-op row was retired by this
    axis, and step 6's input_ancestors write is declared on its evidence. If
    a capture or model change ever stops the fallback from being
    content-effective (e.g. something upstream starts pre-propagating
    ancestry with depths off), the retirement goes vacuous silently — this
    pins the trigger.
    """

    from support.postprocess_axes import _axis_buffer_from_input

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
        trace = _axis_buffer_from_input()
        effective = pp.RECORDED_STEP_EFFECTIVE_WRITES.get("6", set())
        assert {"has_input_ancestor", "input_ancestors"} <= effective, (
            "step 6's buffer-source ancestry fallback must be "
            f"content-effective on this axis; effective: {sorted(effective)}"
        )
        trace.cleanup()
    finally:
        for sink in (
            pp.RECORDED_STEP_WRITES,
            pp.RECORDED_STEP_READS,
            pp.RECORDED_STEP_CLONE_READS,
            pp.RECORDED_STEP_EFFECTIVE_WRITES,
        ):
            sink.clear()


@pytest.mark.requires_assertions
def test_reads_before_release_mark_still_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing half of the released-row read suppression (F6).

    Removal husking re-reads every set cell of released rows; suppressing
    those reads is what keeps step 3/6 read sets honest. The suppression
    must NOT swallow reads made BEFORE the release mark — that is exactly
    how the pinned ('3','label') orphan_records finding stays visible. This
    pins both halves on the orphan-removal axis: the pre-release data read
    records; the husking walk does not flood the step with whole-schema
    reads.
    """

    from support.postprocess_axes import _axis_orphan_remove

    import torchlens.postprocess as pp

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "record")
    pp.RECORDED_STEP_READS.clear()
    try:
        trace = _axis_orphan_remove()
        trace.cleanup()
        step3_reads = pp.RECORDED_STEP_READS.get("3", set())
        assert "label" in step3_reads, (
            "the pre-release orphan_records label read must record — losing "
            "it silently disarms the pinned day-1 finding"
        )
        assert "grad_fn_class_name" not in step3_reads, (
            "released-row husking reads must stay suppressed (they would "
            "report the whole schema as step-3 reads)"
        )
    finally:
        pp.RECORDED_STEP_READS.clear()
        pp.RECORDED_STEP_CLONE_READS.clear()
        pp.RECORDED_STEP_EFFECTIVE_WRITES.clear()
        pp.RECORDED_STEP_WRITES.clear()


def test_write_effectiveness_classifier_is_finding_favoring() -> None:
    """Guard 2's ambiguity default is NO-OP, never effective.

    An EFFECTIVE verdict discharges read-before-write findings, so the
    classifier may return it only when the change is provable; a distinct
    same-class rich object (the tensor-rewrite residual) must classify as
    a no-op even when its contents differ.
    """

    import torch

    from torchlens._trace_core.op_store import (
        _MISSING,
        _write_is_content_effective,
    )

    same = [1, 2]
    assert not _write_is_content_effective(same, same)  # identity rebind
    assert _write_is_content_effective(_MISSING, None)  # first write
    assert _write_is_content_effective(1, 2)  # scalar change
    assert not _write_is_content_effective(1, 1)  # scalar-equal rewrite
    assert _write_is_content_effective("a", 3)  # class change
    assert _write_is_content_effective([1], [1, 2])  # container content change
    assert not _write_is_content_effective([1], [1])  # equal-content rebind
    # The documented residual, pinned: distinct same-class rich objects are
    # AMBIGUOUS and must read as no-op (finding-favoring) even when the
    # values genuinely differ.
    assert not _write_is_content_effective(torch.zeros(2), torch.ones(2))


@pytest.mark.requires_assertions
@pytest.mark.heavy
def test_matrix_union_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phantom-declaration and no-op-writer reports over the full matrix."""

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
        for _, axis_fn in iter_axes():
            trace = axis_fn()
            if trace is not None:
                trace.cleanup()

        phantom: set[tuple[str, str]] = set()
        phantom_reads: set[tuple[str, str]] = set()
        noop: dict[str, frozenset[str]] = {}
        for step, contract in pp.POSTPROCESS_STEP_CONTRACTS.items():
            if step == "0":
                continue
            observed = pp.RECORDED_STEP_WRITES.get(step, set())
            effective = pp.RECORDED_STEP_EFFECTIVE_WRITES.get(step, set())
            observed_reads = pp.RECORDED_STEP_READS.get(step, set())
            for column in contract.writes - observed:
                phantom.add((step, column))
            for column in (contract.reads | contract.placeholder_probes) - observed_reads:
                phantom_reads.add((step, column))
            never_effective = frozenset((observed & contract.writes) - effective)
            if never_effective:
                noop[step] = never_effective

        assert phantom == EXPECTED_PHANTOM_WRITES, (
            "declared-never-observed writes drifted; a NEW phantom "
            "declaration is the cheapest laundering path — root-cause it, "
            f"never exempt it silently. Diff: {phantom ^ EXPECTED_PHANTOM_WRITES}"
        )
        # The phantom-READ report (opus impl-review F6): read enforcement
        # is observed ⊆ declared, so a regression that stops RECORDING
        # reads (e.g. an over-broad released-row suppression in
        # _CombinedAuditOpRowStore.cell_get) makes the leg quieter, never
        # red. Every declared read outside the named exemption set is
        # observed on >=1 axis today; a declared read no axis observes is
        # either a stale declaration or a recording hole — both reviewed,
        # never exempted silently (the one current exemption carries its
        # reason in PHANTOM_READ_EXEMPTIONS, test_postprocess_dag.py).
        assert phantom_reads == EXPECTED_PHANTOM_READS, (
            "declared-never-observed READS drifted; either a declaration "
            "is stale or read recording lost coverage (suppression "
            f"regression). Diff: {sorted(phantom_reads ^ EXPECTED_PHANTOM_READS)}"
        )
        assert noop == PINNED_NOOP_WRITERS, (
            "the permanent no-op writer report drifted; a new no-op writer "
            "cannot discharge read-before-write findings and is reviewed, "
            "never silently accepted."
        )
    finally:
        for sink in (
            pp.RECORDED_STEP_WRITES,
            pp.RECORDED_STEP_READS,
            pp.RECORDED_STEP_CLONE_READS,
            pp.RECORDED_STEP_EFFECTIVE_WRITES,
        ):
            sink.clear()


@pytest.mark.smoke
def test_audit_env_knob_refusals_are_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The audit env-knob refusals carry stable codes, not bare RuntimeError.

    b6 R25 (4th round): the three closed-vocabulary knob parsers and the
    armed-under--O guard were the territory's only branch-on-text refusals.
    """

    from torchlens._errors import InvalidArgumentError
    from torchlens.postprocess import (
        _postprocess_assertions_enabled,
        _read_audit_mode,
        _write_audit_record_mode,
    )

    cases = [
        ("TORCHLENS_POSTPROCESS_ASSERTIONS", "bogus", _postprocess_assertions_enabled),
        ("TORCHLENS_POSTPROCESS_WRITE_AUDIT", "recrod", _write_audit_record_mode),
        ("TORCHLENS_POSTPROCESS_READ_AUDIT", "tru", _read_audit_mode),
    ]
    for env_name, bad_value, parser in cases:
        monkeypatch.setenv(env_name, bad_value)
        with pytest.raises(InvalidArgumentError) as exc_info:
            parser()
        assert exc_info.value.fields["code"] == "postprocess_audit_env_invalid", env_name
        assert exc_info.value.fields["argument"] == env_name
        monkeypatch.delenv(env_name)


def test_read_audit_without_assertions_refuses_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """r7 R04-1: a requested read audit that cannot run refuses at capture.

    ``TORCHLENS_POSTPROCESS_READ_AUDIT=enforce`` with
    ``TORCHLENS_POSTPROCESS_ASSERTIONS`` unset was silently inert -- the read
    audit acts only inside the assertion-armed windows, so a CI leg exporting
    just the read knob got a green run while a live undeclared read (the
    step-16 R04-2 finding) sailed through. The combination now refuses typed
    at postprocess entry, matching the armed-under--O precedent (red-capable:
    pre-fix this capture completes green).
    """

    import torch
    from torch import nn

    import torchlens as tl
    from torchlens._errors import InvalidArgumentError

    monkeypatch.delenv("TORCHLENS_POSTPROCESS_ASSERTIONS", raising=False)
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "enforce")
    with pytest.raises(InvalidArgumentError) as exc_info:
        tl.trace(nn.ReLU(), torch.ones(2))
    assert exc_info.value.fields["code"] == "postprocess_audit_env_invalid"
    assert exc_info.value.fields["argument"] == "TORCHLENS_POSTPROCESS_READ_AUDIT"


def test_multi_output_module_axis_passes_full_enforcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """r7 R04-2: the LSTM-cell / tuple-submodule shape enforces green.

    Step 16 read two undeclared op-store columns (``multi_output_name``,
    ``_source_trace_ref``) on every module call with more than one output
    entry; the enforcement matrix never covered that family, so the audit's
    own tripwire only fired for end users who armed it (red-capable: pre-fix
    this trips 'Step 16 ... read undeclared op-store columns'). The reviewed
    contract diff declares both reads (no ordering edges: both columns are
    ingest-written, never step-written) and the axis pins the family.
    """

    from support.postprocess_axes import _axis_multi_output_module

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "enforce")
    trace = _axis_multi_output_module()
    assert trace is not None
    trace.cleanup()


def exc_code_line(stdout: str) -> str:
    """Return the refusal code printed by the -O provocation child."""

    return stdout.strip().split()[-1]


@pytest.mark.heavy  # measured 7.0-7.9s across CI rows (round-2/3 CI triage, 2026-10-01):
# consistently over the smoke ceiling on every row (a real -O child-process spawn), not
# a one-off load spike.
def test_audit_armed_under_stripped_asserts_refuses_typed() -> None:
    """Arming the audit under -O refuses with a stable code (real -O child)."""

    import os
    import subprocess
    import sys

    script = (
        "import os\n"
        "os.environ['TORCHLENS_POSTPROCESS_ASSERTIONS'] = '1'\n"
        "from torchlens.postprocess import _postprocess_assertions_enabled\n"
        "try:\n"
        "    _postprocess_assertions_enabled()\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__, getattr(exc, 'fields', {}).get('code'))\n"
        "else:\n"
        "    print('NO_RAISE')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "InvalidArgumentError" in completed.stdout
    assert exc_code_line(completed.stdout) == "postprocess_audit_asserts_stripped"
