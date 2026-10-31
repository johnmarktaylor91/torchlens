"""P4 amendment lane: dual-leg reducer, watermark transport, lifecycle.

Covers DoR 4.3-4.6 plus the binding reviewer notes:

* append-only op lane with lane-local amendment seq (Tier-F event ``seq``
  facts never shift when an amendment lands),
* canonical reducer fold (amendment order, last-wins per path, absent-facet
  materialization from ``FACET_DEFAULTS`` — with the planted S5 violation),
* runtime validation at append AND fold; fail-closed target resolution,
* sealed-source refusal vs working-projection acceptance (DoR 4.5.5),
* seal watermark dual-stamp incl. the pre-seal projection clone (S-N2) and
  the ``copy_for_replay`` seq-filter (no double-application),
* multi-domain by-label resolution with the target_seq cross-check gated to
  single-domain journals (O-N6),
* concat: lane-local domain validation, target_seq rebind via the merge seq
  map, re-stamp + re-fold in the target journal (``REBINDABLE_TARGET_LANES``).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.capture_events import (
    REBINDABLE_TARGET_LANES,
    AmendmentTargetError,
    CaptureEvents,
    LaneMergePolicyError,
    SealedJournalAmendmentError,
    SealedJournalAppendError,
    SealedJournalWriteError,
    _clone_op_event_for_replay,
)
from torchlens.ir.op_record import (
    FACET_DEFAULTS,
    AmendmentValidationError,
    OpAmendment,
    OpRecord,
    amend_graph_edge_insertion,
    amend_raw_hook_intervention,
    validate_amendment,
)

from ._journal_probe import grab_journal_events

# Journal SHAPES, not producers (P7 deleted the legacy torch producer):
# "decomposed" = captured OpRecord rows; "legacy" = genuine compat OpEvent
# rows (the preview-journal stand-in until S15), synthesized by the probe
# through the retained inverse adapter. The dual-shape fold legs guard
# PATH_TO_FLAT until it dies with OpEvent in S15.
_LEGS = ("legacy", "decomposed")


def _fresh_journal(template_events: list[Any], count: int = 4) -> CaptureEvents:
    """Return a live journal populated with re-stamped template clones."""

    journal = CaptureEvents()
    for event in template_events[:count]:
        journal.append(_clone_op_event_for_replay(event))
    return journal


def _target(journal: CaptureEvents, position: int = 1) -> Any:
    return journal.op_events[position]


@pytest.fixture(scope="module")
def leg_templates(request: pytest.FixtureRequest) -> dict[str, list[Any]]:
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    return {leg: grab_journal_events(monkeypatch, leg) for leg in _LEGS}


@pytest.mark.parametrize("leg", _LEGS)
def test_append_stamps_folds_and_keeps_raw_lane_append_only(
    leg: str, leg_templates: dict[str, list[Any]]
) -> None:
    journal = _fresh_journal(leg_templates[leg])
    raw_before = list(journal.op_events)
    event_seq_before = journal.event_seq
    target = _target(journal)

    folded = journal.append_amendment(
        amend_raw_hook_intervention(target.seq, target.label_raw, intervention_replaced=True)
    )

    amendment = journal.op_amendments[0]
    assert amendment.seq == 1 and journal.amendment_seq == 1
    assert amendment.run_nonce == journal.run_nonce
    # Lane-local domain: appending an amendment must NOT consume the shared
    # event counter (Tier-F event seq facts are pinned by the baselines).
    assert journal.event_seq == event_seq_before
    # Raw lane genuinely append-only: same objects at same positions.
    assert all(a is b for a, b in zip(journal.op_events, raw_before, strict=True))
    assert not getattr(target, "intervention_replaced")
    # Reducer + one-record view + live index all present the folded state.
    view = journal.amended_op_records()
    assert view[1] is not target and view[1].intervention_replaced
    assert journal.amended_op_record(target.label_raw).intervention_replaced
    assert journal.live_index.by_raw_label[target.label_raw].intervention_replaced
    assert folded.intervention_replaced
    # Unamended entries pass through by identity.
    assert view[0] is journal.op_events[0]
    if leg == "decomposed":
        assert isinstance(view[1], OpRecord)


@pytest.mark.parametrize("leg", _LEGS)
def test_last_wins_per_path_in_amendment_order(
    leg: str, leg_templates: dict[str, list[Any]]
) -> None:
    journal = _fresh_journal(leg_templates[leg])
    target = _target(journal)
    journal.append_amendment(
        amend_raw_hook_intervention(target.seq, target.label_raw, intervention_replaced=True)
    )
    journal.append_amendment(
        amend_raw_hook_intervention(target.seq, target.label_raw, intervention_replaced=False)
    )
    assert journal.amended_op_record(target.label_raw).intervention_replaced is False


@pytest.mark.smoke_cells("test_validation_refusals[decomposed]")
@pytest.mark.parametrize("leg", _LEGS)
def test_validation_refusals(leg: str, leg_templates: dict[str, list[Any]]) -> None:
    journal = _fresh_journal(leg_templates[leg])
    target = _target(journal)

    # Unregistered family on a forged raw record.
    with pytest.raises(AmendmentValidationError, match="unregistered"):
        validate_amendment(OpAmendment(0, 0, target.seq, target.label_raw, "no_such_family", ()))
    # Path set must equal the registry's exactly (ordered).
    with pytest.raises(AmendmentValidationError, match="exact"):
        validate_amendment(
            OpAmendment(
                0,
                0,
                target.seq,
                target.label_raw,
                "raw_hook_intervention",
                (("intervention.intervention_fired", True),),
            )
        )
    # Duplicate paths refuse before the exact-set check.
    with pytest.raises(AmendmentValidationError, match="duplicate"):
        validate_amendment(
            OpAmendment(
                0,
                0,
                target.seq,
                target.label_raw,
                "raw_hook_intervention",
                (
                    ("intervention.intervention_replaced", True),
                    ("intervention.intervention_replaced", False),
                ),
            )
        )
    # Per-path value type check.
    with pytest.raises(AmendmentValidationError, match="type"):
        validate_amendment(
            OpAmendment(
                0,
                0,
                target.seq,
                target.label_raw,
                "raw_hook_intervention",
                (("intervention.intervention_replaced", "yes"),),
            )
        )
    # Identity fields are unpatchable even when forged into a known family's
    # shape via a raw construction (append re-validates).
    forged = OpAmendment(
        0,
        0,
        target.seq,
        target.label_raw,
        "late_buffer_output_parent",
        (("core.seq", 99),),
    )
    with pytest.raises(AmendmentValidationError):
        journal.append_amendment(forged)
    # Fail-closed target resolution.
    with pytest.raises(AmendmentTargetError, match="unknown op"):
        journal.append_amendment(
            amend_raw_hook_intervention(1, "nonexistent_1_1_raw", intervention_replaced=True)
        )
    # Single-domain journals cross-check target_seq.
    with pytest.raises(AmendmentTargetError, match="single-seq-domain"):
        journal.append_amendment(
            amend_raw_hook_intervention(
                target.seq + 7, target.label_raw, intervention_replaced=True
            )
        )


def test_absent_facet_materializes_from_defaults_with_planted_s5_violation(
    leg_templates: dict[str, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _fresh_journal(leg_templates["decomposed"])
    target = next(
        event for event in journal.op_events if getattr(event, "intervention", None) is None
    )

    def _assert_untouched_fields_are_defaults(folded_record: Any) -> None:
        materialized = folded_record.intervention
        assert materialized is not None
        defaults = FACET_DEFAULTS["intervention"]()
        assert materialized.intervention_replaced is True
        for name in ("intervention_fired", "fire_results"):
            assert getattr(materialized, name) == getattr(defaults, name), (
                f"materialized facet field {name!r} drifted from FACET_DEFAULTS"
            )

    journal.append_amendment(
        amend_raw_hook_intervention(target.seq, target.label_raw, intervention_replaced=True)
    )
    _assert_untouched_fields_are_defaults(journal.amended_op_record(target.label_raw))

    # Planted S5 violation: a drifted defaults table must be CAUGHT by the
    # assertion above — proving the check is sensitive, not vacuous.
    from torchlens.ir import op_record as op_record_module
    from torchlens.ir.op_record import InterventionFacet

    planted = dict(op_record_module.FACET_DEFAULTS)
    planted["intervention"] = lambda: InterventionFacet(
        intervention_fired=True,  # wrong default
        intervention_replaced=False,
        fire_results=(),
    )
    monkeypatch.setattr(op_record_module, "FACET_DEFAULTS", planted)
    journal2 = _fresh_journal(leg_templates["decomposed"])
    target2 = next(
        event for event in journal2.op_events if getattr(event, "intervention", None) is None
    )
    journal2.append_amendment(
        amend_raw_hook_intervention(target2.seq, target2.label_raw, intervention_replaced=True)
    )
    with pytest.raises(AssertionError, match="drifted"):
        _assert_untouched_fields_are_defaults(journal2.amended_op_record(target2.label_raw))


@pytest.mark.parametrize("leg", _LEGS)
def test_sealed_source_refuses_working_projection_accepts(
    leg: str, leg_templates: dict[str, list[Any]]
) -> None:
    journal = _fresh_journal(leg_templates[leg])
    target = _target(journal)
    journal.core_seal_watermark = journal.amendment_seq
    journal.amendments_sealed = True
    with pytest.raises(SealedJournalAmendmentError):
        journal.append_amendment(
            amend_raw_hook_intervention(target.seq, target.label_raw, intervention_replaced=True)
        )
    working = journal.copy_for_replay()
    assert working.amendments_sealed is False
    assert working.core_seal_watermark == journal.core_seal_watermark
    working_target = working.op_event_by_label_raw[target.label_raw]
    working.append_amendment(
        amend_raw_hook_intervention(
            working_target.seq, working_target.label_raw, intervention_replaced=True
        )
    )
    assert working.amended_op_record(target.label_raw).intervention_replaced


@pytest.mark.parametrize("leg", _LEGS)
def test_watermark_filters_seeded_copies_only(
    leg: str, leg_templates: dict[str, list[Any]]
) -> None:
    journal = _fresh_journal(leg_templates[leg])
    first, second = journal.op_events[0], journal.op_events[1]
    journal.append_amendment(
        amend_raw_hook_intervention(first.seq, first.label_raw, intervention_replaced=True)
    )
    folded_at_seal = list(journal.amended_op_records())
    journal.core_seal_watermark = journal.amendment_seq  # "seal" consumed #1
    journal.append_amendment(
        amend_raw_hook_intervention(second.seq, second.label_raw, intervention_replaced=True)
    )

    # Plain structural copy: the whole lane rides along.
    plain = journal.copy_for_replay()
    assert [a.seq for a in plain.op_amendments] == [1, 2]

    # Folded-seed copy: seal-consumed amendments are filtered out; genuinely
    # post-seal knowledge survives.
    seeded = journal.copy_for_replay(projected_op_events=folded_at_seal)
    assert [a.seq for a in seeded.op_amendments] == [2]
    # And the surviving amendment still folds on the copy.
    assert seeded.amended_op_record(second.label_raw).intervention_replaced


@pytest.mark.parametrize("leg", _LEGS)
def test_multi_domain_resolves_by_label_last_occurrence(
    leg: str, leg_templates: dict[str, list[Any]]
) -> None:
    journal = _fresh_journal(leg_templates[leg])
    folded = list(journal.amended_op_records())
    # Simulate a multi-pass concatenated projection: the same sealed fold
    # twice — colliding seqs, duplicate labels, no single seq domain.
    seeded = journal.copy_for_replay(projected_op_events=folded + folded)
    assert seeded.single_seq_domain is False
    target = folded[1]
    # target_seq disagreement is NOT cross-checked in a multi-domain journal
    # (O-N6): resolution is by label.
    seeded.append_amendment(
        amend_raw_hook_intervention(
            target.seq + 12345, target.label_raw, intervention_replaced=True
        )
    )
    view = seeded.amended_op_records()
    positions = [index for index, event in enumerate(view) if event.label_raw == target.label_raw]
    assert len(positions) == 2
    first_occurrence, last_occurrence = positions
    assert view[last_occurrence].intervention_replaced is True
    assert view[first_occurrence].intervention_replaced is False
    # Single-domain journals DO cross-check (established by the refusal test).


@pytest.mark.smoke_cells("test_merged_multi_pass_journal_binds_amendments_by_seq[legacy]")
@pytest.mark.parametrize("leg", _LEGS)
def test_merged_multi_pass_journal_binds_amendments_by_seq(
    leg: str, leg_templates: dict[str, list[Any]]
) -> None:
    """Colliding raw labels on a MERGED single-domain journal bind by seq.

    The recorder accumulator concats per-pass journals whose raw labels
    repeat (each pass re-emits the same ``label_raw``); after restamping the
    result is one seq domain where a label's LIVE (last) occurrence is not
    necessarily an amendment's target. Resolution must be seq-exact there:
    by-label last-occurrence binding folds a pass-1 amendment onto the
    pass-2 op, and re-concat of the merged journal (the failed-partial
    recovery copy in ``Recorder._mark_recording_failed``) replays all ops
    before all amendments and refused fail-closed (the P5 fastlog-partial
    regression).
    """

    pass_one = _fresh_journal(leg_templates[leg], count=2)
    target_one = _target(pass_one)
    pass_one.append_amendment(
        amend_raw_hook_intervention(
            target_one.seq, target_one.label_raw, intervention_replaced=True
        )
    )
    pass_two = _fresh_journal(leg_templates[leg], count=2)
    target_two = _target(pass_two)
    pass_two.append_amendment(
        amend_raw_hook_intervention(
            target_two.seq, target_two.label_raw, intervention_replaced=False
        )
    )

    accumulator = CaptureEvents()
    accumulator.concat(pass_one)
    accumulator.concat(pass_two)
    # Restamped into ONE domain: seqs unique, labels collide across passes.
    assert accumulator.single_seq_domain is True
    assert len({a.target_seq for a in accumulator.op_amendments}) == 2

    view = accumulator.amended_op_records()
    positions = [
        index for index, event in enumerate(view) if event.label_raw == target_one.label_raw
    ]
    assert len(positions) == 2
    first_occurrence, last_occurrence = positions
    # Each pass's amendment folds onto ITS OWN occurrence (seq-exact).
    assert view[first_occurrence].intervention_replaced is True
    assert view[last_occurrence].intervention_replaced is False

    # Re-concat of the merged journal: every amendment re-binds through the
    # merge seq map and appends cleanly (previously AmendmentTargetError).
    combined = CaptureEvents()
    combined.concat(accumulator)
    combined_view = combined.amended_op_records()
    combined_positions = [
        index
        for index, event in enumerate(combined_view)
        if event.label_raw == target_one.label_raw
    ]
    assert len(combined_positions) == 2
    assert combined_view[combined_positions[0]].intervention_replaced is True
    assert combined_view[combined_positions[1]].intervention_replaced is False


@pytest.mark.parametrize("leg", _LEGS)
def test_concat_rebinds_target_seq_and_refolds(
    leg: str, leg_templates: dict[str, list[Any]]
) -> None:
    assert {"intervention_events", "op_amendments"} == REBINDABLE_TARGET_LANES
    source = _fresh_journal(leg_templates[leg], count=3)
    target_journal = _fresh_journal(leg_templates[leg], count=0)
    source_target = source.op_events[2]
    source.append_amendment(
        amend_raw_hook_intervention(
            source_target.seq, source_target.label_raw, intervention_replaced=True
        )
    )
    target_journal.concat(source)
    merged = target_journal.op_amendments
    assert len(merged) == 1
    merged_amendment = merged[0]
    # Re-stamped into the target's lane-local domain and run nonce.
    assert merged_amendment.seq == 1
    assert merged_amendment.run_nonce == target_journal.run_nonce
    # target_seq rebound through the merge seq map onto the merged clone.
    merged_target = target_journal.op_event_by_label_raw[source_target.label_raw]
    assert merged_amendment.target_seq == merged_target.seq
    assert target_journal.amended_op_record(source_target.label_raw).intervention_replaced


class _RecordModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(20260812 + 70)
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def test_seal_dual_stamps_live_journal_and_projection_clone() -> None:
    """S-N2 integration: the pre-seal snapshot clone carries the watermark.

    ``snapshot_recording_projection`` clones the journal BEFORE ``seal()``
    runs (capture/trace.py), so the seal must stamp BOTH the live journal and
    the stored clone; the RecordingProjector additionally sources the core's
    own watermark as a fallback. A fresh recording then converts to a trace
    with the carried lane filtered empty (nothing folded twice).
    """

    torch.manual_seed(20260812 + 71)
    recording = tl.record(_RecordModel(), torch.randn(2, 4), save=tl.func("relu"))
    cores = object.__getattribute__(recording, "_captured_run_cores")
    assert cores and cores[0].amendment_watermark is not None
    # The projection clone was made BEFORE the capture-end seal; the seal's
    # dual-stamp must still have reached it (S-N2). It stays a working
    # projection: unsealed, append-capable.
    projection_clone = cores[0].projection_facts.get("capture_events")
    assert projection_clone is not None
    assert projection_clone.core_seal_watermark == cores[0].amendment_watermark
    assert projection_clone.amendments_sealed is False
    # The recorder's accumulation buffer is a separate unsealed stream (the
    # sealed live journal is trace-side and released with the session); the
    # sealed-source refusal mechanics are covered by the unit test above.
    trace = recording.to_trace()
    assert trace is not None and trace.layer_list


@pytest.mark.parametrize("leg", _LEGS)
def test_graph_edge_insertion_folds_parents(leg: str, leg_templates: dict[str, list[Any]]) -> None:
    """A registered-connection amendment folds parent facts on both legs."""

    journal = _fresh_journal(leg_templates[leg])
    parent, child = journal.op_events[0], journal.op_events[1]
    from torchlens.ir import ParentEdge

    new_parents = (
        *child.parents,
        ParentEdge(parent_label_raw=parent.label_raw, arg_position=None, edge_use="output"),
    )
    positions = {"args": {0: parent.label_raw}}
    journal.append_amendment(
        amend_graph_edge_insertion(
            child.seq,
            child.label_raw,
            parents=new_parents,
            parent_arg_positions=positions,
        )
    )
    folded = journal.amended_op_record(child.label_raw)
    assert folded.parents == new_parents
    assert folded.parent_arg_positions is positions


# ---------------------------------------------------------------------------
# FINDING B1-15: sealed-journal enforcement was HALF-HEIGHT.
#
# Only ``append_amendment`` checked ``amendments_sealed``; the op lane's
# ``append`` and every forward sibling appender wrote to a sealed source
# unconditionally, so code holding a sealed journal could diverge the live
# stream from its sealed ``CapturedRunCore`` snapshot with no signal.
# ---------------------------------------------------------------------------

#: Every appender the seal now fences, with the lane it names in the refusal.
_FENCED_APPENDERS: tuple[tuple[str, str], ...] = (
    ("append", "op_events"),
    ("append_module_prep", "module_prep_events"),
    ("append_module_enter", "module_enter_events"),
    ("append_module_exit", "module_exit_events"),
    ("append_pre_hook", "pre_hook_events"),
    ("append_buffer_write", "buffer_write_events"),
    ("append_intervention", "intervention_events"),
    ("append_output_version", "output_version_events"),
)


class _DummyEvent:
    """Stand-in event: the seal fence fires before any field is stamped."""

    seq: int = 0


@pytest.mark.smoke_cells(
    "test_sealed_journal_refuses_every_forward_lane_append[append-op_events]",
    "test_sealed_journal_refuses_every_forward_lane_append[append_module_prep-module_prep_events]",
)
@pytest.mark.parametrize(("appender", "lane"), _FENCED_APPENDERS, ids=lambda value: str(value))
def test_sealed_journal_refuses_every_forward_lane_append(
    appender: str, lane: str, leg_templates: dict[str, list[Any]]
) -> None:
    """Each forward lane gets the fence the amendment lane always had."""

    journal = _fresh_journal(leg_templates["decomposed"])
    lane_length_before = len(getattr(journal, lane))
    counter_before = journal.event_seq
    journal.core_seal_watermark = journal.amendment_seq
    journal.amendments_sealed = True

    with pytest.raises(SealedJournalAppendError, match=lane):
        getattr(journal, appender)(_DummyEvent())

    # Fail-closed: the refusal happens BEFORE the lane or the counter moves.
    assert len(getattr(journal, lane)) == lane_length_before
    assert journal.event_seq == counter_before


def test_sealed_journal_refusals_share_one_catchable_base(
    leg_templates: dict[str, list[Any]],
) -> None:
    """Amendment and forward-lane refusals are one class of defect."""

    journal = _fresh_journal(leg_templates["decomposed"])
    target = _target(journal)
    journal.core_seal_watermark = journal.amendment_seq
    journal.amendments_sealed = True
    with pytest.raises(SealedJournalWriteError):
        journal.append(_DummyEvent())
    with pytest.raises(SealedJournalWriteError):
        journal.append_amendment(
            amend_raw_hook_intervention(target.seq, target.label_raw, intervention_replaced=True)
        )
    assert issubclass(SealedJournalAppendError, SealedJournalWriteError)
    assert issubclass(SealedJournalAmendmentError, SealedJournalWriteError)


def test_sealed_journal_still_accepts_backward_events(
    leg_templates: dict[str, list[Any]],
) -> None:
    """The ONE documented exception: backward capture runs after the seal.

    ``Trace.log_backward`` / ``Recording.log_backward`` append to the sealed
    journal by design and ``backward_events`` is not part of the sealed core, so
    fencing that lane would break real backward capture. Measured across the
    smoke tier: ``append_backward`` is the only appender that ever fires on a
    sealed journal.
    """

    from torchlens.ir.events import BackwardCoverageGap

    journal = _fresh_journal(leg_templates["decomposed"])
    journal.core_seal_watermark = journal.amendment_seq
    journal.amendments_sealed = True
    journal.append_backward(
        BackwardCoverageGap(
            pass_index=1,
            object_id=None,
            class_qualname=None,
            reason="dead_node",
            detail=None,
            timestamp=1.0,
        )
    )
    assert len(journal.backward_events) == 1
    assert journal.backward_events[0].seq >= 1


def test_working_projection_of_a_sealed_journal_accepts_forward_appends(
    leg_templates: dict[str, list[Any]],
) -> None:
    """The fence names the escape hatch, and the escape hatch works."""

    journal = _fresh_journal(leg_templates["decomposed"])
    journal.core_seal_watermark = journal.amendment_seq
    journal.amendments_sealed = True
    working = journal.copy_for_replay()
    assert working.amendments_sealed is False
    template = journal.op_events[0]
    working.append(_clone_op_event_for_replay(template))
    assert len(working.op_events) == len(journal.op_events) + 1
    # And the sealed source is untouched.
    assert len(journal.op_events) == len(working.op_events) - 1


def test_extend_is_fenced_through_append(leg_templates: dict[str, list[Any]]) -> None:
    """``extend`` delegates to ``append``, so it inherits the fence."""

    journal = _fresh_journal(leg_templates["decomposed"])
    template = journal.op_events[0]
    journal.core_seal_watermark = journal.amendment_seq
    journal.amendments_sealed = True
    with pytest.raises(SealedJournalAppendError):
        journal.extend([_clone_op_event_for_replay(template)])


def test_concat_refuses_amendments_without_the_op_lane(
    leg_templates: dict[str, list[Any]],
) -> None:
    """Companion LOW (fable R03-F2): an empty seq map cannot rebind targets.

    Merging ``op_amendments`` without ``op_events`` left ``seq_map`` empty, so
    every ``target_seq`` silently kept its SOURCE-domain value and then bound
    against whatever the target journal holds at that seq. The lane selection is
    now refused instead.
    """

    source = _fresh_journal(leg_templates["decomposed"], count=3)
    source_target = source.op_events[2]
    source.append_amendment(
        amend_raw_hook_intervention(
            source_target.seq, source_target.label_raw, intervention_replaced=True
        )
    )
    target_journal = _fresh_journal(leg_templates["decomposed"], count=0)
    with pytest.raises(LaneMergePolicyError, match="without 'op_events'"):
        target_journal.concat(source, lanes=("op_amendments",))
    assert not target_journal.op_amendments

    # An amendment-free source with the same lane selection stays legal: there
    # is nothing whose target could dangle.
    empty_source = _fresh_journal(leg_templates["decomposed"], count=2)
    target_journal.concat(empty_source, lanes=("op_amendments",))
    assert not target_journal.op_amendments


def test_concat_refuses_an_unrebindable_amendment_target(
    leg_templates: dict[str, list[Any]],
) -> None:
    """A target_seq with no merged op is refused, never silently carried."""

    import dataclasses

    source = _fresh_journal(leg_templates["decomposed"], count=3)
    source_target = source.op_events[2]
    source.append_amendment(
        amend_raw_hook_intervention(
            source_target.seq, source_target.label_raw, intervention_replaced=True
        )
    )
    # Forge a target_seq the merge map cannot possibly contain.
    source.op_amendments[0] = dataclasses.replace(source.op_amendments[0], target_seq=10_000)
    target_journal = _fresh_journal(leg_templates["decomposed"], count=0)
    with pytest.raises(LaneMergePolicyError, match="no merged op"):
        target_journal.concat(source)
