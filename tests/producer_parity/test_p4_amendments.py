"""P4 amendment lane: dual-leg reducer, watermark transport, lifecycle.

Covers DoR 4.3-4.6 plus the binding reviewer notes:

* append-only op lane with lane-local amendment seq (Tier-F event ``seq``
  facts never shift when an amendment lands),
* canonical reducer fold (amendment order, last-wins per path, absent-facet
  materialization from ``FACET_DEFAULTS`` — with the planted S5 violation),
* runtime validation at append AND fold; fail-closed target resolution,
* sealed-source refusal vs working-projection acceptance (DoR 4.5.5),
* seal watermark dual-stamp incl. the pre-seal projection clone (S-N2) and
  the ``_copy_for_replay`` seq-filter (no double-application),
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
from torchlens._ir.capture_events import (
    AmendmentTargetError,
    CaptureEvents,
    LaneMergePolicyError,
    SealedJournalAmendmentError,
    SealedJournalAppendError,
    SealedJournalWriteError,
    _clone_op_event_for_replay,
)
from torchlens._ir.op_record import (
    AmendmentValidationError,
    OpAmendment,
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
                    ("intervention.is_edited", True),
                    ("intervention.is_edited", False),
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
                (("intervention.is_edited", "yes"),),
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
            amend_raw_hook_intervention(1, "nonexistent_1_1_raw", is_edited=True)
        )
    # Single-domain journals cross-check target_seq.
    with pytest.raises(AmendmentTargetError, match="single-seq-domain"):
        journal.append_amendment(
            amend_raw_hook_intervention(target.seq + 7, target.label_raw, is_edited=True)
        )


@pytest.mark.parametrize("leg", _LEGS)
def test_multi_domain_resolves_by_label_last_occurrence(
    leg: str, leg_templates: dict[str, list[Any]]
) -> None:
    journal = _fresh_journal(leg_templates[leg])
    folded = list(journal.amended_op_records())
    # Simulate a multi-pass concatenated projection: the same sealed fold
    # twice — colliding seqs, duplicate labels, no single seq domain.
    seeded = journal._copy_for_replay(projected_op_events=folded + folded)
    assert seeded.single_seq_domain is False
    target = folded[1]
    # target_seq disagreement is NOT cross-checked in a multi-domain journal
    # (O-N6): resolution is by label.
    seeded.append_amendment(
        amend_raw_hook_intervention(target.seq + 12345, target.label_raw, is_edited=True)
    )
    view = seeded.amended_op_records()
    positions = [index for index, event in enumerate(view) if event.label_raw == target.label_raw]
    assert len(positions) == 2
    first_occurrence, last_occurrence = positions
    assert view[last_occurrence].is_edited is True
    assert view[first_occurrence].is_edited is False
    # Single-domain journals DO cross-check (established by the refusal test).


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
        amend_raw_hook_intervention(target_one.seq, target_one.label_raw, is_edited=True)
    )
    pass_two = _fresh_journal(leg_templates[leg], count=2)
    target_two = _target(pass_two)
    pass_two.append_amendment(
        amend_raw_hook_intervention(target_two.seq, target_two.label_raw, is_edited=False)
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
    assert view[first_occurrence].is_edited is True
    assert view[last_occurrence].is_edited is False

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
    assert combined_view[combined_positions[0]].is_edited is True
    assert combined_view[combined_positions[1]].is_edited is False


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
    runs (_capture/trace.py), so the seal must stamp BOTH the live journal and
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
    assert trace is not None and trace._layer_list


@pytest.mark.parametrize("leg", _LEGS)
def test_graph_edge_insertion_folds_parents(leg: str, leg_templates: dict[str, list[Any]]) -> None:
    """A registered-connection amendment folds parent facts on both legs."""

    journal = _fresh_journal(leg_templates[leg])
    parent, child = journal.op_events[0], journal.op_events[1]
    from torchlens._ir import ParentEdge

    new_parents = (
        *child.parents,
        ParentEdge(parent_label_raw=parent.label_raw, arg_position=None, _edge_use="output"),
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
            amend_raw_hook_intervention(target.seq, target.label_raw, is_edited=True)
        )
    assert issubclass(SealedJournalAppendError, SealedJournalWriteError)
    assert issubclass(SealedJournalAmendmentError, SealedJournalWriteError)


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
        amend_raw_hook_intervention(source_target.seq, source_target.label_raw, is_edited=True)
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
        amend_raw_hook_intervention(source_target.seq, source_target.label_raw, is_edited=True)
    )
    # Forge a target_seq the merge map cannot possibly contain.
    source.op_amendments[0] = dataclasses.replace(source.op_amendments[0], target_seq=10_000)
    target_journal = _fresh_journal(leg_templates["decomposed"], count=0)
    with pytest.raises(LaneMergePolicyError, match="no merged op"):
        target_journal.concat(source)
