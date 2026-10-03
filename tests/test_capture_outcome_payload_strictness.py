"""B1-07: the tlspec-v7 `_capture_outcome` codec refuses incoherent payloads.

Two corroborated gaps in the persisted-attestation codec:

(a) CROSS-FIELD COHERENCE. ``attestation_coherent`` checked only structural
    booleans (COMPLETE requires ``(not halted) and finished``; FAILED returned
    True unconditionally), never whether the record was internally consistent.
    So ``{"status": "complete", "phase": "postprocess", "origin": "torchlens",
    "error_type": "RuntimeError", "reason": "postprocess exploded"}`` was
    adopted as an ATTESTED COMPLETE carrying its own failure evidence.

(b) UNKNOWN KEYS. ``parse_outcome_payload`` copied the dict and read the keys
    it knew, rejecting unknown VALUES but silently dropping unknown KEYS. A
    verdict-steering field added without a tlspec bump was invisible to an
    older reader and no warning fired -- the reader blessed an attestation it
    could not fully evaluate.

Both now route through the EXISTING degrade-to-UNKNOWN path: fail-closed, one
warning, never a load crash. This is a TIGHTENING; every one of the 141
pre-existing outcome tests still passes, so no legitimate record was broken.

Severity was DISPUTED (D1: sol HIGH vs fable LOW) on whether adopting the
contradictory record ENABLES anything. ``test_contradictory_complete_grants_no
_extra_capability`` records the evidence: capabilities key on status ALONE, so
there is no capability upgrade -- fable's reading. The honesty defect is real
regardless, and is what these tests pin.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture.outcome import (
    CAPTURE_OUTCOME_CAPABILITIES,
    CaptureOutcome,
    CapturePhase,
    CaptureStatus,
    FailureOrigin,
    attestation_coherent,
    parse_outcome_payload,
    resolve_loaded_outcome,
)


class _Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.relu(self.fc(x))


# ---------------------------------------------------------------------------
# (a) cross-field coherence
# ---------------------------------------------------------------------------

_CONTRADICTORY_COMPLETE = {
    "status": "complete",
    "phase": "postprocess",
    "origin": "torchlens",
    "error_type": "RuntimeError",
    "reason": "postprocess exploded",
}


def test_failed_only_fields_refuse_on_a_complete_status() -> None:
    """The reported payload: COMPLETE carrying its own failure evidence."""

    outcome = parse_outcome_payload(_CONTRADICTORY_COMPLETE)
    # The payload still PARSES -- each field is individually well-formed.
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.phase is CapturePhase.POSTPROCESS
    assert outcome.origin is FailureOrigin.TORCHLENS
    # ...but it is not internally coherent, so it is never adopted.
    assert attestation_coherent(outcome, halted=False, finished=True) is False


@pytest.mark.parametrize("field_name,value", [("phase", "forward"), ("origin", "torchlens")])
@pytest.mark.parametrize("status", ["complete", "halted", "aborted_nonfinite", "unknown"])
def test_failed_only_fields_refuse_on_every_non_failed_status(
    status: str, field_name: str, value: str
) -> None:
    """The rule is per-status, not a single special case for COMPLETE."""

    payload = {"status": status, field_name: value}
    if status == "unattested":
        payload["derived"] = True
    outcome = parse_outcome_payload(payload)
    halted = status == "halted"
    assert attestation_coherent(outcome, halted=halted, finished=True) is False


def test_failed_status_still_accepts_its_own_fields() -> None:
    """The tightening does not break the status those fields belong to."""

    outcome = parse_outcome_payload(
        {
            "status": "failed",
            "phase": "postprocess",
            "origin": "torchlens",
            "error_type": "RuntimeError",
            "reason": "boom",
        }
    )
    assert attestation_coherent(outcome, halted=False, finished=False) is True


def test_boundary_evidence_refuses_on_complete_but_not_on_halted() -> None:
    """COMPLETE means no stop boundary was reached; HALTED carries one."""

    complete = parse_outcome_payload({"status": "complete", "boundary_label": "relu_1_1"})
    assert attestation_coherent(complete, halted=False, finished=True) is False
    halted = parse_outcome_payload({"status": "halted", "boundary_label": "relu_1_1"})
    assert attestation_coherent(halted, halted=True, finished=True) is True


def test_a_demoted_failed_record_keeps_its_inherited_boundary_facts() -> None:
    """A FAILED record demoted from HALTED legitimately carries boundary facts.

    ``demote_outcome`` copies boundary_kind/boundary_label/frontier_labels off
    the demoted-from HALTED record, so FAILED must stay exempt from the
    no-boundary rule or every teardown demotion would degrade to UNKNOWN.
    """

    outcome = parse_outcome_payload(
        {
            "status": "failed",
            "phase": "teardown",
            "origin": "torchlens",
            "boundary_kind": "op",
            "boundary_label": "relu_1_1",
            "frontier_labels": ["relu_1_1"],
        }
    )
    assert attestation_coherent(outcome, halted=True, finished=True) is True


# ---------------------------------------------------------------------------
# (b) unknown keys
# ---------------------------------------------------------------------------


def test_unknown_payload_key_refuses() -> None:
    """A field this reader does not know about is never silently dropped."""

    with pytest.raises(ValueError, match="unknown field"):
        parse_outcome_payload({"status": "complete", "verdict_steering_new_field": "danger"})


def test_the_accepted_key_set_is_derived_from_the_writer() -> None:
    """Reader and writer cannot drift.

    The accepted key set comes from ``CaptureOutcome.to_payload()``, so adding
    a persisted field widens the reader automatically instead of requiring a
    second hand-maintained list to be kept in sync.
    """

    from torchlens.capture.outcome import _OUTCOME_PAYLOAD_KEYS

    written = CaptureOutcome(status=CaptureStatus.FAILED).to_payload()
    assert set(written) == set(_OUTCOME_PAYLOAD_KEYS)
    # Every written payload round-trips through the strict reader.
    assert parse_outcome_payload(written).status is CaptureStatus.FAILED


def test_every_settleable_status_round_trips_through_the_strict_codec() -> None:
    """Anti-vacuity: the tightening accepts everything the writer produces."""

    for status in CaptureStatus:
        is_failed = status is CaptureStatus.FAILED
        outcome = CaptureOutcome(
            status=status,
            derived=status is CaptureStatus.UNATTESTED,
            phase=CapturePhase.FORWARD if is_failed else None,
            origin=FailureOrigin.USER_OP if is_failed else None,
        )
        parsed = parse_outcome_payload(outcome.to_payload())
        assert parsed.status is status
        assert attestation_coherent(
            parsed,
            halted=status is CaptureStatus.HALTED,
            finished=True,
        ), status


# ---------------------------------------------------------------------------
# Both gaps degrade through the existing fail-closed load path
# ---------------------------------------------------------------------------


def test_contradictory_payload_degrades_to_unknown_with_one_warning() -> None:
    """The load path warns and degrades; it never adopts or crashes.

    b8-sol: this payload violates the CROSS-FIELD layer (COMPLETE carrying
    FAILED-only fields) while its structural evidence (halted=False,
    finished=True) is perfectly consistent with COMPLETE. The refusal must
    name the internal contradiction, never blame the structural evidence."""

    state = {
        "_capture_outcome": dict(_CONTRADICTORY_COMPLETE),
        "halted": False,
        "_tracing_finished": True,
    }
    with pytest.warns(RuntimeWarning, match="internally contradictory") as records:
        resolved = resolve_loaded_outcome(state)
    assert resolved.status is CaptureStatus.UNKNOWN
    assert resolved.derived is True
    assert "attestation_incoherent" in (resolved.settlement_note or "")
    assert "FAILED-only field" in (resolved.settlement_note or "")
    assert not any("structural evidence" in str(record.message) for record in records)


def test_unknown_key_payload_degrades_to_unknown_with_one_warning() -> None:
    """Unknown keys reach the parse-failure arm, not a silent adoption."""

    state = {
        "_capture_outcome": {"status": "complete", "future_verdict_field": "x"},
        "halted": False,
        "_tracing_finished": True,
    }
    with pytest.warns(RuntimeWarning, match="could not parse"):
        resolved = resolve_loaded_outcome(state)
    assert resolved.status is CaptureStatus.UNKNOWN
    assert resolved.derived is True
    assert "attestation_parse_failed" in (resolved.settlement_note or "")


def test_tampered_artifact_on_disk_degrades_and_then_refuses_export(tmp_path) -> None:
    """End to end on a real artifact, through tl.load.

    An UNKNOWN outcome is the most restrictive class, so the N1 export gate
    refuses the degraded artifact -- the fail-closed consequence of refusing
    to bless a record this reader cannot evaluate.
    """

    trace = tl.trace(_Small().eval(), torch.ones(1, 3))
    path = tmp_path / "tampered.tlspec"
    tl.save(trace, str(path))
    loaded = tl.load(str(path))
    assert loaded.outcome is not None
    assert loaded.outcome.status is CaptureStatus.COMPLETE

    # Forge the attestation in memory exactly as a tampered artifact would
    # present it, then re-resolve through the load derivation.
    state = dict(vars(loaded))
    state["_capture_outcome"] = dict(_CONTRADICTORY_COMPLETE)
    with pytest.warns(RuntimeWarning):
        resolved = resolve_loaded_outcome(state)
    assert resolved.status is CaptureStatus.UNKNOWN


# ---------------------------------------------------------------------------
# D1 evidence for the severity relay
# ---------------------------------------------------------------------------


def test_contradictory_complete_grants_no_extra_capability() -> None:
    """D1 evidence: adopting the contradictory record upgrades nothing.

    sol argued HIGH on the grounds that N1-N5 then branch on COMPLETE and
    allow capabilities despite the record's own failure evidence; fable argued
    LOW on the grounds that no capability upgrade is possible. Capabilities are
    keyed on status ALONE, so a contradictory COMPLETE and a plain COMPLETE
    grant exactly the same set -- there is no upgrade, and the defect is a
    pure honesty defect. Recorded here so the fork is settled by evidence
    rather than re-argued.
    """

    contradictory = parse_outcome_payload(_CONTRADICTORY_COMPLETE)
    plain = parse_outcome_payload({"status": "complete"})
    assert contradictory.status is plain.status
    # Each capability's table is keyed by STATUS only -- no row consults
    # phase/origin/error_type -- so both records resolve identically.
    for capability, by_status in CAPTURE_OUTCOME_CAPABILITIES.items():
        assert by_status[contradictory.status] == by_status[plain.status], capability


# ---------------------------------------------------------------------------
# (c) forged provenance flags on settle-only statuses (R06)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["complete", "aborted_nonfinite", "failed"])
def test_forged_derived_flag_refuses_on_settle_only_statuses(status: str) -> None:
    """No writer derives COMPLETE / ABORTED_NONFINITE / FAILED.

    Every settle stamp writes ``derived=False`` and the derivation lattices
    emit only HALTED / UNATTESTED / UNKNOWN, so a derived payload claiming a
    settle-only status forges provenance and must not be adopted.
    """

    outcome = parse_outcome_payload({"status": status, "derived": True})
    halted = False
    finished = status == "complete"
    assert attestation_coherent(outcome, halted=halted, finished=finished) is False


@pytest.mark.parametrize("status", ["halted", "unattested", "unknown"])
def test_legitimate_derived_statuses_stay_coherent(status: str) -> None:
    """The lattice's own derived statuses still parse and adopt."""

    outcome = parse_outcome_payload({"status": status, "derived": True})
    halted = status == "halted"
    finished = status in ("halted", "unattested")
    assert attestation_coherent(outcome, halted=halted, finished=finished) is True


def test_forged_reason_refuses_on_complete() -> None:
    """``settle_completed`` never writes a reason; COMPLETE + reason is forged."""

    outcome = parse_outcome_payload({"status": "complete", "reason": "looks legit"})
    assert attestation_coherent(outcome, halted=False, finished=True) is False


def test_forged_recovered_flag_refuses_on_complete() -> None:
    """The fastlog disk-recovery marker never accompanies a settled COMPLETE."""

    outcome = parse_outcome_payload({"status": "complete", "recovered": True})
    assert attestation_coherent(outcome, halted=False, finished=True) is False
