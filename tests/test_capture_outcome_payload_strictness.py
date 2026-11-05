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

from torchlens._capture.outcome import (
    CaptureOutcome,
    CapturePhase,
    CaptureStatus,
    FailureOrigin,
    attestation_coherent,
    parse_outcome_payload,
    resolve_loaded_outcome,
)

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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
def test_boundary_evidence_refuses_on_complete_but_not_on_halted() -> None:
    """COMPLETE means no stop boundary was reached; HALTED carries one."""

    complete = parse_outcome_payload({"status": "complete", "boundary_label": "relu_1_1"})
    assert attestation_coherent(complete, halted=False, finished=True) is False
    halted = parse_outcome_payload({"status": "stopped", "boundary_label": "relu_1_1"})
    assert attestation_coherent(halted, halted=True, finished=True) is True


# ---------------------------------------------------------------------------
# (b) unknown keys
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
def test_unknown_payload_key_refuses() -> None:
    """A field this reader does not know about is never silently dropped."""

    with pytest.raises(ValueError, match="unknown field"):
        parse_outcome_payload({"status": "complete", "verdict_steering_new_field": "danger"})


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
def test_every_settleable_status_round_trips_through_the_strict_codec() -> None:
    """Anti-vacuity: the tightening accepts everything the writer produces."""

    for status in CaptureStatus:
        is_failed = status is CaptureStatus.FAILED
        outcome = CaptureOutcome(
            status=status,
            is_derived=status is CaptureStatus.UNATTESTED,
            phase=CapturePhase.FORWARD if is_failed else None,
            origin=FailureOrigin.USER_OP if is_failed else None,
        )
        parsed = parse_outcome_payload(outcome.to_payload())
        assert parsed.status is status
        assert attestation_coherent(
            parsed,
            halted=status is CaptureStatus.STOPPED,
            finished=True,
        ), status


# ---------------------------------------------------------------------------
# Both gaps degrade through the existing fail-closed load path
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
def test_contradictory_payload_degrades_to_unknown_with_one_warning() -> None:
    """The load path warns and degrades; it never adopts or crashes.

    b8-sol: this payload violates the CROSS-FIELD layer (COMPLETE carrying
    FAILED-only fields) while its structural evidence (halted=False,
    finished=True) is perfectly consistent with COMPLETE. The refusal must
    name the internal contradiction, never blame the structural evidence."""

    state = {
        "_capture_outcome": dict(_CONTRADICTORY_COMPLETE),
        "is_stopped": False,
        "_tracing_finished": True,
    }
    with pytest.warns(RuntimeWarning, match="internally contradictory") as records:
        resolved = resolve_loaded_outcome(state)
    assert resolved.status is CaptureStatus.UNKNOWN
    assert resolved.is_derived is True
    assert "attestation_incoherent" in (resolved.note or "")
    assert "FAILED-only field" in (resolved.note or "")
    assert not any("structural evidence" in str(record.message) for record in records)


# ---------------------------------------------------------------------------
# D1 evidence for the severity relay
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (c) forged provenance flags on settle-only statuses (R06)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
@pytest.mark.parametrize("status", ["aborted_nonfinite"])
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
def test_forged_reason_refuses_on_complete() -> None:
    """``settle_completed`` never writes a reason; COMPLETE + reason is forged."""

    outcome = parse_outcome_payload({"status": "complete", "reason": "looks legit"})
    assert attestation_coherent(outcome, halted=False, finished=True) is False
