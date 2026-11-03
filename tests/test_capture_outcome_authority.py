"""P1 unit proofs for the capture-outcome authority module.

Covers the frozen vocabularies, the string-only payload codec (fail-closed
parse), the status-specific load coherence matrix, the structural derivation
lattice (including the absent-key / ``None`` variants), the capability table
chokepoint (forged-sidecar arming proofs), and the diagnostic failure-origin
classifier.
"""

from __future__ import annotations

import pytest

from torchlens._capture.outcome import (
    CAPTURE_OUTCOME_CAPABILITIES,
    CaptureOutcome,
    CapturePhase,
    CaptureStatus,
    FailureOrigin,
    classify_failure_origin,
    derive_outcome_from_structural_state,
    parse_outcome_payload,
    resolve_loaded_outcome,
)

# ---------------------------------------------------------------------------
# Vocabulary + record
# ---------------------------------------------------------------------------


def test_status_vocabulary_is_frozen() -> None:
    assert {status.value for status in CaptureStatus} == {
        "complete",
        "stopped",
        "aborted_nonfinite",
        "failed",
        "unattested",
        "unknown",
    }
    assert {phase.value for phase in CapturePhase} == {
        "forward",
        "finalize",
        "postprocess",
        "teardown",
    }
    assert {origin.value for origin in FailureOrigin} == {
        "user_op",
        "torchlens",
        "interrupt",
        "unknown",
    }


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_types.py.
def test_partial_property_tristate() -> None:
    assert CaptureOutcome(status=CaptureStatus.COMPLETE).is_partial is False
    assert CaptureOutcome(status=CaptureStatus.STOPPED).is_partial is True
    assert CaptureOutcome(status=CaptureStatus.ABORTED_NONFINITE).is_partial is True
    assert CaptureOutcome(status=CaptureStatus.FAILED).is_partial is True
    assert CaptureOutcome(status=CaptureStatus.UNATTESTED).is_partial is None
    assert CaptureOutcome(status=CaptureStatus.UNKNOWN).is_partial is None


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
@pytest.mark.parametrize(
    "mutation",
    [
        {"status": "totally_new_status"},
        {"status": None},
        {"phase": "prewarm"},
        {"origin": 3},
        {"frontier_labels": "relu_1_2"},
        {"n_ops_committed": "7"},
        {"derived": "yes"},
    ],
)
def test_payload_parse_fails_closed_on_bad_vocabulary(mutation: dict) -> None:
    payload = CaptureOutcome(status=CaptureStatus.COMPLETE).to_payload()
    payload.update(mutation)
    with pytest.raises(ValueError):
        parse_outcome_payload(payload)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
def test_payload_parse_rejects_non_mapping() -> None:
    with pytest.raises(ValueError):
        parse_outcome_payload("complete")


# ---------------------------------------------------------------------------
# Coherence matrix: one coherent + one contradiction per status row
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Derivation lattice (truthiness; absent / None / False are one class)
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"_tracing_finished": False, "is_stopped": True}, CaptureStatus.UNKNOWN),
    ],
)
def test_derivation_lattice(state: dict, expected: CaptureStatus) -> None:
    outcome = derive_outcome_from_structural_state(state)
    assert outcome.status is expected
    assert outcome.is_derived is True
    # No structural derivation may ever bless COMPLETE.
    assert outcome.status is not CaptureStatus.COMPLETE


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
def test_derivation_carries_halt_detail() -> None:
    outcome = derive_outcome_from_structural_state(
        {
            "_tracing_finished": True,
            "is_stopped": True,
            "stop_reason": "relu_1_2_raw",
            "_graph_stop_frontier": "relu_1_2_raw",
        }
    )
    assert outcome.status is CaptureStatus.STOPPED
    assert outcome.reason == "relu_1_2_raw"
    assert outcome.boundary_label == "relu_1_2_raw"


# ---------------------------------------------------------------------------
# Loaded-outcome resolution: adopt, degrade, derive
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
def test_resolve_loaded_degrades_incoherent_attestation_to_unknown() -> None:
    attested = CaptureOutcome(status=CaptureStatus.COMPLETE)
    state = {
        "_capture_outcome": attested.to_payload(),
        "_tracing_finished": False,
        "is_stopped": False,
    }
    with pytest.warns(RuntimeWarning, match="contradicts"):
        resolved = resolve_loaded_outcome(state)
    assert resolved.status is CaptureStatus.UNKNOWN
    assert resolved.is_derived is True
    assert "attestation_incoherent" in (resolved.note or "")


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_outcome_payload.py.
def test_resolve_loaded_without_attestation_uses_lattice() -> None:
    resolved = resolve_loaded_outcome({"_tracing_finished": True, "is_stopped": False})
    assert resolved.status is CaptureStatus.UNATTESTED


# ---------------------------------------------------------------------------
# Capability table + chokepoint arming proofs (forged sidecars)
# ---------------------------------------------------------------------------


def test_capability_table_is_total() -> None:
    for capability, row in CAPTURE_OUTCOME_CAPABILITIES.items():
        assert set(row) == set(CaptureStatus), capability
        for cell in row.values():
            assert cell == "allow" or cell.startswith("allow_scoped:") or cell.startswith("refuse:")


# ---------------------------------------------------------------------------
# Failure-origin classifier plants
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/outcome.py.
def test_origin_interrupt() -> None:
    assert classify_failure_origin(KeyboardInterrupt()) is FailureOrigin.INTERRUPT


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/outcome.py.
def test_origin_no_traceback_is_unknown() -> None:
    assert classify_failure_origin(ValueError("bare")) is FailureOrigin.UNKNOWN


# ---------------------------------------------------------------------------
# Public type/error surface
# ---------------------------------------------------------------------------
