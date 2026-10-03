"""P1 unit proofs for the capture-outcome authority module.

Covers the frozen vocabularies, the string-only payload codec (fail-closed
parse), the status-specific load coherence matrix, the structural derivation
lattice (including the absent-key / ``None`` variants), the capability table
chokepoint (forged-sidecar arming proofs), and the diagnostic failure-origin
classifier.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture.outcome import (
    CAPTURE_OUTCOME_CAPABILITIES,
    CaptureOutcome,
    CaptureOutcomeError,
    CapturePhase,
    CaptureStatus,
    FailureOrigin,
    attestation_coherent,
    classify_failure_origin,
    derive_outcome_from_structural_state,
    parse_outcome_payload,
    require_capture_capability,
    resolve_loaded_outcome,
)

# ---------------------------------------------------------------------------
# Vocabulary + record
# ---------------------------------------------------------------------------


def test_status_vocabulary_is_frozen() -> None:
    assert {status.value for status in CaptureStatus} == {
        "complete",
        "halted",
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


def test_partial_property_tristate() -> None:
    assert CaptureOutcome(status=CaptureStatus.COMPLETE).partial is False
    assert CaptureOutcome(status=CaptureStatus.HALTED).partial is True
    assert CaptureOutcome(status=CaptureStatus.ABORTED_NONFINITE).partial is True
    assert CaptureOutcome(status=CaptureStatus.FAILED).partial is True
    assert CaptureOutcome(status=CaptureStatus.UNATTESTED).partial is None
    assert CaptureOutcome(status=CaptureStatus.UNKNOWN).partial is None


def test_payload_round_trip() -> None:
    outcome = CaptureOutcome(
        status=CaptureStatus.FAILED,
        phase=CapturePhase.POSTPROCESS,
        origin=FailureOrigin.TORCHLENS,
        reason="planted",
        error_type="ValueError",
        boundary_kind="op",
        boundary_label="relu_1_2_raw",
        frontier_labels=("relu_1_2",),
        n_ops_committed=7,
        inference_only=True,
        settlement_note="secondary: cleanup",
    )
    parsed = parse_outcome_payload(outcome.to_payload())
    assert parsed == outcome


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


def test_payload_parse_rejects_non_mapping() -> None:
    with pytest.raises(ValueError):
        parse_outcome_payload("complete")


# ---------------------------------------------------------------------------
# Coherence matrix: one coherent + one contradiction per status row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "derived", "halted", "finished", "coherent"),
    [
        (CaptureStatus.COMPLETE, False, False, True, True),
        (CaptureStatus.COMPLETE, False, True, True, False),
        (CaptureStatus.COMPLETE, False, False, False, False),
        # Attested HALTED tolerates falsy finished (the re-raise path).
        (CaptureStatus.HALTED, False, True, False, True),
        (CaptureStatus.HALTED, False, True, True, True),
        (CaptureStatus.HALTED, False, False, True, False),
        (CaptureStatus.ABORTED_NONFINITE, False, False, False, True),
        (CaptureStatus.ABORTED_NONFINITE, False, True, False, False),
        # Attested FAILED tolerates the pre-stamped halted=True secondary class.
        (CaptureStatus.FAILED, False, True, False, True),
        (CaptureStatus.FAILED, False, False, True, True),
        (CaptureStatus.UNKNOWN, False, True, True, True),
        (CaptureStatus.UNKNOWN, False, False, False, True),
        # UNATTESTED requires derived=True (a settle stamp can never write it).
        (CaptureStatus.UNATTESTED, True, False, True, True),
        (CaptureStatus.UNATTESTED, False, False, True, False),
        (CaptureStatus.UNATTESTED, True, True, True, False),
    ],
)
def test_coherence_matrix(
    status: CaptureStatus, derived: bool, halted: bool, finished: bool, coherent: bool
) -> None:
    outcome = CaptureOutcome(status=status, derived=derived)
    assert attestation_coherent(outcome, halted=halted, finished=finished) is coherent


# ---------------------------------------------------------------------------
# Derivation lattice (truthiness; absent / None / False are one class)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"_tracing_finished": True, "halted": True}, CaptureStatus.HALTED),
        ({"_tracing_finished": True, "halted": False}, CaptureStatus.UNATTESTED),
        ({"_tracing_finished": True}, CaptureStatus.UNATTESTED),
        ({"_tracing_finished": False, "halted": True}, CaptureStatus.UNKNOWN),
        ({"_tracing_finished": None, "halted": True}, CaptureStatus.UNKNOWN),
        ({"halted": True}, CaptureStatus.UNKNOWN),
        ({"_tracing_finished": False, "halted": False}, CaptureStatus.UNKNOWN),
        ({"_tracing_finished": None}, CaptureStatus.UNKNOWN),
        ({}, CaptureStatus.UNKNOWN),
    ],
)
def test_derivation_lattice(state: dict, expected: CaptureStatus) -> None:
    outcome = derive_outcome_from_structural_state(state)
    assert outcome.status is expected
    assert outcome.derived is True
    # No structural derivation may ever bless COMPLETE.
    assert outcome.status is not CaptureStatus.COMPLETE


def test_derivation_carries_halt_detail() -> None:
    outcome = derive_outcome_from_structural_state(
        {
            "_tracing_finished": True,
            "halted": True,
            "halt_reason": "relu_1_2_raw",
            "halt_frontier": "relu_1_2_raw",
        }
    )
    assert outcome.status is CaptureStatus.HALTED
    assert outcome.reason == "relu_1_2_raw"
    assert outcome.boundary_label == "relu_1_2_raw"


# ---------------------------------------------------------------------------
# Loaded-outcome resolution: adopt, degrade, derive
# ---------------------------------------------------------------------------


def test_resolve_loaded_adopts_coherent_attestation() -> None:
    attested = CaptureOutcome(status=CaptureStatus.COMPLETE)
    state = {
        "_capture_outcome": attested.to_payload(),
        "_tracing_finished": True,
        "halted": False,
    }
    assert resolve_loaded_outcome(state) == attested


def test_resolve_loaded_degrades_incoherent_attestation_to_unknown() -> None:
    attested = CaptureOutcome(status=CaptureStatus.COMPLETE)
    state = {
        "_capture_outcome": attested.to_payload(),
        "_tracing_finished": False,
        "halted": False,
    }
    with pytest.warns(RuntimeWarning, match="contradicts"):
        resolved = resolve_loaded_outcome(state)
    assert resolved.status is CaptureStatus.UNKNOWN
    assert resolved.derived is True
    assert "attestation_incoherent" in (resolved.settlement_note or "")


def test_resolve_loaded_degrades_unparseable_attestation_to_unknown() -> None:
    state = {
        "_capture_outcome": {"status": "not_in_vocabulary"},
        "_tracing_finished": True,
        "halted": False,
    }
    with pytest.warns(RuntimeWarning, match="could not parse"):
        resolved = resolve_loaded_outcome(state)
    assert resolved.status is CaptureStatus.UNKNOWN
    assert "attestation_parse_failed" in (resolved.settlement_note or "")


def test_resolve_loaded_without_attestation_uses_lattice() -> None:
    resolved = resolve_loaded_outcome({"_tracing_finished": True, "halted": False})
    assert resolved.status is CaptureStatus.UNATTESTED


# ---------------------------------------------------------------------------
# Capability table + chokepoint arming proofs (forged sidecars)
# ---------------------------------------------------------------------------


class _Product:
    """Bare product stand-in carrying a forged outcome sidecar."""

    def __init__(self, outcome: CaptureOutcome | None) -> None:
        if outcome is not None:
            self._capture_outcome = outcome


def test_capability_table_is_total() -> None:
    for capability, row in CAPTURE_OUTCOME_CAPABILITIES.items():
        assert set(row) == set(CaptureStatus), capability
        for cell in row.values():
            assert cell == "allow" or cell.startswith("allow_scoped:") or cell.startswith("refuse:")


@pytest.mark.parametrize(
    ("capability", "status", "code"),
    [
        ("save_analysis", CaptureStatus.FAILED, "N1"),
        ("save_analysis", CaptureStatus.ABORTED_NONFINITE, "N1"),
        ("save_analysis", CaptureStatus.UNKNOWN, "N1"),
        ("validation_entry", CaptureStatus.FAILED, "N2"),
        ("validation_entry", CaptureStatus.UNKNOWN, "N2"),
        ("live_replay", CaptureStatus.HALTED, "N5"),
        ("live_replay", CaptureStatus.FAILED, "N3"),
        ("loaded_sparse_run", CaptureStatus.FAILED, "N3"),
        ("backward", CaptureStatus.UNKNOWN, "N3"),
        ("save_runnable", CaptureStatus.HALTED, "N4"),
        ("save_runnable", CaptureStatus.FAILED, "N1"),
    ],
)
def test_chokepoint_refuses_with_stable_code(
    capability: str, status: CaptureStatus, code: str
) -> None:
    product = _Product(CaptureOutcome(status=status))
    with pytest.raises(CaptureOutcomeError) as exc_info:
        require_capture_capability(product, capability)
    assert exc_info.value.fields["code"] == code
    assert exc_info.value.fields["capability"] == capability


@pytest.mark.parametrize(
    ("capability", "status"),
    [
        ("save_analysis", CaptureStatus.COMPLETE),
        ("save_analysis", CaptureStatus.HALTED),
        ("save_analysis", CaptureStatus.UNATTESTED),
        ("validation_entry", CaptureStatus.HALTED),
        ("validation_entry", CaptureStatus.UNATTESTED),
        ("live_replay", CaptureStatus.UNATTESTED),
        ("loaded_sparse_run", CaptureStatus.HALTED),
        ("backward", CaptureStatus.HALTED),
        ("save_runnable", CaptureStatus.UNATTESTED),
    ],
)
def test_chokepoint_allows(capability: str, status: CaptureStatus) -> None:
    product = _Product(CaptureOutcome(status=status))
    outcome = require_capture_capability(product, capability)
    assert outcome.status is status


def test_chokepoint_treats_missing_sidecar_as_unknown() -> None:
    """Defense-in-depth: no sidecar -> UNKNOWN + warning, fail-closed."""

    product = _Product(None)
    with pytest.warns(RuntimeWarning, match="no settled capture outcome"):
        with pytest.raises(CaptureOutcomeError) as exc_info:
            require_capture_capability(product, "save_analysis")
    assert exc_info.value.fields["code"] == "N1"


def test_n5_refusal_names_the_rearm_follow_on() -> None:
    product = _Product(CaptureOutcome(status=CaptureStatus.HALTED))
    with pytest.raises(CaptureOutcomeError, match="re-arming|halt="):
        require_capture_capability(product, "live_replay")


# ---------------------------------------------------------------------------
# Failure-origin classifier plants
# ---------------------------------------------------------------------------


def test_origin_interrupt() -> None:
    assert classify_failure_origin(KeyboardInterrupt()) is FailureOrigin.INTERRUPT


def test_origin_no_traceback_is_unknown() -> None:
    assert classify_failure_origin(ValueError("bare")) is FailureOrigin.UNKNOWN


def test_origin_user_forward_failure_classifies_user_op() -> None:
    class Exploding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _ = self.linear(x)
            raise RuntimeError("user forward boom")

    try:
        tl.trace(Exploding(), torch.ones(1, 3))
    except RuntimeError as exc:
        assert classify_failure_origin(exc) is FailureOrigin.USER_OP


def test_origin_c_backed_op_failure_classifies_user_op() -> None:
    """A C-backed torch failure inside the wrapper trampoline is the user's op."""

    class ShapeMismatch(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(x, torch.ones(7, 7))

    try:
        tl.trace(ShapeMismatch(), torch.ones(1, 3))
    except RuntimeError as exc:
        origin = classify_failure_origin(exc)
        assert origin is FailureOrigin.USER_OP, origin


def test_origin_torchlens_frame_classifies_torchlens() -> None:
    """An exception whose innermost frame is torchlens code is TORCHLENS."""

    try:
        parse_outcome_payload("not a mapping")
    except ValueError as exc:
        assert classify_failure_origin(exc) is FailureOrigin.TORCHLENS


# ---------------------------------------------------------------------------
# Public type/error surface
# ---------------------------------------------------------------------------


def test_types_and_errors_exports() -> None:
    from torchlens import errors as tl_errors, types as tl_types

    assert tl_types.CaptureOutcome is CaptureOutcome
    assert tl_types.CaptureStatus is CaptureStatus
    assert tl_types.CapturePhase is CapturePhase
    assert tl_types.FailureOrigin is FailureOrigin
    assert tl_errors.CaptureOutcomeError is CaptureOutcomeError
    assert issubclass(tl_errors.StopSignalSwallowedError, tl_errors.CaptureError)
