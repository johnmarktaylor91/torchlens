"""The structure-only evidence envelope writer (weightsfree memo sec 5).

ONE immutable envelope, computed once at settlement, consumed by every
renderer, serializer, agent surface, slice, and persisted artifact. The
closed key set, claim vocabularies, and coherence rules were adjudicated at
C07 (tlspec v9, entry-dark until this writer) and are load-validated
fail-closed by ``torchlens/_io/forgery_validation.py``
(``artifact_structure_evidence_invalid``) — this writer and that validator
must stay in lockstep.

The PERSISTED claim statuses are the capture-time facts: hypotheses stay
hypotheses (D21 — the discharge registry is session-only, so a loaded trace
is HYPOTHESIS again and the persisted ``discharge`` field is ``"absent"``
in v1; the reserved attestation object is a named future, never
opportunistically serialized).

Every spelling is DOCUMENTED-UNSTABLE pending naming-session/S2 ratification.
"""

from __future__ import annotations

from typing import Any

from ._weightsfree_admission import (
    MetaAdmissionRecord,
    admission_record_for,
    enforce_settlement_invariants,
    stamp_wrap_generation,
)

__all__ = ["build_structure_evidence", "settle_weightsfree_capture"]


def settle_weightsfree_capture(trace: Any, admission: MetaAdmissionRecord | None) -> None:
    """Settlement seam: stamp session facts and the evidence envelope.

    Runs on EVERY finished torch capture: the wrap-generation stamp (W1-ORD)
    feeds the discharge preflight for both twins, real oracles included.
    Structure-only captures additionally pass the D22 invariant net and
    receive the persisted evidence envelope; an admitted meta offense whose
    trace lost the structure-only marker fails closed inside the invariants.

    Parameters
    ----------
    trace:
        The finished capture product.
    admission:
        The entry-gate admission record (``None`` on ordinary and
        real-substrate captures).
    """

    from .. import _state

    stamp_wrap_generation(trace, int(getattr(_state, "_wrap_epoch", 0)))
    if admission is not None or bool(getattr(trace, "structure_only", False)):
        enforce_settlement_invariants(trace)
    if bool(getattr(trace, "structure_only", False)):
        trace.structure_evidence = build_structure_evidence(trace)


def build_structure_evidence(trace: Any) -> dict[str, Any]:
    """Build the closed-key evidence envelope for one structure-only capture.

    Parameters
    ----------
    trace:
        A finished structure-only capture (either substrate).

    Returns
    -------
    dict[str, Any]
        The envelope, exactly the C07 closed key set; validates fail-closed
        under the tlspec v9 load rules.
    """

    from .. import _state

    admission = admission_record_for(trace)
    status_value = getattr(getattr(trace, "outcome", None), "status", None)
    status_token = getattr(status_value, "value", None)
    if status_token == "complete":
        outcome = "complete"
    elif status_token == "failed":
        outcome = "failed"
    else:
        outcome = "partial"
    if admission is not None:
        substrate = admission.substrate
        factory_device_policy = admission.factory_device_policy
        ambient_mode_present = admission.ambient_mode_present
        wrap_generation = admission.wrap_generation
    else:
        substrate = "real"
        factory_device_policy = "none"
        ambient_mode_present = False
        wrap_generation = int(getattr(_state, "_wrap_epoch", 0))
    return {
        "capture_mode": "structure_only",
        "substrate": substrate,
        "values_available": False,
        "outcome": outcome,
        "factory_device_policy": factory_device_policy,
        "ambient_mode_present": ambient_mode_present,
        "wrap_generation": wrap_generation,
        "input_plan": _input_plan_facts(trace),
        "claims": {
            "graph_structure": "observed_canonical",
            "shapes_dtypes": "hypothesis",
            "flops_geometry_bytes": "hypothesis_estimate",
            "declared_buffer_mutations": "hypothesis",
            "measured_values": "unavailable",
            "timing_allocator_memory": "unavailable",
            "declared_branch_assumptions": "none",
        },
        "discharge": "absent",
    }


def _input_plan_facts(trace: Any) -> dict[str, Any]:
    """Input-plan provenance facts (quickstart rung where present).

    A gold-rung user-supplied input has no synthesized/omitted members; the
    quickstart resolver's ``InputProvenance`` payload (F17) supplies the
    synthesized-rung disclosure when the capture came through the ladder.
    """

    import contextlib

    source = "user_supplied"
    synthesized: list[str] = []
    # Provenance enrichment, never authority: a resolver absent from this
    # build (or a payload it cannot parse) leaves the gold-rung defaults.
    with contextlib.suppress(Exception):
        from ..quickstart import trace_input_provenance

        provenance = trace_input_provenance(trace)
        if provenance is not None:
            source = getattr(provenance, "rung", None) or "declared"
            synthesized = list(getattr(provenance, "synthesized", ()) or ())
    return {
        "source": str(source),
        "synthesized": synthesized,
        "omitted": [],
        "selection_source": "explicit",
    }
