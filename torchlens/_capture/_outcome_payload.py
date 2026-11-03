"""Persisted-attestation codec and load-time outcome resolution.

Parses one persisted ``_capture_outcome`` payload against the closed
vocabularies, validates it against the cross-field and structural coherence
matrix, and derives a conservative outcome when no usable attestation exists.
Split out of :mod:`torchlens._capture.outcome` (which re-exports every public
and test-visible name) and kept an import-graph leaf: it imports only the
sibling vocabulary module.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from enum import Enum
from typing import Any

from ._outcome_types import (
    CaptureOutcome,
    CapturePhase,
    CaptureStatus,
    FailureOrigin,
)

# ---------------------------------------------------------------------------
# Persistence codec + load derivation
# ---------------------------------------------------------------------------

#: The closed persisted-payload key set (B1-07b). DERIVED from the writer
#: (``CaptureOutcome.to_payload``) rather than hand-listed, so reader and writer
#: cannot drift: adding a persisted field automatically widens the accepted set,
#: and a field this reader does not know about refuses.
_OUTCOME_PAYLOAD_KEYS: frozenset[str] = frozenset(
    CaptureOutcome(status=CaptureStatus.UNKNOWN).to_payload()
)

#: Fields that only a FAILED attestation may carry (B1-07a). Every settle path
#: was enumerated to derive this: ``phase``/``origin``/``error_type`` are
#: written by ``settle_failed``'s FAILED arm, ``demote_outcome``, and the
#: fastlog failed-partial stamps -- and by nothing else. A record claiming
#: COMPLETE while carrying ``phase="postprocess"`` and
#: ``error_type="RuntimeError"`` is internally contradictory and must degrade
#: to UNKNOWN rather than present as attested.
_FAILED_ONLY_OUTCOME_FIELDS: tuple[str, ...] = ("phase", "origin", "error_type")

#: Fields no COMPLETE/UNATTESTED attestation may carry. Both statuses mean "no
#: stop boundary was reached", so boundary/frontier evidence contradicts them.
#: HALTED and ABORTED_NONFINITE legitimately carry boundary facts, and a FAILED
#: record demoted from HALTED inherits them, so those three are exempt.
_NO_BOUNDARY_OUTCOME_FIELDS: tuple[str, ...] = (
    "boundary_kind",
    "boundary_label",
    "frontier_labels",
)


def _refuse_unknown_keys(data: dict[Any, Any]) -> None:
    """Refuse a payload carrying any key outside the closed persisted set.

    Raises
    ------
    ValueError
        If ``data`` has a field the writer-derived key set does not know.
    """

    # B1-07(b): UNKNOWN KEYS REFUSE. The codec used to copy the dict and read
    # only the keys it knew, so a verdict-steering field added in a later
    # torchlens without a tlspec bump was invisible to this reader and no
    # warning fired -- the reader would bless an attestation it could not fully
    # evaluate. That is the exact incident class the lockstep mechanism kills
    # elsewhere. Refusing routes through the existing degrade-to-UNKNOWN path
    # (fail-closed, one warning, never a load crash), so an old torchlens
    # reading a newer artifact says "I cannot verify this" instead of
    # "verified".
    unknown_keys = sorted(set(data) - _OUTCOME_PAYLOAD_KEYS)
    if unknown_keys:
        raise ValueError(
            "capture outcome payload has unknown field(s) "
            f"{unknown_keys} (this artifact was likely written by a newer "
            "torchlens whose attestation this reader cannot fully evaluate)"
        )


def _enum_or_none(data: dict[Any, Any], key: str, enum_cls: type[Enum]) -> Any:
    """Read one closed-vocabulary enum field, or ``None`` when absent.

    Raises
    ------
    ValueError
        If the value is not a string or is outside ``enum_cls``.
    """

    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"capture outcome field {key!r} must be a string, got {value!r}")
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise ValueError(
            f"capture outcome field {key!r} has unknown vocabulary value {value!r}"
        ) from exc


def _str_or_none(data: dict[Any, Any], key: str) -> str | None:
    """Read one optional string field.

    Raises
    ------
    ValueError
        If the value is present and not a string.
    """

    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"capture outcome field {key!r} must be a string or None")
    return value


def _bool(data: dict[Any, Any], key: str, default: bool = False) -> bool:
    """Read one strictly-boolean field, falling back to ``default`` when absent.

    Raises
    ------
    ValueError
        If the value is present and not a ``bool``.
    """

    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"capture outcome field {key!r} must be a bool")
    return value


def _frontier_or_none(data: dict[Any, Any]) -> Any:
    """Read the optional ``frontier_labels`` field as a tuple of strings.

    Raises
    ------
    ValueError
        If the value is present and not a list/tuple of strings.
    """

    frontier = data.get("frontier_labels")
    if frontier is not None:
        if not isinstance(frontier, (list, tuple)) or not all(
            isinstance(item, str) for item in frontier
        ):
            raise ValueError("capture outcome field 'frontier_labels' must be a list of strings")
        frontier = tuple(frontier)
    return frontier


def _n_ops_or_none(data: dict[Any, Any]) -> Any:
    """Read the optional ``n_ops_committed`` field.

    Raises
    ------
    ValueError
        If the value is present and not a (non-bool) ``int``.
    """

    n_ops = data.get("n_ops_committed")
    if n_ops is not None and (isinstance(n_ops, bool) or not isinstance(n_ops, int)):
        raise ValueError("capture outcome field 'n_ops_committed' must be an int or None")
    return n_ops


def parse_outcome_payload(payload: object) -> CaptureOutcome:
    """Parse one persisted attestation payload against the closed vocabularies.

    Parameters
    ----------
    payload:
        The persisted ``_capture_outcome`` value (a string-only dict).

    Returns
    -------
    CaptureOutcome
        Parsed record.

    Raises
    ------
    ValueError
        On any shape, type, or vocabulary violation (the caller degrades to
        UNKNOWN with a warning; a load never crashes on a bad attestation).
    """

    if not isinstance(payload, Mapping):
        raise ValueError(f"capture outcome payload must be a mapping, got {type(payload).__name__}")
    data = dict(payload)
    _refuse_unknown_keys(data)
    status = _enum_or_none(data, "status", CaptureStatus)
    if status is None:
        raise ValueError("capture outcome payload is missing 'status'")
    phase = _enum_or_none(data, "phase", CapturePhase)
    origin = _enum_or_none(data, "origin", FailureOrigin)
    frontier = _frontier_or_none(data)
    n_ops = _n_ops_or_none(data)
    return CaptureOutcome(
        status=status,
        phase=phase,
        origin=origin,
        reason=_str_or_none(data, "reason"),
        error_type=_str_or_none(data, "error_type"),
        boundary_kind=_str_or_none(data, "boundary_kind"),
        boundary_label=_str_or_none(data, "boundary_label"),
        frontier_labels=frontier,
        num_committed_ops=n_ops,
        inference_only=_bool(data, "inference_only"),
        is_recovered=_bool(data, "recovered"),
        is_derived=_bool(data, "derived"),
        note=_str_or_none(data, "settlement_note"),
    )


def _cross_field_incoherence(outcome: CaptureOutcome) -> str | None:
    """Return the first internal-contradiction rule one attestation violates (B1-07a).

    Parameters
    ----------
    outcome:
        Parsed attestation record.

    Returns
    -------
    str | None
        A description of the violated cross-field rule, or ``None`` when the
        record is internally self-consistent. The description names the ACTUAL
        rule so the load-time refusal never misattributes an internal
        contradiction to the artifact's structural evidence (b8-sol).
    """

    if outcome.status is not CaptureStatus.FAILED:
        forbidden = [
            field_name
            for field_name in _FAILED_ONLY_OUTCOME_FIELDS
            if getattr(outcome, field_name) is not None
        ]
        if forbidden:
            return f"carries FAILED-only field(s) {forbidden} on a non-FAILED status"
    # ``derived`` is provenance only the derivation lattices write, and they
    # emit HALTED / UNATTESTED / UNKNOWN exclusively; every settle stamp
    # writes ``derived=False``. A derived COMPLETE / ABORTED_NONFINITE /
    # FAILED payload is therefore internally contradictory (R06).
    if outcome.is_derived and outcome.status in (
        CaptureStatus.COMPLETE,
        CaptureStatus.ABORTED_NONFINITE,
        CaptureStatus.FAILED,
    ):
        return "claims derived=True for a status no derivation lattice emits"
    # ``settle_completed`` writes neither a reason nor the fastlog
    # disk-recovery marker; a COMPLETE payload carrying either forges
    # provenance no writer can produce (R06).
    if outcome.status is CaptureStatus.COMPLETE and (
        outcome.reason is not None or outcome.is_recovered
    ):
        return "carries a reason or recovered marker no COMPLETE writer produces"
    boundary_evidence = [
        field_name
        for field_name in _NO_BOUNDARY_OUTCOME_FIELDS
        if getattr(outcome, field_name) is not None
    ]
    if outcome.status in (CaptureStatus.COMPLETE, CaptureStatus.UNATTESTED) and boundary_evidence:
        return f"carries stop-boundary field(s) {boundary_evidence} on a no-boundary status"
    return None


def _cross_field_coherent(outcome: CaptureOutcome) -> bool:
    """Return whether one attestation is internally self-consistent (B1-07a)."""

    return _cross_field_incoherence(outcome) is None


def attestation_coherent(
    outcome: CaptureOutcome,
    *,
    halted: bool,
    finished: bool,
) -> bool:
    """Validate one parsed attestation against the structural evidence.

    The status-specific coherence matrix (design 2.4): a parsed attestation
    is adopted only when its status is possible given the artifact's
    structural fields, all read by TRUTHINESS.

    Two layers, both required (B1-07a). The structural layer below asks
    "is this status possible for these structural fields?". The CROSS-FIELD
    layer asks "is this record internally consistent?" -- it used to be
    missing entirely, so a payload could claim COMPLETE while carrying
    ``phase="postprocess"``, ``origin="torchlens"`` and
    ``error_type="RuntimeError"`` and be adopted as an attested COMPLETE
    carrying its own failure evidence. This is a TIGHTENING of the matrix; no
    previously-refused record is now accepted.
    """

    if not _cross_field_coherent(outcome):
        return False
    if outcome.status is CaptureStatus.COMPLETE:
        return (not halted) and finished
    if outcome.status is CaptureStatus.STOPPED:
        # The re-raise path settles attested HALTED without postprocess, so
        # either ``_tracing_finished`` value is coherent.
        return halted
    if outcome.status is CaptureStatus.ABORTED_NONFINITE:
        return not halted
    if outcome.status is CaptureStatus.FAILED:
        # A halted-postprocess secondary failure settles FAILED with
        # ``halted=True`` already stamped pre-postprocess.
        return True
    if outcome.status is CaptureStatus.UNKNOWN:
        return True
    if outcome.status is CaptureStatus.UNATTESTED:
        # Only re-saves of DERIVED outcomes may carry UNATTESTED; a settle
        # stamp can never produce it.
        return (not halted) and finished and outcome.is_derived
    return False


def derive_outcome_from_structural_state(state: Mapping[str, Any]) -> CaptureOutcome:
    """Derive a conservative outcome from structural evidence alone.

    The lattice (design 2.4), truthiness throughout so ``False``/``None``/
    absent are one class::

        halted and finished      -> HALTED      (derived)
        halted and not finished  -> UNKNOWN     (halted postprocess never completed)
        finished and not halted  -> UNATTESTED  (complete and tail-failure indistinguishable)
        otherwise                -> UNKNOWN
    """

    finished = bool(state.get("_tracing_finished"))
    halted = bool(state.get("is_stopped"))
    inference_only = bool(state.get("inference_only"))
    if halted and finished:
        reason = state.get("stop_reason")
        frontier = state.get("_graph_stop_frontier")
        return CaptureOutcome(
            status=CaptureStatus.STOPPED,
            reason=reason if isinstance(reason, str) else None,
            boundary_label=frontier if isinstance(frontier, str) else None,
            inference_only=inference_only,
            is_derived=True,
        )
    if finished and not halted:
        return CaptureOutcome(
            status=CaptureStatus.UNATTESTED,
            inference_only=inference_only,
            is_derived=True,
        )
    return CaptureOutcome(
        status=CaptureStatus.UNKNOWN,
        inference_only=inference_only,
        is_derived=True,
    )


def _unknown_attestation_outcome(note: str) -> CaptureOutcome:
    """Return the fail-closed derived UNKNOWN outcome carrying ``note``."""

    return CaptureOutcome(status=CaptureStatus.UNKNOWN, is_derived=True, note=note)


def resolve_loaded_outcome(state: Mapping[str, Any]) -> CaptureOutcome:
    """Resolve the outcome for one loaded Trace state dict (pickle or .tlspec).

    Adopts a persisted attestation when it parses against the closed
    vocabularies AND is coherent per :func:`attestation_coherent`; otherwise
    derives from the structural lattice. Any contradiction, parse failure, or
    unknown vocabulary value degrades to UNKNOWN with ONE warning naming the
    incoherence -- fail-closed, never a load crash.
    """

    payload = state.get("_capture_outcome")
    if payload is None:
        return derive_outcome_from_structural_state(state)
    try:
        # In-process restores may hand back an already-typed record; it still
        # passes through the same coherence matrix below.
        outcome = payload if isinstance(payload, CaptureOutcome) else parse_outcome_payload(payload)
    except ValueError as exc:
        warnings.warn(
            f"TorchLens could not parse this artifact's capture-outcome attestation ({exc}); "
            "treating the capture outcome as UNKNOWN (fail-closed).",
            RuntimeWarning,
            stacklevel=3,
        )
        return _unknown_attestation_outcome(f"attestation_parse_failed: {exc}")
    halted = bool(state.get("is_stopped"))
    finished = bool(state.get("_tracing_finished"))
    # b8-sol: the refusal names the rule that ACTUALLY failed. An internally
    # contradictory record (cross-field layer) used to be blamed on the
    # artifact's structural evidence even when halted/finished were perfectly
    # consistent with the claimed status -- the wrong rule for the operator
    # debugging the artifact.
    internal_violation = _cross_field_incoherence(outcome)
    if internal_violation is not None:
        warnings.warn(
            "TorchLens found an internally contradictory capture-outcome "
            f"attestation ({outcome.status.value!r}): it {internal_violation}; "
            "treating the capture outcome as UNKNOWN (fail-closed).",
            RuntimeWarning,
            stacklevel=3,
        )
        return _unknown_attestation_outcome(
            f"attestation_incoherent_internal: status={outcome.status.value} {internal_violation}"
        )
    if not attestation_coherent(outcome, halted=halted, finished=finished):
        warnings.warn(
            "TorchLens found an attested capture outcome "
            f"({outcome.status.value!r}) that contradicts this artifact's "
            f"structural evidence (halted={halted}, tracing_finished={finished}); "
            "treating the capture outcome as UNKNOWN (fail-closed).",
            RuntimeWarning,
            stacklevel=3,
        )
        return _unknown_attestation_outcome(
            f"attestation_incoherent: status={outcome.status.value} "
            f"halted={halted} finished={finished}"
        )
    return outcome
