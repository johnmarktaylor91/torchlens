"""The capture-outcome authority: one typed lattice for every capture product.

This module is the SINGLE authority for terminal capture truth (early-stopping
unification design of record, v4). It owns:

- the frozen status vocabulary (:class:`CaptureStatus`, :class:`CapturePhase`,
  :class:`FailureOrigin`) and the durable :class:`CaptureOutcome` record;
- the persisted attestation codec (string-only payload, closed-vocabulary
  parse) and the status-specific load coherence matrix;
- the evidence-based derivation lattice for artifacts without attestation;
- the per-outcome capability table and its ONE enforcement chokepoint;
- the diagnostic failure-origin classifier.

Settlement adapters (``settle_capture``, ``demote_outcome``, the cook /
recorder / backend-finalize stamps) also live here so no second writer can
exist. Consumers branch on the typed record or the stable refusal codes,
never on exception text.
"""

from __future__ import annotations

import traceback
import warnings
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import CaptureError, TorchLensError

if TYPE_CHECKING:
    pass


class CaptureStatus(str, Enum):
    """Terminal status vocabulary for one capture product."""

    COMPLETE = "complete"
    HALTED = "halted"
    ABORTED_NONFINITE = "aborted_nonfinite"
    FAILED = "failed"
    UNATTESTED = "unattested"
    UNKNOWN = "unknown"


class CapturePhase(str, Enum):
    """Failure phase attribution (FAILED outcomes only)."""

    FORWARD = "forward"
    FINALIZE = "finalize"
    POSTPROCESS = "postprocess"
    TEARDOWN = "teardown"


class FailureOrigin(str, Enum):
    """Diagnostic failure origin (FAILED outcomes only; never capability-steering)."""

    USER_OP = "user_op"
    TORCHLENS = "torchlens"
    INTERRUPT = "interrupt"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CaptureOutcome:
    """Durable, string-only terminal outcome record for one capture product.

    Parameters
    ----------
    status:
        Settled terminal status.
    phase:
        Failure phase (FAILED only).
    origin:
        Diagnostic failure origin (FAILED only); never steers capability.
    reason:
        Halt reason / abort reason / failure summary, string-only.
    error_type:
        Qualified exception type name for FAILED outcomes.
    boundary_kind:
        Stop-boundary event kind (halt/nonfinite boundaries).
    boundary_label:
        Stop-boundary op label (raw where postprocess never ran).
    frontier_labels:
        Halt-frontier labels (final labels when postprocess ran, else raw).
    n_ops_committed:
        Count of committed op-kind lane entries at settlement.
    inference_only:
        Whether the capture ran under the inference-only public mode.
    recovered:
        True for disk-recovered fastlog products.
    derived:
        True when reconstructed from structural evidence, not a settle stamp.
    settlement_note:
        Secondary-failure / demotion / provenance disclosure, string-only.
    """

    status: CaptureStatus
    phase: CapturePhase | None = None
    origin: FailureOrigin | None = None
    reason: str | None = None
    error_type: str | None = None
    boundary_kind: str | None = None
    boundary_label: str | None = None
    frontier_labels: tuple[str, ...] | None = None
    n_ops_committed: int | None = None
    inference_only: bool = False
    recovered: bool = False
    derived: bool = False
    settlement_note: str | None = None

    @property
    def partial(self) -> bool | None:
        """Return the honest partiality claim for this outcome.

        ``True`` for a known prefix or failure, ``False`` for settled
        complete, ``None`` where partiality is unknowable (UNATTESTED /
        UNKNOWN claims nothing).
        """

        if self.status is CaptureStatus.COMPLETE:
            return False
        if self.status in (
            CaptureStatus.HALTED,
            CaptureStatus.ABORTED_NONFINITE,
            CaptureStatus.FAILED,
        ):
            return True
        return None

    def to_payload(self) -> dict[str, Any]:
        """Return the persisted string-only payload for this record."""

        return {
            "status": self.status.value,
            "phase": None if self.phase is None else self.phase.value,
            "origin": None if self.origin is None else self.origin.value,
            "reason": self.reason,
            "error_type": self.error_type,
            "boundary_kind": self.boundary_kind,
            "boundary_label": self.boundary_label,
            "frontier_labels": (
                None if self.frontier_labels is None else list(self.frontier_labels)
            ),
            "n_ops_committed": self.n_ops_committed,
            "inference_only": self.inference_only,
            "recovered": self.recovered,
            "derived": self.derived,
            "settlement_note": self.settlement_note,
        }


class CaptureOutcomeError(TorchLensError, ValueError):
    """Typed capability refusal from the capture-outcome chokepoint.

    ``fields["code"]`` carries the stable gate code (``"N1"``..``"N5"``);
    branch on it, never on message text.

    ``ValueError`` lineage (R64-4): the sibling refusals at the same public
    doors (``tl.save`` raises ``RunnablePreflightError`` /
    ``InvalidArgumentError`` for its other three refusal reasons) are all
    ``ValueError``-lineage, so a caller's natural ``except ValueError`` used
    to catch three of the door's four refusals and silently miss the N-gate.
    """


class StopSignalSwallowedError(CaptureError, RuntimeError):
    """A halt/nonfinite stop request was swallowed by user code (F6).

    Raised at the capture boundary when the stop-request latch is set but the
    forward returned normally: a user ``except``/``except BaseException`` ate
    the control signal, so the capture can never be blessed COMPLETE.
    ``RuntimeError`` lineage (R64-4): it reports an invalid runtime state at
    the capture boundary, like its ``CaptureContextError`` sibling; it was one
    of only two concrete publicly-raised classes with no builtin lineage.
    """


@dataclass(frozen=True, slots=True)
class StopRequest:
    """One latched stop request (F6), set on the active trace at raise time.

    Parameters
    ----------
    kind:
        ``"halt"`` or ``"nonfinite"``.
    reason:
        The stop reason recorded at the raise site.
    boundary_kind:
        Event kind at the stop boundary, when known.
    boundary_label:
        Op label at the stop boundary, when known.
    error_ref:
        Weak reference to the exact exception raised at the latch site.
        Settlement matches the terminal exception against it by IDENTITY, so
        an unrelated later ``CaptureError`` (after the latched abort was
        swallowed) settles FAILED with its own diagnostics instead of being
        misattributed as the clean ABORTED_NONFINITE (R06). Weak so the
        latch never extends the exception's (traceback-carrying) lifetime;
        within the raise-to-settle window the terminal exception is alive.
    """

    kind: str
    reason: str | None = None
    boundary_kind: str | None = None
    boundary_label: str | None = None
    error_ref: weakref.ref[BaseException] | None = None


# ---------------------------------------------------------------------------
# Capability table + chokepoint
# ---------------------------------------------------------------------------

#: OUTCOME-DEGRADING load conditions (persisted-state contract, ecosystem MEMO
#: 3.3/8.2): condition code -> the MOST OPTIMISTIC status a load carrying that
#: condition may attest. Registered from day one as inert plumbing -- the v1
#: contract refuses unknown persisted fields before any object exists, so no
#: shipped path consults this table yet -- but any future accept/disclose mode
#: MUST fold it into settlement: a loaded object carrying state the reader
#: could not interpret can never attest COMPLETE (the D-ECO-10 absorption
#: measurement: a real capture with an absorbed unknown field attested
#: CaptureStatus.COMPLETE with zero warnings).
OUTCOME_DEGRADING_LOAD_CONDITIONS: dict[str, CaptureStatus] = {
    "unknown_persisted_field": CaptureStatus.UNKNOWN,
}

_ALLOW = "allow"

CAPTURE_OUTCOME_CAPABILITIES: dict[str, dict[CaptureStatus, str]] = {
    # N2 -- validation ENTRY refusal only; tripwire bodies stay byte-untouched.
    # UNATTESTED deliberately keeps entry OPEN: validation is the tripwire that
    # catches a freeze-corrupted legacy representation, so refusing it would
    # disarm the check for exactly the most-suspect artifacts.
    "validation_entry": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: "allow_scoped:halted ground-truth output block skipped",
        CaptureStatus.ABORTED_NONFINITE: "refuse:N2",
        CaptureStatus.FAILED: "refuse:N2",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N2",
    },
    # N1 -- analysis-save refusal. Replaces today's ungated pass-through of
    # provably-unfinished artifacts; HALTED and UNATTESTED re-export stay open.
    "save_analysis": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: _ALLOW,
        CaptureStatus.ABORTED_NONFINITE: "refuse:N1",
        CaptureStatus.FAILED: "refuse:N1",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N1",
    },
    # N4 -- runnable-save refusal for HALTED (fail-closed default, JMT veto).
    # ABORTED/FAILED/UNKNOWN are subsumed by N1 at the same entry.
    "save_runnable": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: "refuse:N4",
        CaptureStatus.ABORTED_NONFINITE: "refuse:N1",
        CaptureStatus.FAILED: "refuse:N1",
        CaptureStatus.UNATTESTED: "allow_scoped:existing runnable preflights are the gate",
        CaptureStatus.UNKNOWN: "refuse:N1",
    },
    # N5 (HALTED) / N3 (ABORTED/FAILED/UNKNOWN) -- LIVE providers only:
    # refresh, save_new_outs, push, replay, rerun, run() on a live trace, and
    # fast=True. The recorded prefix cannot re-drive a full native forward.
    "live_replay": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: "refuse:N5",
        CaptureStatus.ABORTED_NONFINITE: "refuse:N3",
        CaptureStatus.FAILED: "refuse:N3",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N3",
    },
    # Loaded-sparse run() executes exactly the recorded taken-path prefix DAG
    # under pause_logging(): the HALTED failure mode cannot occur there.
    "loaded_sparse_run": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: _ALLOW,
        CaptureStatus.ABORTED_NONFINITE: "refuse:N3",
        CaptureStatus.FAILED: "refuse:N3",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N3",
    },
    # log_backward / backward / recording_backward: the autograd graph of a
    # halted capture IS the captured prefix, so HALTED stays allowed.
    "backward": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.HALTED: _ALLOW,
        CaptureStatus.ABORTED_NONFINITE: "refuse:N3",
        CaptureStatus.FAILED: "refuse:N3",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N3",
    },
}
"""Frozen per-outcome capability cells; consumers branch ONLY through
:func:`require_capture_capability`. Cells are ``"allow"``,
``"allow_scoped:<note>"``, or ``"refuse:<code>"``."""


_REFUSAL_HINTS: dict[str, str] = {
    "N1": (
        "This capture did not settle as a saveable product. Inspect it live "
        "(draw, report.explain, tl.partial.from_failed_capture) instead of "
        "exporting it."
    ),
    "N2": (
        "Validation entry requires a settled COMPLETE, HALTED, or legacy "
        "finished capture; a failed or unproven capture has no replayable "
        "ground truth to validate against."
    ),
    "N3": (
        "Replay and backward surfaces require a structurally complete "
        "captured graph; this capture's graph is failed or unproven."
    ),
    "N4": (
        "A halted capture records a PREFIX of the forward; the sparse "
        "runnable contract requires the complete taken path. Re-capture "
        "without halt= to save a runnable artifact."
    ),
    "N5": (
        "Live-provider replay re-drives the FULL native forward, but a "
        "halted capture recorded only a prefix; results would silently "
        "disagree with the recorded graph. Re-capture without halt= "
        "(re-arming raise_on_nan/halt options as needed), or load the "
        "artifact and use loaded-sparse run()."
    ),
}


# Ceiling on persisted failure reasons (R67): generous for real messages,
# small next to any artifact.
_MAX_REASON_CHARS = 4096


def safe_exception_str(exc: BaseException) -> str:
    """Return ``str(exc)`` without letting a hostile ``__str__`` escape.

    Exception stringification runs arbitrary user code: settlement sits in
    ``finally`` blocks that promise a settled product and byte-identical
    exception identity/chaining, so a ``__str__`` that itself raises must
    degrade to the type name -- never break guaranteed settlement or mask the
    original exception with the secondary stringification failure.
    """

    try:
        text = str(exc)
    except BaseException:  # noqa: BLE001 -- hostile __str__; disclosed fallback below
        return f"<unprintable {type(exc).__name__}: __str__ raised>"
    if len(text) > _MAX_REASON_CHARS:
        # The reason persists verbatim into every artifact (tlspec v7): a user
        # exception embedding a tensor repr or a multi-MB assert message must
        # not bloat the artifact. Truncation is disclosed, never silent (R67).
        text = f"{text[:_MAX_REASON_CHARS]}... [truncated, {len(text)} chars total]"
    return text or type(exc).__name__


def safe_exception_repr(exc: BaseException) -> str:
    """Return ``repr(exc)`` with the same hostile-``__repr__`` guarantee."""

    try:
        return repr(exc)
    except BaseException:  # noqa: BLE001 -- hostile __repr__; disclosed fallback below
        return f"<unrepresentable {type(exc).__name__}: __repr__ raised>"


def outcome_for(trace: object) -> CaptureOutcome | None:
    """Return the settled outcome sidecar attached to ``trace``, if any.

    A wrapper product that carries no sidecar of its own may declare ONE
    sanctioned delegation hop via the ``_OUTCOME_DELEGATE_FIELD`` class
    attribute naming the attribute that holds the settled inner product
    (``PartialTrace.trace``). Without it every capability gate read the
    WRAPPER's empty ``__dict__`` and treated a shipped FAILED partial as
    UNKNOWN with a false hand-built-object warning, while ``p.outcome``
    forwarded the inner trace's FAILED stamp -- two answers for one product
    (b1-opus-R06-1). The hop is single-level by construction: the delegate is
    read through its own ``__dict__`` only, never recursed.

    A SLOTS-BACKED product whose validating ``outcome`` property is itself
    the settlement-truth reader declares ``_OUTCOME_SELF_AUTHORITY = True``
    (``Recording``: R10-5 re-parse of the stamped record plus the
    conservative construction-status derivation, never blessing COMPLETE).
    Without it the gate read the missing ``__dict__`` as UNKNOWN and refused
    every settled Recording with a false hand-built-object warning while
    ``recording.outcome`` reported the real status -- the same two-answers
    disease, WT1 A-IV item 21 (lane A08).

    A CONTAINER product declares the same self-authority flag over a DERIVED
    fold (``Bundle``: the worst-of-members fold, COMPLETE never blessed above
    the weakest member, ``derived=True``, disclosure-only). Without it the
    gate treated every ``tl.bundle()``-built product as UNKNOWN and
    ``tl.save`` false-refused every bundle with the same false hand-built
    warning while ``bundle.save()`` worked -- the same two-answers disease,
    instance three (foldB D9). The fold is reported beside results and never
    gates a read on a member it does not describe: capability gating stays
    PER MEMBER at the member whose facts a read cites (foldB D8).
    """

    outcome = getattr(trace, "__dict__", {}).get("_capture_outcome")
    if isinstance(outcome, CaptureOutcome):
        return outcome
    if getattr(type(trace), "_OUTCOME_SELF_AUTHORITY", False):
        candidate = getattr(trace, "outcome", None)
        if isinstance(candidate, CaptureOutcome):
            return candidate
    delegate_field = getattr(type(trace), "_OUTCOME_DELEGATE_FIELD", None)
    if isinstance(delegate_field, str):
        inner = getattr(trace, delegate_field, None)
        if inner is not None and inner is not trace:
            inner_outcome = getattr(inner, "__dict__", {}).get("_capture_outcome")
            if isinstance(inner_outcome, CaptureOutcome):
                return inner_outcome
    return None


def require_capture_capability(
    trace: object,
    capability: str,
    *,
    detail: str | None = None,
) -> CaptureOutcome:
    """Enforce one capability cell for ``trace``'s settled outcome.

    Parameters
    ----------
    trace:
        Product whose ``_capture_outcome`` sidecar is consulted. A live trace
        with no sidecar is treated as UNKNOWN with a warning (fail-closed
        defense-in-depth; unreachable for shipped constructors per the P0
        settlement inventory).
    capability:
        Row key in :data:`CAPTURE_OUTCOME_CAPABILITIES`.
    detail:
        Optional gate-site context appended to the refusal message.

    Returns
    -------
    CaptureOutcome
        The consulted outcome, for callers that scope behavior on it.

    Raises
    ------
    CaptureOutcomeError
        When the cell refuses; ``fields["code"]`` carries the stable code.
    """

    outcome = outcome_for(trace)
    if outcome is None:
        warnings.warn(
            "TorchLens capability check found no settled capture outcome on "
            f"this {type(trace).__name__}; treating it as UNKNOWN (fail-closed). "
            "Every shipped product constructor settles an outcome, so this "
            "usually means a hand-built or pre-settlement object.",
            RuntimeWarning,
            stacklevel=3,
        )
        outcome = CaptureOutcome(status=CaptureStatus.UNKNOWN, derived=True)
    cell = CAPTURE_OUTCOME_CAPABILITIES[capability][outcome.status]
    if cell.startswith("refuse:"):
        code = cell.split(":", 1)[1]
        message = (
            f"TorchLens refuses {capability!r} for a capture with outcome "
            f"status {outcome.status.value!r} (gate {code}). "
            f"{_REFUSAL_HINTS.get(code, '')}"
        )
        if outcome.reason:
            message += f" Capture outcome reason: {outcome.reason}"
        if detail:
            message += f" {detail}"
        raise CaptureOutcomeError(
            message,
            code=code,
            capability=capability,
            status=outcome.status.value,
        )
    return outcome


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

    def _enum_or_none(key: str, enum_cls: type[Enum]) -> Any:
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

    status = _enum_or_none("status", CaptureStatus)
    if status is None:
        raise ValueError("capture outcome payload is missing 'status'")
    phase = _enum_or_none("phase", CapturePhase)
    origin = _enum_or_none("origin", FailureOrigin)

    def _str_or_none(key: str) -> str | None:
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

    frontier = data.get("frontier_labels")
    if frontier is not None:
        if not isinstance(frontier, (list, tuple)) or not all(
            isinstance(item, str) for item in frontier
        ):
            raise ValueError("capture outcome field 'frontier_labels' must be a list of strings")
        frontier = tuple(frontier)
    n_ops = data.get("n_ops_committed")
    if n_ops is not None and (isinstance(n_ops, bool) or not isinstance(n_ops, int)):
        raise ValueError("capture outcome field 'n_ops_committed' must be an int or None")

    def _bool(key: str, default: bool = False) -> bool:
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

    return CaptureOutcome(
        status=status,
        phase=phase,
        origin=origin,
        reason=_str_or_none("reason"),
        error_type=_str_or_none("error_type"),
        boundary_kind=_str_or_none("boundary_kind"),
        boundary_label=_str_or_none("boundary_label"),
        frontier_labels=frontier,
        n_ops_committed=n_ops,
        inference_only=_bool("inference_only"),
        recovered=_bool("recovered"),
        derived=_bool("derived"),
        settlement_note=_str_or_none("settlement_note"),
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
    if outcome.derived and outcome.status in (
        CaptureStatus.COMPLETE,
        CaptureStatus.ABORTED_NONFINITE,
        CaptureStatus.FAILED,
    ):
        return "claims derived=True for a status no derivation lattice emits"
    # ``settle_completed`` writes neither a reason nor the fastlog
    # disk-recovery marker; a COMPLETE payload carrying either forges
    # provenance no writer can produce (R06).
    if outcome.status is CaptureStatus.COMPLETE and (
        outcome.reason is not None or outcome.recovered
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
    if outcome.status is CaptureStatus.HALTED:
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
        return (not halted) and finished and outcome.derived
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
    halted = bool(state.get("halted"))
    inference_only = bool(state.get("inference_only"))
    if halted and finished:
        reason = state.get("halt_reason")
        frontier = state.get("halt_frontier")
        return CaptureOutcome(
            status=CaptureStatus.HALTED,
            reason=reason if isinstance(reason, str) else None,
            boundary_label=frontier if isinstance(frontier, str) else None,
            inference_only=inference_only,
            derived=True,
        )
    if finished and not halted:
        return CaptureOutcome(
            status=CaptureStatus.UNATTESTED,
            inference_only=inference_only,
            derived=True,
        )
    return CaptureOutcome(
        status=CaptureStatus.UNKNOWN,
        inference_only=inference_only,
        derived=True,
    )


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
        return CaptureOutcome(
            status=CaptureStatus.UNKNOWN,
            derived=True,
            settlement_note=f"attestation_parse_failed: {exc}",
        )
    halted = bool(state.get("halted"))
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
        return CaptureOutcome(
            status=CaptureStatus.UNKNOWN,
            derived=True,
            settlement_note=(
                f"attestation_incoherent_internal: status={outcome.status.value} "
                f"{internal_violation}"
            ),
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
        return CaptureOutcome(
            status=CaptureStatus.UNKNOWN,
            derived=True,
            settlement_note=(
                f"attestation_incoherent: status={outcome.status.value} "
                f"halted={halted} finished={finished}"
            ),
        )
    return outcome


# ---------------------------------------------------------------------------
# Failure-origin classification (diagnostic only)
# ---------------------------------------------------------------------------

_TORCHLENS_PKG_DIR = Path(__file__).resolve().parent.parent

# C-backed user-op failures surface with the wrapper trampoline as their
# innermost Python frame; those lines execute the USER's op, so they classify
# as USER_OP, not TORCHLENS. Matched on source text to survive line drift.
_TRAMPOLINE_SOURCE_MARKER = "out_orig = func("


def _frame_zone(filename: str) -> str:
    """Classify one frame filename as 'torchlens', 'library', or 'user'."""

    if filename.startswith("<frozen"):
        return "library"
    try:
        path = Path(filename).resolve()
    except (OSError, ValueError):
        return "library"
    try:
        path.relative_to(_TORCHLENS_PKG_DIR)
        return "torchlens"
    except ValueError:
        pass
    if "site-packages" in path.parts or "dist-packages" in path.parts:
        return "library"
    return "user"


def _qualified_type_name(exc: BaseException) -> str:
    """Return ``module.qualname`` for a failure's exception type (R67).

    The docstring contract promises a QUALIFIED name; the bare ``__name__``
    made same-named exception classes from different libraries
    indistinguishable in a persisted v7 artifact. Builtins keep their bare
    name (the ``builtins.`` prefix is noise).
    """

    exc_type = type(exc)
    module = getattr(exc_type, "__module__", None)
    qualname = getattr(exc_type, "__qualname__", exc_type.__name__)
    if not module or module == "builtins":
        return str(qualname)
    return f"{module}.{qualname}"


def classify_failure_origin(exc: BaseException) -> FailureOrigin:
    """Classify one terminal exception's origin (diagnostic only).

    Walks the traceback from the innermost frame outward. The innermost
    attributable frame wins: user code -> USER_OP; a torchlens frame ->
    TORCHLENS, except the wrapper trampoline (executing the user's op) ->
    USER_OP. Interrupts classify INTERRUPT; no traceback or nothing
    attributable -> UNKNOWN. Misclassification is capability-safe by
    construction -- origin never steers a gate.
    """

    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return FailureOrigin.INTERRUPT
    if isinstance(exc, StopSignalSwallowedError):
        return FailureOrigin.TORCHLENS
    # A user-supplied intervention payload failing validation is the USER's
    # bad tensor, not a TorchLens bug -- but the raise happens inside a
    # torchlens frame, so the innermost-frame walk below misattributed it as
    # TORCHLENS (R67). Classify by exception type instead of frame zone.
    try:
        from ..intervention.errors import HookValueError
    except Exception:  # noqa: BLE001 - classifier must never raise
        pass
    else:
        if isinstance(exc, HookValueError):
            return FailureOrigin.USER_OP
    tb = exc.__traceback__
    if tb is None:
        return FailureOrigin.UNKNOWN
    try:
        frames = traceback.extract_tb(tb)
    except Exception:
        return FailureOrigin.UNKNOWN
    for frame in reversed(frames):
        zone = _frame_zone(frame.filename)
        if zone == "user":
            return FailureOrigin.USER_OP
        if zone == "torchlens":
            line = frame.line or ""
            if _TRAMPOLINE_SOURCE_MARKER in line:
                return FailureOrigin.USER_OP
            return FailureOrigin.TORCHLENS
    return FailureOrigin.UNKNOWN


# ---------------------------------------------------------------------------
# Settlement: the ONE exception-safe termination protocol
# ---------------------------------------------------------------------------

_TERMINAL_STATE_FOR_STATUS: dict[CaptureStatus, str] = {
    CaptureStatus.COMPLETE: "complete",
    CaptureStatus.HALTED: "halted",
    CaptureStatus.ABORTED_NONFINITE: "failed",
    CaptureStatus.FAILED: "failed",
    CaptureStatus.UNATTESTED: "failed",
    CaptureStatus.UNKNOWN: "failed",
}


def set_capture_phase(trace: object, phase: CapturePhase) -> None:
    """Advance the transient settlement phase marker for one capture run.

    Leaving FORWARD also caches the committed-op count: postprocess later pops
    and releases the event stream, so a tail failure's settlement stamp would
    otherwise read an empty working projection instead of the real count.
    """

    trace.__dict__["_capture_phase"] = phase
    if phase in (CapturePhase.FINALIZE, CapturePhase.POSTPROCESS):
        live_count = _count_live_committed_ops(trace)
        if live_count is not None:
            trace.__dict__["_settlement_ops_committed"] = live_count


def current_capture_phase(trace: object) -> CapturePhase:
    """Return the current settlement phase marker (FORWARD before any update)."""

    phase = trace.__dict__.get("_capture_phase")
    return phase if isinstance(phase, CapturePhase) else CapturePhase.FORWARD


def _count_live_committed_ops(trace: object) -> int | None:
    """Count op-kind lane entries on the live event stream, fail-soft."""

    events = trace.__dict__.get("capture_events") or trace.__dict__.get("_capture_events")
    if events is None:
        return None
    try:
        count = 0
        for event in getattr(events, "op_events", ()):
            record_context = getattr(event, "record_context", None)
            if record_context is None or getattr(record_context, "kind", None) == "op":
                count += 1
        return count
    except Exception:
        return None


def count_committed_ops(trace: object) -> int | None:
    """Return the committed op-kind entry count for settlement stamps.

    Exhaustive op records carry no ``record_context`` (all op-kind by
    construction); fastlog-projected events are filtered on
    ``record_context.kind == "op"``. A live nonzero count wins; a released or
    already-popped stream falls back to the phase-transition cache.
    """

    live = _count_live_committed_ops(trace)
    if live:
        return live
    cached = trace.__dict__.get("_settlement_ops_committed")
    if isinstance(cached, int):
        return cached
    return live


def _stamp(trace: object, session: Any, outcome: CaptureOutcome) -> CaptureOutcome:
    """Write one settled outcome to both homes and perform the one transition.

    The trace sidecar is rebound unconditionally (settle and demotion are the
    only writers); the session transition runs only when the session has not
    already reached its first terminal state (``TerminalState`` is the
    first-transition log and stays monotonic).
    """

    trace.__dict__["_capture_outcome"] = outcome
    trace.__dict__.pop("_capture_phase", None)
    trace.__dict__.pop("_settlement_ops_committed", None)
    if session is not None and getattr(session, "outcome", None) is None:
        session.transition(
            _TERMINAL_STATE_FOR_STATUS[outcome.status],
            capture_outcome=outcome,
        )
    return outcome


def settle_completed(trace: object, session: object) -> CaptureOutcome:
    """Settle one successfully completed capture (paths 1-2)."""

    return _stamp(
        trace,
        session,
        CaptureOutcome(
            status=CaptureStatus.COMPLETE,
            n_ops_committed=count_committed_ops(trace),
            inference_only=bool(getattr(trace, "inference_only", False)),
        ),
    )


def settle_halted(
    trace: object,
    session: object,
    halt_exc: BaseException,
    *,
    finalize_partial: bool,
    postprocess_ran: bool,
) -> CaptureOutcome:
    """Settle one halted capture (paths 3-4), attested.

    Frontier labels come from the post-postprocess ``output_layers`` when the
    halted postprocess ran (final labels); otherwise the raw boundary label is
    the only honest frontier fact and ``frontier_labels`` stays ``None``.
    """

    frontier: tuple[str, ...] | None = None
    reason = getattr(halt_exc, "reason", None)
    boundary_label = getattr(halt_exc, "boundary_label", None) or (
        reason if isinstance(reason, str) else None
    )
    if finalize_partial and postprocess_ran:
        try:
            frontier = tuple(str(label) for label in getattr(trace, "output_layers", ()))
        except Exception:
            frontier = None
        # The labeling remap rewrote the persisted halt fields to FINAL labels
        # during the halted postprocess; the settled record mirrors them so
        # the boundary resolves through ``trace[...]`` on the finished
        # product. Raw-label boundaries stay only where postprocess never ran.
        remapped_reason = getattr(trace, "halt_reason", None)
        if isinstance(remapped_reason, str):
            reason = remapped_reason
        remapped_frontier = getattr(trace, "halt_frontier", None)
        if isinstance(remapped_frontier, str):
            boundary_label = remapped_frontier
    return _stamp(
        trace,
        session,
        CaptureOutcome(
            status=CaptureStatus.HALTED,
            reason=reason if isinstance(reason, str) else None,
            boundary_kind=getattr(halt_exc, "boundary_kind", None),
            boundary_label=boundary_label,
            frontier_labels=frontier,
            n_ops_committed=count_committed_ops(trace),
            inference_only=bool(getattr(trace, "inference_only", False)),
        ),
    )


def settle_failed(
    trace: object,
    session: object,
    exc: BaseException,
    *,
    interrupted: bool = False,
    settlement_note: str | None = None,
    n_ops_committed: int | None = None,
) -> CaptureOutcome:
    """Settle one failed capture (paths 5-6, 8; halted-secondary via note).

    A latched nonfinite stop request (the ``raise_nonfinite`` structural
    marker) classifies ABORTED_NONFINITE when the terminal exception is the
    nonfinite ``CaptureError`` itself; everything else is FAILED with phase
    and diagnostic origin attribution.
    """

    if n_ops_committed is None:
        n_ops_committed = count_committed_ops(trace)
    stop_request = trace.__dict__.get("_stop_requested")
    if (
        isinstance(stop_request, StopRequest)
        and stop_request.kind == "nonfinite"
        and isinstance(exc, CaptureError)
        # The latch records the raised exception's IDENTITY: only the exact
        # nonfinite CaptureError classifies ABORTED_NONFINITE. An unrelated
        # CaptureError arriving after a swallowed abort settles FAILED below
        # with its own reason/error_type instead of being misattributed (R06).
        and stop_request.error_ref is not None
        and stop_request.error_ref() is exc
        # A SWALLOWED nonfinite abort is a FAILED capture, never a clean
        # ABORTED_NONFINITE: the abort did not actually stop the forward.
        and not isinstance(exc, StopSignalSwallowedError)
        and not interrupted
    ):
        return _stamp(
            trace,
            session,
            CaptureOutcome(
                status=CaptureStatus.ABORTED_NONFINITE,
                reason=stop_request.reason,
                boundary_kind=stop_request.boundary_kind,
                boundary_label=stop_request.boundary_label,
                n_ops_committed=n_ops_committed,
                inference_only=bool(getattr(trace, "inference_only", False)),
                settlement_note=settlement_note,
            ),
        )
    origin = FailureOrigin.INTERRUPT if interrupted else classify_failure_origin(exc)
    return _stamp(
        trace,
        session,
        CaptureOutcome(
            status=CaptureStatus.FAILED,
            phase=current_capture_phase(trace),
            origin=origin,
            reason=safe_exception_str(exc),
            error_type=_qualified_type_name(exc),
            n_ops_committed=n_ops_committed,
            inference_only=bool(getattr(trace, "inference_only", False)),
            settlement_note=settlement_note,
        ),
    )


def demote_outcome(
    trace: object,
    session: Any,
    *,
    note: str,
    exc: BaseException | None = None,
) -> CaptureOutcome | None:
    """Demote an already-settled outcome after a post-settlement teardown failure.

    The sole sanctioned post-settlement writer: permitted transitions are
    downgrades only (COMPLETE/HALTED -> FAILED/TEARDOWN). Atomically REPLACES
    the frozen record in both homes (trace sidecar and the session outcome's
    record slot); the session's ``TerminalState`` first-transition log is
    never revised. Anything already FAILED/ABORTED stays as settled.

    ``exc`` is the teardown exception itself: it populates the demoted
    record's structured ``error_type`` so consumers branch on the field, not
    the free-text ``reason`` (the only legal pre-demotion statuses are
    COMPLETE/HALTED, whose ``error_type`` is always None).
    """

    settled = outcome_for(trace)
    if settled is None or settled.status not in (
        CaptureStatus.COMPLETE,
        CaptureStatus.HALTED,
    ):
        return None
    demoted = CaptureOutcome(
        status=CaptureStatus.FAILED,
        phase=CapturePhase.TEARDOWN,
        origin=FailureOrigin.TORCHLENS,
        reason=note,
        error_type=type(exc).__name__ if exc is not None else settled.error_type,
        boundary_kind=settled.boundary_kind,
        boundary_label=settled.boundary_label,
        frontier_labels=settled.frontier_labels,
        n_ops_committed=settled.n_ops_committed,
        inference_only=settled.inference_only,
        recovered=settled.recovered,
        settlement_note=f"demoted_from={settled.status.value}: {note}",
    )
    trace.__dict__["_capture_outcome"] = demoted
    run_outcome = getattr(session, "outcome", None)
    if run_outcome is not None and getattr(run_outcome, "capture_outcome", None) is not None:
        from dataclasses import replace as dataclass_replace

        session.outcome = dataclass_replace(run_outcome, capture_outcome=demoted)
    return demoted


def stamp_forked(fork: object, parent: object) -> CaptureOutcome:
    """Settle one fork with a DERIVED outcome, never the parent's attestation.

    A fork is TorchLens's sanctioned mutation surface: its contents may be
    hand-edited after creation, so inheriting the parent's blessed settle
    stamp by identity would let a mutated fork save as a bit-identical
    attested product. The fork settles through the same structural
    derivation lattice an attestation-less artifact uses (R06 doctrine:
    derivation emits HALTED / UNATTESTED / UNKNOWN, never a blessed
    COMPLETE), keeping the parent's committed-op count and recording fork
    provenance in the settlement note. Capability parity is preserved: a
    COMPLETE parent's fork settles UNATTESTED, whose every capability cell
    is allow/allow-scoped.
    """

    from dataclasses import replace as dataclass_replace

    parent_outcome = outcome_for(parent)
    derived = derive_outcome_from_structural_state(getattr(fork, "__dict__", {}))
    parent_status = "unsettled" if parent_outcome is None else parent_outcome.status.value
    derived = dataclass_replace(
        derived,
        n_ops_committed=(None if parent_outcome is None else parent_outcome.n_ops_committed),
        settlement_note=f"forked_from={parent_status}",
    )
    fork.__dict__["_capture_outcome"] = derived
    return derived


def stamp_cooked(
    trace: object,
    *,
    halted: bool,
    reason: str | None = None,
    boundary_kind: str | None = None,
    boundary_label: str | None = None,
    frontier_label: str | None = None,
) -> CaptureOutcome:
    """Settle one Trace cooked from a Recording (path 9), attested.

    ``reason`` and ``boundary_label`` are distinct facts (R06: the single
    ``reason`` parameter formerly fed both, and the cooked path passed the
    Recording-space RAW label while ``frontier_labels`` was final -- one
    outcome mixing label spaces, with ``boundary_kind`` dropped entirely).
    Callers pass the post-postprocess FINAL labels, mirroring
    ``settle_halted``'s remap read-back.
    """

    if halted:
        outcome = CaptureOutcome(
            status=CaptureStatus.HALTED,
            reason=reason,
            boundary_kind=boundary_kind,
            boundary_label=boundary_label if boundary_label is not None else reason,
            frontier_labels=None if frontier_label is None else (frontier_label,),
            n_ops_committed=count_committed_ops(trace),
            settlement_note="cooked_from=recording",
        )
    else:
        outcome = CaptureOutcome(
            status=CaptureStatus.COMPLETE,
            n_ops_committed=count_committed_ops(trace),
            settlement_note="cooked_from=recording",
        )
    return _stamp(trace, None, outcome)


def stamp_recording_outcome(recording: object, outcome: CaptureOutcome) -> CaptureOutcome:
    """Stamp one settled outcome onto a frozen Recording (paths 11-14)."""

    object.__setattr__(recording, "_outcome", outcome)
    return outcome


def stamp_backend_finalized(trace: object) -> CaptureOutcome:
    """Settle one preview-backend capture at its true product boundary (path 20).

    Called as the LAST act of each preview backend's capture entry, after ALL
    tail work (module attachment, compaction, relation freeze, derived grads,
    cleanup, depth flood). A failure anywhere between the structural
    ``_tracing_finished`` write and this stamp is productless: the exception
    propagates, no stamp exists, and a hypothetical pickle of the escaped
    object derives UNATTESTED -- never COMPLETE. A backend that ever needs an
    early stamp must demote through :func:`demote_outcome` on post-stamp
    teardown failure; premature stamping is closed by rule.

    HALT-AWARE BY CONSTRUCTION (B1-01). A preview backend that supports
    ``halt=`` (paddle's shipped capability, MLX's ``_mlx_halt_selector``)
    catches its ``HaltSignal``, writes the structural halt fields, and then
    falls through the SAME tail to this one stamp. Reading ``trace.halted``
    here mirrors :func:`settle_halted` at the chokepoint, so no preview
    backend can stamp a halted product COMPLETE -- a wrongly-blessed settled
    status that the load-time coherence matrix would have to degrade to
    UNKNOWN, taking N1 re-save and N2 validation entry down with it. Backends
    with no halt path are unaffected: ``halted`` is falsey and the COMPLETE
    arm is byte-identical to before.

    SWALLOW-PROOF (F6 parity, R06 round 3). The halt-capable preview raise
    sites latch a :class:`StopRequest` on the trace before raising their
    ``HaltSignal`` -- exactly the torch ``evaluate_halt`` contract. A user
    broad-``except`` that eats the signal leaves the latch set with no
    structural ``halted`` write, so this stamp is the boundary checkpoint:
    it settles FAILED and raises :class:`StopSignalSwallowedError`, never
    blessing the swallowed capture COMPLETE. The latch is consumed on every
    arm, so a normally-halted capture stamps HALTED exactly as before.
    """

    stop_request = trace.__dict__.pop("_stop_requested", None)
    if isinstance(stop_request, StopRequest) and not bool(getattr(trace, "halted", False)):
        swallowed = StopSignalSwallowedError(
            "TorchLens raised a halt stop signal during this forward, but the "
            "capture reached its settlement stamp un-halted: user code "
            "swallowed the control signal (typically a broad `except:` or "
            "`except BaseException:` around the model body). The capture "
            "cannot be trusted as complete. Stop boundary: "
            f"{stop_request.boundary_label or stop_request.reason!r}.",
            kind=stop_request.kind,
            boundary_label=stop_request.boundary_label,
        )
        settle_failed(trace, None, swallowed)
        raise swallowed
    if bool(getattr(trace, "halted", False)):
        reason = getattr(trace, "halt_reason", None)
        frontier_label = getattr(trace, "halt_frontier", None)
        # The preview tail ran to completion (module attachment, relation
        # freeze), so ``output_layers`` holds this backend's final labels for
        # the halt frontier -- the same fact settle_halted reads on the
        # postprocess-ran path. Absence stays None (unknown), never ().
        frontier: tuple[str, ...] | None = None
        try:
            frontier = tuple(str(label) for label in getattr(trace, "output_layers", ())) or None
        except Exception:
            frontier = None
        return _stamp(
            trace,
            None,
            CaptureOutcome(
                status=CaptureStatus.HALTED,
                reason=reason if isinstance(reason, str) else None,
                boundary_label=frontier_label if isinstance(frontier_label, str) else None,
                frontier_labels=frontier,
                n_ops_committed=count_committed_ops(trace),
                inference_only=bool(getattr(trace, "inference_only", False)),
            ),
        )
    return _stamp(
        trace,
        None,
        CaptureOutcome(
            status=CaptureStatus.COMPLETE,
            n_ops_committed=count_committed_ops(trace),
            inference_only=bool(getattr(trace, "inference_only", False)),
        ),
    )


__all__ = [
    "CAPTURE_OUTCOME_CAPABILITIES",
    "CaptureOutcome",
    "CaptureOutcomeError",
    "CapturePhase",
    "CaptureStatus",
    "FailureOrigin",
    "StopRequest",
    "StopSignalSwallowedError",
    "attestation_coherent",
    "classify_failure_origin",
    "demote_outcome",
    "derive_outcome_from_structural_state",
    "outcome_for",
    "parse_outcome_payload",
    "require_capture_capability",
    "resolve_loaded_outcome",
    "safe_exception_repr",
    "safe_exception_str",
    "settle_completed",
    "settle_failed",
    "settle_halted",
    "stamp_backend_finalized",
    "stamp_cooked",
    "stamp_forked",
]
