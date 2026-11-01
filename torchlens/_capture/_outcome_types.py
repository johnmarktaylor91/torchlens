"""Leaf vocabulary, capability table and sidecar readers of the outcome authority.

This module holds the frozen status vocabulary, the durable
:class:`CaptureOutcome` record, the typed refusal/stop exceptions, the
per-outcome capability table with its refusal hints, and the two hostile-
``__str__`` / ``__repr__`` guards plus :func:`outcome_for`. It is an
import-graph leaf (it imports only the error hierarchy) so the loaded-payload
codec can live in its own sibling module without joining the package's import
cycle. The public home of every name is :mod:`torchlens._capture.outcome`,
which re-exports them; existing by-reference pickles that name the authority
module keep resolving through those re-exports.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..errors import CaptureError, TorchLensError


class CaptureStatus(str, Enum):
    """Terminal status vocabulary for one capture product."""

    COMPLETE = "complete"
    STOPPED = "stopped"
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
    num_committed_ops: int | None = None
    inference_only: bool = False
    is_recovered: bool = False
    is_derived: bool = False
    note: str | None = None

    @property
    def is_partial(self) -> bool | None:
        """Return the honest partiality claim for this outcome.

        ``True`` for a known prefix or failure, ``False`` for settled
        complete, ``None`` where partiality is unknowable (UNATTESTED /
        UNKNOWN claims nothing).
        """

        if self.status is CaptureStatus.COMPLETE:
            return False
        if self.status in (
            CaptureStatus.STOPPED,
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
            "n_ops_committed": self.num_committed_ops,
            "inference_only": self.inference_only,
            "recovered": self.is_recovered,
            "derived": self.is_derived,
            "settlement_note": self.note,
        }


class CaptureOutcomeError(TorchLensError, ValueError):
    """Typed capability refusal from the capture-outcome chokepoint.

    ``fields["code"]`` carries the stable gate code (``"N1"``..``"N5"``);
    branch on it, never on message text.

    ``ValueError`` lineage (R64-4): the sibling refusals at the same public
    doors (``Trace.save`` raises ``RunnablePreflightError`` /
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
    the capture boundary, like its ``CaptureNotActiveError`` sibling; it was one
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
        CaptureStatus.STOPPED: "allow_scoped:halted ground-truth output block skipped",
        CaptureStatus.ABORTED_NONFINITE: "refuse:N2",
        CaptureStatus.FAILED: "refuse:N2",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N2",
    },
    # N1 -- analysis-save refusal. Replaces today's ungated pass-through of
    # provably-unfinished artifacts; HALTED and UNATTESTED re-export stay open.
    "save_analysis": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.STOPPED: _ALLOW,
        CaptureStatus.ABORTED_NONFINITE: "refuse:N1",
        CaptureStatus.FAILED: "refuse:N1",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N1",
    },
    # N4 -- runnable-save refusal for HALTED (fail-closed default, maintainer veto).
    # ABORTED/FAILED/UNKNOWN are subsumed by N1 at the same entry.
    "save_runnable": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.STOPPED: "refuse:N4",
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
        CaptureStatus.STOPPED: "refuse:N5",
        CaptureStatus.ABORTED_NONFINITE: "refuse:N3",
        CaptureStatus.FAILED: "refuse:N3",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N3",
    },
    # Loaded-sparse run() executes exactly the recorded taken-path prefix DAG
    # under pause_logging(): the HALTED failure mode cannot occur there.
    "loaded_sparse_run": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.STOPPED: _ALLOW,
        CaptureStatus.ABORTED_NONFINITE: "refuse:N3",
        CaptureStatus.FAILED: "refuse:N3",
        CaptureStatus.UNATTESTED: _ALLOW,
        CaptureStatus.UNKNOWN: "refuse:N3",
    },
    # log_backward / backward / recording_backward: the autograd graph of a
    # halted capture IS the captured prefix, so HALTED stays allowed.
    "backward": {
        CaptureStatus.COMPLETE: _ALLOW,
        CaptureStatus.STOPPED: _ALLOW,
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
        "(draw, report.explain, tl._partial.from_failed_capture) instead of "
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
        "without stop_after= to save a runnable artifact."
    ),
    "N5": (
        "Live-provider replay re-drives the FULL native forward, but a "
        "halted capture recorded only a prefix; results would silently "
        "disagree with the recorded graph. Re-capture without stop_after= "
        "(re-arming on_nonfinite/stop_after options as needed), or load the "
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
    gate treated every ``tl.Bundle()``-built product as UNKNOWN and
    ``Trace.save`` false-refused every bundle with the same false hand-built
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
