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
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from .._errors import InvalidArgumentError
from ..errors import CaptureError
from ._outcome_payload import (
    _OUTCOME_PAYLOAD_KEYS,  # noqa: F401 -- test-visible re-export
    attestation_coherent,
    derive_outcome_from_structural_state,
    parse_outcome_payload,
    resolve_loaded_outcome,
)
from ._outcome_types import (
    _REFUSAL_HINTS,
    CAPTURE_OUTCOME_CAPABILITIES,
    OUTCOME_DEGRADING_LOAD_CONDITIONS,  # noqa: F401 -- public re-export
    CaptureOutcome,
    CaptureOutcomeError,
    CapturePhase,
    CaptureStatus,
    FailureOrigin,
    StopRequest,
    StopSignalSwallowedError,
    outcome_for,
    safe_exception_repr,
    safe_exception_str,
)

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _refusal_message(
    capability: str,
    outcome: CaptureOutcome,
    code: str,
    detail: str | None,
) -> str:
    """Build the refusal message for one refused capability cell."""

    message = (
        f"TorchLens refuses {capability!r} for a capture with outcome "
        f"status {outcome.status.value!r} (gate {code}). "
        f"{_REFUSAL_HINTS.get(code, '')}"
    )
    if outcome.reason:
        message += f" Capture outcome reason: {outcome.reason}"
    if detail:
        message += f" {detail}"
    return message


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
        outcome = CaptureOutcome(status=CaptureStatus.UNKNOWN, is_derived=True)
    cell = CAPTURE_OUTCOME_CAPABILITIES[capability][outcome.status]
    if cell.startswith("refuse:"):
        code = cell.split(":", 1)[1]
        message = _refusal_message(capability, outcome, code, detail)
        raise CaptureOutcomeError(
            message,
            code=code,
            capability=capability,
            status=outcome.status.value,
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
# The completeness-witness dispatch mode's unchanged redispatch of the user's
# aten op (installed for runnable census and for armed plane-P captures): an
# exception raised there is the user's op failing, exactly like the trampoline.
_WITNESS_REDISPATCH_FILE = "completeness_witness.py"
_WITNESS_REDISPATCH_MARKER = "result = func(*args, **(kwargs or {}))"
# The declined-protocol redispatch's re-offer to a third-party handler: an
# exception the handler raises there is the user's op failing in library code.
_PROTOCOL_REDISPATCH_FILE = "_protocol_redispatch.py"
_PROTOCOL_REDISPATCH_MARKER = "result = value.__torch_function__(wrapper, types, args, kwargs)"
# The ``torch.ops`` call-class recorder's pass-through to the original ``__call__``
# (the census redispatch and unrecorded calls reach the operator through it).
_TORCH_OPS_PASSTHROUGH_FILE = "_torch_ops_calls.py"
_TORCH_OPS_PASSTHROUGH_MARKER = "return original(self, *args, **kwargs)"
# Its in-place replay callable for a receiver-mutating operator (``mutates_args``).
_TORCH_OPS_MUTATION_MARKER = "call_operator(*args, **kwargs)"


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
    TORCHLENS, except the wrapper trampoline, the completeness-witness
    redispatch and the declined-protocol re-offer (each executing the user's op) -> USER_OP. Interrupts classify INTERRUPT; no traceback or nothing
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
            if (
                _TRAMPOLINE_SOURCE_MARKER in line
                or (
                    frame.filename.endswith(_WITNESS_REDISPATCH_FILE)
                    and _WITNESS_REDISPATCH_MARKER in line
                )
                or (
                    frame.filename.endswith(_PROTOCOL_REDISPATCH_FILE)
                    and _PROTOCOL_REDISPATCH_MARKER in line
                )
                or (
                    frame.filename.endswith(_TORCH_OPS_PASSTHROUGH_FILE)
                    and (
                        _TORCH_OPS_PASSTHROUGH_MARKER in line or _TORCH_OPS_MUTATION_MARKER in line
                    )
                )
            ):
                return FailureOrigin.USER_OP
            return FailureOrigin.TORCHLENS
    return FailureOrigin.UNKNOWN


# ---------------------------------------------------------------------------
# Settlement: the ONE exception-safe termination protocol
# ---------------------------------------------------------------------------

_TERMINAL_STATE_FOR_STATUS: dict[CaptureStatus, str] = {
    CaptureStatus.COMPLETE: "complete",
    CaptureStatus.STOPPED: "is_stopped",
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


def settle_completed(trace: Any, session: object) -> CaptureOutcome:
    """Settle one successfully completed capture (paths 1-2)."""

    return _stamp(
        trace,
        session,
        CaptureOutcome(
            status=CaptureStatus.COMPLETE,
            num_committed_ops=count_committed_ops(trace),
            inference_only=bool(trace._options_inference_only),
        ),
    )


def settle_halted(
    trace: Any,
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
        remapped_reason = getattr(trace, "stop_reason", None)
        if isinstance(remapped_reason, str):
            reason = remapped_reason
        remapped_frontier = trace._graph_stop_frontier
        if isinstance(remapped_frontier, str):
            boundary_label = remapped_frontier
    return _stamp(
        trace,
        session,
        CaptureOutcome(
            status=CaptureStatus.STOPPED,
            reason=reason if isinstance(reason, str) else None,
            boundary_kind=getattr(halt_exc, "boundary_kind", None),
            boundary_label=boundary_label,
            frontier_labels=frontier,
            num_committed_ops=count_committed_ops(trace),
            inference_only=bool(trace._options_inference_only),
        ),
    )


def _is_clean_nonfinite_abort(
    stop_request: Any, exc: BaseException, interrupted: bool
) -> TypeGuard[StopRequest]:
    """Return whether ``exc`` is the exact latched nonfinite abort (ABORTED_NONFINITE)."""

    return (
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
    )


def settle_failed(
    trace: Any,
    session: object,
    exc: BaseException,
    *,
    interrupted: bool = False,
    note: str | None = None,
    num_committed_ops: int | None = None,
) -> CaptureOutcome:
    """Settle one failed capture (paths 5-6, 8; halted-secondary via note).

    A latched nonfinite stop request (the ``raise_nonfinite`` structural
    marker) classifies ABORTED_NONFINITE when the terminal exception is the
    nonfinite ``CaptureError`` itself; everything else is FAILED with phase
    and diagnostic origin attribution.
    """

    if num_committed_ops is None:
        num_committed_ops = count_committed_ops(trace)
    stop_request = trace.__dict__.get("_stop_requested")
    if _is_clean_nonfinite_abort(stop_request, exc, interrupted):
        return _stamp(
            trace,
            session,
            CaptureOutcome(
                status=CaptureStatus.ABORTED_NONFINITE,
                reason=stop_request.reason,
                boundary_kind=stop_request.boundary_kind,
                boundary_label=stop_request.boundary_label,
                num_committed_ops=num_committed_ops,
                inference_only=bool(trace._options_inference_only),
                note=note,
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
            num_committed_ops=num_committed_ops,
            inference_only=bool(trace._options_inference_only),
            note=note,
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
        CaptureStatus.STOPPED,
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
        num_committed_ops=settled.num_committed_ops,
        inference_only=settled.inference_only,
        is_recovered=settled.is_recovered,
        note=f"demoted_from={settled.status.value}: {note}",
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
        num_committed_ops=(None if parent_outcome is None else parent_outcome.num_committed_ops),
        note=f"forked_from={parent_status}",
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
            status=CaptureStatus.STOPPED,
            reason=reason,
            boundary_kind=boundary_kind,
            boundary_label=boundary_label if boundary_label is not None else reason,
            frontier_labels=None if frontier_label is None else (frontier_label,),
            num_committed_ops=count_committed_ops(trace),
            note="cooked_from=recording",
        )
    else:
        outcome = CaptureOutcome(
            status=CaptureStatus.COMPLETE,
            num_committed_ops=count_committed_ops(trace),
            note="cooked_from=recording",
        )
    return _stamp(trace, None, outcome)


def stamp_recording_outcome(recording: object, outcome: CaptureOutcome) -> CaptureOutcome:
    """Stamp one settled outcome onto a frozen Recording (paths 11-14)."""

    object.__setattr__(recording, "_outcome", outcome)
    return outcome


def _swallowed_stop_error(stop_request: StopRequest) -> StopSignalSwallowedError:
    """Build the swallowed-stop-signal error for one latched ``stop_request``."""

    return StopSignalSwallowedError(
        "TorchLens raised a halt stop signal during this forward, but the "
        "capture reached its settlement stamp un-halted: user code "
        "swallowed the control signal (typically a broad `except:` or "
        "`except BaseException:` around the model body). The capture "
        "cannot be trusted as complete. Stop boundary: "
        f"{stop_request.boundary_label or stop_request.reason!r}.",
        kind=stop_request.kind,
        boundary_label=stop_request.boundary_label,
    )


def _preview_halt_frontier(trace: object) -> tuple[str, ...] | None:
    """Read the halted preview product's final frontier labels (None when unknown)."""

    # The preview tail ran to completion (module attachment, relation
    # freeze), so ``output_layers`` holds this backend's final labels for
    # the halt frontier -- the same fact settle_halted reads on the
    # postprocess-ran path. Absence stays None (unknown), never ().
    frontier: tuple[str, ...] | None = None
    try:
        frontier = tuple(str(label) for label in getattr(trace, "output_layers", ())) or None
    except Exception:
        frontier = None
    return frontier


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
    falls through the SAME tail to this one stamp. Reading ``trace.is_stopped``
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
    if isinstance(stop_request, StopRequest) and not bool(getattr(trace, "is_stopped", False)):
        swallowed = _swallowed_stop_error(stop_request)
        settle_failed(trace, None, swallowed)
        raise swallowed
    if bool(getattr(trace, "is_stopped", False)):
        reason = getattr(trace, "stop_reason", None)
        frontier_label = cast("Trace", trace)._graph_stop_frontier
        frontier = _preview_halt_frontier(trace)
        return _stamp(
            trace,
            None,
            CaptureOutcome(
                status=CaptureStatus.STOPPED,
                reason=reason if isinstance(reason, str) else None,
                boundary_label=frontier_label if isinstance(frontier_label, str) else None,
                frontier_labels=frontier,
                num_committed_ops=count_committed_ops(trace),
                inference_only=bool(cast("Trace", trace)._options_inference_only),
            ),
        )
    return settle_completed(trace, None)


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


def refuse_streamed_trace_run(trace: Any) -> None:
    """Refuse ``run()`` on a trace captured with ``storage=tl.to_disk(...)``.

    Every run provider re-captures the payloads into RAM while the streamed
    bundle on disk keeps the first capture, so the trace and its bundle would
    silently disagree (behaviour-net IO-8). Re-streaming a rerun into the
    bundle is not built; until it is, a streamed trace does not run. A trace
    loaded from a bundle is not streamed (its payloads are the bundle's) and
    keeps its loaded providers.
    """

    state = trace.__dict__
    if state.get("_loaded_from_bundle"):
        return
    bundle_path = state.get("_source_bundle_path")
    if bundle_path is None and state.get("_out_writer") is None:
        return
    where = f" at {bundle_path}" if bundle_path is not None else ""
    raise InvalidArgumentError(
        "this trace was captured with storage=tl.to_disk(...), so its payloads live in the "
        f"bundle{where}; run() re-captures every payload into RAM while that bundle keeps the "
        "first capture, so a streamed trace does not run",
        code="trace_option_unsupported",
        remedy=(
            "capture again with storage=tl.to_disk(...) to a new path, or capture without "
            "to_disk, and run that trace"
        ),
        argument="storage",
        bundle_path=None if bundle_path is None else str(bundle_path),
    )
