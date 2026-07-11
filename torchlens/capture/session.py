"""Run-owned capture state and legacy compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal
from weakref import ReferenceType, WeakKeyDictionary, ref

from ..ir.events import OpEvent
from .kernel import CaptureKernel
from .ledgers import CompletenessManifest, DecisionLedger, EventJournal, PayloadLedger
from .plan import CapturePlan, EnrichmentLevel

TerminalState = Literal["complete", "halted", "failed"]
CleanupCallback = Callable[[], None]

_LEGACY_CAPTURE_SESSIONS: "WeakKeyDictionary[object, CaptureSession]" = WeakKeyDictionary()
_LEGACY_EVENT_SESSIONS: dict[int, tuple[ReferenceType[object], ReferenceType[CaptureSession]]] = {}


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Unified terminal capture outcome and optional partial products.

    Parameters
    ----------
    state
        The single terminal state for the run.
    output
        Raw forward or halt-frontier output when available.
    product
        Completed compatibility product when one exists.
    partial_product
        Partial product attached by an existing compatibility path.
    exception
        Terminal failure or halt exception when one exists.
    """

    state: TerminalState
    output: Any = None
    product: Any = None
    partial_product: Any = None
    exception: BaseException | None = None


@dataclass(slots=True)
class _CleanupEntry:
    """One session-owned teardown action.

    Parameters
    ----------
    name
        Stable cleanup action name.
    callback
        Existing teardown callback whose timing is preserved by the adapter.
    completed
        Whether the callback has already run.
    """

    name: str
    callback: CleanupCallback
    completed: bool = False


@dataclass(slots=True, weakref_slot=True)
class CaptureSession:
    """The only mutable owner of one capture run's new spine state.

    The Stage 2 adapter deliberately leaves legacy mutable capture fields on
    ``Trace`` in place.  This session mirrors their durable ownership boundary
    without exposing any public lookup surface or changing producer behavior.

    Parameters
    ----------
    plan
        Immutable intent compiled before the run begins.
    backend_token
        Opaque backend session token or adapter object.
    """

    plan: CapturePlan
    backend_token: object | None = None
    event_journal: EventJournal = field(default_factory=EventJournal)
    decision_ledger: DecisionLedger = field(default_factory=DecisionLedger)
    payload_ledger: PayloadLedger = field(default_factory=PayloadLedger)
    completeness: CompletenessManifest = field(default_factory=CompletenessManifest)
    output_bindings: dict[str, object] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    module_state: dict[str, object] = field(default_factory=dict)
    history_state: dict[str, object] = field(default_factory=dict)
    builders: dict[str, object] = field(default_factory=dict)
    cleanup_stack: list[_CleanupEntry] = field(default_factory=list)
    outcome: RunOutcome | None = None
    kernel: CaptureKernel = field(init=False)

    def __post_init__(self) -> None:
        """Compile the session's fixed-order capture kernel."""

        self.kernel = CaptureKernel(self)

    def release(self) -> None:
        """Release all run-local compatibility sidecars.

        This is called only after legacy forward-memory cleanup has completed.
        It keeps the Stage-2 adapter lifetime-neutral by releasing its event,
        payload, cleanup, and outcome references at the same point as the
        legacy capture path.

        Returns
        -------
        None
            This operation is idempotent.
        """

        self.event_journal.clear()
        self.decision_ledger.clear()
        self.payload_ledger.clear()
        self.completeness.clear()
        self.output_bindings.clear()
        self.counters.clear()
        self.module_state.clear()
        self.history_state.clear()
        self.builders.clear()
        self.cleanup_stack.clear()
        self.backend_token = None
        if self.outcome is not None:
            self.outcome = RunOutcome(state=self.outcome.state)

    def observe_event(self, event: OpEvent) -> None:
        """Populate stage-2 sidecars from an existing producer event.

        Parameters
        ----------
        event
            Frozen event already appended to the legacy ``CaptureEvents``
            buffer.  No event fields, payloads, selectors, or interventions are
            recomputed here.
        """

        event_id = self.event_journal.append(event)
        self.decision_ledger.append_from_event(event_id, event)
        self.payload_ledger.append_from_event(event_id, event)
        self.counters["events"] = self.counters.get("events", 0) + 1

    def note_legacy_emission(self) -> None:
        """Record entry through the Stage-1 producer compatibility seam.

        Returns
        -------
        None
            Updates only session-local instrumentation; the legacy producer
            remains solely responsible for capture behavior.
        """

        self.counters["producer_emissions"] = self.counters.get("producer_emissions", 0) + 1

    def replace_event(self, event: OpEvent) -> None:
        """Mirror an existing immutable producer-event replacement.

        Parameters
        ----------
        event
            Replacement event produced by a legacy compatibility helper.
        """

        event_id = self.event_journal.replace(event)
        self.decision_ledger.append_from_event(event_id, event)
        self.payload_ledger.append_from_event(event_id, event)

    def register_cleanup(self, name: str, callback: CleanupCallback) -> None:
        """Register one teardown action on the session-owned cleanup stack.

        Parameters
        ----------
        name
            Stable action name.  Re-registering an existing action preserves
            the original callback so cleanup remains exactly once.
        callback
            Existing teardown callback to invoke.
        """

        if any(entry.name == name for entry in self.cleanup_stack):
            return
        self.cleanup_stack.append(_CleanupEntry(name=name, callback=callback))

    def run_cleanup(self, name: str, callback: CleanupCallback) -> bool:
        """Run one registered teardown action exactly once.

        Parameters
        ----------
        name
            Stable cleanup action name.
        callback
            Existing teardown callback.  It is retained only on its first
            registration and is never invoked a second time.

        Returns
        -------
        bool
            ``True`` when this invocation ran the callback, otherwise ``False``.
        """

        self.register_cleanup(name, callback)
        for entry in reversed(self.cleanup_stack):
            if entry.name != name:
                continue
            if entry.completed:
                return False
            entry.completed = True
            entry.callback()
            return True
        raise RuntimeError(f"CaptureSession cleanup action was not registered: {name!r}")

    def transition(
        self,
        state: TerminalState,
        *,
        output: Any = None,
        product: Any = None,
        partial_product: Any = None,
        exception: BaseException | None = None,
    ) -> RunOutcome:
        """Perform the single terminal transition for this session.

        Parameters
        ----------
        state
            Terminal state to record.
        output, product, partial_product, exception
            Existing compatibility outcome fields to mirror.

        Returns
        -------
        RunOutcome
            Frozen terminal outcome.

        Raises
        ------
        RuntimeError
            If a caller attempts a second, conflicting terminal transition.
        """

        candidate = RunOutcome(
            state=state,
            output=output,
            product=product,
            partial_product=partial_product,
            exception=exception,
        )
        if self.outcome is None:
            self.outcome = candidate
            return candidate
        if self.outcome != candidate:
            raise RuntimeError("CaptureSession already reached a terminal state.")
        return self.outcome


def compile_legacy_capture_plan(
    trace: object,
    *,
    backend_name: str,
    layers_to_save: Any,
    grad_layers_to_save: Any,
    random_seed: int | None,
    postprocess: bool,
) -> CapturePlan:
    """Compile a no-behavior-change plan from legacy trace configuration.

    Parameters
    ----------
    trace
        Existing trace-like compatibility owner.
    backend_name
        Selected backend name.
    layers_to_save
        Existing activation selector argument.
    grad_layers_to_save
        Existing gradient selector argument.
    random_seed
        Existing forward RNG seed.
    postprocess
        Whether legacy orchestration will materialize a ``Trace`` now.

    Returns
    -------
    CapturePlan
        Immutable mirror of today's already-validated intent.
    """

    capture_mode = str(getattr(trace, "capture_mode", "exhaustive"))
    projection_target = "trace" if postprocess else "recording"
    if capture_mode == "fast":
        projection_target = "refresh"
    # This is a compatibility-only demand declaration.  Existing producers
    # continue to choose their historical work; no extra metadata or payload
    # work is requested from them in Stage 2.
    default_enrichment = (
        EnrichmentLevel.METADATA if capture_mode == "exhaustive" else EnrichmentLevel.SHELL
    )
    options = getattr(trace, "_predicate_save_options", None)
    return CapturePlan.compile(
        projection_target=projection_target,
        available_capabilities=(),
        required_completeness=(),
        default_enrichment=default_enrichment,
        selectors={"layers": layers_to_save, "grad_layers": grad_layers_to_save},
        interventions=getattr(trace, "_intervention_plan", None),
        storage=getattr(options, "storage", None),
        history={
            "size": getattr(trace, "_predicate_history_size", None),
            "lookback": getattr(trace, "_predicate_lookback", None),
            "lookback_payload_policy": getattr(trace, "_predicate_lookback_payload_policy", None),
        },
        backward={
            "backward_ready": getattr(trace, "backward_ready", False),
            "save_gradients": getattr(trace, "save_gradients", False),
        },
        execution_context={
            "random_seed": random_seed,
            "inference_only": getattr(trace, "inference_only", False),
        },
        stop_policy=getattr(trace, "_stop_directive", None),
        backend_name=backend_name,
    )


def attach_legacy_capture_session(
    trace: object,
    *,
    backend_token: object | None,
    backend_name: str,
    layers_to_save: Any,
    grad_layers_to_save: Any,
    random_seed: int | None,
    postprocess: bool,
) -> CaptureSession:
    """Attach a new run-owned session behind the legacy trace adapter.

    Parameters
    ----------
    trace
        Existing trace-like compatibility owner.
    backend_token
        Opaque selected backend adapter.
    backend_name
        Selected backend name.
    layers_to_save, grad_layers_to_save, random_seed, postprocess
        Existing orchestration arguments mirrored into the plan.

    Returns
    -------
    CaptureSession
        Newly attached session for this one forward run.
    """

    session = CaptureSession(
        plan=compile_legacy_capture_plan(
            trace,
            backend_name=backend_name,
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            random_seed=random_seed,
            postprocess=postprocess,
        ),
        backend_token=backend_token,
    )
    _LEGACY_CAPTURE_SESSIONS[trace] = session
    return session


def capture_session_for(owner: object) -> CaptureSession | None:
    """Return the stage-2 session attached to a legacy compatibility owner.

    Parameters
    ----------
    owner
        Trace-like owner that may carry an active capture session.

    Returns
    -------
    CaptureSession | None
        Attached session when the owner is on the Stage 2 adapter path.
    """

    try:
        return _LEGACY_CAPTURE_SESSIONS.get(owner)
    except TypeError:
        return None


def detach_capture_session(trace: object, events: object, session: CaptureSession) -> None:
    """Detach and release a completed legacy compatibility session.

    Parameters
    ----------
    trace
        Legacy trace compatibility owner for the completed run.
    events
        Legacy event buffer associated with the completed run.
    session
        Stage-2 session to detach.  Mismatched registry entries are retained
        to avoid disturbing a subsequent run.

    Returns
    -------
    None
        Removes both compatibility registrations and clears the session.  The
        operation is safe to invoke more than once.
    """

    try:
        if _LEGACY_CAPTURE_SESSIONS.get(trace) is session:
            _LEGACY_CAPTURE_SESSIONS.pop(trace, None)
    except TypeError:
        pass

    event_id = id(events)
    entry = _LEGACY_EVENT_SESSIONS.get(event_id)
    if entry is not None:
        events_ref, session_ref = entry
        if events_ref() is events and session_ref() is session:
            _LEGACY_EVENT_SESSIONS.pop(event_id, None)
    session.release()


def attach_capture_events_session(events: object, session: CaptureSession) -> None:
    """Associate a legacy event buffer with its session outside serialized state.

    Parameters
    ----------
    events
        Existing mutable ``CaptureEvents`` buffer for the active run.
    session
        Stage-2 run owner that mirrors producer facts into its ledgers.
    """

    event_id = id(events)

    def discard_events(
        _events_ref: ReferenceType[object],
        _registry: dict[int, tuple[ReferenceType[object], ReferenceType[CaptureSession]]] = (
            _LEGACY_EVENT_SESSIONS
        ),
    ) -> None:
        """Drop the compatibility association when its event buffer is collected."""

        _registry.pop(event_id, None)

    _LEGACY_EVENT_SESSIONS[event_id] = (ref(events, discard_events), ref(session))


def capture_session_for_events(events: object) -> CaptureSession | None:
    """Return the session associated with one legacy event buffer.

    Parameters
    ----------
    events
        Existing mutable ``CaptureEvents`` buffer.

    Returns
    -------
    CaptureSession | None
        Active compatibility session, if one is registered.
    """

    entry = _LEGACY_EVENT_SESSIONS.get(id(events))
    if entry is None:
        return None
    events_ref, session_ref = entry
    if events_ref() is events:
        session = session_ref()
        if session is not None:
            return session
    _LEGACY_EVENT_SESSIONS.pop(id(events), None)
    return None
