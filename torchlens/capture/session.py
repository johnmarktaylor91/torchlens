"""Run-owned capture state and legacy compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping
import warnings
from weakref import ReferenceType, WeakKeyDictionary, ref

from .. import _state
from ..ir.events import OpEvent
from ..utils.tensor_utils import safe_copy
from .kernel import CaptureKernel
from .ledgers import (
    CompletenessManifest,
    CompletenessState,
    DecisionLedger,
    DecisionRecord,
    EventFact,
    EventId,
    EventJournal,
    PayloadLedger,
    PayloadRecord,
)
from .plan import CapturePlan, EnrichmentLevel, RetentionKind, RetentionProfile

TerminalState = Literal["complete", "halted", "failed"]
CleanupCallback = Callable[[], None]

_LEGACY_CAPTURE_SESSIONS: "WeakKeyDictionary[object, CaptureSession]" = WeakKeyDictionary()
_LEGACY_EVENT_SESSIONS: dict[int, tuple[ReferenceType[object], ReferenceType[CaptureSession]]] = {}


@dataclass(slots=True)
class ActivationEscrowPayload:
    """One detached activation retained in RAM or a temporary spill file."""

    tensor: Any | None
    nbytes: int
    spill_path: Path | None = None

    def materialize(self) -> Any:
        """Return the retained tensor, loading a temporary spill when necessary."""

        if self.tensor is not None:
            return self.tensor
        if self.spill_path is None:
            raise RuntimeError("Activation escrow payload has neither RAM nor spill storage.")
        import torch

        with _state.pause_logging():
            return torch.load(self.spill_path, weights_only=True)


@dataclass(frozen=True, slots=True)
class CapturedRunCore:
    """Sealed, repeatedly readable facts produced by one capture execution.

    Parameters
    ----------
    event_facts
        Immutable operation facts in producer order.
    decisions
        Selection and intervention sidecars keyed by stable event identity.
    payloads
        Payload leases keyed by stable event identity.
    completeness
        Truthful observability states recorded for the run.
    """

    event_facts: tuple[EventFact, ...]
    decisions: Mapping[EventId, DecisionRecord]
    payloads: Mapping[EventId, PayloadRecord]
    completeness: Mapping[str, CompletenessState]
    projection_facts: Mapping[str, Any]


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
    _sealed_core: CapturedRunCore | None = field(default=None, init=False, repr=False)
    projection_facts: dict[str, Any] = field(default_factory=dict)
    activation_escrow: dict[int, ActivationEscrowPayload] = field(default_factory=dict)
    gradient_reference_escrow: dict[int, Any] = field(default_factory=dict)
    activation_escrow_ram_bytes: int = 0
    activation_escrow_peak_ram_bytes: int = 0
    activation_escrow_spilled_bytes: int = 0
    gradient_reference_logical_bytes: int = 0
    gradient_reference_peak_count: int = 0
    live_gradient_labels: set[str] = field(default_factory=set)
    _activation_spill_dir: tempfile.TemporaryDirectory[str] | None = field(
        default=None, init=False, repr=False
    )
    _gradient_warning_emitted: bool = field(default=False, init=False, repr=False)

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
        self.activation_escrow.clear()
        self.gradient_reference_escrow.clear()
        self.activation_escrow_ram_bytes = 0
        self.activation_escrow_peak_ram_bytes = 0
        self.activation_escrow_spilled_bytes = 0
        self.gradient_reference_logical_bytes = 0
        self.gradient_reference_peak_count = 0
        self.live_gradient_labels.clear()
        if self._activation_spill_dir is not None:
            self._activation_spill_dir.cleanup()
            self._activation_spill_dir = None
        self._gradient_warning_emitted = False
        self.backend_token = None
        if self.outcome is not None:
            self.outcome = RunOutcome(state=self.outcome.state)

    def escrow_candidate(
        self,
        raw_index: int,
        tensor: Any,
        fields_dict: Mapping[str, Any] | None = None,
    ) -> None:
        """Retain one selector candidate before its immutable event is appended.

        Parameters
        ----------
        raw_index
            Reserved raw operation index.
        tensor
            Live backend tensor for the operation.
        fields_dict
            Live event fields, used to identify positive-index gradient targets.
        """

        profile = self.plan.retention_profile
        if raw_index in profile.gradient_live_indices and fields_dict is not None:
            self.live_gradient_labels.add(str(fields_dict["_label_raw"]))
        if profile.activation_kind is RetentionKind.ACTIVATION:
            with _state.pause_logging():
                payload = safe_copy(tensor, detach_tensor=True)
                nbytes = int(payload.nelement() * payload.element_size())
            self.activation_escrow[raw_index] = ActivationEscrowPayload(payload, nbytes)
            self.activation_escrow_ram_bytes += nbytes
            self.activation_escrow_peak_ram_bytes = max(
                self.activation_escrow_peak_ram_bytes,
                self.activation_escrow_ram_bytes,
            )
            window = profile.activation_window
            if window is not None:
                while len(self.activation_escrow) > window:
                    evicted = self.activation_escrow.pop(next(iter(self.activation_escrow)))
                    if evicted.tensor is not None:
                        self.activation_escrow_ram_bytes -= evicted.nbytes
            self._spill_activation_escrow_to_budget()
        if profile.gradient_kind is RetentionKind.GRADIENT_REFERENCE:
            self.gradient_reference_escrow[raw_index] = tensor
            with _state.pause_logging():
                self.gradient_reference_logical_bytes += int(
                    tensor.nelement() * tensor.element_size()
                )
            self.gradient_reference_peak_count = max(
                self.gradient_reference_peak_count,
                len(self.gradient_reference_escrow),
            )
            window = profile.gradient_window
            if window is not None:
                while len(self.gradient_reference_escrow) > window:
                    oldest_index = next(iter(self.gradient_reference_escrow))
                    evicted = self.gradient_reference_escrow.pop(oldest_index)
                    with _state.pause_logging():
                        self.gradient_reference_logical_bytes -= int(
                            evicted.nelement() * evicted.element_size()
                        )
            self._warn_for_extreme_gradient_retention()

    def _spill_activation_escrow_to_budget(self) -> None:
        """Move oldest detached payloads to temporary files until RAM is within budget."""

        profile = self.plan.retention_profile
        if not profile.spillable:
            return
        while self.activation_escrow_ram_bytes > profile.activation_ram_budget_bytes:
            candidate = next(
                (entry for entry in self.activation_escrow.values() if entry.tensor is not None),
                None,
            )
            if candidate is None:
                return
            if self._activation_spill_dir is None:
                self._activation_spill_dir = tempfile.TemporaryDirectory(
                    prefix="torchlens-activation-escrow-"
                )
            spill_path = Path(self._activation_spill_dir.name) / (
                f"payload-{self.activation_escrow_spilled_bytes:020d}.pt"
            )
            import torch

            with _state.pause_logging():
                torch.save(candidate.tensor, spill_path)
            candidate.tensor = None
            candidate.spill_path = spill_path
            self.activation_escrow_ram_bytes -= candidate.nbytes
            self.activation_escrow_spilled_bytes += candidate.nbytes

    def _warn_for_extreme_gradient_retention(self) -> None:
        """Warn once when unwindowable live references cross the compiled byte bound."""

        profile = self.plan.retention_profile
        if (
            self._gradient_warning_emitted
            or profile.gradient_window is not None
            or profile.spillable
            or self.gradient_reference_logical_bytes <= profile.gradient_warning_threshold_bytes
        ):
            return
        self._gradient_warning_emitted = True
        retained_mib = self.gradient_reference_logical_bytes / (1024 * 1024)
        warnings.warn(
            "Unwindowable gradient selection is retaining graph-connected tensor references "
            f"covering approximately {retained_mib:.1f} MiB of logical tensor payload. "
            "Narrow the selector or use an explicit trace refresh to reduce peak memory.",
            RuntimeWarning,
            stacklevel=3,
        )

    def resolve_deferred_retention(self, trace: Any, output_tensors: list[Any]) -> None:
        """Resolve final selectors, project activation winners, and install grad hooks."""

        from ..backends.torch.tensor_tracking import _add_tensor_backward_hook
        from .trace import _get_op_nums_from_user_labels

        activation_selector = getattr(trace, "_deferred_retention_selector", None)
        if activation_selector is not None:
            live_output_by_raw_index: dict[int, Any] = {}
            for output_label, output_tensor in zip(trace.output_layers, output_tensors):
                output_op = trace.layer_dict_all_keys[output_label]
                live_output_by_raw_index[output_op.raw_index] = output_tensor
                for parent_label in output_op.parents:
                    parent = trace.layer_dict_all_keys[parent_label]
                    live_output_by_raw_index[parent.raw_index] = output_tensor
            selected = _get_op_nums_from_user_labels(trace, activation_selector)
            selected_nums = set() if selected == "all" else set(selected)
            selected_nums.update(
                op.raw_index
                for op in trace.layer_list
                if getattr(op, "layer_type", None) == "output"
            )
            for op in trace.layer_list:
                if op.raw_index in selected_nums and getattr(op, "layer_type", None) == "output":
                    selected_nums.update(
                        trace.layer_dict_all_keys[parent_label].raw_index
                        for parent_label in op.parents
                    )
            for op in trace.layer_list:
                if op.raw_index not in selected_nums:
                    continue
                payload = live_output_by_raw_index.get(op.raw_index)
                payload_entry = self.activation_escrow.get(op.raw_index)
                if payload is None and payload_entry is not None:
                    payload = payload_entry.materialize()
                if trace.backward_ready:
                    payload = self.gradient_reference_escrow.get(op.raw_index, payload)
                if payload is None and getattr(op, "layer_type", None) == "output":
                    for parent_label in op.parents:
                        parent = trace.layer_dict_all_keys[parent_label]
                        payload = live_output_by_raw_index.get(parent.raw_index)
                        parent_payload_entry = self.activation_escrow.get(parent.raw_index)
                        if payload is None and parent_payload_entry is not None:
                            payload = parent_payload_entry.materialize()
                        if trace.backward_ready:
                            payload = self.gradient_reference_escrow.get(parent.raw_index, payload)
                        if payload is not None:
                            break
                if payload is None:
                    continue
                op.save_activation(
                    payload,
                    (),
                    {},
                    False,
                    getattr(trace, "activation_transform", None),
                )
            from ..postprocess import _refresh_fast_saved_summary

            _refresh_fast_saved_summary(trace)
            trace._layer_nums_to_save = sorted(selected_nums)

        gradient_selector = getattr(trace, "_deferred_gradient_selector", None)
        if gradient_selector is None:
            trace.__dict__.pop("_deferred_retention_selector", None)
            return
        selected_grads = _get_op_nums_from_user_labels(trace, gradient_selector)
        trace._grad_op_nums_to_save = selected_grads
        hook_nums = (
            {op.raw_index for op in trace.layer_list}
            if selected_grads == "all"
            else set(selected_grads)
        )
        for op in trace.layer_list:
            if op.raw_index in hook_nums and getattr(op, "layer_type", None) == "output":
                hook_nums.update(
                    trace.layer_dict_all_keys[parent_label].raw_index for parent_label in op.parents
                )
        trace._installing_deferred_gradient_hooks = True
        try:
            for op in trace.layer_list:
                tensor = self.gradient_reference_escrow.get(op.raw_index)
                if tensor is not None and op.raw_index in hook_nums:
                    _add_tensor_backward_hook(trace, tensor, op._label_raw)
        finally:
            trace.__dict__.pop("_installing_deferred_gradient_hooks", None)
        trace.__dict__.pop("_deferred_retention_selector", None)
        trace.__dict__.pop("_deferred_gradient_selector", None)

    def observe_event(self, event: OpEvent) -> None:
        """Populate stage-2 sidecars from an existing producer event.

        Parameters
        ----------
        event
            Frozen event already appended to the legacy ``CaptureEvents``
            buffer.  No event fields, payloads, selectors, or interventions are
            recomputed here.
        """

        if self._sealed_core is not None:
            raise RuntimeError("Cannot append capture facts after the run core is sealed.")
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

        if self._sealed_core is not None:
            raise RuntimeError("Cannot replace capture facts after the run core is sealed.")
        event_id = self.event_journal.replace(event)
        self.decision_ledger.append_from_event(event_id, event)
        self.payload_ledger.append_from_event(event_id, event)

    def seal(self) -> CapturedRunCore:
        """Seal and return the repeatedly readable projection source.

        Returns
        -------
        CapturedRunCore
            Immutable snapshots of facts and stable-id sidecars. Repeated calls
            return the same core object.
        """

        if self._sealed_core is None:
            self._sealed_core = CapturedRunCore(
                event_facts=self.event_journal.facts,
                decisions=MappingProxyType(dict(self.decision_ledger.records)),
                payloads=MappingProxyType(dict(self.payload_ledger.records)),
                completeness=MappingProxyType(dict(self.completeness.states)),
                projection_facts=MappingProxyType(dict(self.projection_facts)),
            )
        return self._sealed_core

    def snapshot_recording_projection(
        self,
        trace: object,
        *,
        output_tensors: list[Any] | None = None,
        output_tensor_addresses: list[str] | None = None,
    ) -> None:
        """Snapshot legacy facts needed by Recording projections before sealing.

        Parameters
        ----------
        trace
            Live predicate-mode trace that measured the run facts.
        output_tensors
            Attributed model outputs for a completed run, when available.
        output_tensor_addresses
            Structural addresses corresponding to ``output_tensors``.
        """

        if self._sealed_core is not None:
            raise RuntimeError("Cannot snapshot projection facts after the run core is sealed.")
        capture_events = getattr(trace, "capture_events", None)
        recording = getattr(trace, "_fastlog_recording", None)
        records = ()
        if recording is not None:
            records = tuple(object.__getattribute__(recording, "records"))
        self.projection_facts.update(
            {
                "capture_events": (
                    None if capture_events is None else capture_events.copy_for_replay()
                ),
                "records": records,
                "output_tensors": tuple(output_tensors or ()),
                "output_tensor_addresses": tuple(output_tensor_addresses or ()),
                "output_labels": tuple(
                    getattr(getattr(tensor, "_tl", None), "label_raw", None)
                    for tensor in (output_tensors or ())
                ),
                "capture_start_time": getattr(trace, "capture_start_time", 0),
                "setup_duration": getattr(trace, "setup_duration", 0),
                "forward_duration": getattr(trace, "forward_duration", 0),
                "forward_peak_memory": getattr(trace, "forward_peak_memory", None),
                "forward_memory_backend": getattr(trace, "forward_memory_backend", None),
                "random_seed": getattr(trace, "random_seed", None),
                "source_model_ref": getattr(trace, "_source_model_ref", None),
                "layer_counter": getattr(trace, "_layer_counter", 0),
            }
        )

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
    # This is a compatibility-only demand declaration.  Existing producers
    # continue to choose their historical work; no extra metadata or payload
    # work is requested from them in Stage 2.
    default_enrichment = (
        EnrichmentLevel.METADATA if capture_mode == "exhaustive" else EnrichmentLevel.SHELL
    )
    options = getattr(trace, "_predicate_save_options", None)
    deferred_activation = bool(getattr(trace, "_deferred_retention_selector", None))
    deferred_gradients = bool(getattr(trace, "_deferred_gradient_selector", None))
    graph_connected = bool(getattr(trace, "backward_ready", False))
    negative_windows = _negative_selector_windows(layers_to_save)
    grad_negative_windows = _negative_selector_windows(grad_layers_to_save)
    grad_positive_indices = _positive_selector_indices(grad_layers_to_save)
    activation_window = max(negative_windows) if negative_windows else None
    gradient_window = max(grad_negative_windows) if grad_negative_windows else None
    retention_profile = RetentionProfile(
        activation_kind=(
            RetentionKind.ACTIVATION
            if deferred_activation and not graph_connected
            else RetentionKind.NONE
        ),
        activation_window=activation_window if deferred_activation else 0,
        gradient_kind=(
            RetentionKind.GRADIENT_REFERENCE
            if deferred_gradients
            and not _contains_only_positive_integer_selectors(grad_layers_to_save)
            else RetentionKind.NONE
        ),
        gradient_window=gradient_window if deferred_gradients else 0,
        spillable=deferred_activation and not graph_connected,
        gradient_live_indices=grad_positive_indices,
    )
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
        retention_profile=retention_profile,
    )


def _negative_selector_windows(selector: Any) -> tuple[int, ...]:
    """Return rolling-window bounds declared by negative integer selectors.

    Parameters
    ----------
    selector
        Legacy selector value or nested selector container.

    Returns
    -------
    tuple[int, ...]
        Positive tail-window sizes in encounter order.
    """

    if isinstance(selector, int) and selector < 0:
        return (-selector,)
    if isinstance(selector, (list, tuple, set, frozenset)):
        return tuple(window for item in selector for window in _negative_selector_windows(item))
    return ()


def _positive_selector_indices(selector: Any) -> tuple[int, ...]:
    """Return raw indices declared by positive integer gradient selectors."""

    if isinstance(selector, int) and not isinstance(selector, bool) and selector >= 0:
        return (selector + 1,)
    if isinstance(selector, (list, tuple, set, frozenset)):
        return tuple(index for item in selector for index in _positive_selector_indices(item))
    return ()


def _contains_only_positive_integer_selectors(selector: Any) -> bool:
    """Return whether a gradient selection can install every hook live."""

    if isinstance(selector, int) and not isinstance(selector, bool):
        return selector >= 0
    if isinstance(selector, (list, tuple, set, frozenset)) and selector:
        return all(_contains_only_positive_integer_selectors(item) for item in selector)
    return False


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
