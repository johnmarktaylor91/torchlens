"""Run-owned capture state and legacy compatibility adapters."""

from __future__ import annotations

import tempfile
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from weakref import ref

from .. import _state
from ..ir.events import OpEvent
from ..utils.tensor_utils import safe_copy
from .outcome import CaptureOutcome
from .plan import CapturePlan, EnrichmentLevel, RetentionKind, RetentionProfile

TerminalState = Literal["complete", "halted", "failed"]
"""First-transition log vocabulary. The settlement authority passes
``CaptureStatus``-derived values that collapse onto these three states; the
full six-status truth rides ``RunOutcome.capture_outcome``."""
CleanupCallback = Callable[[], None]


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
    events
        Canonical immutable operation event spine in producer order, folded
        through the journal's amendment reducer at seal time.
    projection_facts
        Snapshot of legacy run facts needed by Recording projections.
    amendment_watermark
        Highest amendment seq (lane-local domain) consumed by the seal fold,
        or ``None`` when the session had no bound journal. Projectors source
        working-copy watermarks from here (reviewer note S-N2) so carried
        amendments the seal already folded can never apply twice.
    """

    events: tuple[OpEvent, ...]
    projection_facts: Mapping[str, Any]
    amendment_watermark: int | None = None


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
    capture_outcome
        Typed settled outcome record written by the settlement authority
        (``torchlens.capture.outcome``); the authority is its only writer.
    """

    state: TerminalState
    output: Any = None
    product: Any = None
    partial_product: Any = None
    exception: BaseException | None = None
    capture_outcome: CaptureOutcome | None = None


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


# A plain (non-slotted) dataclass stays weakref-able on every supported
# interpreter -- ``weakref_slot=True`` needs py3.11+, and the lifetime-neutral
# weakref contract only requires that ``CaptureSession`` be weakref-able, which
# ``__dict__``/``__weakref__`` provide here without a version-specific slots combo.
@dataclass
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
    output_bindings: dict[str, object] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    module_state: dict[str, object] = field(default_factory=dict)
    history_state: dict[str, object] = field(default_factory=dict)
    builders: dict[str, object] = field(default_factory=dict)
    cleanup_stack: list[_CleanupEntry] = field(default_factory=list)
    outcome: RunOutcome | None = None
    _event_spine: list[OpEvent] | None = field(default=None, init=False, repr=False)
    _sealed_core: CapturedRunCore | None = field(default=None, init=False, repr=False)
    _event_journal: Any | None = field(default=None, init=False, repr=False)
    projection_facts: dict[str, Any] = field(default_factory=dict)
    activation_escrow: dict[int, ActivationEscrowPayload] = field(default_factory=dict)
    gradient_reference_escrow: dict[int, Any] = field(default_factory=dict)
    activation_escrow_ram_bytes: int = 0
    activation_escrow_peak_ram_bytes: int = 0
    activation_escrow_spilled_bytes: int = 0
    _activation_escrow_spill_index: int = field(default=0, init=False, repr=False)
    gradient_reference_logical_bytes: int = 0
    gradient_reference_peak_count: int = 0
    _activation_spill_dir: tempfile.TemporaryDirectory[str] | None = field(
        default=None, init=False, repr=False
    )
    _gradient_warning_emitted: bool = field(default=False, init=False, repr=False)

    def bind_event_spine(self, events: list[OpEvent]) -> None:
        """Bind the session to the active ``CaptureEvents`` operation list.

        Parameters
        ----------
        events
            Canonical mutable operation-event list for this capture run.
        """

        if self._sealed_core is not None:
            raise RuntimeError("Cannot bind a capture spine after the run core is sealed.")
        self._event_spine = events

    def bind_event_journal(self, events: Any) -> None:
        """Bind the session to the whole ``CaptureEvents`` journal object.

        The seal reads the op lane through the journal's amendment reducer
        and stamps the seal watermark back onto it; the raw list binding
        above remains the fallback for spine-only callers.

        Parameters
        ----------
        events
            Canonical mutable ``CaptureEvents`` buffer for this capture run.
        """

        if self._sealed_core is not None:
            raise RuntimeError("Cannot bind a capture journal after the run core is sealed.")
        self._event_journal = events

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

        self._event_spine = None
        self._event_journal = None
        self.output_bindings.clear()
        self.counters.clear()
        self.module_state.clear()
        self.history_state.clear()
        self.builders.clear()
        self.projection_facts.clear()
        self._sealed_core = None
        self.cleanup_stack.clear()
        self.activation_escrow.clear()
        self.gradient_reference_escrow.clear()
        self.activation_escrow_ram_bytes = 0
        self.activation_escrow_peak_ram_bytes = 0
        self.activation_escrow_spilled_bytes = 0
        self._activation_escrow_spill_index = 0
        self.gradient_reference_logical_bytes = 0
        self.gradient_reference_peak_count = 0
        if self._activation_spill_dir is not None:
            try:
                self._activation_spill_dir.cleanup()
            except Exception:
                pass
            finally:
                self._activation_spill_dir = None
        self._gradient_warning_emitted = False
        self.backend_token = None
        if self.outcome is not None:
            self.outcome = RunOutcome(
                state=self.outcome.state,
                capture_outcome=self.outcome.capture_outcome,
            )

    def escrow_candidate(
        self,
        raw_index: int,
        tensor: Any,
        *,
        retain_activation: bool = True,
    ) -> None:
        """Retain one selector candidate before its immutable event is appended.

        Parameters
        ----------
        raw_index
            Reserved raw operation index.
        tensor
            Live backend tensor for the operation.
        retain_activation
            Whether this candidate still needs detached deferred retention.
        """

        profile = self.plan.retention_profile
        if (
            profile.activation_kind is RetentionKind.ACTIVATION
            and retain_activation
            and profile.activation_module_exit_addresses is None
        ):
            self._retain_activation(raw_index, tensor)
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
                    self.gradient_reference_escrow.pop(oldest_index)
            self._warn_for_extreme_gradient_retention()

    def escrow_module_exit_outputs(
        self,
        module_call_label: str,
        outputs: tuple[tuple[int, Any], ...],
    ) -> None:
        """Retain the outputs of one module pass for a pure ``tl.module`` selector.

        A deferred selector built only from ``tl.module`` terms joined by
        ``|`` resolves to module-output ops and nothing else, so retaining
        each matching module pass's outputs at its exit replaces escrowing
        every op's output. Exit time is the copy point the per-op escrow
        would have used for these ops: no tracked op runs between an output
        op's creation and its module's exit without minting a new label.

        Parameters
        ----------
        module_call_label
            Pass-qualified module call label (``"address:call_index"``).
        outputs
            ``(raw_index, tensor)`` for each labeled output of the module pass.
        """

        profile = self.plan.retention_profile
        addresses = profile.activation_module_exit_addresses
        if addresses is None or profile.activation_kind is not RetentionKind.ACTIVATION:
            return
        from ..ir.selector_eval import module_address_matches

        if not any(module_address_matches(module_call_label, address) for address in addresses):
            return
        for raw_index, tensor in outputs:
            if raw_index not in self.activation_escrow:
                self._retain_activation(raw_index, tensor)

    def _retain_activation(self, raw_index: int, tensor: Any) -> None:
        """Copy one detached activation into escrow, windowed and spilled to budget.

        Parameters
        ----------
        raw_index
            Raw operation index that keys the escrow.
        tensor
            Live backend tensor to copy.
        """

        profile = self.plan.retention_profile
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
                elif evicted.spill_path is not None:
                    evicted.spill_path.unlink(missing_ok=True)
                    self.activation_escrow_spilled_bytes -= evicted.nbytes
        self._spill_activation_escrow_to_budget()

    def _spill_activation_escrow_to_budget(self) -> None:
        """Move oldest detached payloads to temporary files until RAM is within budget.

        Raises
        ------
        SaveBudgetExceededError
            ``escrow_budget_exceeded`` when the DECLARED total spill bound
            (leverage B9) would be crossed: an unbounded implicit disk write
            is refused typed, never performed silently.
        """

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
            budget = profile.escrow_spill_budget_bytes
            if budget is not None and self.activation_escrow_spilled_bytes + candidate.nbytes > (
                budget
            ):
                from .._save_budget import SaveBudgetExceededError, format_bytes

                raise SaveBudgetExceededError(
                    "Deferred-selector escrow spill crossed its declared bound: "
                    f"{format_bytes(self.activation_escrow_spilled_bytes)} already "
                    f"spilled to temporary disk plus {format_bytes(candidate.nbytes)} "
                    f"pending exceeds the {format_bytes(budget)} budget. Escrow "
                    "holds candidate payloads until postprocess resolves the "
                    "selector. A selector that needs FINAL graph numbering (an "
                    "op label, a positive ordinal, or an output spelling) or "
                    "that mixes `tl.module(...)` with other terms escrows every "
                    "op's output. A selector made only of `tl.module(...)` terms "
                    "joined by `|` escrows only the matching module outputs, so "
                    "reaching this bound with one means those outputs alone "
                    "exceed it. Address the same sites with a live spelling "
                    "instead -- `tl.module(path)` on its own or unioned with "
                    "other module paths, `tl.func(name)`, a bare op-type name, "
                    "or a `save=` predicate (see torchlens.capture.preflight."
                    "address_preflight for the equivalent spelling).",
                    code="escrow_budget_exceeded",
                    spilled_bytes=self.activation_escrow_spilled_bytes,
                    pending_bytes=candidate.nbytes,
                    budget_bytes=budget,
                )
            if self._activation_spill_dir is None:
                self._activation_spill_dir = tempfile.TemporaryDirectory(
                    prefix="torchlens-activation-escrow-"
                )
            spill_path = Path(self._activation_spill_dir.name) / (
                f"payload-{self._activation_escrow_spill_index:020d}.pt"
            )
            self._activation_escrow_spill_index += 1
            import torch

            from ..backends.torch._tl import clear_meta

            with _state.pause_logging():
                # Spill the payload WITHOUT its TorchLens sidecar: a captured
                # payload's ``TensorMeta.label_storage`` pins its own
                # ``UntypedStorage``, and ``torch.save`` serializes the
                # tensor's ``__dict__``, so the same bytes reach the writer
                # both as the typed tensor and as the untyped pin ("Cannot
                # save multiple tensors or storages that view the same data
                # as different types"). Escrow consumption is raw-index-keyed
                # (never label-keyed) and ``materialize`` loads with
                # ``weights_only=True``, which would refuse the sidecar
                # anyway. Stripping metadata from the payload the escrow
                # exclusively owns is NOT a clone: the storage is written
                # once, unchanged.
                clear_meta(candidate.tensor)
                torch.save(candidate.tensor, spill_path)
            candidate.tensor = None
            candidate.spill_path = spill_path
            self.activation_escrow_ram_bytes -= candidate.nbytes
            self.activation_escrow_spilled_bytes += candidate.nbytes

    def _warn_for_extreme_gradient_retention(self) -> None:
        """Warn once when graph-pinned references cross the compiled byte bound."""

        profile = self.plan.retention_profile
        if (
            self._gradient_warning_emitted
            or self.gradient_reference_logical_bytes <= profile.gradient_warning_threshold_bytes
        ):
            return
        self._gradient_warning_emitted = True
        retained_mib = self.gradient_reference_logical_bytes / (1024 * 1024)
        warnings.warn(
            "Graph-connected gradient selection is retaining tensor references "
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
            for output_label, output_tensor in zip(
                trace.output_layers, output_tensors, strict=True
            ):
                output_op = trace.layer_dict_all_keys[output_label]
                live_output_by_raw_index[output_op.raw_index] = output_tensor
                for parent_label in output_op.parents:
                    parent = trace.layer_dict_all_keys[parent_label]
                    live_output_by_raw_index[parent.raw_index] = output_tensor
            from ..intervention.selectors import BaseSelector
            from ..ir.selector_eval import selector_contains_kind

            selected: list[int] | str
            if isinstance(activation_selector, BaseSelector):
                from ..intervention.resolver import _resolve_unchecked

                selected = sorted(
                    {
                        raw_index
                        for site in _resolve_unchecked(
                            tuple(getattr(trace, "layer_list", ())),
                            activation_selector,
                            strict=False,
                        )
                        if isinstance((raw_index := getattr(site, "raw_index", None)), int)
                    }
                )
                trace._tl_save_selector_fire_count = len(selected)
            else:
                selected = _get_op_nums_from_user_labels(trace, activation_selector)
            requested_nums = set() if selected == "all" else set(selected)
            selected_nums = set(requested_nums)
            exact_selector = isinstance(activation_selector, BaseSelector) and (
                selector_contains_kind(activation_selector, "module")
            )
            if not exact_selector:
                selected_nums.update(
                    op.raw_index
                    for op in trace.layer_list
                    if getattr(op, "layer_type", None) == "output"
                )
                for op in trace.layer_list:
                    if (
                        op.raw_index in selected_nums
                        and getattr(op, "layer_type", None) == "output"
                    ):
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
                    if op.has_saved_activation:
                        continue
                    if op.raw_index in requested_nums:
                        raise RuntimeError(
                            "TorchLens could not retain an explicitly requested activation "
                            f"for {op.layer_label!r} (raw index {op.raw_index})."
                        )
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
        from ..intervention.selectors import BaseSelector as _BaseSelector

        selected_grads: list[int] | str
        if isinstance(gradient_selector, _BaseSelector):
            if _selector_needs_backward_context(gradient_selector):
                # Backward-context selectors (in_backward_pass, grad_input/
                # grad_output, grad_fn kinds) are undecidable in site space --
                # site-lifecycle evaluation reads them as False, which used to
                # resolve ZERO hook sites and silently record nothing. Hook
                # placement takes the honest superset (every op); retention
                # stays exact because the fire-time policy re-evaluates the
                # selector per grad event with the backward context attached.
                selected_grads = "all"
            else:
                # Selector-shaped save_grads resolves like the activation
                # branch: the public resolve_sites gate refuses mid-capture,
                # so the unchecked resolver walks the finalized layer list
                # directly.
                from ..intervention.resolver import _resolve_unchecked

                selected_grads = sorted(
                    {
                        raw_index
                        for site in _resolve_unchecked(
                            tuple(getattr(trace, "layer_list", ())),
                            gradient_selector,
                            strict=False,
                        )
                        if isinstance((raw_index := getattr(site, "raw_index", None)), int)
                    }
                )
        else:
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

    def seal(self) -> CapturedRunCore:
        """Seal and return the repeatedly readable projection source.

        Returns
        -------
        CapturedRunCore
            Immutable snapshots of facts and stable-id sidecars. Repeated calls
            return the same core object.
        """

        if self._sealed_core is None:
            journal = self._event_journal
            if journal is not None:
                # Fold the amendment lane into the sealed spine and stamp the
                # watermark on BOTH the live journal and the pre-seal
                # projection clone stored by snapshot_recording_projection —
                # the clone predates this seal, so stamping only the live
                # object would strand it without a filter anchor (S-N2).
                events = tuple(journal.amended_op_records())
                watermark = int(journal.amendment_seq or 0)
                journal.core_seal_watermark = watermark
                journal.amendments_sealed = True
                projection_clone = self.projection_facts.get("capture_events")
                if projection_clone is not None:
                    projection_clone.core_seal_watermark = watermark
            else:
                events = tuple(self._event_spine or ())
                watermark = None
            self._sealed_core = CapturedRunCore(
                events=events,
                projection_facts=MappingProxyType(dict(self.projection_facts)),
                amendment_watermark=watermark,
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
                "layer_counter": getattr(getattr(trace, "_raw_graph_ws", None), "layer_counter", 0),
            }
        )

    def register_cleanup(self, name: str, callback: CleanupCallback) -> _CleanupEntry:
        """Register one teardown action on the session-owned cleanup stack.

        Parameters
        ----------
        name
            Stable action name.  Re-registering an existing action preserves
            the original callback so cleanup remains exactly once.
        callback
            Existing teardown callback to invoke.

        Returns
        -------
        _CleanupEntry
            Existing or newly registered cleanup entry.
        """

        for entry in self.cleanup_stack:
            if entry.name == name:
                return entry
        entry = _CleanupEntry(name=name, callback=callback)
        self.cleanup_stack.append(entry)
        return entry

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

        entry = self.register_cleanup(name, callback)
        if entry.completed:
            return False
        entry.completed = True
        entry.callback()
        return True

    def transition(
        self,
        state: TerminalState,
        *,
        output: Any = None,
        product: Any = None,
        partial_product: Any = None,
        exception: BaseException | None = None,
        capture_outcome: CaptureOutcome | None = None,
    ) -> RunOutcome:
        """Perform the single terminal transition for this session.

        Parameters
        ----------
        state
            Terminal state to record.
        output, product, partial_product, exception
            Existing compatibility outcome fields to mirror.
        capture_outcome
            Typed settled outcome from the settlement authority.

        Returns
        -------
        RunOutcome
            Frozen terminal outcome.

        Raises
        ------
        RuntimeError
            If a caller attempts a second terminal transition. The guard is
            unconditional: ``RunOutcome.output`` can hold a tensor, so an
            equality-based "same transition" carve-out would raise the
            ambiguous-bool ``RuntimeError`` from tensor ``__eq__`` instead.
        """

        if self.outcome is not None:
            raise RuntimeError("CaptureSession already reached a terminal state.")
        self.outcome = RunOutcome(
            state=state,
            output=output,
            product=product,
            partial_product=partial_product,
            exception=exception,
            capture_outcome=capture_outcome,
        )
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
    from .._trace_selector_helpers import _selector_requires_unwindowed_escrow

    negative_windows = _negative_selector_windows(layers_to_save)
    grad_negative_windows = _negative_selector_windows(grad_layers_to_save)
    activation_window = max(negative_windows) if negative_windows else None
    gradient_window = max(grad_negative_windows) if grad_negative_windows else None
    # Final-numbering components (integer ordinals, indexed labels) resolve
    # post-postprocess at arbitrary graph positions: a mixed selection such as
    # ``[1, -1]`` must not let the tail window evict the payload the positive
    # component needs before deferred resolution runs.
    if _selector_requires_unwindowed_escrow(layers_to_save):
        activation_window = None
    if _selector_requires_unwindowed_escrow(grad_layers_to_save):
        gradient_window = None
    from ..ir.selector_eval import module_union_addresses

    # A pure ``tl.module`` union resolves only to module-output ops, so the
    # session retains those at module exit instead of escrowing every op.
    module_exit_addresses = module_union_addresses(
        getattr(trace, "_deferred_retention_selector", None)
    )
    retention_profile = RetentionProfile(
        activation_kind=(
            RetentionKind.ACTIVATION
            if deferred_activation and not graph_connected
            else RetentionKind.NONE
        ),
        activation_window=activation_window if deferred_activation else 0,
        # Every deferred gradient selector retains references and installs its
        # hooks post-postprocess: positive integer ordinals are FINAL layer
        # numbers, so no raw-index "prediction" can place their hooks live.
        gradient_kind=(
            RetentionKind.GRADIENT_REFERENCE if deferred_gradients else RetentionKind.NONE
        ),
        gradient_window=gradient_window if deferred_gradients else 0,
        spillable=deferred_activation and not graph_connected,
        activation_module_exit_addresses=module_exit_addresses,
    )
    return CapturePlan.compile(
        projection_target=projection_target,
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


#: Selector kinds decidable only with backward-fire context (pass number,
#: grad kind, grad_fn identity); site-lifecycle evaluation reads them as
#: False, so hook placement must not consult site resolution for them.
_BACKWARD_CONTEXT_SELECTOR_KINDS = frozenset(
    {"backward_pass", "grad_kind", "grad_fn", "grad_fn_handle", "grad_fn_label"}
)


def _selector_needs_backward_context(selector: Any) -> bool:
    """Return whether a selector tree contains a backward-context leaf.

    Parameters
    ----------
    selector
        Selector (possibly composite/negated) supplied as ``save_grads``.

    Returns
    -------
    bool
        Whether any leaf requires backward-fire context to evaluate.
    """

    stack: list[Any] = [selector]
    while stack:
        node = stack.pop()
        if getattr(node, "selector_kind", None) in _BACKWARD_CONTEXT_SELECTOR_KINDS:
            return True
        children = getattr(node, "selectors", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        wrapped = getattr(node, "selector", None)
        if wrapped is not None:
            stack.append(wrapped)
    return False


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
    # The trace is the SOLE strong owner of its run session; no side registry
    # may be an ownership head.
    trace._capture_session = session  # type: ignore[attr-defined]
    return session


def capture_session_for(owner: object) -> CaptureSession | None:
    """Return the run session attached to a trace-like owner.

    Parameters
    ----------
    owner
        Trace-like owner that may carry an active capture session.

    Returns
    -------
    CaptureSession | None
        Attached session when the owner is on an active capture run.
    """

    session = getattr(owner, "_capture_session", None)
    return session if isinstance(session, CaptureSession) else None


def detach_capture_session(trace: object, events: object, session: CaptureSession) -> None:
    """Detach and release a completed capture session.

    Parameters
    ----------
    trace
        Trace owner for the completed run.
    events
        Event buffer associated with the completed run.
    session
        Session to detach.  Mismatched attachments are retained to avoid
        disturbing a subsequent run.

    Returns
    -------
    None
        Removes both attachments and clears the session.  The operation is
        safe to invoke more than once.
    """

    if getattr(trace, "_capture_session", None) is session:
        try:
            trace.__dict__.pop("_capture_session", None)
        except AttributeError:
            pass
    events_session_ref = getattr(events, "_tl_capture_session_ref", None)
    if events_session_ref is not None and events_session_ref() is session:
        events.__dict__.pop("_tl_capture_session_ref", None)
    session.release()


def attach_capture_events_session(events: object, session: CaptureSession) -> None:
    """Associate an event buffer with its owning run session.

    Parameters
    ----------
    events
        Existing mutable ``CaptureEvents`` buffer for the active run.
    session
        Run owner whose sealed core snapshots this buffer's operation spine.
    """

    op_events = getattr(events, "op_events", None)
    if not isinstance(op_events, list):
        raise TypeError("Capture event buffers must expose a mutable op_events list.")
    session.bind_event_spine(op_events)
    if hasattr(events, "amended_op_records"):
        session.bind_event_journal(events)
    # Weak backref only: the session (via the trace) owns the run; the buffer
    # must never keep a completed session alive.
    events._tl_capture_session_ref = ref(session)  # type: ignore[attr-defined]


def capture_session_for_events(events: object) -> CaptureSession | None:
    """Return the session associated with one event buffer.

    Parameters
    ----------
    events
        Existing mutable ``CaptureEvents`` buffer.

    Returns
    -------
    CaptureSession | None
        Active owning session, if the buffer is still attached to one.
    """

    session_ref = getattr(events, "_tl_capture_session_ref", None)
    if session_ref is None:
        return None
    session = session_ref()
    return session if isinstance(session, CaptureSession) else None
