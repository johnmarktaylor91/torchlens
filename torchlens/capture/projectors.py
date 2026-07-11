"""Sibling projections over a sealed capture run core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable, cast
import warnings
import weakref
from weakref import WeakKeyDictionary

from ..ir.events import OpEvent
from .session import CapturedRunCore

if TYPE_CHECKING:
    from ..fastlog.types import ActivationRecord

_REFRESH_SOURCES: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


def _event_from_core(core: CapturedRunCore, fact_index: int) -> OpEvent:
    """Resolve one event fact with its authoritative stable-id sidecars.

    Parameters
    ----------
    core
        Sealed source for the projection.
    fact_index
        Producer-order index of the event fact.

    Returns
    -------
    OpEvent
        Immutable event view with decision and payload fields sourced from the
        ledgers keyed by the fact's stable event identity.
    """

    fact = core.event_facts[fact_index]
    event = fact.event
    decision = core.decisions.get(fact.event_id)
    payload = core.payloads.get(fact.event_id)
    if decision is not None:
        event = replace(
            event,
            predicate_matched=decision.predicate_matched,
            intervention_fired=decision.intervention_fired,
            intervention_replaced=decision.intervention_replaced,
            fire_results=decision.fire_results,
        )
    if payload is not None:
        event = replace(event, output=payload.output)
    return event


@dataclass(frozen=True, slots=True)
class TraceProjector:
    """Read Trace Step-0 operation events from a sealed run core."""

    core: CapturedRunCore

    def events(self) -> tuple[OpEvent, ...]:
        """Return operation events in producer order.

        Returns
        -------
        tuple[OpEvent, ...]
            Repeatedly readable operation facts resolved through stable-id
            decision and payload ledgers.
        """

        return tuple(
            _event_from_core(self.core, index) for index in range(len(self.core.event_facts))
        )


@dataclass(frozen=True, slots=True)
class RefreshProjector:
    """Project a same-graph captured rerun onto an existing Trace."""

    target: Any
    layer_nums_to_save: str | tuple[int, ...] = "all"
    grad_layer_nums_to_save: str | tuple[int, ...] = "all"

    _DYNAMIC_OP_FIELDS = frozenset(
        {
            "out",
            "transformed_out",
            "has_saved_activation",
            "saved_args",
            "saved_kwargs",
            "has_saved_args",
            "shape",
            "dtype",
            "activation_memory",
            "transformed_out_shape",
            "transformed_out_dtype",
            "transformed_activation_memory",
            "grad",
            "transformed_grad",
            "has_grad",
            "transformed_grad_shape",
            "transformed_grad_dtype",
            "transformed_gradient_memory",
            "grad_fn_class_name",
            "grad_fn_class_qualname",
            "grad_fn_object_id",
            "grad_fn_handle",
            "autograd_memory",
            "num_autograd_tensors",
            "bytes_delta_at_call",
            "bytes_peak_at_call",
            "func_duration",
            "func_rng_states",
            "func_autocast_state",
            "non_tensor_pos_args",
            "non_tensor_kwargs",
            "func_config",
            "has_out_variations",
            "out_versions_by_child",
        }
    )

    def project(self, refreshed: Any) -> None:
        """Validate and apply refreshed payloads without replacing graph containers.

        Parameters
        ----------
        refreshed
            Fully postprocessed Trace captured by the fixed-order kernel.

        Raises
        ------
        ValueError
            If the refreshed computational graph differs from the target graph.
        """

        target_signature = self._graph_signature(self.target)
        refreshed_signature = self._graph_signature(refreshed)
        if target_signature != refreshed_signature:
            raise self._graph_change_error(refreshed)
        if getattr(self.target, "internal_sink_ops", ()):
            raise self._graph_change_error(refreshed)
        refreshed_by_raw = {layer._layer_label_raw: layer for layer in refreshed.layer_list}
        for layer in self.target.layer_list:
            new_shape = refreshed_by_raw[layer._layer_label_raw].shape
            if layer.shape is not None and new_shape != layer.shape:
                warnings.warn(
                    f"Tensor shape changed for '{layer.layer_label}': "
                    f"expected {layer.shape}, got {new_shape}. "
                    "The computational graph may have changed between ops."
                )
        from ..data_classes._state_adapter import state_items

        preserved_states = [dict(state_items(layer)) for layer in self.target.layer_list]
        if not self.target._refresh_matching_rerun_state_from(refreshed):
            raise self._graph_change_error(refreshed)
        for layer, preserved in zip(self.target.layer_list, preserved_states):
            for field_name, value in preserved.items():
                if field_name not in self._DYNAMIC_OP_FIELDS:
                    layer._internal_set(field_name, value)
        refreshed._refresh_projection_target_ref = weakref.ref(self.target)
        _REFRESH_SOURCES[self.target] = refreshed
        self.target._layer_nums_to_save = self.layer_nums_to_save
        self.target._grad_op_nums_to_save = self.grad_layer_nums_to_save
        self._rebind_backward_hooks()
        self._separate_output_payloads()
        if self.layer_nums_to_save != "all":
            selected = set(self.layer_nums_to_save)
            for output_label in self.target.output_layers:
                output = self.target.layer_dict_all_keys[output_label]
                selected.add(output.raw_index)
                selected.update(
                    self.target.layer_dict_all_keys[parent].raw_index for parent in output.parents
                )
            for layer in self.target.layer_list:
                if layer.raw_index not in selected:
                    self._clear_payload(layer)

    @staticmethod
    def _graph_change_error(refreshed: Any) -> ValueError:
        """Build the legacy graph-change exception with partial-capture metadata."""

        from ..partial import PartialTrace

        error = ValueError(
            "The computational graph changed for this forward pass compared to the original "
            "call to trace (either due to different inputs or a different "
            "random seed), so save_new_outs failed. Please re-run "
            "trace with the desired inputs."
        )
        error.partial_log = PartialTrace(refreshed, error)  # type: ignore[attr-defined]
        return error

    def _rebind_backward_hooks(self) -> None:
        """Bind refreshed live tensors and grad-fn registry entries to the target Trace."""

        from ..backends.torch.backward import _register_forward_grad_fn
        from ..backends.torch.tensor_tracking import _add_tensor_backward_hook

        self.target.__dict__["_tl_backward_hooked_tensor_keys"] = set()
        for layer in self.target.layer_list:
            _register_forward_grad_fn(
                self.target,
                layer.grad_fn_handle,
                layer._layer_label_raw,
            )
            if layer.out is not None:
                _add_tensor_backward_hook(self.target, layer.out, layer._layer_label_raw)

    def _separate_output_payloads(self) -> None:
        """Copy output-node payloads so they do not alias their parent payloads."""

        from ..utils.tensor_utils import safe_copy

        for output_label in self.target.output_layers:
            output = self.target.layer_dict_all_keys[output_label]
            if not output.parents or output.out is None:
                continue
            output._internal_set(
                "out",
                safe_copy(
                    output.out,
                    detach_tensor=self.target.detach_saved_activations,
                ),
            )

    @staticmethod
    def _clear_payload(layer: Any) -> None:
        """Clear refresh payload fields for one unselected operation.

        Parameters
        ----------
        layer
            Existing operation whose refreshed payload was not requested.
        """

        layer._internal_set("out", None)
        layer._internal_set("transformed_out", None)
        layer.transformed_out_shape = None
        layer.transformed_out_dtype = None
        layer.transformed_activation_memory = None
        layer.has_saved_activation = False
        layer.has_out_variations = False
        layer.out_versions_by_child = {}

    @staticmethod
    def _graph_signature(trace: Any) -> tuple[tuple[Any, ...], ...]:
        """Return the refresh graph-change tripwire signature for a Trace.

        Parameters
        ----------
        trace
            Completed Trace whose operation topology should be summarized.

        Returns
        -------
        tuple[tuple[Any, ...], ...]
            Ordered raw identity, type, and parent facts for every operation.
        """

        return tuple(
            (
                layer._layer_label_raw,
                layer.layer_type,
                tuple(sorted(layer.parents)),
            )
            for layer in trace.layer_list
        )


@dataclass(frozen=True, slots=True)
class RecordingProjection:
    """Sparse activation records and lookup indexes projected from sealed facts."""

    records: tuple["ActivationRecord", ...]
    by_pass: dict[int, list[int]]
    by_label: dict[str, list[tuple[int, int]]]
    by_address: dict[str, list[int]]
    events: tuple[OpEvent, ...]
    capture_events: object | None
    output_tensors: tuple[object, ...]
    output_tensor_addresses: tuple[str, ...]
    output_labels: tuple[str | None, ...]
    trace_facts: dict[str, object]
    buffer_layers: tuple[str, ...]
    internal_source_ops: tuple[str, ...]

    def prepare_trace(self, trace: Any) -> None:
        """Apply sealed run facts required before Trace postprocessing.

        Parameters
        ----------
        trace
            Fresh Trace projection shell to initialize from core facts.
        """

        trace.buffer_layers = list(self.buffer_layers)
        trace.internal_source_ops = list(self.internal_source_ops)
        trace.capture_start_time = self.trace_facts["capture_start_time"]
        trace.setup_duration = self.trace_facts["setup_duration"]
        trace.forward_duration = self.trace_facts["forward_duration"]
        trace.forward_peak_memory = self.trace_facts["forward_peak_memory"]
        trace.forward_memory_backend = self.trace_facts["forward_memory_backend"]
        trace._source_model_ref = self.trace_facts["source_model_ref"]
        trace.random_seed = self.trace_facts["random_seed"]
        trace._layer_counter = cast(int, self.trace_facts["layer_counter"])

        from ..backends.torch._tl import get_tensor_label, set_tensor_label

        for tensor, label in zip(self.output_tensors, self.output_labels):
            if label is not None and get_tensor_label(tensor) is None:
                set_tensor_label(tensor, label)

    def bind_halt_frontier(self, tensor: object, label: str) -> None:
        """Restore a core-attributed halt-frontier label before projection.

        Parameters
        ----------
        tensor
            Retained halt-frontier tensor selected from projected records.
        label
            Authoritative raw event label for that tensor.
        """

        from ..backends.torch._tl import get_tensor_label, set_tensor_label

        if get_tensor_label(tensor) is None:
            set_tensor_label(tensor, label)


class RecordingProjector:
    """Build Recording activation indexes directly from sealed run cores."""

    def project(self, cores: Iterable[CapturedRunCore]) -> RecordingProjection:
        """Build the sparse activation projection without first building a Trace.

        Parameters
        ----------
        cores
            Sealed cores in Recorder pass order.

        Returns
        -------
        RecordingProjection
            Retained records and the current public lookup indexes.
        """

        from .projections import activation_record_from_event

        records: list[ActivationRecord] = []
        by_pass: dict[int, list[int]] = {}
        by_label: dict[str, list[tuple[int, int]]] = {}
        by_address: dict[str, list[int]] = {}
        all_events: list[OpEvent] = []
        captured_cores = tuple(cores)
        for core in captured_cores:
            core_events = TraceProjector(core).events()
            all_events.extend(core_events)
            stored_records = core.projection_facts.get("records", ())
            if stored_records:
                projected_records = stored_records
            else:
                projected_records = tuple(
                    record
                    for event in core_events
                    if (record := activation_record_from_event(event)) is not None
                )
            for record in projected_records:
                index = len(records)
                records.append(record)
                by_pass.setdefault(record.ctx.pass_index, []).append(index)
                by_label.setdefault(record.ctx.label, []).append((record.ctx.pass_index, index))
                if record.ctx.raw_label is not None:
                    by_label.setdefault(record.ctx.raw_label, []).append(
                        (record.ctx.pass_index, index)
                    )
                if record.ctx.address is not None:
                    by_address.setdefault(record.ctx.address, []).append(index)
        last_facts = captured_cores[-1].projection_facts if captured_cores else {}
        capture_events = next(
            (
                core.projection_facts.get("capture_events")
                for core in captured_cores
                if core.projection_facts.get("capture_events") is not None
            ),
            None,
        )
        if capture_events is not None:
            capture_events = capture_events.copy_for_replay()
            capture_events.op_events = list(all_events)
            capture_events.op_event_by_label_raw = {event.label_raw: event for event in all_events}
            capture_events.op_event_index_by_label_raw = {
                event.label_raw: index for index, event in enumerate(all_events)
            }
            capture_events.raw_layer_counter = max(
                (event.raw_index for event in all_events), default=0
            )
        return RecordingProjection(
            tuple(records),
            by_pass,
            by_label,
            by_address,
            tuple(all_events),
            capture_events,
            tuple(last_facts.get("output_tensors", ())),
            tuple(last_facts.get("output_tensor_addresses", ())),
            tuple(last_facts.get("output_labels", ())),
            dict(last_facts),
            tuple(event.label_raw for event in all_events if event.layer_type == "buffer"),
            tuple(
                event.label_raw
                for event in all_events
                if event.layer_type != "input" and not event.parents
            ),
        )
