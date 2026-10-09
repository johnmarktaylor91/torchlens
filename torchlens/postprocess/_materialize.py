"""Step 0 of postprocess: materialize raw Trace state from capture events."""

from __future__ import annotations

import importlib
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from torchlens._io import BlobRef as PortableBlobRef
from torchlens.intervention.types import EdgeUseRecord
from torchlens.ir import CaptureEvents
from torchlens.ir.events import (
    ModuleEnterEvent,
    ModuleExitEvent,
    ModulePrepEvent,
    OpEvent,
)

from ..backends.torch._tl import get_buffer_address, get_tensor_label, get_tensor_meta
from ..constants import LAYER_PASS_LOG_FIELD_ORDER
from ..data_classes._module_role_hints import (
    multi_output_role_from_path,
    role_hints_for_module_class,
)
from ..ir.workspaces import _init_module_hierarchy_data
from ..utils import get_vars_of_type_from_obj
from ..utils._torch_compat import tensor_version_or_none
from ..utils._torch_symbols import torch_attr
from ..utils.display import _record_phase_timing
from ._buffer_addresses import (
    _buffer_addresses_by_label,
    _buffer_alias_snapshots_by_address,
    _recorded_buffer_address,
)
from ._ingest_contract import IngestInputs, JournalView, Step0Result
from ._primitive_profile import _materialize_forward_primitive_profile

if TYPE_CHECKING:
    from torchlens.data_classes.op import Op
    from torchlens.data_classes.trace import Trace


from torchlens.ir.op_record import IngestExtras as _IngestExtras
from torchlens.ir.op_record_scatter import CELL_SOURCES

_EMPTY_INGEST_EXTRAS = _IngestExtras()


def materialize_log_from_fields(fields_dict: dict[str, object], store: object | None = None) -> Op:
    """Construct the live log object for one captured operation.

    Parameters
    ----------
    fields_dict
        Raw field mapping populated by the backend hot path.
    store
        The owning trace's ``OpRowStore`` (the M5 builder ingress): the op is
        appended as a shared columnar row. ``None`` constructs a detached
        single-row op (legacy/preview callers).

    Returns
    -------
    Op
        Materialized operation or buffer log.
    """

    from torchlens.data_classes.op import Op

    pending_blob_ids = _pop_pending_blob_ids(fields_dict)
    # r83 C2: the display ``address`` and the backend-native ``backend_address``
    # are resolved SEPARATELY for buffers (a plain-attribute constant has a
    # display address but is not backend-native state, so its backend_address
    # stays None). ``Op.__init__`` otherwise defaults ``backend_address`` from
    # ``address`` whenever it is None, which would recouple them; the resolved
    # value is stashed under an extra key by ``_fields_from_event`` and applied
    # here after construction so the default cannot override it. r85: the extra
    # key is set ONLY for a BUFFER-addressed record (see ``_fields_from_event``);
    # an op/input node carries no override and keeps the coupled default, so it
    # is genuinely unaffected -- the r83 comment claiming EVERY record was popped
    # (forcing op nodes to None) was FALSE and is corrected there.
    has_backend_override = "_materialized_backend_address" in fields_dict
    backend_address_override = fields_dict.pop("_materialized_backend_address", None)
    op_log = Op(fields_dict, _store=store)
    for field_name, blob_id in pending_blob_ids.items():
        setattr(op_log, field_name, blob_id)
    if has_backend_override:
        op_log.backend_address = cast("str | None", backend_address_override)
    return op_log


def _pop_pending_blob_ids(fields_dict: dict[str, object]) -> dict[str, object]:
    """Remove streaming-only pending blob ids before Op construction.

    Parameters
    ----------
    fields_dict
        Raw field mapping populated by the backend hot path.

    Returns
    -------
    dict[str, object]
        Pending blob-id values keyed by Op attribute name.
    """

    pending_fields = (
        "_pending_blob_id",
        "_pending_transformed_out_blob_id",
        "_pending_grad_blob_id",
        "_pending_transformed_grad_blob_id",
    )
    return {
        field_name: fields_dict.pop(field_name)
        for field_name in pending_fields
        if field_name in fields_dict
    }


def _journal_buffer_write_events(trace: Trace) -> tuple[Any, ...]:
    """Return the journal's buffer-write lane for one trace.

    Buffer writes live in the capture journal (``CaptureEvents.buffer_write_events``),
    not in a Trace-side list. During Step 0 the live stream is still attached as
    ``trace.capture_events``; afterwards the trace owns the released stream via
    ``_capture_events``.
    """

    stream = getattr(trace, "capture_events", None)
    if stream is None:
        stream = getattr(trace, "_capture_events", None)
    return tuple(getattr(stream, "buffer_write_events", ()) or ())


@dataclass(slots=True)
class _ModuleSideChannel:
    """Module side-channel rebuild output (``Step0Result.module_side_channel``).

    Built LOCALLY by ingest from journal lanes; the orchestrator applies it
    onto the trace's ``ModuleCaptureWorkspace`` after ingest returns.
    """

    module_build_data: dict[str, Any]
    module_metadata: dict[str, Any]
    module_forward_args: dict[Any, Any]


def build_ingest_inputs(trace: Trace, events: CaptureEvents) -> IngestInputs:
    """Orchestrator-side construction of the frozen step-0 input bundle.

    Owns the lazy ``TraceCore``/``OpRowStore`` creation (ppdag v3 section 9.1
    orchestrator row) and the trace-backed ``timing_sink``; ingest itself
    reads nothing off the trace beyond this bundle (contract I7).
    """

    core = trace.__dict__.get("_trace_core")
    if core is None:
        from torchlens._trace_core import OpRowStore, TraceCore
        from torchlens.data_classes.op import _OP_STORE_LAYOUT

        core = TraceCore()
        core.ops = OpRowStore(_OP_STORE_LAYOUT)
        trace._trace_core = core

    def timing_sink(bucket: str, elapsed: float) -> None:
        """Record one ingest phase's elapsed time on the trace."""

        _record_phase_timing(trace, bucket, elapsed)

    return IngestInputs(
        journal=JournalView(
            # The folded reducer view IS the op lane step 0 consumes; the
            # amendment lane rides along as read-only provenance facts
            # (already folded into op_events — consumers must never re-apply).
            op_events=tuple(events.amended_op_records()),
            op_amendments=tuple(getattr(events, "op_amendments", ()) or ()),
            module_prep_events=tuple(events.module_prep_events),
            module_enter_events=tuple(events.module_enter_events),
            module_exit_events=tuple(events.module_exit_events),
            pre_hook_events=tuple(events.pre_hook_events),
            buffer_write_events=_journal_buffer_write_events(trace),
            output_version_events=tuple(events.output_version_events),
            grad_fn_handles_by_label_raw=events.grad_fn_handles_by_label_raw,
            aten_events=tuple(events.aten_events),
        ),
        module_workspace=trace._module_capture_ws,
        raw_graph_workspace=trace._raw_graph_ws,
        buffer_initial_values=dict(getattr(trace, "_buffer_initial_values", {}) or {}),
        op_equivalence_classes=trace.op_equivalence_classes,
        source_model_ref=getattr(trace, "_source_model_ref", None),
        param_logs=trace.param_logs,
        owning_trace=trace,
        op_row_store=core.ops,
        trace_core=core,
        input_layers_initial=tuple(trace.input_layers),
        timing_sink=timing_sink,
        # Orchestrator-side option read (contract I7: ingest itself never
        # touches the trace): structure-only captures strip buffer-write
        # payloads to declared geometry (W3, weightsfree memo defect L6).
        scatter_options={"structure_only": bool(getattr(trace, "structure_only", False))},
    )


def apply_step0_result(trace: Trace, result: Step0Result) -> None:
    """Orchestrator-side application of ingest's restaged mutation payloads."""

    for label_raw, op_log in result.raw_log_registrations:
        trace._raw_graph_ws.raw_layer_dict[label_raw] = op_log
        trace._raw_graph_ws.raw_layer_labels_list.append(label_raw)
    for label_raw in result.input_layer_labels:
        if label_raw not in trace.input_layers:
            trace.input_layers.append(label_raw)
    trace.op_equivalence_classes.clear()
    trace.op_equivalence_classes.update(result.equivalence_class_map)
    side_channel = result.module_side_channel
    trace._module_capture_ws.module_build_data = side_channel.module_build_data
    trace._module_capture_ws.module_metadata = side_channel.module_metadata
    trace._module_capture_ws.module_forward_args = side_channel.module_forward_args


def materialize_from_events(trace: Trace, events: CaptureEvents) -> None:
    """Materialize capture events into raw build-state logs.

    Compatibility composition of the frozen step-0 seam: build the
    ``IngestInputs`` bundle, run ``ingest_op_records`` (late-bound through
    this module's namespace so the oracle interception seams stay
    monkeypatchable), and apply the returned ``Step0Result``. Preview
    backends and partial-capture recovery call this wrapper; the torch
    orchestrator reaches it through ``torchlens.postprocess``.

    Parameters
    ----------
    trace
        Trace whose transient build state was populated during capture.
    events
        Mutable event accumulator owned by the active capture session.
    Returns
    -------
    None
        Populates raw trace lookup structures without consuming the sealed source lanes.
    """

    import torchlens.postprocess._materialize as _materialize_module

    inputs = build_ingest_inputs(trace, events)
    result = _materialize_module.ingest_op_records(inputs, CELL_SOURCES)
    apply_step0_result(trace, result)
    _materialize_forward_primitive_profile(
        trace,
        inputs.journal.aten_events,
        recording_enabled=bool(events.aten_recording_enabled),
    )


def ingest_op_records(inputs: IngestInputs, manifest: Mapping[str, str]) -> Step0Result:
    """Step 0 ingest: journal records -> raw op store rows (frozen seam v1).

    One record loop over the folded journal view: each entry adapts to the
    decomposed shape at the ONE ingest boundary (compat ``OpEvent`` through
    ``op_record_from_event``; a decomposed ``OpRecord`` passes through with
    empty extras), the generated scatter produces every record-sourced cell,
    and the JOIN cells are computed from the declared input lanes. Mutations
    are restaged as ``Step0Result`` payloads (X2: the raw-log registration
    map is LOCAL and the payload derives from it).
    """

    from torchlens.ir.op_record import OpRecord, op_record_from_event
    from torchlens.ir.op_record_scatter import scatter_record_to_cells

    journal = inputs.journal
    op_store = inputs.op_row_store
    module_workspace_forward_args = dict(inputs.module_workspace.module_forward_args)
    side_channel = _rebuild_module_side_channels(journal)
    module_enter_addresses = _module_enter_addresses(
        list(journal.module_prep_events),
        list(journal.module_enter_events),
        list(journal.module_exit_events),
    )
    op_events = _op_events_in_raw_order(list(journal.op_events))
    output_version_lane = journal.output_version_events
    input_layer_labels: list[str] = []
    for event in op_events:
        if (
            event.layer_type == "input"
            and event.label_raw not in inputs.input_layers_initial
            and event.label_raw not in input_layer_labels
        ):
            input_layer_labels.append(event.label_raw)
    op_event_labels = {event.label_raw for event in op_events}
    children_by_parent = _children_by_parent(
        journal.buffer_write_events, op_events, op_event_labels
    )
    buffer_addresses_by_label = _buffer_addresses_by_label(
        inputs.buffer_initial_values, journal.buffer_write_events, op_events
    )
    equivalence_class_map, equivalent_ops_by_label = _equivalent_ops_by_label(
        journal.buffer_write_events, op_events, buffer_addresses_by_label
    )
    # W3 (weightsfree memo defect L6): the aliased-buffer payload backfill
    # hydrates every unwritten registered buffer's value into its source
    # node — a VALUE payload a structure-only capture must never retain
    # (declared geometry only; the leak made every buffer-holding
    # structure-only artifact fail its own load at M-C2).
    if inputs.scatter_options.get("structure_only", False):
        buffer_alias_snapshots: dict[str, torch.Tensor] = {}
    else:
        buffer_alias_snapshots = _buffer_alias_snapshots_by_address(
            journal.buffer_write_events, inputs.source_model_ref
        )
    module_input_fields = _module_input_fields(
        list(journal.module_enter_events),
        module_enter_addresses,
        module_workspace_forward_args,
    )
    op_events_by_label = {event.label_raw: event for event in op_events}
    input_io_roles = _input_io_roles(inputs.raw_graph_workspace, op_events)
    # Count ops per innermost module call so a single-op (atomic) leaf module can
    # be told apart from a multi-op one. The innermost module of an op is the last
    # frame of its capture-time module stack.
    innermost_module_op_counts: Counter[tuple[str, int]] = Counter(
        (event.module_stack[-1].address, event.module_stack[-1].call_index)
        for event in op_events
        if event.module_stack
    )
    module_output_fields = _module_output_fields(
        list(journal.module_exit_events),
        op_events_by_label,
        _module_role_hints_by_address(list(journal.module_prep_events)),
        innermost_module_op_counts,
    )
    buffer_write_fields = _buffer_write_fields(
        journal.buffer_write_events,
        op_event_labels,
        structure_only=bool(inputs.scatter_options.get("structure_only", False)),
    )
    output_versions = _output_versions_by_parent(output_version_lane)
    registered_buffer_names = set(inputs.buffer_initial_values or {})

    local_registrations: dict[str, Any] = {}
    for event in op_events:
        if isinstance(event, OpRecord):
            record, extras = event, _EMPTY_INGEST_EXTRAS
        else:
            record, extras = op_record_from_event(event)
        fields_dict: dict[str, object] = dict.fromkeys(LAYER_PASS_LOG_FIELD_ORDER)
        fields_dict.update(scatter_record_to_cells(record, extras, inputs.owning_trace))
        _apply_join_cells(
            fields_dict,
            record,
            extras,
            journal,
            op_event_labels,
            children_by_parent.get(event.label_raw, []),
            equivalent_ops_by_label.get(event.label_raw, {event.label_raw}),
            buffer_addresses_by_label.get(event.label_raw),
            buffer_alias_snapshots,
            module_input_fields.get(event.label_raw, _empty_module_input_fields()),
            module_output_fields.get(event.label_raw, _empty_module_output_fields()),
            buffer_write_fields.get(event.label_raw, {}),
            input_io_roles.get(event.label_raw),
            output_versions.get(event.label_raw, {}),
            op_events_by_label,
            inputs.param_logs,
            registered_buffer_names,
        )
        start = time.perf_counter()
        op_log = materialize_log_from_fields(fields_dict, op_store)
        inputs.timing_sink("object_construction:op", time.perf_counter() - start)
        local_registrations[event.label_raw] = op_log
    _drop_missing_buffer_sources(local_registrations)
    return Step0Result(
        raw_log_registrations=tuple(local_registrations.items()),
        input_layer_labels=tuple(input_layer_labels),
        equivalence_class_map=equivalence_class_map,
        module_side_channel=side_channel,
    )


def _op_events_in_raw_order(op_events: list[OpEvent]) -> list[OpEvent]:
    """Return operation events sorted by their reserved raw index.

    Parameters
    ----------
    op_events
        Operation events in backend append order.

    Returns
    -------
    list[OpEvent]
        Events in graph raw-index order.
    """

    return sorted(op_events, key=lambda event: event.raw_index)


def _drop_missing_buffer_sources(local_registrations: dict[str, Any]) -> None:
    """Clear buffer source labels that are absent from raw materialized logs.

    X2 resolution: consumes ingest's LOCAL registration map, never a
    workspace read-back of state ingest itself created mid-loop.

    Parameters
    ----------
    local_registrations
        Ingest's local (label_raw -> op_log) registration map.

    Returns
    -------
    None
        Mutates buffer logs in place.
    """

    raw_labels = set(local_registrations)
    for op_log in local_registrations.values():
        buffer_source = getattr(op_log, "buffer_source", None)
        if buffer_source is not None and buffer_source not in raw_labels:
            op_log.buffer_source = None


def _output_versions_by_parent(
    output_version_events: tuple[Any, ...],
) -> dict[str, dict[str, object]]:
    """Return output-version payloads grouped by parent raw label.

    Parameters
    ----------
    output_version_events
        The journal's output-version lane.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping from parent raw label to child raw label to pre-child payload.
    """

    grouped: dict[str, dict[str, object]] = defaultdict(dict)
    for event in output_version_events:
        grouped[event.parent_raw_label][event.child_raw_label] = event.payload
    return grouped


def _apply_join_cells(
    fields_dict: dict[str, object],
    event: Any,
    extras: Any,
    journal: JournalView,
    op_event_labels: set[str],
    children: list[str],
    equivalent_ops: set[str],
    buffer_address: str | None,
    buffer_alias_snapshots: dict[str, torch.Tensor],
    module_input_fields: dict[str, object],
    module_output_fields: dict[str, object],
    buffer_write_fields: dict[str, object],
    input_io_role: str | None,
    output_versions_by_child: dict[str, object],
    op_events_by_label: Mapping[str, Any],
    param_logs_registry: Any,
    registered_buffer_names: set[str],
) -> None:
    """Apply the JOIN-class cells for one record onto its scatter output.

    The record-sourced cells (CORE/FACET/EXTRAS/DEFAULT classes) come from
    the generated scatter; everything here joins a non-record input lane
    (children edges, grad-fn index, param registry, buffer address pool,
    module and buffer-write siblings, output versions, io roles, payload
    disposition) exactly as the deleted ``_fields_from_event`` literal did.

    Parameters
    ----------
    fields_dict
        Field mapping pre-seeded with the scatter's record-sourced cells.
    event
        Journal op record (either shape; read through the strict protocol).
    extras
        ``IngestExtras`` for the record (adapter-carried compat channels).
    journal
        The enumerated read-only journal lane view.
    op_event_labels
        Raw labels present in the materialized event stream.
    children
        Raw child labels joined from later operation parent edges.
    equivalent_ops
        Raw labels with the same event equivalence class.
    buffer_address
        Buffer address joined from buffer initial-value state, when applicable.
    buffer_alias_snapshots
        Refreshed snapshots for indirectly updated aliased buffers.
    module_input_fields
        Per-op module-entry sibling fields.
    module_output_fields
        Per-op module-exit sibling fields.
    buffer_write_fields
        Per-op buffer-write sibling fields.
    input_io_role
        Reconstructed input role for source input events.
    output_versions_by_child
        Child-specific output snapshots keyed by child label.
    op_events_by_label
        Operation events keyed by raw label.
    param_logs_registry
        The trace's ``param_logs`` registry (declared input).
    registered_buffer_names
        Names of the model's declared registered-buffer universe.

    Returns
    -------
    None
        Mutates ``fields_dict`` in place.
    """

    output = event.output
    tensor = output.tensor
    transformed = output.transformed_tensor
    semantics = event.backend_semantics
    if semantics is not None and semantics.unknown_aliasing:
        raise ValueError(
            "Cannot materialize capture events for "
            f"{event.label_raw}: backend aliasing semantics are unknown. "
            "Replay and validation require an explicit alias contract."
        )
    params = tuple(event.params)
    param_logs = _param_logs_for_event(param_logs_registry, params)
    resolved_param_addresses = {log.address for log in param_logs}
    resolved_params = tuple(param for param in params if param.address in resolved_param_addresses)
    resolved_parent_params = [
        parent_param
        for param, parent_param in zip(params, event.parent_params, strict=False)
        if param.address in resolved_param_addresses
    ]
    parent_param_ops = {param.barcode: event.pass_index for param in resolved_params}
    param_shapes = [param.shape for param in resolved_params]
    parent_params = resolved_parent_params
    grad_fn_handle = journal.grad_fn_handles_by_label_raw.get(event.label_raw)
    # Index-first single ownership; the adapter-carried compat channel keeps
    # detached legacy streams whole until the OpEvent field dies in S15.
    grad_handle = grad_fn_handle if grad_fn_handle is not None else extras.grad_fn_handle
    # r83 C2: resolve the DISPLAY address and the backend-native address
    # SEPARATELY.
    #
    # ``buffer_address`` (from ``_buffer_addresses_by_label``) is REGISTERED-only
    # -- it names an entry of the model's declared state universe -- and is the
    # backend-native address. The display ``address`` additionally falls back to
    # the address the CAPTURE recorded for this event
    # (``_recorded_buffer_address``): a plain-attribute or list-element buffer is
    # not registered state, but it still has a meaningful display address (its
    # attribute path, e.g. ``child.q``). Using the recorded address here is what
    # makes a NESTED plain-attribute buffer show its full path rather than the
    # containing module's address that ``_event_address`` would otherwise return,
    # and it is authoritative -- the value/shape heuristics never reach it (see
    # ``_buffer_addresses_by_label``). ``backend_address`` is decoupled to the
    # registered-only value and applied post-construction in
    # ``materialize_log_from_fields`` (None for a non-registered buffer, as a
    # tensor outside the declared state universe should have).
    resolved_address = buffer_address or _recorded_buffer_address(event) or _event_address(event)
    tensor_payload = _event_tensor_payload(event, resolved_address, buffer_alias_snapshots)
    fields_dict.update(
        {
            "out": tensor_payload,
            "shape": _shape_from_payload(tensor_payload, tensor.shape),
            "dtype": _dtype_from_payload(tensor_payload, tensor.dtype),
            "activation_memory": _memory_from_payload(tensor_payload, tensor.memory),
            "has_out_variations": bool(output_versions_by_child or output.child_versions),
            "out_versions_by_child": {
                **dict(output.child_versions),
                **output_versions_by_child,
            },
            "grad_fn_object_id": None if grad_handle is None else id(grad_handle),
            "grad_fn_handle": grad_handle,
            "parent_params": parent_params,
            "_param_barcodes": [param.barcode for param in resolved_params],
            "parent_param_ops": parent_param_ops,
            "_param_logs": param_logs,
            "param_shapes": param_shapes,
            "num_params": sum(prod(shape) for shape in param_shapes if shape is not None),
            "num_params_trainable": sum(log.num_params for log in param_logs if log.is_trainable),
            "num_params_frozen": sum(log.num_params for log in param_logs if not log.is_trainable),
            "param_memory": sum(int(log.param_memory) for log in param_logs),
            "equivalent_ops": equivalent_ops,
            "children": children,
            "has_children": bool(children),
            "_edge_uses": _edge_use_records_from_event(event, op_events_by_label),
            "io_role": _event_io_role(event, input_io_role),
            "address": resolved_address,
            "buffer_source": _event_buffer_source(event, op_event_labels),
        }
    )
    if isinstance(tensor.blob_ref, PortableBlobRef):
        fields_dict["_pending_blob_id"] = tensor.blob_ref.blob_id
        if tensor.payload is None:
            fields_dict["out"] = None
    if transformed is not None and isinstance(transformed.blob_ref, PortableBlobRef):
        fields_dict["_pending_transformed_out_blob_id"] = transformed.blob_ref.blob_id
        if transformed.payload is None:
            fields_dict["transformed_out"] = None
    fields_dict.update(buffer_write_fields)
    _restamp_swapped_reference_out(fields_dict, buffer_write_fields)
    fields_dict.update(module_input_fields)
    fields_dict.update(module_output_fields)
    # r83 C2 / r85 (free FINDING-1): the registered-only backend-native address,
    # decoupled from the display ``address`` above -- but ONLY for a BUFFER-addressed
    # node. ``materialize_log_from_fields`` pops this extra key before construction
    # and applies it afterwards, so ``Op.__init__``'s ``backend_address <- address``
    # default cannot recouple them. A registered buffer keeps its registered address;
    # a plain-attribute / list-element buffer (recorded display address but NOT
    # declared state) is pinned to ``None``, which is the whole C2 intent.
    #
    # A regular op / activation / input node is NOT a buffer, so it gets NO override
    # and keeps the ``Op.__init__`` default (``backend_address == address``). The
    # original r83 code set this key UNCONDITIONALLY, which forced EVERY op node to
    # ``None`` -- contradicting its own commit message ("an op/input ... is
    # unaffected") and leaving the model-output layer (built by copying the last op
    # through a path that keeps the default) reporting the coupled address while the
    # aliasing op node reported ``None``. Gating on buffer-addressedness makes the
    # commit's claim TRUE, restores intra-trace consistency (op node and output layer
    # for the same module agree), and matches every other backend, where an op node
    # carries a backend-native handle (``jaxpr:``/``uop:``) that the output layer
    # inherits. Verdicts and replay fingerprints are byte-unchanged: no runnable-load
    # / param_source / resolver consumer reads an op node's ``backend_address`` (the
    # only reader is the string-when-present metadata invariant, satisfied either way).
    recorded_buffer_address = _recorded_buffer_address(event)
    if buffer_address is not None:
        fields_dict["_materialized_backend_address"] = buffer_address
    elif recorded_buffer_address is not None:
        # A buffer-addressed node whose registered address was NOT claimed from
        # the one-claim pool in ``_buffer_addresses_by_label``. Two disjoint kinds
        # land here:
        #   1. the WRITE-SIDE node of a REGISTERED buffer whose READ-SIDE sibling
        #      already claimed the address (any in-place ``add_``/``copy_``/
        #      ``mul_``, every BatchNorm/InstanceNorm running-stat) -- the read and
        #      write node share one equivalence class, so the write node's recorded
        #      address is that SAME registered buffer's name; and
        #   2. a plain-attribute / list-element buffer that is not declared state at
        #      all (its recorded address is its own display path, e.g. ``child.q``).
        #
        # r87: kind 1 must report the SAME registered ``backend_address`` as its
        # read-side node -- r85's Option-a rule is "a registered buffer keeps its
        # registered address" and intra-trace consistency was the fix's stated goal;
        # previously the write node fell through to ``buffer_address`` (None). The
        # discriminator is REGISTERED-ness -- does the recorded address genuinely
        # name a buffer in the model's declared-state universe
        # (``trace._buffer_initial_values``, the SAME set the runnable preflight
        # refuses against, ``_io/runnable.py``) -- NOT pool-membership. This is
        # never a heuristic borrow: the address used is the node's OWN recorded
        # address, and only when it is registered.
        #
        # kind 2 stays None -- the whole r83 C2 intent (a non-registered tensor is
        # outside the declared state universe, so it has a display address but no
        # backend-native one). A foreign tensor whose recorded display address is
        # not in the registered universe therefore never receives a registered
        # ``backend_address``, so C2's silent wrong-bind cannot be re-opened; and
        # because ``backend_address`` is verdict-neutral (no runnable-load /
        # param_source / resolver consumer reads it -- only the string-when-present
        # metadata invariant does), this clause moves no verdict and no replay
        # fingerprint. The display ``address`` and the save binding that DRIVE the
        # C2 refusal are resolved above and are untouched here.
        if recorded_buffer_address in registered_buffer_names:
            fields_dict["_materialized_backend_address"] = recorded_buffer_address
        else:
            fields_dict["_materialized_backend_address"] = None


def _children_by_parent(
    buffer_write_events: tuple[Any, ...],
    op_events: list[OpEvent],
    op_event_labels: set[str],
) -> dict[str, list[str]]:
    """Join raw child labels from operation parent edges.

    Parameters
    ----------
    buffer_write_events
        The journal's buffer-write lane.
    op_events
        Ordered operation events for the capture.
    op_event_labels
        Raw labels present in the materialized event stream.

    Returns
    -------
    dict[str, list[str]]
        Child labels keyed by raw parent label.
    """

    children: dict[str, list[str]] = defaultdict(list)
    for event in op_events:
        for edge in event.parents:
            if event.label_raw not in children[edge.parent_label_raw]:
                children[edge.parent_label_raw].append(event.label_raw)
    for event in buffer_write_events:
        producer_label_raw = getattr(event, "producer_label_raw", None)
        version_label_raw = getattr(event, "version_label_raw", None)
        if producer_label_raw not in op_event_labels or version_label_raw not in op_event_labels:
            continue
        if version_label_raw not in children[producer_label_raw]:
            children[producer_label_raw].append(version_label_raw)
    return children


def _edge_use_records_from_event(
    event: OpEvent,
    op_events_by_label: Mapping[str, OpEvent],
) -> list[object]:
    """Return normalized edge-use records for a materialized event.

    Parameters
    ----------
    event
        Operation event whose edge uses should be persisted.
    op_events_by_label
        Operation events keyed by raw label, used to annotate parent function
        call ids when available.

    Returns
    -------
    list[object]
        Edge-use records in the stable dataclass shape used by replay and
        intervention metadata.
    """

    records: list[object] = []
    for record in event._edge_uses:
        if isinstance(record, EdgeUseRecord):
            records.append(record)
            continue
        normalized = _normalize_edge_use_record(record, event, op_events_by_label)
        if normalized is not None:
            records.append(normalized)
    return records


def _normalize_edge_use_record(
    record: object,
    event: OpEvent,
    op_events_by_label: Mapping[str, OpEvent],
) -> EdgeUseRecord | None:
    """Normalize a legacy tuple edge-use record.

    Parameters
    ----------
    record
        Edge-use record emitted by a backend.
    event
        Child operation event receiving the materialized record.
    op_events_by_label
        Operation events keyed by raw label.

    Returns
    -------
    EdgeUseRecord | None
        Normalized edge-use record, or ``None`` when ``record`` is not a known
        edge-use shape.
    """

    if not isinstance(record, tuple) or len(record) < 3:
        return None
    parent_label, arg_position, edge_use = record[:3]
    if not isinstance(parent_label, str) or not isinstance(edge_use, str):
        return None
    parent_event = op_events_by_label.get(parent_label)
    # F10 (lovely item 8): the storage verdict populates from the closed
    # function-semantics table when this event carries a function name;
    # keyword edges and unknown functions stay the honest "unknown".
    from ..intervention.edge_semantics import classify_view_or_copy

    arg_path = _edge_arg_position_to_path(arg_position)
    event_func_name = getattr(getattr(event, "function", None), "func_name", None)
    if edge_use == "kwarg":
        storage_relation = "unknown"
    else:
        storage_relation = classify_view_or_copy(event_func_name, arg_path)
    return EdgeUseRecord(
        parent_label=parent_label,
        child_label=event.label_raw,
        arg_kind="keyword" if edge_use == "kwarg" else "positional",
        arg_path=cast(Any, arg_path),
        view_or_copy=cast(Any, storage_relation),
        parent_func_call_id=None if parent_event is None else parent_event.func_call_id,
        child_func_call_id=event.func_call_id or 0,
        edge_use=edge_use,
    )


def _edge_arg_position_to_path(arg_position: object) -> tuple[object, ...]:
    """Return a stable tuple path for a backend edge argument position.

    Parameters
    ----------
    arg_position
        Backend-provided argument position.

    Returns
    -------
    tuple[object, ...]
        Tuple path equivalent.
    """

    if isinstance(arg_position, tuple):
        return arg_position
    if arg_position is None:
        return ()
    return (arg_position,)


def _equivalent_ops_by_label(
    buffer_write_events: tuple[Any, ...],
    op_events: list[OpEvent],
    buffer_addresses_by_label: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Group raw labels by event equivalence class.

    Parameters
    ----------
    buffer_write_events
        The journal's buffer-write lane.
    op_events
        Ordered operation events for the capture.
    buffer_addresses_by_label
        Resolved registered-buffer addresses keyed by raw buffer-source label.

    Returns
    -------
    tuple[dict[str, set[str]], dict[str, set[str]]]
        The populated equivalence-class map (``Step0Result`` payload; shares
        its set objects with the membership view) and the per-label
        membership view.
    """

    groups: dict[str, set[str]] = defaultdict(set)
    buffer_address_by_label = dict(buffer_addresses_by_label)
    for event in buffer_write_events:
        label_raw = getattr(event, "version_label_raw", None)
        address = getattr(event, "address", None)
        if isinstance(label_raw, str) and isinstance(address, str):
            buffer_address_by_label[label_raw] = address
    for event in op_events:
        buffer_address = buffer_address_by_label.get(event.label_raw)
        key = (
            f"buffer:{buffer_address}"
            if event.kind == "source" and event.layer_type == "buffer" and buffer_address
            else _base_equivalence_class(event) or event.label_raw
        )
        groups[key].add(event.label_raw)
    return dict(groups), {label_raw: group for group in groups.values() for label_raw in group}


def _base_equivalence_class(event: OpEvent) -> str | None:
    """Return the legacy pre-module-suffix equivalence-class key for an event.

    Parameters
    ----------
    event
        Operation event whose ``equivalence_class`` includes module suffixes.

    Returns
    -------
    str | None
        Equivalence class with the module-address suffix removed when possible.
    """

    equivalence_class = event.equivalence_class
    if event.is_transform and event.transform_kind is not None:
        # Legacy events smuggle fn_code_location inside transform_config;
        # decomposed records carry it as a first-class transform-facet field.
        code_location = event.transform_config.get("fn_code_location")
        if code_location is None:
            transform_facet = getattr(event, "transform", None)
            code_location = getattr(transform_facet, "fn_code_location", None)
        fingerprint = code_location if code_location is not None else event.transform_fn_qualname
        return f"{event.transform_kind}:{fingerprint}"
    if equivalence_class is None or not event.modules:
        return equivalence_class
    module_suffix = "_".join(address for address, _call_index in event.modules)
    if module_suffix and equivalence_class.endswith(module_suffix):
        return equivalence_class[: -len(module_suffix)]
    return equivalence_class


def _event_tensor_payload(
    event: OpEvent,
    resolved_address: str | None,
    buffer_alias_snapshots: dict[str, torch.Tensor],
) -> object:
    """Return event payload, replacing stale aliased-buffer reads when needed.

    Parameters
    ----------
    event
        Operation event being materialized.
    resolved_address
        Resolved buffer address, when this is a buffer source.
    buffer_alias_snapshots
        Refreshed snapshots for indirectly updated aliased buffers.

    Returns
    -------
    object
        Payload to store in the raw Op fields.
    """

    payload = event.output.tensor.payload
    if (
        event.kind == "source"
        and event.layer_type == "buffer"
        and resolved_address in buffer_alias_snapshots
        and payload is None
    ):
        return buffer_alias_snapshots[resolved_address]
    return payload


def _shape_from_payload(
    payload: object, fallback: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    """Return tensor shape from payload or fallback metadata.

    Parameters
    ----------
    payload
        Candidate tensor payload.
    fallback
        Event metadata fallback.

    Returns
    -------
    tuple[int, ...] | None
        Tensor shape.
    """

    return tuple(payload.shape) if isinstance(payload, torch.Tensor) else fallback


def _dtype_from_payload(payload: object, fallback: object | None) -> object | None:
    """Return tensor dtype from payload or fallback metadata.

    Parameters
    ----------
    payload
        Candidate tensor payload.
    fallback
        Event metadata fallback.

    Returns
    -------
    object | None
        Runtime dtype when available.
    """

    return payload.dtype if isinstance(payload, torch.Tensor) else _resolve_dtype(fallback)


def _memory_from_payload(payload: object, fallback: int | None) -> int | None:
    """Return tensor memory from payload or fallback metadata.

    Parameters
    ----------
    payload
        Candidate tensor payload.
    fallback
        Event metadata fallback.

    Returns
    -------
    int | None
        Tensor memory in bytes.
    """

    if isinstance(payload, torch.Tensor):
        return int(payload.nelement() * payload.element_size())
    return fallback


def _rebuild_module_side_channels(journal: JournalView) -> _ModuleSideChannel:
    """Rebuild module postprocess side channels from journal module lanes.

    Builds into a LOCAL bundle; the orchestrator applies it onto the trace's
    ``ModuleCaptureWorkspace`` via ``Step0Result.module_side_channel``.

    Parameters
    ----------
    journal
        Journal lane view containing prep, enter, exit, and pre-hook records.

    Returns
    -------
    _ModuleSideChannel
        Freshly rebuilt module side-channel state.
    """

    side_channel = _ModuleSideChannel(
        module_build_data=_init_module_hierarchy_data(),
        module_metadata={},
        module_forward_args={},
    )
    for prep_event in journal.module_prep_events:
        _apply_module_prep_event(side_channel, prep_event)
    module_enter_addresses = _module_enter_addresses(
        list(journal.module_prep_events),
        list(journal.module_enter_events),
        list(journal.module_exit_events),
    )
    for enter_event in journal.module_enter_events:
        _apply_module_enter_event(
            side_channel, enter_event, module_enter_addresses[id(enter_event)]
        )
    for exit_event in journal.module_exit_events:
        _apply_module_exit_event(side_channel, exit_event)
    provenance = side_channel.module_build_data.setdefault("module_pre_hook_provenance", {})
    for pre_hook_event in journal.pre_hook_events:
        if pre_hook_event.call_index is None:
            continue
        call_label = f"{pre_hook_event.address}:{pre_hook_event.call_index}"
        provenance[call_label] = (
            pre_hook_event.inputs_before_pre_hooks,
            pre_hook_event.inputs_after_pre_hooks,
            pre_hook_event.effects,
        )
    if not journal.module_enter_events:
        _fill_module_call_stacks_from_op_events(side_channel, list(journal.op_events))
    return side_channel


def _fill_module_call_stacks_from_op_events(
    side_channel: _ModuleSideChannel, op_events: list[OpEvent]
) -> None:
    """Rebuild ``module_call_stacks`` from op module stacks (predicate path).

    The exhaustive torch capture records each module call's ancestor chain into
    ``mbd["module_call_stacks"]`` at prep time (``backends/torch/model_prep.py``:
    ``call_stack = [f"{f.address}:{f.pass_index}" for f in stack[:-1]]``, where the
    undecorated root ``self`` never appears in the stack). ``_apply_module_enter_event``
    replays that into the same dict during materialize. The predicate/fastlog
    capture path emits no typed ``ModuleEnterEvent``s, so that dict stays empty and
    every reconstructed ``ModuleCall.module_call_stack`` is ``[]`` -- which fails the
    ``module_hierarchy`` invariant's call-tree-link check (e.g. "ModuleCall
    'block.0:1' module_call_stack=[] does not start with ['block:1']").

    Each predicate ``OpEvent`` carries its full capture-time ``module_stack``
    (``tuple[ModuleFrame, ...]`` of ``(address, call_index)`` from root ``self`` down
    to the innermost module). For a frame at position ``i`` the module call's
    ancestor stack is ``module_stack[:i]`` with the reserved root ``self`` dropped --
    reproducing exactly the exhaustive ``stack[:-1]`` value (which also excludes the
    never-pushed root). Fill only missing entries so this stays a faithful
    reconstruction, never an override of any authoritative enter-event value.
    """

    stacks = side_channel.module_build_data["module_call_stacks"]
    for event in op_events:
        module_stack = event.module_stack
        for index, frame in enumerate(module_stack):
            if frame.address == "self":
                continue
            call_label = f"{frame.address}:{frame.call_index}"
            if call_label in stacks:
                continue
            stacks[call_label] = [
                f"{ancestor.address}:{ancestor.call_index}"
                for ancestor in module_stack[:index]
                if ancestor.address != "self"
            ]


def _apply_module_prep_event(side_channel: _ModuleSideChannel, event: ModulePrepEvent) -> None:
    """Apply one module prep event to transient module metadata.

    Parameters
    ----------
    side_channel
        Local module side-channel bundle being rebuilt.
    event
        Prep-time module metadata event.

    Returns
    -------
    None
        Mutates module metadata and module type maps.
    """

    side_channel.module_metadata[event.address] = {
        "cls": None,
        "class_name": event.class_name,
        "class_qualname": event.cls_qualname,
        "class_source_file": event.class_source_file,
        "class_source_line": event.class_source_line,
        "init_source_file": event.init_source_file,
        "init_source_line": event.init_source_line,
        "forward_source_file": event.forward_source_file,
        "forward_source_line": event.forward_source_line,
        "class_docstring": event.class_docstring,
        "init_signature": event.init_signature,
        "init_docstring": event.init_docstring,
        "forward_signature": event.forward_signature,
        "forward_docstring": event.forward_docstring,
        "address_children": list(event.address_children),
        "all_addresses": list(event.all_addresses),
        "training": event.training_at_prep,
        "forward_pre_hooks": _list_or_empty(event.forward_pre_hooks),
        "forward_hooks": _list_or_empty(event.forward_hooks),
        "backward_pre_hooks": _list_or_empty(event.backward_pre_hooks),
        "backward_hooks": _list_or_empty(event.backward_hooks),
        "full_backward_pre_hooks": _list_or_empty(event.full_backward_pre_hooks),
        "full_backward_hooks": _list_or_empty(event.full_backward_hooks),
        "custom_attributes": dict(event.custom_attributes),
        "custom_methods": list(event.custom_methods),
    }
    if event.address != "self":
        side_channel.module_build_data["module_types"][event.address] = event.module_type_str


def _list_or_empty(value: object | None) -> list[Any]:
    """Return ``value`` as a list, or an empty list for ``None``.

    Parameters
    ----------
    value
        Optional iterable hook summary payload.

    Returns
    -------
    list[Any]
        List copy when possible, otherwise an empty list.
    """

    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _apply_module_enter_event(
    side_channel: _ModuleSideChannel, event: ModuleEnterEvent, address: str
) -> None:
    """Apply one module-enter event to transient module side channels.

    Parameters
    ----------
    side_channel
        Local module side-channel bundle being rebuilt.
    event
        Module-enter event.
    address
        Resolved module address for this event.

    Returns
    -------
    None
        Mutates module build data and forward-argument maps.
    """

    mbd = side_channel.module_build_data
    call_label = f"{address}:{event.call_index}"
    mbd["module_training_modes"][address] = event.training
    mbd["module_forward_start_times"][call_label] = event.forward_start_time
    mbd["module_code_contexts"][call_label] = list(event.code_context)
    mbd["module_call_stacks"][call_label] = list(event.call_stack)
    mbd.setdefault("module_forward_templates", {})[call_label] = (
        event.forward_args_template,
        event.forward_kwargs_template,
    )
    mbd["module_layer_argnames"][call_label].extend(list(event.layer_argnames))
    side_channel.module_forward_args[(address, event.call_index)] = (
        event.forward_args,
        event.forward_kwargs,
    )


def _apply_module_exit_event(side_channel: _ModuleSideChannel, event: ModuleExitEvent) -> None:
    """Apply one module-exit event to transient module side channels.

    Parameters
    ----------
    side_channel
        Local module side-channel bundle being rebuilt.
    event
        Module-exit event.

    Returns
    -------
    None
        Mutates module build data.
    """

    mbd = side_channel.module_build_data
    mbd["module_forward_durations"][event.call_label] = event.forward_duration
    if event.output_structure is not None:
        mbd["module_output_structures"][event.call_label] = event.output_structure
    if event.output_paths:
        mbd.setdefault("module_output_paths", {})[event.call_label] = tuple(event.output_paths)


def _module_enter_addresses(
    prep_events: list[ModulePrepEvent],
    enter_events: list[ModuleEnterEvent],
    exit_events: list[ModuleExitEvent],
) -> dict[int, str]:
    """Resolve module-enter addresses, filling missing addresses from exits.

    Parameters
    ----------
    prep_events
        Module-prep events defining valid module addresses.
    enter_events
        Module-enter events in capture order.
    exit_events
        Module-exit events in capture order.

    Returns
    -------
    dict[int, str]
        Resolved module address keyed by ``id(enter_event)``.
    """

    valid_addresses = {event.address for event in prep_events}
    explicit_enter_keys = {
        (enter_event.address, enter_event.call_index)
        for enter_event in enter_events
        if enter_event.address in valid_addresses
    }
    unmatched_exit_by_index: dict[int, deque[ModuleExitEvent]] = defaultdict(deque)
    for exit_event in exit_events:
        if (exit_event.address, exit_event.call_index) not in explicit_enter_keys:
            unmatched_exit_by_index[exit_event.call_index].append(exit_event)

    resolved: dict[int, str] = {}
    for enter_event in enter_events:
        call_label_address = enter_event.call_label.rsplit(":", 1)[0]
        if call_label_address in valid_addresses:
            resolved[id(enter_event)] = call_label_address
            continue
        if enter_event.address in valid_addresses:
            resolved[id(enter_event)] = enter_event.address
            continue
        candidates = unmatched_exit_by_index.get(enter_event.call_index)
        if candidates:
            resolved[id(enter_event)] = candidates.popleft().address
        else:
            resolved[id(enter_event)] = str(enter_event.address)
    return resolved


def _module_input_fields(
    enter_events: list[ModuleEnterEvent],
    enter_addresses: dict[int, str],
    module_forward_args: dict[Any, Any],
) -> dict[str, dict[str, object]]:
    """Fold module-enter events into per-op sibling fields.

    Parameters
    ----------
    enter_events
        Module enter events in capture order.
    enter_addresses
        Resolved module addresses keyed by event identity.
    module_forward_args
        Live module forward args/kwargs side channel keyed by module call.

    Returns
    -------
    dict[str, dict[str, object]]
        Module-entry Op fields keyed by raw op label.
    """

    by_label: dict[str, dict[str, object]] = {}
    for event in enter_events:
        address = enter_addresses[id(event)]
        call_tuple = (address, event.call_index)
        call_label = f"{address}:{event.call_index}"
        input_labels = list(event.input_labels)
        if not input_labels:
            forward_args, forward_kwargs = module_forward_args.get(
                (address, event.call_index),
                (event.forward_args, event.forward_kwargs),
            )
            input_tensors = get_vars_of_type_from_obj(
                [forward_args, forward_kwargs],
                torch.Tensor,
                [torch.nn.Parameter],
                search_depth=5,
            )
            input_labels = [
                label_raw
                for tensor in input_tensors
                if (label_raw := get_tensor_label(tensor)) is not None
            ]
        # B3R7-R05-1: this enter lane must NOT touch ``module_call_stack``.
        # It used to append the entered call's address here, which stamped the
        # FED call's stack onto ops that never ran inside it (the model input,
        # top-level producers) -- the fed-call fact is ``input_to_module_calls``;
        # containment comes from the op's own modules facet at ingest.
        for label_raw in input_labels:
            fields = by_label.setdefault(label_raw, _empty_module_input_fields())
            cast(list[Any], fields["input_to_module_calls"]).append(call_tuple)
        for label_raw, arg_key in event.layer_argnames:
            fields = by_label.setdefault(label_raw, _empty_module_input_fields())
            cast(defaultdict[str, list[Any]], fields["module_entry_arg_keys"])[call_label].append(
                arg_key
            )
    return by_label


def _module_output_fields(
    exit_events: list[ModuleExitEvent],
    op_events_by_label: dict[str, OpEvent],
    role_hints_by_address: dict[str, object],
    innermost_module_op_counts: Counter[tuple[str, int]],
) -> dict[str, dict[str, object]]:
    """Fold module-exit events into per-op sibling fields.

    Parameters
    ----------
    exit_events
        Module exit events in capture order.
    op_events_by_label
        Operation events keyed by raw label.
    role_hints_by_address
        Semantic output-role hints keyed by module address.
    innermost_module_op_counts
        Number of ops whose innermost module is each ``(address, call_index)``.
        A module call containing exactly one op is an atomic (single-op) leaf.

    Returns
    -------
    dict[str, dict[str, object]]
        Module-exit Op fields keyed by raw op label.
    """

    by_label: dict[str, dict[str, object]] = {}
    for event in exit_events:
        call_tuple = (event.address, event.call_index)
        role_hints = role_hints_by_address.get(event.address)
        for output_index, label_raw in enumerate(event.output_tensor_labels_raw):
            fields = by_label.setdefault(label_raw, _empty_module_output_fields())
            fields["is_module_output"] = True
            # NOTE: do NOT mark ``intervention_replaced`` here from
            # ``has_user_forward_hooks``. That is a PROXY, not proof: a module can
            # carry a purely observational forward hook (returns ``None``, never
            # substitutes) and this overlay runs during plain postprocess with no
            # evidence a substitution happened. Forcing it True would (a) mislabel
            # a plain-capture module output as a user intervention (disarming the
            # tripwire, see project CLAUDE.md "Validation Integrity") and (b) even
            # for a genuine replacement mark the PRE-replacement tensor. Genuine
            # replacements are marked with proof on the actual replacement op by
            # ``_make_user_forward_hook_wrapper`` and flow in via the authoritative
            # ``event.intervention_replaced`` base field above.
            cast(list[Any], fields["output_of_modules"]).append(event.address)
            cast(list[Any], fields["output_of_module_calls"]).append(call_tuple)
            if output_index < len(event.output_names):
                fields["multi_output_name"] = event.output_names[output_index]
                continue
            op_event = op_events_by_label.get(label_raw)
            if op_event is not None:
                fields["multi_output_name"] = _multi_output_name_from_event(
                    op_event,
                    role_hints,
                    output_index,
                )
        # B3R7-R05-1: the exit lane no longer overwrites ``module_call_stack``
        # (its containment value now comes uniformly from the op's own modules
        # facet at ingest); this loop keeps only the atomic-leaf detection.
        for label_raw, stack, _is_atomic, _atomic_call in event.per_output_atomic:
            fields = by_label.setdefault(label_raw, _empty_module_output_fields())
            # A module output op is an atomic (single-op leaf) module exit when its
            # innermost module call contains exactly one op. This is computed from
            # the finalized op-to-module map rather than at capture time so that
            # sibling/side ops (e.g. a BatchNorm ``num_batches_tracked`` bump) are
            # counted and multi-op leaves are not mis-flagged as atomic.
            if not stack:
                continue
            innermost = stack[-1]
            innermost_call = (innermost.address, innermost.call_index)
            if innermost_module_op_counts.get(innermost_call, 0) == 1:
                fields["is_atomic_module"] = True
                fields["atomic_module_call"] = innermost_call
    return by_label


def _restamp_swapped_reference_out(
    fields_dict: dict[str, object], buffer_write_fields: Mapping[str, object]
) -> None:
    """Re-stamp a reference-mode version for a buffer version node's swapped payload.

    At capture a buffer version node is logged like any source tensor, so in
    ``save_mode="reference"`` its payload is the live buffer and its
    ``saved_out_version`` stamp is that buffer's version counter.
    :func:`_buffer_write_fields` then replaces the payload with the write
    journal's private copy of the written value, a different tensor with its
    own counter. Left alone, the stamp describes a tensor the op no longer
    holds, and the reference tripwire
    (``op._validate_reference_out_not_mutated``) misreads the mismatch as a
    mutation (``saved _version=2, current _version=0`` on every eval
    BatchNorm). The new stamp is the copy's counter at materialization, so the
    tripwire still fires if the stored copy is written in place later.

    Parameters
    ----------
    fields_dict
        Op fields being materialized; ``annotations`` is replaced, never mutated.
    buffer_write_fields
        The buffer-write sibling fields just applied to ``fields_dict``.
    """

    out = buffer_write_fields.get("out")
    annotations = fields_dict.get("annotations")
    if not isinstance(out, torch.Tensor) or not isinstance(annotations, Mapping):
        return
    if annotations.get("save_mode") != "reference":
        return
    fields_dict["annotations"] = {
        **annotations,
        "saved_out_version": tensor_version_or_none(out),
    }


def _buffer_write_fields(
    buffer_write_events: tuple[Any, ...],
    op_event_labels: set[str],
    *,
    structure_only: bool = False,
) -> dict[str, dict[str, object]]:
    """Fold buffer write events into per-op sibling fields.

    Parameters
    ----------
    buffer_write_events
        The journal's buffer-write lane.
    op_event_labels
        Raw labels present in the materialized event stream.
    structure_only
        W3 (weightsfree memo defect L6): under the structure-only contract a
        buffer version node keeps its DECLARED geometry (shape / dtype /
        byte estimate) but never a value payload — retaining the write
        journal's copied value made every buffer-holding structure-only
        artifact save-then-fail-to-load at the M-C2 coherence gate, and BN
        running stats are training-derived state a weights-free artifact
        must not carry.

    Returns
    -------
    dict[str, dict[str, object]]
        Buffer-write fields keyed by raw version-node label.
    """

    by_label: dict[str, dict[str, object]] = {}
    for event in buffer_write_events:
        label_raw = getattr(event, "version_label_raw", None)
        if label_raw is None:
            continue
        producer_label_raw = getattr(event, "producer_label_raw", None)
        fields: dict[str, object] = {
            "address": getattr(event, "address", None),
            "buffer_write_kind": getattr(event, "kind", None),
            "buffer_value_changed": getattr(event, "value_changed", None),
            "buffer_source": producer_label_raw if producer_label_raw in op_event_labels else None,
            "buffer_source_func_name": getattr(event, "source_func_name", None),
        }
        value = getattr(event, "value", None)
        if isinstance(value, torch.Tensor):
            geometry: dict[str, object] = {
                "shape": tuple(value.shape),
                "dtype": value.dtype,
                "activation_memory": value.nelement() * value.element_size(),
            }
            if not structure_only:
                geometry.update({"out": value, "has_saved_activation": True})
            fields.update(geometry)
        if producer_label_raw in op_event_labels:
            fields["parents"] = [producer_label_raw]
            fields["parent_arg_positions"] = {"args": {0: producer_label_raw}, "kwargs": {}}
        by_label[label_raw] = fields
    return by_label


def _empty_module_input_fields() -> dict[str, object]:
    """Return empty per-op module-entry fields.

    Returns
    -------
    dict[str, object]
        Fresh mutable field defaults.
    """

    return {
        "input_to_module_calls": [],
        "module_entry_arg_keys": defaultdict(list),
    }


def _empty_module_output_fields() -> dict[str, object]:
    """Return empty per-op module-exit fields.

    Returns
    -------
    dict[str, object]
        Fresh mutable field defaults.
    """

    return {
        "output_of_modules": [],
        "output_of_module_calls": [],
        "is_module_output": False,
        "is_atomic_module": False,
        "atomic_module_call": None,
    }


def _module_role_hints_by_address(
    prep_events: list[ModulePrepEvent],
) -> dict[str, object]:
    """Build semantic output-role hints keyed by module address.

    Parameters
    ----------
    prep_events
        Module prep events with importable class qualnames.

    Returns
    -------
    dict[str, object]
        Role-hint mappings keyed by module address.
    """

    hints_by_address: dict[str, object] = {}
    for event in prep_events:
        module_class = _resolve_module_class(event.cls_qualname)
        if module_class is None:
            # `torch.nn` also exposes non-class submodules (e.g. `nn.init`,
            # `nn.functional`, `nn.utils`). A user module class can legitimately
            # share one of those names (e.g. a class literally named `init`),
            # so this fallback must reject non-class matches instead of handing
            # them to `issubclass()` below.
            candidate = getattr(nn, event.class_name, None)
            module_class = candidate if isinstance(candidate, type) else None
        hints = role_hints_for_module_class(module_class)
        if hints is not None:
            hints_by_address[event.address] = hints
    return hints_by_address


def _resolve_module_class(class_qualname: str | None) -> type[Any] | None:
    """Resolve a module class qualname to a runtime class when importable.

    Parameters
    ----------
    class_qualname
        Fully qualified class name from a module prep event.

    Returns
    -------
    type[Any] | None
        Resolved class, or ``None`` when unavailable.
    """

    if class_qualname is None or "." not in class_qualname:
        return None
    module_name, _, qualname = class_qualname.rpartition(".")
    # SECURITY (not attacker-controlled): ``class_qualname`` here originates from a
    # ``ModulePrepEvent`` built during LIVE model preparation from the real in-process
    # ``nn.Module`` subclass (``model_prep._module_type`` -> ``{cls.__module__}.{cls.__qualname__}``).
    # ``materialize_from_events`` is invoked ONLY from live-capture backends and partial-capture
    # recovery -- NEVER from ``tl.load()`` -- so this resolver never sees a deserialized,
    # bundle-controlled string. It is therefore intentionally NOT the default-deny
    # ``sys.modules``-only resolver used for the portable, attacker-influenceable
    # ``ContainerSpec`` (see ``torchlens.ir.container.resolve_container_type``): the module
    # of a live model class is already imported, and hardening this hot path would only add
    # cost without closing an attack surface. If a future load path ever re-materializes a
    # deserialized event stream, this MUST be routed through a default-deny resolver instead.
    try:
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (AttributeError, ImportError):
        return None
    return obj if isinstance(obj, type) else None


def _multi_output_name_from_event(
    op_event: OpEvent,
    role_hints: object | None,
    fallback_index: int | None,
) -> str | None:
    """Return the semantic multi-output name for an event output.

    Parameters
    ----------
    op_event
        Output operation event.
    role_hints
        Optional semantic role hints for the owning module.
    fallback_index
        Output order from the module-exit event.

    Returns
    -------
    str | None
        Semantic or fallback multi-output name.
    """

    container_path = op_event.output.container_path
    output_index = op_event.output.multi_output_index
    if output_index is None:
        output_index = fallback_index
    if not container_path and role_hints is not None and output_index is not None:
        container_path = (output_index,)
    return multi_output_role_from_path(
        container_path,
        output_index,
        hints=role_hints,  # type: ignore[arg-type]
    )


def _input_io_roles(raw_graph_workspace: Any, op_events: list[OpEvent]) -> dict[str, str]:
    """Reconstruct legacy input role strings by source-input event order.

    Parameters
    ----------
    raw_graph_workspace
        The raw-graph workspace read handle (X1: ``input_tensor_addresses``).
    op_events
        Ordered operation events from the capture.

    Returns
    -------
    dict[str, str]
        Input role strings keyed by raw input label.
    """

    input_events = [event for event in op_events if event.layer_type == "input"]
    input_addresses = raw_graph_workspace.input_tensor_addresses
    if isinstance(input_addresses, list) and len(input_addresses) == len(input_events):
        return {
            event.label_raw: address
            for event, address in zip(input_events, input_addresses, strict=True)
            if isinstance(address, str)
        }
    if len(input_events) == 1:
        return {input_events[0].label_raw: "input.input"}
    return {event.label_raw: f"input.{index}" for index, event in enumerate(input_events)}


def _param_logs_for_event(param_logs_registry: Any, params: tuple[object, ...]) -> list[Any]:
    """Resolve event parameter refs to existing Trace ``Param`` logs.

    Parameters
    ----------
    param_logs_registry
        The trace's ``param_logs`` registry (declared input).
    params
        Parameter refs carried by an operation event.

    Returns
    -------
    list[object]
        Param logs in event parameter order when resolvable.
    """

    param_logs: list[Any] = []
    for param in params:
        address = getattr(param, "address", None)
        if address is not None and address in param_logs_registry:
            param_logs.append(param_logs_registry[address])
    return param_logs


def _event_address(event: OpEvent) -> str | None:
    """Return the raw address field for source-like events.

    Parameters
    ----------
    event
        Operation event being materialized.

    Returns
    -------
    str | None
        Buffer/input address when carried by module context, otherwise ``None``.
    """

    buffer_address = get_buffer_address(event.output.tensor.payload)
    if buffer_address is not None:
        return buffer_address
    record_context = getattr(event, "record_context", None)
    input_output_address = getattr(record_context, "input_output_address", None)
    if isinstance(input_output_address, str):
        return input_output_address
    if event.module_stack:
        return event.module_stack[-1].address
    return None


def _event_buffer_source(event: OpEvent, op_event_labels: set[str]) -> str | None:
    """Return the promoted source label for a buffer event.

    Parameters
    ----------
    event
        Operation event being materialized.
    op_event_labels
        Raw labels present in the materialized event stream.

    Returns
    -------
    str | None
        Raw producer label for a promoted buffer, when tensor metadata carries it.
    """

    tensor_meta = get_tensor_meta(event.output.tensor.payload)
    if tensor_meta is None or tensor_meta.buffer_source not in op_event_labels:
        return None
    return tensor_meta.buffer_source


def _event_io_role(event: OpEvent, input_io_role: str | None) -> str | None:
    """Return the raw input/output role for source input events.

    Parameters
    ----------
    event
        Operation event being materialized.
    input_io_role
        Reconstructed role for a source input event.

    Returns
    -------
    str | None
        Source input role when derivable, otherwise ``None``.
    """

    if event.layer_type == "input":
        return input_io_role
    return None


def _resolve_dtype(dtype: object | None) -> object | None:
    """Convert serialized torch dtype names back to dtype objects.

    Parameters
    ----------
    dtype
        Dtype object or serialized dtype name from an event ref.

    Returns
    -------
    object | None
        Runtime dtype when resolvable, otherwise the original value.
    """

    if dtype is None or not isinstance(dtype, str):
        return dtype
    if dtype.startswith("torch."):
        dtype_name = dtype.split(".", 1)[1]
        # r47 secD_1: ``torch_attr`` reads ``torch.__dict__`` (no lazy ``torch.__getattr__``);
        # fall back to the original string when the name is not a real top-level torch symbol.
        return torch_attr(dtype_name) or dtype
    return dtype
