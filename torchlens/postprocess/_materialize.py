"""Step 0 of postprocess: materialize raw Trace state from capture events."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import importlib
from math import prod
from typing import TYPE_CHECKING, Any, cast

from torch import nn
import torch
from torchlens._io import BlobRef as PortableBlobRef
from torchlens.ir import CaptureEvents
from torchlens.ir.events import (
    ModuleEnterEvent,
    ModuleExitEvent,
    ModuleFrame,
    ModulePrepEvent,
    OpEvent,
)

from ..backends.torch._tl import get_buffer_address, get_tensor_label, get_tensor_meta
from ..constants import LAYER_PASS_LOG_FIELD_ORDER
from ..data_classes._module_role_hints import (
    multi_output_role_from_path,
    role_hints_for_module_class,
)
from ..data_classes.trace import _init_module_hierarchy_data
from ..utils import get_vars_of_type_from_obj, safe_copy
from ..utils.display import _timed_phase

if TYPE_CHECKING:
    from torchlens.data_classes.trace import Trace
    from torchlens.data_classes.op import Op


def materialize_log_from_fields(fields_dict: dict[str, object]) -> "Op":
    """Construct the live log object for one captured operation.

    Parameters
    ----------
    fields_dict
        Raw field mapping populated by the backend hot path.

    Returns
    -------
    Op
        Materialized operation or buffer log.
    """

    from torchlens.data_classes.op import Op

    pending_blob_ids = _pop_pending_blob_ids(fields_dict)
    op_log = Op(fields_dict)  # type: ignore[arg-type]
    for field_name, blob_id in pending_blob_ids.items():
        setattr(op_log, field_name, blob_id)
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


def _annotations_from_event(event: OpEvent) -> dict[str, object]:
    """Return annotations preserved on a materialized event."""

    raw_annotations = event.transform_config.get("_tl_annotations")
    return dict(raw_annotations) if isinstance(raw_annotations, Mapping) else {}


def register_materialized_event(
    trace: "Trace",
    event: OpEvent,
    op_log: "Op",
) -> None:
    """Append an event and expose its live log to in-flight hooks.

    Parameters
    ----------
    trace
        Active trace receiving the event.
    event
        Operation event emitted for the new log.
    op_log
        Ignored legacy parameter retained for call-site compatibility.

    Returns
    -------
    None
        Mutates ``trace.capture_events`` and the raw capture indexes.
    """

    events = getattr(trace, "capture_events", None)
    if events is None:
        events = CaptureEvents()
        trace.capture_events = events
    events.append(event)


def materialize_from_events(trace: "Trace", events: CaptureEvents) -> None:
    """Materialize capture events into raw build-state logs.

    Parameters
    ----------
    trace
        Trace whose transient build state was populated during capture.
    events
        Mutable event accumulator owned by the active capture session.

    Returns
    -------
    None
        Populates raw trace lookup structures and destructively consumes event lists.
    """

    live_module_forward_args = dict(getattr(trace, "_module_forward_args", {}))
    _rebuild_module_side_channels(trace, events)
    module_enter_addresses = _module_enter_addresses(
        events.module_prep_events,
        events.module_enter_events,
        events.module_exit_events,
    )
    op_event_labels = {event.label_raw for event in events.op_events}
    children_by_parent = _children_by_parent(trace, events.op_events, op_event_labels)
    buffer_addresses_by_label = _buffer_addresses_by_label(trace, events.op_events)
    equivalent_ops_by_label = _equivalent_ops_by_label(
        trace,
        events.op_events,
        buffer_addresses_by_label,
    )
    buffer_alias_snapshots = _buffer_alias_snapshots_by_address(trace)
    module_input_fields = _module_input_fields(
        events.module_enter_events,
        module_enter_addresses,
        live_module_forward_args,
    )
    input_io_roles = _input_io_roles(trace, events.op_events)
    # Count ops per innermost module call so a single-op (atomic) leaf module can
    # be told apart from a multi-op one. The innermost module of an op is the last
    # frame of its capture-time module stack.
    innermost_module_op_counts: Counter[tuple[str, int]] = Counter(
        (event.module_stack[-1].address, event.module_stack[-1].call_index)
        for event in events.op_events
        if event.module_stack
    )
    module_output_fields = _module_output_fields(
        events.module_exit_events,
        {event.label_raw: event for event in events.op_events},
        _module_role_hints_by_address(events.module_prep_events),
        innermost_module_op_counts,
    )
    buffer_write_fields = _buffer_write_fields(trace, op_event_labels)
    output_versions = _output_versions_by_parent(events)

    for event in events.op_events:
        fields_dict = _fields_from_event(
            trace,
            event,
            op_event_labels,
            children_by_parent.get(event.label_raw, []),
            equivalent_ops_by_label.get(event.label_raw, {event.label_raw}),
            buffer_addresses_by_label.get(event.label_raw),
            buffer_alias_snapshots,
            module_input_fields.get(event.label_raw, _empty_module_input_fields()),
            module_output_fields.get(event.label_raw, _empty_module_output_fields()),
            buffer_write_fields.get(event.label_raw, {}),
            events.grad_fn_handles_by_label_raw.get(event.label_raw),
            input_io_roles.get(event.label_raw),
            output_versions.get(event.label_raw, {}),
        )
        with _timed_phase(trace, "object_construction:op"):
            op_log = materialize_log_from_fields(fields_dict)
        _register_raw_log(trace, event, op_log)
    _drop_missing_buffer_sources(trace)

    events.op_events.clear()
    events.module_events.clear()
    events.module_prep_events.clear()
    events.module_enter_events.clear()
    events.module_exit_events.clear()
    events.conditional_events.clear()
    events.output_version_events.clear()
    events.live_by_raw_label.clear()
    events.op_event_by_label_raw.clear()
    events.live_index.clear()
    events.grad_fn_handles_by_label_raw.clear()


def _drop_missing_buffer_sources(trace: "Trace") -> None:
    """Clear buffer source labels that are absent from raw materialized logs.

    Parameters
    ----------
    trace
        Trace with Step-0 raw logs populated.

    Returns
    -------
    None
        Mutates buffer logs in place.
    """

    raw_labels = set(trace._raw_layer_dict)
    for op_log in trace._raw_layer_dict.values():
        buffer_source = getattr(op_log, "buffer_source", None)
        if buffer_source is not None and buffer_source not in raw_labels:
            op_log.buffer_source = None


def _output_versions_by_parent(events: CaptureEvents) -> dict[str, dict[str, object]]:
    """Return output-version payloads grouped by parent raw label.

    Parameters
    ----------
    events
        Capture event buffer containing sibling output-version events.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping from parent raw label to child raw label to pre-child payload.
    """

    grouped: dict[str, dict[str, object]] = defaultdict(dict)
    for event in events.output_version_events:
        grouped[event.parent_raw_label][event.child_raw_label] = event.payload
    return grouped


def _fields_from_event(
    trace: "Trace",
    event: OpEvent,
    op_event_labels: set[str],
    children: list[str],
    equivalent_ops: set[str],
    buffer_address: str | None,
    buffer_alias_snapshots: dict[str, torch.Tensor],
    module_input_fields: dict[str, object],
    module_output_fields: dict[str, object],
    buffer_write_fields: dict[str, object],
    grad_fn_handle: object | None,
    input_io_role: str | None,
    output_versions_by_child: dict[str, object],
) -> dict[str, object]:
    """Build a complete raw ``Op`` field dictionary from one operation event.

    Parameters
    ----------
    trace
        Trace being postprocessed.
    event
        Operation event to materialize.
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
    grad_fn_handle
        Live autograd handle side-table entry, when present.
    input_io_role
        Reconstructed input role for source input events.

    Returns
    -------
    dict[str, object]
        Complete pre-postprocess field mapping accepted by ``Op``.
    """

    output = event.output
    tensor = output.tensor
    transformed = output.transformed_tensor
    function = event.function
    semantics = event.backend_semantics
    if semantics.unknown_aliasing:
        raise ValueError(
            "Cannot materialize capture events for "
            f"{event.label_raw}: backend aliasing semantics are unknown. "
            "Replay and validation require an explicit alias contract."
        )
    templates = event.templates
    params = tuple(event.params)
    param_logs = _param_logs_for_event(trace, params)
    parent_param_ops = {param.barcode: event.pass_index for param in params}
    param_shapes = [param.shape for param in params]
    parent_params = list(event.parent_params)
    grad_handle = grad_fn_handle if grad_fn_handle is not None else event.grad_fn_handle
    module = event.modules[-1] if event.modules else None
    resolved_address = buffer_address or _event_address(event)
    tensor_payload = _event_tensor_payload(event, resolved_address, buffer_alias_snapshots)
    fields_dict: dict[str, object] = {field_name: None for field_name in LAYER_PASS_LOG_FIELD_ORDER}
    fields_dict.update(
        {
            "_label_raw": event.label_raw,
            "_layer_label_raw": event.layer_label_raw,
            "step_index": event.step_index,
            "raw_index": event.raw_index,
            "ordinal_index": -1,
            "source_trace": event.source_trace or trace,
            "_tracing_finished": event.tracing_finished,
            "_construction_done": event.construction_done,
            "label": None,
            "label_short": None,
            "layer_label": None,
            "layer_label_short": None,
            "type": event.layer_type,
            "type_index": event.type_index,
            "pass_index": event.pass_index,
            "num_passes": 1,
            "lookup_keys": [],
            "out": tensor_payload,
            "has_saved_activation": output.has_saved_activation,
            "output_device": output.output_device,
            "activation_transform": output.activation_transform,
            "annotations": _annotations_from_event(event),
            "interventions": [
                result.fire_record
                for result in event.fire_results
                if result.fire_record is not None
            ],
            "intervention_replaced": event.intervention_replaced,
            "detach_saved_activations": output.detach_saved_activations,
            "has_saved_args": False if templates is None else templates.has_saved_args,
            "saved_args": None if templates is None else templates.saved_args,
            "saved_kwargs": None if templates is None else templates.saved_kwargs,
            "args_template": None if templates is None else templates.args_template,
            "kwargs_template": None if templates is None else templates.kwargs_template,
            "input_ops": None,
            "input_activations": None,
            "input_shapes": None,
            "input_dtypes": None,
            "input_memory": None,
            "num_inputs": None,
            "shape": _shape_from_payload(tensor_payload, tensor.shape),
            "transformed_out_shape": None if transformed is None else transformed.shape,
            "dtype": _dtype_from_payload(tensor_payload, tensor.dtype),
            "transformed_out_dtype": None
            if transformed is None
            else _resolve_dtype(transformed.dtype),
            "activation_memory": _memory_from_payload(tensor_payload, tensor.memory),
            "transformed_activation_memory": None if transformed is None else transformed.memory,
            "visualizer_path": output.visualizer_path,
            "bytes_delta_at_call": semantics.bytes_delta_at_call,
            "bytes_peak_at_call": semantics.bytes_peak_at_call,
            "transformed_out": None if transformed is None else transformed.payload,
            "autograd_memory": semantics.autograd_memory,
            "num_autograd_tensors": semantics.num_autograd_tensors,
            "has_out_variations": bool(output_versions_by_child or output.child_versions),
            "out_versions_by_child": {
                **dict(output.child_versions),
                **output_versions_by_child,
            },
            "grad": None,
            "transformed_grad": None,
            "save_grads": event.policy.save_grad,
            "has_grad": False,
            "grad_shape": None,
            "transformed_grad_shape": None,
            "grad_dtype": None,
            "transformed_grad_dtype": None,
            "gradient_memory": 0,
            "transformed_gradient_memory": None,
            "func": function.func,
            "func_call_id": function.func_call_id,
            "func_name": function.func_name,
            "func_qualname": function.func_qualname,
            "code_context": list(function.code_context),
            "func_duration": function.func_duration or 0,
            "flops_forward": function.flops_forward or 0,
            "flops_backward": function.flops_backward or 0,
            "func_rng_states": function.func_rng_states,
            "func_autocast_state": function.func_autocast_state,
            "arg_names": tuple(function.arg_names),
            "num_args_total": function.num_args_total,
            "num_pos_args": function.num_pos_args,
            "num_kwargs": function.num_kwargs,
            "non_tensor_pos_args": list(function.non_tensor_pos_args),
            "non_tensor_kwargs": dict(function.non_tensor_kwargs),
            "func_non_tensor_args": list(function.func_non_tensor_args),
            "is_inplace": function.is_inplace,
            "grad_fn_class_name": semantics.grad_fn_class_name,
            "grad_fn_class_qualname": event.grad_fn_class_qualname,
            "grad_fn_object_id": None if grad_handle is None else id(grad_handle),
            "grad_fn_handle": grad_handle,
            "grad_fn": None,
            "in_multi_output": output.in_multi_output,
            "multi_output_index": output.multi_output_index,
            "multi_output_name": None,
            "container_path": tuple(output.container_path),
            "container_spec": output.container_spec,
            "is_transform": event.is_transform,
            "transform_kind": event.transform_kind,
            "transform_chain": tuple(event.transform_chain),
            "transform_config": dict(event.transform_config),
            "transform_fn_name": event.transform_fn_name,
            "transform_fn_qualname": event.transform_fn_qualname,
            "transform_fn_source": event.transform_fn_source,
            "unattributed_tensor_args": tuple(event.unattributed_tensor_args),
            "parent_params": parent_params,
            "_param_barcodes": [param.barcode for param in params],
            "parent_param_ops": parent_param_ops,
            "_param_logs": param_logs,
            "param_shapes": param_shapes,
            "num_params": sum(prod(shape) for shape in param_shapes if shape is not None),
            "num_params_trainable": sum(log.num_params for log in param_logs if log.is_trainable),
            "num_params_frozen": sum(log.num_params for log in param_logs if not log.is_trainable),
            "param_memory": sum(int(log.param_memory) for log in param_logs),
            "equivalence_class": event.equivalence_class,
            "equivalent_ops": equivalent_ops,
            "recurrent_ops": [],
            "parents": [edge.parent_label_raw for edge in event.parents],
            "parent_arg_positions": event.parent_arg_positions,
            "_edge_uses": list(event._edge_uses) if trace.intervention_ready else [],
            "root_ancestors": set(event.root_ancestors),
            "children": children,
            "has_children": bool(children),
            "is_input": event.kind == "source" and event.layer_type == "input",
            "has_input_ancestor": bool(event.input_ancestors),
            "input_ancestors": set(event.input_ancestors),
            "min_distance_from_input": None,
            "max_distance_from_input": None,
            "is_output": False,
            "is_output_parent": event.is_output_parent,
            "is_final_output": False,
            "has_output_descendant": False,
            "output_descendants": set(),
            "is_orphan": False,
            "io_role": _event_io_role(event, input_io_role),
            "min_distance_to_output": None,
            "max_distance_to_output": None,
            "is_buffer": event.kind == "source" and event.layer_type == "buffer",
            "address": resolved_address,
            "buffer_pass": None,
            "buffer_source": _event_buffer_source(event, op_event_labels),
            "buffer_write_kind": None,
            "buffer_value_changed": None,
            "buffer_replay_validated": None,
            "buffer_source_func_name": None,
            "is_internal_source": event.layer_type != "input" and not event.parents,
            "has_internal_source_ancestor": event.has_internal_source_ancestor,
            "internal_source_parents": [],
            "internal_source_ancestors": set(event.internal_source_ancestors),
            "is_internal_sink": False,
            "is_terminal_bool": False,
            "is_terminal_conditional_bool": False,
            "conditional_context_kind": None,
            "conditional_wrapper_kind": None,
            "terminal_conditional_id": None,
            "is_scalar_bool": bool(event.is_scalar_bool),
            "bool_value": event.bool_value,
            "in_conditionals": [],
            "terminal_bool_for": None,
            "is_in_conditional_body": False,
            "conditional_branch_stack": [],
            "conditional_branch_depth": 0,
            "conditional_entry_children": [],
            "conditional_then_children": [],
            "conditional_elif_children": {},
            "conditional_else_children": [],
            "conditional_arm_children": {},
            "module": module,
            "_address_normalized": None,
            "modules": list(event.modules),
            "fx_qualpath": None,
            "fx_call_index": 0,
            "module_call_stack": [],
            "input_to_module_calls": [],
            "module_entry_arg_keys": defaultdict(list),
            "output_of_modules": [],
            "output_of_module_calls": [],
            "is_module_output": False,
            "is_atomic_module": False,
            "atomic_module_call": None,
            "func_config": dict(function.func_config),
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
    fields_dict.update(module_input_fields)
    fields_dict.update(module_output_fields)
    return fields_dict


def _children_by_parent(
    trace: "Trace",
    op_events: list[OpEvent],
    op_event_labels: set[str],
) -> dict[str, list[str]]:
    """Join raw child labels from operation parent edges.

    Parameters
    ----------
    trace
        Trace holding buffer-write events captured by the torch backend.
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
    for event in getattr(trace, "_buffer_write_events", []):
        producer_label_raw = getattr(event, "producer_label_raw", None)
        version_label_raw = getattr(event, "version_label_raw", None)
        if producer_label_raw not in op_event_labels or version_label_raw not in op_event_labels:
            continue
        if version_label_raw not in children[producer_label_raw]:
            children[producer_label_raw].append(version_label_raw)
    return children


def _equivalent_ops_by_label(
    trace: "Trace",
    op_events: list[OpEvent],
    buffer_addresses_by_label: dict[str, str],
) -> dict[str, set[str]]:
    """Group raw labels by event equivalence class.

    Parameters
    ----------
    trace
        Trace whose equivalence-class index is restored for postprocess.
    op_events
        Ordered operation events for the capture.
    buffer_addresses_by_label
        Resolved registered-buffer addresses keyed by raw buffer-source label.

    Returns
    -------
    dict[str, set[str]]
        Equivalent raw labels keyed by each member raw label.
    """

    groups: dict[str, set[str]] = defaultdict(set)
    buffer_address_by_label = dict(buffer_addresses_by_label)
    for event in getattr(trace, "_buffer_write_events", []):
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
    trace.op_equivalence_classes.clear()
    trace.op_equivalence_classes.update(groups)
    return {label_raw: group for group in groups.values() for label_raw in group}


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
        code_location = event.transform_config.get("fn_code_location")
        fingerprint = code_location if code_location is not None else event.transform_fn_qualname
        return f"{event.transform_kind}:{fingerprint}"
    if equivalence_class is None or not event.modules:
        return equivalence_class
    module_suffix = "_".join(address for address, _call_index in event.modules)
    if module_suffix and equivalence_class.endswith(module_suffix):
        return equivalence_class[: -len(module_suffix)]
    return equivalence_class


def _buffer_addresses_by_label(trace: "Trace", op_events: list[OpEvent]) -> dict[str, str]:
    """Join initial registered-buffer addresses to source buffer events.

    Parameters
    ----------
    trace
        Trace carrying initial registered-buffer value snapshots.
    op_events
        Ordered operation events for the capture.

    Returns
    -------
    dict[str, str]
        Buffer addresses keyed by raw buffer source label.
    """

    unmatched_addresses = list(getattr(trace, "_buffer_initial_values", {}).items())
    by_label: dict[str, str] = {}
    source_buffer_events = [
        event for event in op_events if event.kind == "source" and event.layer_type == "buffer"
    ]
    op_events_by_label = {event.label_raw: event for event in op_events}
    for event in op_events:
        if event.function.func_name not in {"batch_norm", "batchnorm"}:
            continue
        module_address = event.modules[-1][0] if event.modules else None
        if module_address is None:
            continue
        args_positions = event.parent_arg_positions.get("args", {})
        for arg_position, buffer_name in ((3, "running_mean"), (4, "running_var")):
            label_raw = args_positions.get(arg_position)
            if isinstance(label_raw, str):
                by_label[label_raw] = f"{module_address}.{buffer_name}"

    for write_event in getattr(trace, "_buffer_write_events", []):
        producer_label_raw = getattr(write_event, "producer_label_raw", None)
        if not isinstance(producer_label_raw, str):
            continue
        producer_event = op_events_by_label.get(producer_label_raw)
        if producer_event is None:
            continue
        address = getattr(write_event, "address", None)
        if not isinstance(address, str) or not address.endswith(".num_batches_tracked"):
            continue
        for edge in producer_event.parents:
            parent_event = op_events_by_label.get(edge.parent_label_raw)
            if (
                parent_event is not None
                and parent_event.kind == "source"
                and parent_event.layer_type == "buffer"
            ):
                by_label.setdefault(edge.parent_label_raw, address)

    used_addresses = set(by_label.values())
    unmatched_addresses = [
        (address, value)
        for address, value in unmatched_addresses
        if address not in used_addresses and not address.endswith(".num_batches_tracked")
    ]
    unmatched_address_names = {address for address, _value in unmatched_addresses}
    for event in source_buffer_events:
        if event.label_raw in by_label:
            continue
        address = _buffer_address_from_equivalence_class(
            event.equivalence_class,
            unmatched_address_names,
        )
        if address is None:
            address = _unique_buffer_address_by_value(
                event.output.tensor.payload, unmatched_addresses
            )
        if address is None:
            address = _unique_buffer_address_by_shape(
                event.output.tensor.shape, unmatched_addresses
            )
        if address is not None:
            by_label[event.label_raw] = address
            unmatched_addresses = [
                (candidate_address, value)
                for candidate_address, value in unmatched_addresses
                if candidate_address != address
            ]
            unmatched_address_names.discard(address)
    return by_label


def _buffer_address_from_equivalence_class(
    equivalence_class: str | None,
    unmatched_addresses: set[str],
) -> str | None:
    """Return a registered-buffer address encoded in a buffer equivalence class.

    Parameters
    ----------
    equivalence_class
        Event equivalence class for a source buffer.
    unmatched_addresses
        Registered buffer addresses that have not yet been assigned.

    Returns
    -------
    str | None
        Matching registered-buffer address, if the equivalence class names one.
    """

    if equivalence_class is None or not equivalence_class.startswith("buffer_"):
        return None
    candidate = equivalence_class.removeprefix("buffer_")
    return candidate if candidate in unmatched_addresses else None


def _buffer_alias_snapshots_by_address(trace: "Trace") -> dict[str, torch.Tensor]:
    """Return refreshed snapshots for indirectly updated aliased buffers.

    Parameters
    ----------
    trace
        Trace with an active buffer-write tracker.

    Returns
    -------
    dict[str, torch.Tensor]
        Final snapshots for buffer addresses that were updated only through an alias.
    """

    directly_written = {
        getattr(event, "address", None) for event in getattr(trace, "_buffer_write_events", [])
    }
    model_ref = getattr(trace, "_source_model_ref", None)
    model = None if model_ref is None else model_ref()
    if model is None or not hasattr(model, "named_buffers"):
        return {}
    snapshots: dict[str, torch.Tensor] = {}
    for address, tensor in model.named_buffers():
        if address in directly_written or not isinstance(tensor, torch.Tensor):
            continue
        snapshots[address] = safe_copy(tensor, detach_tensor=True)
    return snapshots


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


def _unique_buffer_address_by_value(
    payload: object,
    candidates: list[tuple[str, object]],
) -> str | None:
    """Return the unique candidate address whose value matches the payload.

    Parameters
    ----------
    payload
        Event payload value.
    candidates
        Candidate buffer address/value pairs.

    Returns
    -------
    str | None
        Unique matching address, otherwise ``None``.
    """

    matches = [address for address, value in candidates if _tensor_values_match(payload, value)]
    return matches[0] if len(matches) == 1 else None


def _unique_buffer_address_by_shape(
    shape: tuple[int, ...] | None,
    candidates: list[tuple[str, object]],
) -> str | None:
    """Return the unique candidate address whose tensor shape matches.

    Parameters
    ----------
    shape
        Event tensor shape.
    candidates
        Candidate buffer address/value pairs.

    Returns
    -------
    str | None
        Unique matching address, otherwise ``None``.
    """

    if shape is None:
        return None
    matches = [
        address
        for address, value in candidates
        if isinstance(value, torch.Tensor) and tuple(value.shape) == shape
    ]
    return matches[0] if len(matches) == 1 else None


def _tensor_values_match(left: object, right: object) -> bool:
    """Return whether two tensor-like values have identical contents.

    Parameters
    ----------
    left
        First value.
    right
        Second value.

    Returns
    -------
    bool
        True when both values are tensors with equal shape, dtype, and values.
    """

    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    return bool(
        left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    )


def _rebuild_module_side_channels(trace: "Trace", events: CaptureEvents) -> None:
    """Rebuild module postprocess side channels from module events.

    Parameters
    ----------
    trace
        Trace whose transient module state will be reset.
    events
        Capture events containing prep, enter, and exit module records.

    Returns
    -------
    None
        Mutates ``trace._module_build_data``, ``trace._module_metadata``, and
        ``trace._module_forward_args``.
    """

    trace._module_build_data = _init_module_hierarchy_data()
    trace._module_metadata = {}
    trace._module_forward_args = {}
    for prep_event in events.module_prep_events:
        _apply_module_prep_event(trace, prep_event)
    module_enter_addresses = _module_enter_addresses(
        events.module_prep_events,
        events.module_enter_events,
        events.module_exit_events,
    )
    for enter_event in events.module_enter_events:
        _apply_module_enter_event(trace, enter_event, module_enter_addresses[id(enter_event)])
    for exit_event in events.module_exit_events:
        _apply_module_exit_event(trace, exit_event)


def _apply_module_prep_event(trace: "Trace", event: ModulePrepEvent) -> None:
    """Apply one module prep event to transient module metadata.

    Parameters
    ----------
    trace
        Trace whose side-channel state is being rebuilt.
    event
        Prep-time module metadata event.

    Returns
    -------
    None
        Mutates module metadata and module type maps.
    """

    trace._module_metadata[event.address] = {
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
        trace._module_build_data["module_types"][event.address] = event.module_type_str


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


def _apply_module_enter_event(trace: "Trace", event: ModuleEnterEvent, address: str) -> None:
    """Apply one module-enter event to transient module side channels.

    Parameters
    ----------
    trace
        Trace whose side-channel state is being rebuilt.
    event
        Module-enter event.
    address
        Resolved module address for this event.

    Returns
    -------
    None
        Mutates module build data and forward-argument maps.
    """

    mbd = trace._module_build_data
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
    trace._module_forward_args[(address, event.call_index)] = (
        event.forward_args,
        event.forward_kwargs,
    )


def _apply_module_exit_event(trace: "Trace", event: ModuleExitEvent) -> None:
    """Apply one module-exit event to transient module side channels.

    Parameters
    ----------
    trace
        Trace whose side-channel state is being rebuilt.
    event
        Module-exit event.

    Returns
    -------
    None
        Mutates module build data.
    """

    mbd = trace._module_build_data
    mbd["module_forward_durations"][event.call_label] = event.forward_duration
    if event.output_structure is not None:
        mbd["module_output_structures"][event.call_label] = event.output_structure


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
    unmatched_exit_by_index: dict[int, list[ModuleExitEvent]] = defaultdict(list)
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
        candidates = unmatched_exit_by_index.get(enter_event.call_index, [])
        if candidates:
            resolved[id(enter_event)] = candidates.pop(0).address
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
        for label_raw in input_labels:
            fields = by_label.setdefault(label_raw, _empty_module_input_fields())
            cast(list[Any], fields["module_call_stack"]).append(address)
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
    innermost_module_op_counts: "Counter[tuple[str, int]]",
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
            if event.has_user_forward_hooks:
                fields["intervention_replaced"] = True
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
        for label_raw, stack, _is_atomic, _atomic_call in event.per_output_atomic:
            fields = by_label.setdefault(label_raw, _empty_module_output_fields())
            fields["module_call_stack"] = _module_stack_addresses(stack)
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


def _buffer_write_fields(
    trace: "Trace",
    op_event_labels: set[str],
) -> dict[str, dict[str, object]]:
    """Fold buffer write events into per-op sibling fields.

    Parameters
    ----------
    trace
        Trace holding buffer-write events captured by the torch backend.
    op_event_labels
        Raw labels present in the materialized event stream.

    Returns
    -------
    dict[str, dict[str, object]]
        Buffer-write fields keyed by raw version-node label.
    """

    by_label: dict[str, dict[str, object]] = {}
    for event in getattr(trace, "_buffer_write_events", []):
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
            fields.update(
                {
                    "out": value,
                    "has_saved_activation": True,
                    "shape": tuple(value.shape),
                    "dtype": value.dtype,
                    "activation_memory": value.nelement() * value.element_size(),
                }
            )
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
        "module_call_stack": [],
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
            module_class = getattr(nn, event.class_name, None)
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


def _module_stack_addresses(stack: tuple[ModuleFrame, ...]) -> list[str]:
    """Convert module frames to the raw module-call-stack field format.

    Parameters
    ----------
    stack
        Module frames carried by a module-exit event.

    Returns
    -------
    list[str]
        Module addresses in stack order.
    """

    return [frame.address for frame in stack]


def _input_io_roles(trace: "Trace", op_events: list[OpEvent]) -> dict[str, str]:
    """Reconstruct legacy input role strings by source-input event order.

    Parameters
    ----------
    trace
        Trace carrying capture-time input tensor addresses when available.
    op_events
        Ordered operation events from the capture.

    Returns
    -------
    dict[str, str]
        Input role strings keyed by raw input label.
    """

    input_events = [
        event for event in op_events if event.kind == "source" and event.layer_type == "input"
    ]
    input_addresses = getattr(trace, "_input_tensor_addresses", None)
    if isinstance(input_addresses, list) and len(input_addresses) == len(input_events):
        return {
            event.label_raw: address
            for event, address in zip(input_events, input_addresses, strict=True)
            if isinstance(address, str)
        }
    if len(input_events) == 1:
        return {input_events[0].label_raw: "input.input"}
    return {event.label_raw: f"input.{index}" for index, event in enumerate(input_events)}


def _param_logs_for_event(trace: "Trace", params: tuple[object, ...]) -> list[Any]:
    """Resolve event parameter refs to existing Trace ``Param`` logs.

    Parameters
    ----------
    trace
        Trace with populated ``param_logs``.
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
        if address is not None and address in trace.param_logs:
            param_logs.append(trace.param_logs[address])
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

    if event.kind == "source" and event.layer_type == "input":
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
        return getattr(torch, dtype_name, dtype)
    return dtype


def _register_raw_log(trace: "Trace", event: OpEvent, op_log: "Op") -> None:
    """Register one live log in transient raw lookup structures.

    Parameters
    ----------
    trace
        Trace receiving the raw log.
    event
        Operation event corresponding to ``op_log``.
    op_log
        Live operation log to register.

    Returns
    -------
    None
        Mutates ``trace._raw_layer_dict`` and ``trace._raw_layer_labels_list``.
    """

    trace._raw_layer_dict[event.label_raw] = op_log
    trace._raw_layer_labels_list.append(event.label_raw)
