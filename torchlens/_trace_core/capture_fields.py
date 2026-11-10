"""Capture event schema and the stored-field table (pure data).

Every column of the op row store has one row in :data:`FIELD_TABLE` saying how
its value comes to exist:

* ``call``: the value stores an observation taken while the wrapped call runs
  (the live arguments, the output tensor, the frames, the clock). It cannot be
  recovered afterwards, so a capture event has to carry it in the named slot.
* ``derived``: computed from this event's own slots and static tables.
* ``graph``: computed from other events too (a pass over the whole trace).
* ``option``: a per-trace capture setting, the same for every row.
* ``later``: has no forward-capture value; a backward pass, an intervention,
  a save or a user edit writes it.
* ``state``: storage bookkeeping (references, caches, lifecycle flags).

``slots`` names the :data:`EVENT_SLOTS` entries that carry a ``call`` value, or
that a ``derived`` value is computed from. ``settled`` is the last phase that
changes the stored value during a forward capture and its postprocessing:
``emit`` (the row commit at the call), ``forward`` (later in the same forward
pass), ``ingest`` (row adoption at the start of postprocessing) or the name of
a postprocess pipeline step. An empty ``settled`` means no phase writes it.
Representation-only phases (compaction, view freezing) are not counted.

The table is keyed on by field resolvers and by field-level differential
checks; ``tests/test_capture_fields.py`` fails when a store column has no row.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import NamedTuple

#: Slots of one capture event, in record order, with what each one holds.
EVENT_SLOTS: tuple[tuple[str, str], ...] = (
    ("kind", "function call, or an input, buffer or parameter source"),
    ("fid", "id of the function's static plan: callable, names, argument names"),
    ("call", "call ordinal; the outputs of one call share it"),
    ("out_index", "path of this tensor inside the call's output container"),
    ("parents", "node ids of the tensors the call consumed"),
    ("shape", "shape read off the output"),
    ("dtype", "dtype read off the output"),
    ("device", "device read off the output"),
    ("module", "the enclosing module-call record and its entry and exit facts"),
    ("site", "interned call-site record: user frames and source policy facts"),
    ("t0", "clock read before the call"),
    ("dur", "elapsed time of the call"),
    ("args", "argument structure: per slot a node id, a value or a copy"),
    ("facts", "per-call facts gone after the call: autograd node, parameters"),
    ("payload", "the saved activation and its transformed form, when kept"),
    ("flags", "in-place, multi-output and nesting flags"),
    ("extra", "facts only unusual calls have: mutation, buffers, random state"),
)
#: Pseudo-slot for facts held once per trace (model outputs, input roles).
TRACE_SLOT = "trace"
SLOT_NAMES: frozenset[str] = frozenset(name for name, _ in EVENT_SLOTS) | {TRACE_SLOT}

SOURCES: tuple[str, ...] = ("call", "derived", "graph", "option", "later", "state")
FAMILIES: tuple[str, ...] = (
    "labels",
    "function",
    "site",
    "args",
    "tensor",
    "stats",
    "payload",
    "autograd",
    "params",
    "relations",
    "buffers",
    "control",
    "module",
    "options",
    "intervention",
    "state",
)
#: Phases before the postprocess pipeline; every other ``settled`` value is a step name.
FIXED_PHASES: tuple[str, ...] = ("emit", "forward", "ingest")


class FieldRow(NamedTuple):
    """Classification of one op-store column."""

    name: str
    family: str
    source: str
    slots: tuple[str, ...]
    settled: str


class Observation(NamedTuple):
    """A call-time observation that is not itself a store column."""

    name: str
    slot: str
    feeds: tuple[str, ...]


_ROWS: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    ("_label_raw", "labels", "graph", ("fid",), "emit"),
    ("_layer_label_raw", "labels", "graph", (), "group_recurrent_layers"),
    ("step_index", "labels", "graph", (), "log_final_layer_info"),
    ("raw_index", "labels", "graph", ("fid",), "emit"),
    ("ordinal_index", "labels", "graph", (), "build_lookup_keys"),
    ("_tracing_finished", "state", "state", (), "mark_tracing_finished"),
    ("_construction_done", "state", "state", (), "ingest"),
    ("label", "labels", "graph", (), "map_final_labels"),
    ("short_label", "labels", "graph", (), "map_final_labels"),
    ("layer_label", "labels", "graph", (), "map_final_labels"),
    ("short_layer_label", "labels", "graph", (), "map_final_labels"),
    ("type", "function", "derived", ("fid",), "emit"),
    ("type_index", "labels", "graph", ("fid",), "map_final_labels"),
    ("pass_number", "labels", "graph", (), "group_recurrent_layers"),
    ("num_passes", "labels", "graph", (), "group_recurrent_layers"),
    ("lookup_keys", "labels", "graph", (), "build_lookup_keys"),
    ("out", "payload", "call", ("payload",), "ingest"),
    ("has_saved_activation", "payload", "derived", ("payload",), "emit"),
    ("output_device", "options", "option", (), "emit"),
    ("activation_transform", "options", "option", (), "emit"),
    ("annotations", "intervention", "call", ("extra",), "build_layer_logs"),
    ("edits", "intervention", "later", (), "emit"),
    ("is_edited", "intervention", "call", ("extra",), "emit"),
    ("detach_saved_activations", "options", "option", (), "emit"),
    ("has_saved_args", "args", "derived", ("args",), "emit"),
    ("saved_args", "args", "call", ("args",), "emit"),
    ("saved_kwargs", "args", "call", ("args",), "emit"),
    ("args_template", "args", "derived", ("args",), "log_final_layer_info"),
    ("kwargs_template", "args", "derived", ("args",), "log_final_layer_info"),
    ("shape", "tensor", "call", ("shape",), "ingest"),
    ("transformed_out_shape", "payload", "call", ("payload",), "emit"),
    ("dtype", "tensor", "call", ("dtype",), "ingest"),
    ("dtype_ref", "tensor", "derived", ("dtype",), "ingest"),
    ("transformed_out_dtype", "payload", "call", ("payload",), "emit"),
    ("device_ref", "tensor", "call", ("device",), "ingest"),
    ("backend_address", "buffers", "derived", ("extra",), "ingest"),
    ("resolver_status", "intervention", "later", (), "ingest"),
    ("activation_memory", "tensor", "call", ("facts",), "ingest"),
    ("transformed_activation_memory", "payload", "call", ("payload",), "emit"),
    ("visualizer_path", "intervention", "later", (), "emit"),
    ("bytes_delta_at_call", "stats", "call", ("extra",), "emit"),
    ("bytes_peak_at_call", "stats", "call", ("extra",), "emit"),
    ("transformed_out", "payload", "call", ("payload",), "emit"),
    ("autograd_memory", "autograd", "call", ("facts",), "ingest"),
    ("num_autograd_tensors", "autograd", "call", ("facts",), "emit"),
    ("has_out_variations", "payload", "derived", ("extra",), "ingest"),
    ("out_versions_by_child", "payload", "call", ("extra",), "ingest"),
    ("grad", "autograd", "later", (), "emit"),
    ("transformed_grad", "autograd", "later", (), "emit"),
    ("save_grads", "options", "option", (), "emit"),
    ("has_grad", "autograd", "later", (), "emit"),
    ("grad_shape", "autograd", "later", (), "emit"),
    ("transformed_grad_shape", "autograd", "later", (), "emit"),
    ("grad_dtype", "autograd", "later", (), "emit"),
    ("transformed_grad_dtype", "autograd", "later", (), "emit"),
    ("gradient_memory", "autograd", "later", (), "ingest"),
    ("transformed_gradient_memory", "autograd", "later", (), "emit"),
    ("func", "function", "derived", ("fid",), "fix_buffer_layers"),
    ("func_id", "args", "derived", ("fid", "args"), "emit"),
    ("func_call_id", "function", "call", ("call",), "emit"),
    ("func_name", "function", "derived", ("fid",), "fix_buffer_layers"),
    ("func_qualname", "function", "derived", ("fid",), "emit"),
    ("code_context", "site", "call", ("site",), "emit"),
    ("var_names", "site", "derived", ("site",), "populate_source_var_names"),
    ("func_duration", "function", "call", ("dur",), "ingest"),
    ("flops_forward", "stats", "derived", ("fid", "shape", "args", "facts"), "ingest"),
    ("flops_backward", "stats", "derived", ("fid", "shape", "args", "facts"), "ingest"),
    ("func_rng_states", "function", "call", ("extra",), "emit"),
    ("func_autocast_state", "function", "call", ("facts",), "emit"),
    ("arg_names", "function", "derived", ("fid",), "emit"),
    ("num_args_total", "args", "derived", ("args",), "emit"),
    ("num_pos_args", "args", "derived", ("args",), "emit"),
    ("num_kwargs", "args", "derived", ("args",), "emit"),
    ("non_tensor_pos_args", "args", "derived", ("args",), "emit"),
    ("non_tensor_kwargs", "args", "derived", ("args",), "emit"),
    ("func_non_tensor_args", "args", "derived", ("args",), "emit"),
    ("is_inplace", "function", "call", ("flags",), "emit"),
    ("grad_fn_class_name", "autograd", "call", ("facts",), "emit"),
    ("grad_fn_class_qualname", "autograd", "call", ("facts",), "emit"),
    ("grad_fn_object_id", "autograd", "derived", ("facts",), "ingest"),
    ("grad_fn_handle", "autograd", "call", ("facts",), "ingest"),
    ("grad_fn", "autograd", "later", (), "emit"),
    ("in_multi_output", "function", "call", ("flags",), "emit"),
    ("multi_output_index", "function", "call", ("out_index",), "emit"),
    ("multi_output_name", "function", "derived", ("out_index", "facts"), "ingest"),
    ("container_path", "function", "call", ("out_index",), "emit"),
    ("container_spec", "function", "call", ("facts",), "emit"),
    ("is_transform", "function", "derived", ("fid",), "emit"),
    ("transform_kind", "function", "derived", ("fid",), "emit"),
    ("transform_chain", "function", "derived", ("fid",), "emit"),
    ("transform_config", "function", "derived", ("fid",), "emit"),
    ("transform_fn_name", "function", "derived", ("fid",), "emit"),
    ("transform_fn_qualname", "function", "derived", ("fid",), "emit"),
    ("transform_fn_source", "function", "derived", ("fid",), "emit"),
    ("unattributed_tensor_args", "args", "derived", ("args",), "emit"),
    ("dropped_edge_tensor_args", "args", "derived", ("args",), "emit"),
    ("parent_params", "params", "call", ("facts",), "finalize_param_logs"),
    ("_param_barcodes", "params", "call", ("facts",), "ingest"),
    ("parent_param_ops", "params", "graph", (), "ingest"),
    ("_param_logs", "params", "call", ("facts",), "ingest"),
    ("param_shapes", "params", "call", ("facts",), "ingest"),
    ("num_params", "params", "derived", ("facts",), "ingest"),
    ("num_params_trainable", "params", "derived", ("facts",), "ingest"),
    ("num_params_frozen", "params", "derived", ("facts",), "ingest"),
    ("param_memory", "params", "derived", ("facts",), "ingest"),
    (
        "equivalence_class",
        "relations",
        "derived",
        ("fid", "args", "facts", "module", "out_index"),
        "emit",
    ),
    ("equivalent_ops", "relations", "graph", (), "log_final_layer_info"),
    ("recurrent_ops", "relations", "graph", (), "log_final_layer_info"),
    ("site_key", "site", "call", ("site",), "group_recurrent_layers"),
    ("injection_provenance", "intervention", "later", (), "ingest"),
    ("episode_step", "intervention", "later", (), "ingest"),
    ("tl_authored_root", "intervention", "later", (), "ingest"),
    ("parents", "relations", "call", ("parents",), "log_final_layer_info"),
    ("parent_arg_positions", "args", "call", ("args",), "log_final_layer_info"),
    ("_edge_uses", "args", "call", ("args",), "log_final_layer_info"),
    ("edge_substitutions", "intervention", "later", (), "ingest"),
    ("edge_replacement_stamps", "intervention", "later", (), "ingest"),
    ("root_ancestors", "relations", "graph", (), "log_final_layer_info"),
    ("children", "relations", "graph", (), "log_final_layer_info"),
    ("has_children", "relations", "graph", (), "ingest"),
    ("is_input", "relations", "derived", ("kind",), "emit"),
    ("input_was_parameter", "buffers", "call", ("extra",), "emit"),
    ("has_input_ancestor", "relations", "graph", (), "mark_layer_depths"),
    ("input_ancestors", "relations", "graph", (), "log_final_layer_info"),
    ("min_distance_from_input", "relations", "graph", (), "mark_layer_depths"),
    ("max_distance_from_input", "relations", "graph", (), "mark_layer_depths"),
    ("is_output", "relations", "derived", ("trace",), "emit"),
    ("is_output_parent", "relations", "derived", ("trace",), "forward"),
    ("is_final_output", "relations", "derived", ("trace",), "emit"),
    ("has_output_descendant", "relations", "graph", (), "find_output_ancestors"),
    ("output_descendants", "relations", "graph", (), "log_final_layer_info"),
    ("is_orphan", "relations", "graph", (), "emit"),
    ("io_role", "relations", "derived", ("trace",), "ingest"),
    ("min_distance_to_output", "relations", "graph", (), "mark_layer_depths"),
    ("max_distance_to_output", "relations", "graph", (), "mark_layer_depths"),
    ("is_buffer", "relations", "derived", ("kind",), "emit"),
    ("is_constant", "relations", "graph", (), "build_layer_logs"),
    ("address", "buffers", "call", ("extra",), "ingest"),
    ("buffer_pass", "buffers", "graph", (), "fix_buffer_layers"),
    ("buffer_source", "buffers", "call", ("extra",), "log_final_layer_info"),
    ("buffer_write_kind", "buffers", "call", ("extra",), "ingest"),
    ("buffer_value_changed", "buffers", "call", ("extra",), "ingest"),
    ("buffer_replay_validated", "buffers", "graph", (), "fix_buffer_layers"),
    ("buffer_source_func_name", "buffers", "call", ("extra",), "ingest"),
    ("is_internal_source", "relations", "derived", ("parents",), "ingest"),
    ("has_internal_source_ancestor", "relations", "graph", (), "emit"),
    ("internal_source_parents", "relations", "graph", (), "log_final_layer_info"),
    ("internal_source_ancestors", "relations", "graph", (), "log_final_layer_info"),
    ("is_internal_sink", "relations", "graph", (), "remove_orphan_nodes"),
    ("is_terminal_bool", "control", "graph", (), "remove_orphan_nodes"),
    ("is_terminal_conditional_bool", "control", "graph", (), "mark_conditional_branches"),
    ("conditional_context_kind", "control", "graph", (), "mark_conditional_branches"),
    ("conditional_wrapper_kind", "control", "graph", (), "emit"),
    ("terminal_conditional_id", "control", "graph", (), "mark_conditional_branches"),
    ("is_scalar_bool", "tensor", "derived", ("shape", "dtype"), "emit"),
    ("bool_value", "tensor", "call", ("facts",), "emit"),
    ("in_conditionals", "control", "graph", (), "build_layer_logs"),
    ("terminal_bool_for", "control", "graph", (), "build_layer_logs"),
    ("conditional_branch_stack", "control", "graph", (), "mark_conditional_branches"),
    ("conditional_branch_depth", "control", "graph", (), "mark_conditional_branches"),
    ("conditional_entry_children", "control", "graph", (), "log_final_layer_info"),
    ("conditional_then_children", "control", "graph", (), "log_final_layer_info"),
    ("conditional_elif_children", "control", "graph", (), "emit"),
    ("conditional_else_children", "control", "graph", (), "emit"),
    ("conditional_arm_children", "control", "graph", (), "log_final_layer_info"),
    ("module", "module", "call", ("module",), "build_lookup_keys"),
    ("_address_normalized", "module", "derived", ("module",), "hash_graph_shape"),
    ("modules", "module", "call", ("module",), "build_lookup_keys"),
    ("fx_qualpath", "module", "graph", (), "log_final_layer_info"),
    ("fx_call_index", "module", "graph", (), "log_final_layer_info"),
    ("module_call_stack", "module", "derived", ("module",), "build_lookup_keys"),
    ("input_to_module_calls", "module", "call", ("module",), "build_lookup_keys"),
    ("module_entry_arg_keys", "module", "call", ("module",), "ingest"),
    ("output_of_modules", "module", "call", ("module",), "ingest"),
    ("output_of_module_calls", "module", "call", ("module",), "build_lookup_keys"),
    ("is_module_output", "module", "call", ("module",), "ingest"),
    ("is_atomic_module", "module", "call", ("module",), "ingest"),
    ("atomic_module_call", "module", "call", ("module",), "log_final_layer_info"),
    ("func_config", "args", "derived", ("fid", "args", "facts"), "emit"),
    ("_source_trace_ref", "state", "state", (), "emit"),
    ("out_ref", "state", "state", (), "ingest"),
    ("grad_ref", "state", "state", (), "ingest"),
    ("_pending_blob_id", "state", "state", (), "ingest"),
    ("_pending_transformed_out_blob_id", "state", "state", (), "ingest"),
    ("_pending_grad_blob_id", "state", "state", (), "ingest"),
    ("_pending_transformed_grad_blob_id", "state", "state", (), "ingest"),
    ("_grad_records", "autograd", "later", (), "ingest"),
    ("_facets_cache", "state", "state", (), ""),
    ("_receptive_field_cache", "state", "state", (), ""),
    ("_projective_field_cache", "state", "state", (), ""),
    ("_arg_expressions_cache", "state", "state", (), "add_output_layers"),
    ("_is_in_conditional_body", "control", "graph", (), "mark_conditional_branches"),
)

FIELD_TABLE: Mapping[str, FieldRow] = MappingProxyType({row[0]: FieldRow(*row) for row in _ROWS})

#: Observations the capture takes at the call that feed columns without being one.
UNSTORED_OBSERVATIONS: tuple[Observation, ...] = (
    Observation("event_kind", "kind", ("is_input", "is_buffer")),
    Observation("same_object_returned", "flags", ("is_inplace",)),
    Observation("bottom_level", "flags", ()),
    Observation("tensor_version", "extra", ("is_inplace", "out_versions_by_child")),
    Observation(
        "arg_slot_state",
        "args",
        ("_edge_uses", "unattributed_tensor_args", "dropped_edge_tensor_args"),
    ),
    Observation("backend_semantics", "extra", ()),
    Observation("fire_results", "extra", ("is_edited", "edits")),
    Observation("source_trace", "extra", ()),
    Observation("materialized_backend_address", "extra", ("backend_address",)),
    Observation(
        "model_outputs",
        TRACE_SLOT,
        ("is_output", "is_final_output", "is_output_parent", "io_role"),
    ),
)


def unclassified_fields(names: Iterable[str]) -> tuple[str, ...]:
    """Return the names that have no row in :data:`FIELD_TABLE`.

    Parameters
    ----------
    names:
        Store column names to look up.

    Returns
    -------
    tuple[str, ...]
        The names without a row, in the order given.
    """
    return tuple(name for name in names if name not in FIELD_TABLE)


def fields_with_source(source: str) -> tuple[str, ...]:
    """Return the classified field names whose source is ``source``.

    Parameters
    ----------
    source:
        One of :data:`SOURCES`.

    Returns
    -------
    tuple[str, ...]
        Field names in table order.
    """
    return tuple(row.name for row in FIELD_TABLE.values() if row.source == source)


def call_time_fields() -> dict[str, tuple[str, ...]]:
    """Group the call-time fields by the event slot that carries them.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Slot name to the ``call`` fields stored from it, slots in record order.
    """
    grouped: dict[str, list[str]] = {name: [] for name, _ in EVENT_SLOTS}
    for row in FIELD_TABLE.values():
        if row.source == "call":
            for slot in row.slots:
                grouped[slot].append(row.name)
    return {slot: tuple(names) for slot, names in grouped.items() if names}
