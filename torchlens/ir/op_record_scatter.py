"""Record -> store-cell scatter and the cell-source classification (v1).

``CELL_SOURCES`` classifies every op store cell (``_OP_SLOT_NAMES`` plus the
extra-key channels) by its step-0 ingest source:

* ``CORE`` — from ``OpCore`` (including ``core.output`` nested reads and
  core-derived flags such as ``is_input``).
* ``FACET:<name>`` — from one typed facet (absent facets read the checked-in
  defaults table).
* ``JOIN:<lane>`` — from a non-record ``IngestInputs`` lane (children edges,
  grad-fn index, param logs, buffer address pool, module enter/exit and
  buffer-write siblings, output versions, io-role assignment, payload
  disposition against buffer alias snapshots).
* ``STEP:<n>`` / ``DERIVED:init`` — written by a later postprocess step or
  derived inside ``Op.__init__``; ingest seeds only the neutral default.
* ``DEFAULT`` — constant ingest seed.
* ``EXTRAS:<key>`` — from ``IngestExtras`` (trace-identity joins).
* ``NO_PRODUCER`` — pure caches.

``scatter_record_to_cells`` produces the CORE/FACET/EXTRAS/DEFAULT cells (and
the payload-disposition cells under NEUTRAL joins) byte-identically to
today's ``_fields_from_event``; the P1 parity gate proves this cell-for-cell
on real captures, and P3 rewires ingest so this scatter IS the single truth.

``torchlens/ir/op_record_manifest.py`` is GENERATED from this module
(regenerate-and-diff gated); regenerate with
``python -m tools.generate_op_record_manifest``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .op_record import FACET_DEFAULTS, IngestExtras, OpRecord

CELL_SOURCE_MANIFEST_VERSION = 2

CELL_SOURCES: dict[str, str] = {
    # ---- identity / core ----------------------------------------------------
    "_label_raw": "CORE",
    "_layer_label_raw": "CORE",
    "_tracing_finished": "FACET:policy",
    "_construction_done": "FACET:policy",
    "step_index": "CORE",
    "raw_index": "CORE",
    "ordinal_index": "DEFAULT",
    "label": "DEFAULT",  # STEP:8 writes the final value
    "label_short": "DEFAULT",
    "layer_label": "DEFAULT",
    "layer_label_short": "DEFAULT",
    "type": "CORE",
    "type_index": "CORE",
    "pass_index": "CORE",
    "num_passes": "DEFAULT",
    "lookup_keys": "DEFAULT",
    # ---- output / payload ----------------------------------------------------
    "out": "JOIN:payload",
    "has_saved_activation": "CORE",  # buffer-write sibling may override (JOIN)
    "output_device": "CORE",
    "activation_transform": "CORE",
    "annotations": "FACET:annotations",
    "interventions": "FACET:intervention",
    "intervention_replaced": "FACET:intervention",
    "detach_saved_activations": "CORE",
    "has_saved_args": "FACET:templates",
    "saved_args": "FACET:templates",
    "saved_kwargs": "FACET:templates",
    "args_template": "FACET:templates",
    "kwargs_template": "FACET:templates",
    "shape": "JOIN:payload",
    "transformed_out_shape": "CORE",
    "dtype": "JOIN:payload",
    "dtype_ref": "DERIVED:init",
    "transformed_out_dtype": "CORE",
    "device_ref": "DERIVED:init",
    "backend_address": "DERIVED:init",
    "resolver_status": "DERIVED:init",
    "activation_memory": "JOIN:payload",
    "transformed_activation_memory": "CORE",
    "visualizer_path": "CORE",
    "bytes_delta_at_call": "FACET:policy",
    "bytes_peak_at_call": "FACET:policy",
    "transformed_out": "CORE",
    "autograd_memory": "FACET:policy",
    "num_autograd_tensors": "FACET:policy",
    "has_out_variations": "JOIN:output_versions",
    "out_versions_by_child": "JOIN:output_versions",
    # ---- gradients (backward-phase writers) ----------------------------------
    "grad": "DEFAULT",
    "transformed_grad": "DEFAULT",
    "save_grads": "FACET:policy",
    "has_grad": "DEFAULT",
    "grad_shape": "DEFAULT",
    "transformed_grad_shape": "DEFAULT",
    "grad_dtype": "DEFAULT",
    "transformed_grad_dtype": "DEFAULT",
    "gradient_memory": "DEFAULT",
    "transformed_gradient_memory": "DEFAULT",
    # ---- function facet -------------------------------------------------------
    "func": "FACET:function",
    "func_id": "FACET:function",
    "func_call_id": "FACET:function",  # nested sibling-sharing value; the
    # top-level core func_call_id feeds call-group JOIN machinery instead
    "func_name": "FACET:function",
    "func_qualname": "FACET:function",
    "code_context": "FACET:function",
    "var_names": "DEFAULT",  # STEP:11.5
    "func_duration": "FACET:function",
    "flops_forward": "FACET:function",
    "flops_backward": "FACET:function",
    "func_rng_states": "FACET:function",
    "func_autocast_state": "FACET:function",
    "arg_names": "FACET:function",
    "num_args_total": "FACET:function",
    "num_pos_args": "FACET:function",
    "num_kwargs": "FACET:function",
    "non_tensor_pos_args": "FACET:function",
    "non_tensor_kwargs": "FACET:function",
    "func_non_tensor_args": "FACET:function",
    "is_inplace": "FACET:function",
    "func_config": "FACET:function",
    # ---- autograd -------------------------------------------------------------
    "grad_fn_class_name": "FACET:policy",  # semantics.grad_fn_class_name
    "grad_fn_class_qualname": "FACET:autograd",
    "grad_fn_object_id": "JOIN:grad_fn_index",
    "grad_fn_handle": "JOIN:grad_fn_index",
    "grad_fn": "DEFAULT",
    # ---- multi-output / container ----------------------------------------------
    "in_multi_output": "CORE",
    "multi_output_index": "CORE",
    "multi_output_name": "DEFAULT",
    "container_path": "CORE",
    "container_spec": "CORE",
    # ---- transform facet ---------------------------------------------------------
    "is_transform": "FACET:transform",
    "transform_kind": "FACET:transform",
    "transform_chain": "FACET:transform",
    "transform_config": "FACET:transform",  # legacy keys re-injected at scatter
    "transform_fn_name": "FACET:transform",
    "transform_fn_qualname": "FACET:transform",
    "transform_fn_source": "FACET:transform",
    # ---- graph facet ----------------------------------------------------------
    "unattributed_tensor_args": "FACET:graph",
    "dropped_edge_tensor_args": "FACET:graph",
    "equivalence_class": "FACET:graph",
    "parents": "CORE",
    "parent_arg_positions": "FACET:graph",
    "_edge_uses": "JOIN:edge_uses",
    # L6 stage-3 tier-(ii) stores: ingest seeds the neutral {}; the do() edge
    # engine writes them session-time on fork replay (FieldPolicy.DROP).
    "edge_substitutions": "DEFAULT",
    "edge_replacement_stamps": "DEFAULT",
    "is_output_parent": "FACET:graph",
    "input_was_parameter": "FACET:graph",
    # ---- params (resolved against the registry) -------------------------------
    "parent_params": "JOIN:param_logs",
    "_param_barcodes": "JOIN:param_logs",
    "parent_param_ops": "JOIN:param_logs",
    "_param_logs": "JOIN:param_logs",
    "param_shapes": "JOIN:param_logs",
    "num_params": "JOIN:param_logs",
    "num_params_trainable": "JOIN:param_logs",
    "num_params_frozen": "JOIN:param_logs",
    "param_memory": "JOIN:param_logs",
    # ---- equivalence / recurrence ------------------------------------------------
    "equivalent_ops": "JOIN:equivalence",
    "recurrent_ops": "DEFAULT",
    "site_key": "DEFAULT",  # STEP:7 mints the site_key_v1 value
    # tlspec v9 entry-dark injected-op identity (C07): default None on every
    # model op; the F01 log_injections writer stamps injected ops post-hoc.
    "injection_provenance": "DEFAULT",
    # ---- ancestry --------------------------------------------------------------
    "root_ancestors": "FACET:ancestry",
    "children": "JOIN:children",
    "has_children": "JOIN:children",
    "is_input": "CORE",
    "has_input_ancestor": "FACET:ancestry",
    "input_ancestors": "FACET:ancestry",
    "min_distance_from_input": "DEFAULT",
    "max_distance_from_input": "DEFAULT",
    "is_output": "DEFAULT",
    "is_final_output": "DEFAULT",
    "has_output_descendant": "DEFAULT",
    "output_descendants": "DEFAULT",
    "is_orphan": "DEFAULT",
    "io_role": "JOIN:io_role",
    "min_distance_to_output": "DEFAULT",
    "max_distance_to_output": "DEFAULT",
    # ---- buffers ----------------------------------------------------------------
    "is_buffer": "CORE",
    "address": "JOIN:buffer_address",
    "buffer_pass": "DEFAULT",
    "buffer_source": "JOIN:buffer_source",
    "buffer_write_kind": "DEFAULT",  # JOIN:buffer_write override when present
    "buffer_value_changed": "DEFAULT",
    "buffer_replay_validated": "DEFAULT",
    "buffer_source_func_name": "DEFAULT",
    # ---- internal sources ----------------------------------------------------------
    "is_internal_source": "CORE",
    "has_internal_source_ancestor": "FACET:ancestry",
    "internal_source_parents": "JOIN:ancestry",
    "internal_source_ancestors": "FACET:ancestry",
    "is_internal_sink": "DEFAULT",
    # ---- conditionals (STEP:5 domain) ----------------------------------------------
    "is_terminal_bool": "DEFAULT",
    "is_terminal_conditional_bool": "DEFAULT",
    "conditional_context_kind": "DEFAULT",
    "conditional_wrapper_kind": "DEFAULT",
    "terminal_conditional_id": "DEFAULT",
    "is_scalar_bool": "FACET:control",
    "bool_value": "FACET:control",
    "in_conditionals": "DEFAULT",
    "terminal_bool_for": "DEFAULT",
    "conditional_branch_stack": "DEFAULT",
    "conditional_branch_depth": "DEFAULT",
    "conditional_entry_children": "DEFAULT",
    "conditional_then_children": "DEFAULT",
    "conditional_elif_children": "DEFAULT",
    "conditional_else_children": "DEFAULT",
    "conditional_arm_children": "DEFAULT",
    "_is_in_conditional_body": "DEFAULT",
    # ---- modules ------------------------------------------------------------------
    "module": "FACET:modules",
    "_address_normalized": "DEFAULT",
    "modules": "FACET:modules",
    "fx_qualpath": "DEFAULT",
    "fx_call_index": "DEFAULT",
    # B3R7-R05-1: ``module_call_stack`` is the op's OWN containment stack (the
    # glossary's "ModuleCall labels active for this Op"), so it derives from
    # the creation-time modules facet -- NOT from the module-enter join, whose
    # fed-call stack made the one persisted field mean three different things
    # (fed-call for module inputs, containment for module outputs, empty
    # otherwise). The fed-call fact lives on ``input_to_module_calls``.
    "module_call_stack": "FACET:modules",
    "input_to_module_calls": "JOIN:module_enter",
    "module_entry_arg_keys": "JOIN:module_enter",
    "output_of_modules": "JOIN:module_exit",
    "output_of_module_calls": "JOIN:module_exit",
    "is_module_output": "JOIN:module_exit",
    "is_atomic_module": "JOIN:module_exit",
    "atomic_module_call": "JOIN:module_exit",
    # ---- identity joins / runtime ---------------------------------------------------
    "source_trace": "EXTRAS:source_trace",
    "_source_trace_ref": "DERIVED:init",
    "out_ref": "DERIVED:init",
    "grad_ref": "DERIVED:init",
    # ---- extra-key channels (conditional; popped before Op construction) -----------
    "_pending_blob_id": "CORE",  # from core.output.tensor.blob_ref
    "_pending_transformed_out_blob_id": "CORE",
    "_pending_grad_blob_id": "DEFAULT",  # backward-phase blob channel
    "_pending_transformed_grad_blob_id": "DEFAULT",
    "_materialized_backend_address": "JOIN:buffer_address",
    # ---- backward-phase / caches ------------------------------------------------------
    "_grad_records": "DEFAULT",
    "_facets_cache": "NO_PRODUCER",
    "_receptive_field_cache": "NO_PRODUCER",
    "_projective_field_cache": "NO_PRODUCER",
    "_arg_expressions_cache": "NO_PRODUCER",
}

# Extra-key channels beyond _OP_SLOT_NAMES (three-way closure universe).
EXTRA_KEY_CHANNELS: frozenset[str] = frozenset(
    {
        "_pending_blob_id",
        "_pending_transformed_out_blob_id",
        "_pending_grad_blob_id",
        "_pending_transformed_grad_blob_id",
        "_materialized_backend_address",
    }
)


def _facet(record: OpRecord, name: str) -> Any:
    """Return one facet of ``record``, materializing its checked-in default if absent."""

    from .op_record import _FACET_ATTRIBUTES

    value = getattr(record, _FACET_ATTRIBUTES[name])
    if value is not None:
        return value
    return FACET_DEFAULTS[name]()


def _internal_source_parent_labels(record: OpRecord, owning_trace: Any) -> list[str]:
    """Return direct parent labels whose paths include an internal source.

    Parameters
    ----------
    record:
        Journal record whose direct parent edges should be classified.
    owning_trace:
        Active trace providing the capture-time ancestry index.

    Returns
    -------
    list[str]
        Direct raw parent labels carrying internal-source ancestry, in edge order.
    """

    live_index = owning_trace.capture_events.live_index
    return [
        edge.parent_label_raw
        for edge in record.core.parents
        if live_index.require_event(edge.parent_label_raw).has_internal_source_ancestor
    ]


def scatter_record_to_cells(record: OpRecord, extras: IngestExtras, owning_trace: Any) -> dict:
    """Produce the record-sourced store cells for one op (byte-parity form).

    Covers the CORE / FACET / EXTRAS / DEFAULT classes plus the
    payload-disposition cells under NEUTRAL joins; JOIN/STEP cells belong to
    the ingest joins and later steps. Values byte-match today's
    ``_fields_from_event`` output for the same journal record (the P1 gate).
    """

    core = record.core
    output = core.output
    tensor = output.tensor
    transformed = output.transformed_tensor
    function = record.function
    templates = record.templates
    graph = _facet(record, "graph")
    modules_facet = _facet(record, "modules")
    ancestry = _facet(record, "ancestry")
    autograd = _facet(record, "autograd")
    transform = _facet(record, "transform")
    control = _facet(record, "control")
    annotations_facet = _facet(record, "annotations")
    policy_facet = _facet(record, "policy")
    intervention = _facet(record, "intervention")
    semantics = policy_facet.backend_semantics

    # legacy transform_config re-injection: `_tl_annotations` UNCONDITIONAL
    # (both producers stamp the key on every op), fn_code_location conditional
    transform_config_cell: dict[str, object] = dict(transform.transform_config)
    transform_config_cell["_tl_annotations"] = dict(annotations_facet.annotations)
    if transform.fn_code_location is not None:
        transform_config_cell["fn_code_location"] = transform.fn_code_location

    cells: dict[str, Any] = {
        "_label_raw": core.label_raw,
        "_layer_label_raw": core.layer_label_raw,
        "step_index": core.step_index,
        "raw_index": core.raw_index,
        "ordinal_index": -1,
        "source_trace": extras.source_trace or owning_trace,
        "_tracing_finished": policy_facet.tracing_finished,
        "_construction_done": policy_facet.construction_done,
        "label": None,
        "label_short": None,
        "layer_label": None,
        "layer_label_short": None,
        "type": core.layer_type,
        "type_index": core.type_index,
        "pass_index": core.pass_index,
        "num_passes": 1,
        "lookup_keys": [],
        "has_saved_activation": output.has_saved_activation,
        "output_device": output.output_device,
        "activation_transform": output.activation_transform,
        "annotations": dict(annotations_facet.annotations),
        "interventions": [
            result.fire_record
            for result in intervention.fire_results
            if getattr(result, "fire_record", None) is not None
        ],
        "intervention_replaced": intervention.intervention_replaced,
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
        "transformed_out_shape": None if transformed is None else transformed.shape,
        "transformed_out_dtype": None if transformed is None else _resolve_dtype(transformed.dtype),
        "transformed_activation_memory": None if transformed is None else transformed.memory,
        "visualizer_path": output.visualizer_path,
        "bytes_delta_at_call": None if semantics is None else semantics.bytes_delta_at_call,
        "bytes_peak_at_call": None if semantics is None else semantics.bytes_peak_at_call,
        "transformed_out": None if transformed is None else transformed.payload,
        "autograd_memory": None if semantics is None else semantics.autograd_memory,
        "num_autograd_tensors": None if semantics is None else semantics.num_autograd_tensors,
        "grad": None,
        "transformed_grad": None,
        "save_grads": None if policy_facet.policy is None else policy_facet.policy.save_grad,
        "has_grad": False,
        "grad_shape": None,
        "transformed_grad_shape": None,
        "grad_dtype": None,
        "transformed_grad_dtype": None,
        "gradient_memory": 0,
        "transformed_gradient_memory": None,
        "func": None if function is None else function.func,
        "func_id": None if function is None else function.func_id,
        "func_call_id": None if function is None else function.func_call_id,
        "func_name": None if function is None else function.func_name,
        "func_qualname": None if function is None else function.func_qualname,
        "code_context": [] if function is None else list(function.code_context),
        "var_names": [],
        "func_duration": (function.func_duration or 0) if function is not None else 0,
        "flops_forward": None if function is None else function.flops_forward,
        "flops_backward": None if function is None else function.flops_backward,
        "func_rng_states": None if function is None else function.func_rng_states,
        "func_autocast_state": None if function is None else function.func_autocast_state,
        "arg_names": () if function is None else tuple(function.arg_names),
        "num_args_total": 0 if function is None else function.num_args_total,
        "num_pos_args": 0 if function is None else function.num_pos_args,
        "num_kwargs": 0 if function is None else function.num_kwargs,
        "non_tensor_pos_args": [] if function is None else list(function.non_tensor_pos_args),
        "non_tensor_kwargs": {} if function is None else dict(function.non_tensor_kwargs),
        "func_non_tensor_args": [] if function is None else list(function.func_non_tensor_args),
        "is_inplace": False if function is None else function.is_inplace,
        "grad_fn_class_name": None if semantics is None else semantics.grad_fn_class_name,
        "grad_fn_class_qualname": autograd.grad_fn_class_qualname,
        "grad_fn": None,
        "in_multi_output": output.in_multi_output,
        "multi_output_index": output.multi_output_index,
        "multi_output_name": None,
        "container_path": tuple(output.container_path),
        "container_spec": output.container_spec,
        "is_transform": transform.is_transform,
        "transform_kind": transform.transform_kind,
        "transform_chain": tuple(transform.transform_chain),
        "transform_config": transform_config_cell,
        "transform_fn_name": transform.transform_fn_name,
        "transform_fn_qualname": transform.transform_fn_qualname,
        "transform_fn_source": transform.transform_fn_source,
        "unattributed_tensor_args": tuple(graph.unattributed_tensor_args),
        "dropped_edge_tensor_args": tuple(graph.dropped_edge_tensor_args),
        "equivalence_class": graph.equivalence_class,
        "recurrent_ops": [],
        "parents": [edge.parent_label_raw for edge in core.parents],
        "parent_arg_positions": graph.parent_arg_positions,
        "root_ancestors": set(ancestry.root_ancestors),
        "is_input": core.layer_type == "input",
        "input_was_parameter": graph.input_was_parameter,
        "has_input_ancestor": bool(ancestry.input_ancestors),
        "input_ancestors": set(ancestry.input_ancestors),
        "min_distance_from_input": None,
        "max_distance_from_input": None,
        "is_output": False,
        "is_output_parent": graph.is_output_parent,
        "is_final_output": False,
        "has_output_descendant": False,
        "output_descendants": set(),
        "is_orphan": False,
        "min_distance_to_output": None,
        "max_distance_to_output": None,
        "is_buffer": core.kind == "source" and core.layer_type == "buffer",
        "buffer_pass": None,
        "buffer_write_kind": None,
        "buffer_value_changed": None,
        "buffer_replay_validated": None,
        "buffer_source_func_name": None,
        "is_internal_source": core.layer_type != "input" and not core.parents,
        "has_internal_source_ancestor": ancestry.has_internal_source_ancestor,
        "internal_source_parents": _internal_source_parent_labels(record, owning_trace),
        "internal_source_ancestors": set(ancestry.internal_source_ancestors),
        "is_internal_sink": False,
        "is_terminal_bool": False,
        "is_terminal_conditional_bool": False,
        "conditional_context_kind": None,
        "conditional_wrapper_kind": None,
        "terminal_conditional_id": None,
        "is_scalar_bool": bool(control.is_scalar_bool),
        "bool_value": control.bool_value,
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
        "module": modules_facet.modules[-1] if modules_facet.modules else None,
        "_address_normalized": None,
        "modules": list(modules_facet.modules),
        "fx_qualpath": None,
        "fx_call_index": 0,
        # Same facet as ``modules``: the containment stack active at op
        # creation. Both cells hold raw ``(address, call_index)`` pairs here
        # and are converted to canonical ``address:N`` ModuleCall labels
        # together by the step-11 relabel (labeling.py).
        "module_call_stack": list(modules_facet.modules),
        "input_to_module_calls": [],
        "module_entry_arg_keys": defaultdict(list),
        "output_of_modules": [],
        "output_of_module_calls": [],
        "is_module_output": False,
        "is_atomic_module": False,
        "atomic_module_call": None,
        "func_config": {} if function is None else dict(function.func_config),
    }

    from .._io import BlobRef as PortableBlobRef

    if isinstance(tensor.blob_ref, PortableBlobRef):
        cells["_pending_blob_id"] = tensor.blob_ref.blob_id
    if transformed is not None and isinstance(transformed.blob_ref, PortableBlobRef):
        cells["_pending_transformed_out_blob_id"] = transformed.blob_ref.blob_id
    return cells


def _resolve_dtype(dtype: object) -> object:
    """Match _materialize's transformed-dtype normalization."""

    from ..postprocess._materialize import _resolve_dtype as materialize_resolve_dtype

    return materialize_resolve_dtype(dtype)
