"""Row-direct op cells: the exhaustive draft straight to store cells (capture C2a).

Today an exhaustive op's cells are produced in two hops: the commit freezes
the draft (``fields_dict``) into an ``OpRecord`` (``_op_record_from_log``) and
ingest scatters the record back into cells (``scatter_record_to_cells``).
:func:`op_row_cells` computes the same cells in one hop, without building the
record; :func:`write_op_row` appends them as one layout-ordered row.

The value contract is exact: for the same arguments, ``op_row_cells`` returns
``scatter_record_to_cells(_op_record_from_log(...), IngestExtras(), trace)``
key for key and value for value (types included), evaluated at the same
moment. Facet presence is mirrored: a facet the freeze would omit reads the
checked-in ``FACET_DEFAULTS`` value, as the scatter does. The shared
function-call ref goes through ``_resolve_call_function_ref`` so sibling
outputs of one call share it exactly as the freeze does (the call box is
updated the same way).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from ..._ir.op_record import FACET_DEFAULTS
from ..._ir.op_record_scatter import _resolve_dtype
from ..._trace_core.op_store import _MISSING
from ._ops_capture_records import _resolve_call_function_ref
from ._ops_container_base import _copy_parent_arg_positions

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ..._ir.events import ModuleFrame
    from ..._ir.intervention import FireResult
    from ..._ir.refs import FunctionCallRef

__all__ = ("op_row_cells", "write_op_row")

#: One default instance per defaultable facet, built on first use. Every
#: default field the cells read is either immutable or copied below, exactly
#: as the scatter copies the per-call default it builds.
_FACET_DEFAULT_CACHE: dict[str, Any] = {}


def _facet_default(name: str) -> Any:
    """Return the cached default instance of one facet."""

    value = _FACET_DEFAULT_CACHE.get(name)
    if value is None:
        value = FACET_DEFAULTS[name]()
        _FACET_DEFAULT_CACHE[name] = value
    return value


def _semantics_cells(fields_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Return the five backend-semantics cells (the freeze's fallback mirrored)."""

    semantics = fields_dict.get("backend_semantics")
    if semantics is None:
        return {
            "bytes_delta_at_call": fields_dict["bytes_delta_at_call"],
            "bytes_peak_at_call": fields_dict["bytes_peak_at_call"],
            "autograd_memory": fields_dict["autograd_memory"],
            "num_autograd_tensors": fields_dict["num_autograd_tensors"],
            "grad_fn_class_name": fields_dict["grad_fn_class_name"],
        }
    return {
        "bytes_delta_at_call": semantics.bytes_delta_at_call,
        "bytes_peak_at_call": semantics.bytes_peak_at_call,
        "autograd_memory": semantics.autograd_memory,
        "num_autograd_tensors": semantics.num_autograd_tensors,
        "grad_fn_class_name": semantics.grad_fn_class_name,
    }


def _transformed_cells(fields_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Return the transformed-output cells (``None`` when no transformed output)."""

    payload = fields_dict["transformed_out"]
    if payload is None:
        return {
            "transformed_out_shape": None,
            "transformed_out_dtype": None,
            "transformed_activation_memory": None,
            "transformed_out": None,
        }
    return {
        "transformed_out_shape": fields_dict["transformed_out_shape"],
        "transformed_out_dtype": _resolve_dtype(str(fields_dict["transformed_out_dtype"])),
        "transformed_activation_memory": fields_dict["transformed_activation_memory"],
        "transformed_out": payload,
    }


def _function_cells(function: FunctionCallRef) -> dict[str, Any]:
    """Return the cells the scatter reads off the function-call ref."""

    return {
        "func": function.func,
        "func_id": function.func_id,
        "func_call_id": function.func_call_id,
        "func_name": function.func_name,
        "func_qualname": function.func_qualname,
        "code_context": list(function.code_context),
        "var_names": [],
        "func_duration": function.func_duration or 0,
        "flops_forward": function.flops_forward,
        "flops_backward": function.flops_backward,
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
        "func_config": dict(function.func_config),
    }


def _transform_cells(fields_dict: Mapping[str, Any], annotations: dict[str, Any]) -> dict[str, Any]:
    """Return the transform cells, mirroring the freeze's facet presence and config split."""

    user_config = dict(fields_dict.get("transform_config") or {})
    user_config.pop("_tl_annotations", None)
    fn_code_location = user_config.pop("fn_code_location", None)
    is_transform = bool(fields_dict.get("is_transform", False))
    kind = fields_dict.get("transform_kind")
    chain = tuple(fields_dict.get("transform_chain") or ())
    fn_name = fields_dict.get("transform_fn_name")
    fn_qualname = fields_dict.get("transform_fn_qualname")
    fn_source = fields_dict.get("transform_fn_source")
    present = (
        is_transform
        or kind is not None
        or chain
        or user_config
        or fn_name is not None
        or fn_qualname is not None
        or fn_source is not None
        or fn_code_location is not None
    )
    if not present:
        default = _facet_default("transform")
        is_transform = default.is_transform
        kind = default.transform_kind
        chain = tuple(default.transform_chain)
        user_config = dict(default.transform_config)
        fn_name = default.transform_fn_name
        fn_qualname = default.transform_fn_qualname
        fn_source = default.transform_fn_source
        fn_code_location = default.fn_code_location
    config_cell: dict[str, object] = dict(user_config)
    config_cell["_tl_annotations"] = dict(annotations)
    if fn_code_location is not None:
        config_cell["fn_code_location"] = fn_code_location
    return {
        "is_transform": is_transform,
        "transform_kind": kind,
        "transform_chain": chain,
        "transform_config": config_cell,
        "transform_fn_name": fn_name,
        "transform_fn_qualname": fn_qualname,
        "transform_fn_source": fn_source,
    }


def _ancestry_cells(fields_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ancestry cells, reading the default facet when the freeze omits it."""

    input_ancestors = frozenset(fields_dict["input_ancestors"])
    internal_source_ancestors = frozenset(fields_dict["internal_source_ancestors"])
    root_ancestors = frozenset(fields_dict["root_ancestors"])
    has_internal = fields_dict["has_internal_source_ancestor"]
    if not (input_ancestors or internal_source_ancestors or root_ancestors or has_internal):
        default = _facet_default("ancestry")
        input_ancestors = default.input_ancestors
        internal_source_ancestors = default.internal_source_ancestors
        root_ancestors = default.root_ancestors
        has_internal = default.has_internal_source_ancestor
    return {
        "root_ancestors": set(root_ancestors),
        "has_input_ancestor": bool(input_ancestors),
        "input_ancestors": set(input_ancestors),
        "has_internal_source_ancestor": has_internal,
        "internal_source_ancestors": set(internal_source_ancestors),
    }


def _internal_source_parent_labels(trace: Trace, parents: list[str]) -> list[str]:
    """Return the parent labels whose committed op carries internal-source ancestry."""

    require_event = trace.capture_events.live_index.require_event
    return [label for label in parents if require_event(label).has_internal_source_ancestor]


def _facet_value_cells(
    fields_dict: Mapping[str, Any], fire_results: tuple[FireResult, ...]
) -> dict[str, Any]:
    """Return the control, autograd, intervention and annotation cells."""

    is_scalar_bool = fields_dict["is_scalar_bool"]
    bool_value = fields_dict["bool_value"]
    if is_scalar_bool is None and bool_value is None:
        control = _facet_default("control")
        is_scalar_bool, bool_value = control.is_scalar_bool, control.bool_value
    qualname = fields_dict["grad_fn_class_qualname"]
    if qualname is None:
        qualname = _facet_default("autograd").grad_fn_class_qualname
    is_edited = fields_dict["is_edited"]
    if not (fire_results or is_edited):
        intervention = _facet_default("intervention")
        fire_results, is_edited = intervention.fire_results, intervention.is_edited
    annotations = dict(fields_dict.get("annotations") or {})
    if not annotations:
        annotations = dict(_facet_default("annotations").annotations)
    return {
        "is_scalar_bool": bool(is_scalar_bool),
        "bool_value": bool_value,
        "grad_fn_class_qualname": qualname,
        "is_edited": is_edited,
        "edits": [
            result.fire_record
            for result in fire_results
            if getattr(result, "fire_record", None) is not None
        ],
        "annotations": annotations,
    }


def _output_cells(fields_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Return the output-ref cells and the pending blob ids."""

    cells = {
        "has_saved_activation": fields_dict["has_saved_activation"],
        "output_device": fields_dict["output_device"],
        "activation_transform": fields_dict["activation_transform"],
        "detach_saved_activations": fields_dict["detach_saved_activations"],
        "visualizer_path": fields_dict["visualizer_path"],
        "in_multi_output": fields_dict["in_multi_output"],
        "multi_output_index": fields_dict["multi_output_index"],
        "container_path": tuple(fields_dict["container_path"]),
        "container_spec": fields_dict["container_spec"],
    }
    if fields_dict.get("_pending_blob_id") is not None:
        cells["_pending_blob_id"] = fields_dict["_pending_blob_id"]
    if (
        fields_dict["transformed_out"] is not None
        and fields_dict.get("_pending_transformed_out_blob_id") is not None
    ):
        cells["_pending_transformed_out_blob_id"] = fields_dict["_pending_transformed_out_blob_id"]
    return cells


def _core_graph_cells(trace: Trace, fields_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Return the core, graph, template and policy cells."""

    layer_type = fields_dict["type"]
    parents = list(fields_dict["parents"])
    is_source = bool(fields_dict["is_input"] or fields_dict["is_buffer"])
    modules = tuple(fields_dict["modules"])
    return {
        "_label_raw": fields_dict["_label_raw"],
        "_layer_label_raw": fields_dict["_layer_label_raw"],
        "step_index": fields_dict["step_index"] or 0,
        "raw_index": fields_dict["raw_index"],
        "type": layer_type,
        "type_index": fields_dict["type_index"],
        "pass_number": fields_dict["pass_number"],
        "source_trace": trace,
        "_tracing_finished": fields_dict["_tracing_finished"],
        "_construction_done": fields_dict["_construction_done"],
        "save_grads": fields_dict["save_grads"],
        "has_saved_args": fields_dict["has_saved_args"],
        "saved_args": fields_dict["saved_args"],
        "saved_kwargs": fields_dict["saved_kwargs"],
        "args_template": fields_dict["args_template"],
        "kwargs_template": fields_dict["kwargs_template"],
        "unattributed_tensor_args": tuple(fields_dict.get("unattributed_tensor_args") or ()),
        "dropped_edge_tensor_args": tuple(fields_dict.get("dropped_edge_tensor_args") or ()),
        "equivalence_class": fields_dict["equivalence_class"],
        "parents": parents,
        "parent_arg_positions": _copy_parent_arg_positions(fields_dict["parent_arg_positions"]),
        "is_input": layer_type == "input",
        "input_was_parameter": bool(fields_dict.get("input_was_parameter", False)),
        "is_output_parent": fields_dict["is_output_parent"],
        "is_buffer": is_source and layer_type == "buffer",
        "is_internal_source": layer_type != "input" and not parents,
        "internal_source_parents": _internal_source_parent_labels(trace, parents),
        "module": modules[-1] if modules else None,
        "modules": list(modules),
        "module_call_stack": list(modules),
    }


def _constant_cells() -> dict[str, Any]:
    """Return the cells the scatter seeds with constants (fresh containers per op)."""

    return {
        "ordinal_index": -1,
        "label": None,
        "short_label": None,
        "layer_label": None,
        "short_layer_label": None,
        "num_passes": 1,
        "lookup_keys": [],
        "input_ops": None,
        "input_activations": None,
        "input_shapes": None,
        "input_dtypes": None,
        "input_memory": None,
        "num_inputs": None,
        "grad": None,
        "transformed_grad": None,
        "has_grad": False,
        "grad_shape": None,
        "transformed_grad_shape": None,
        "grad_dtype": None,
        "transformed_grad_dtype": None,
        "gradient_memory": 0,
        "transformed_gradient_memory": None,
        "grad_fn": None,
        "multi_output_name": None,
        "recurrent_ops": [],
        "min_distance_from_input": None,
        "max_distance_from_input": None,
        "is_output": False,
        "is_final_output": False,
        "has_output_descendant": False,
        "output_descendants": set(),
        "is_orphan": False,
        "min_distance_to_output": None,
        "max_distance_to_output": None,
        "buffer_pass": None,
        "buffer_write_kind": None,
        "buffer_value_changed": None,
        "buffer_replay_validated": None,
        "buffer_source_func_name": None,
        "is_internal_sink": False,
        "is_terminal_bool": False,
        "is_terminal_conditional_bool": False,
        "conditional_context_kind": None,
        "conditional_wrapper_kind": None,
        "terminal_conditional_id": None,
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
        "_address_normalized": None,
        "fx_qualpath": None,
        "fx_call_index": 0,
        "input_to_module_calls": [],
        "module_entry_arg_keys": defaultdict(list),
        "output_of_modules": [],
        "output_of_module_calls": [],
        "is_module_output": False,
        "is_atomic_module": False,
        "atomic_module_call": None,
    }


def op_row_cells(
    trace: Trace,
    fields_dict: Mapping[str, Any],
    tensor: torch.Tensor,
    fire_results: tuple[FireResult, ...] = (),
    module_stack: tuple[ModuleFrame, ...] | None = None,
    call_ref_box: list[FunctionCallRef] | None = None,
) -> dict[str, Any]:
    """Return one exhaustive op's record-sourced store cells without building its record.

    Parameters
    ----------
    trace
        Active capture trace (the ``source_trace`` cell and the live index).
    fields_dict
        The final exhaustive draft, as passed to ``_op_record_from_log``.
    tensor
        The op's output tensor (accepted for signature parity with the freeze;
        every cell it feeds is already on the draft).
    fire_results
        Intervention results for this op.
    module_stack
        Module frames (signature parity; the cells read ``modules`` only).
    call_ref_box
        The per-call shared function-ref box, updated as the freeze updates it.

    Returns
    -------
    dict[str, Any]
        Cell name to value, equal to the scatter of the frozen record.
    """

    del tensor, module_stack
    facet_cells = _facet_value_cells(fields_dict, fire_results)
    cells = _constant_cells()
    cells.update(_core_graph_cells(trace, fields_dict))
    cells.update(_output_cells(fields_dict))
    cells.update(_transformed_cells(fields_dict))
    cells.update(_semantics_cells(fields_dict))
    cells.update(_ancestry_cells(fields_dict))
    cells.update(_transform_cells(fields_dict, facet_cells["annotations"]))
    cells.update(facet_cells)
    cells.update(_function_cells(_resolve_call_function_ref(fields_dict, call_ref_box)))
    return cells


def write_op_row(store: Any, cells: Mapping[str, Any]) -> int:
    """Append one layout-ordered row holding ``cells``; unset fields stay ``_MISSING``.

    Parameters
    ----------
    store
        A building ``OpRowStore`` (or any store with ``layout`` and ``adopt_row``).
    cells
        Cell name to value; every name must be a layout field.

    Returns
    -------
    int
        The new row's index.

    Raises
    ------
    KeyError
        If a cell name is not a field of the store's layout.
    """

    fid_by_name = store.layout.fid_by_name
    row: list[Any] = [_MISSING] * store.layout.n_fields
    for name, value in cells.items():
        row[fid_by_name[name]] = value
    return int(store.adopt_row(row))
