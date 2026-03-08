"""Functions for logging output tensors produced by decorated torch operations.

This module handles the creation and population of LayerPassLog entries for every
tensor produced during a forward pass.  It covers both *exhaustive* mode (full
metadata collection) and *fast* mode (re-use of a previously logged graph with
new activations).

Architecture overview:
    Every decorated torch function wrapper calls ``log_function_output_tensors``
    after executing the original function.  This dispatcher routes to either:

    - ``log_function_output_tensors_exhaustive``: builds a complete
      ``fields_dict`` of ~80 fields per tensor, creates a LayerPassLog entry,
      updates family links (parent/child/sibling/spouse), and optionally
      saves the activation value.

    - ``log_function_output_tensors_fast``: skips metadata collection entirely.
      Increments counters to maintain alignment with the exhaustive pass,
      verifies the graph hasn't changed, and saves new activation values into
      the existing LayerPassLog entries.

Label format convention:
    Raw labels follow ``{layer_type}_{type_num}_{realtime_num}_raw``, e.g.
    ``"conv2d_3_47_raw"``.  During postprocessing, these are mapped to final
    labels like ``"conv2d_3:1"`` (layer 3, pass 1).

pause_logging usage:
    ``pause_logging()`` temporarily disables the logging toggle so that
    utility operations (e.g., ``get_tensor_memory_amount``, ``safe_copy``,
    ``activation_postfunc``) don't get logged as model operations.  It is
    used inside ``save_tensor_data`` and wherever helper functions call
    decorated torch methods on tensors.
"""

import copy
from collections import defaultdict
from math import prod
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import torch

from .. import _state as _st
from .._state import pause_logging
from ..utils.introspection import (
    _get_func_call_stack,
    get_attr_values_from_tensor_list,
    get_vars_of_type_from_obj,
)
from ..utils.tensor_utils import get_tensor_memory_amount, safe_copy, tensor_nanequal
from ..utils.collections import index_nested, ensure_iterable
from .flops import compute_backward_flops, compute_forward_flops
from ..data_classes.buffer_log import BufferLog
from ..data_classes.layer_pass_log import LayerPassLog
from .arg_positions import (
    FUNC_ARG_SPECS,
    ArgSpec,
    extract_tensors_and_params,
    _cache_dynamic_spec,
    _normalize_func_name,
)

from .tensor_tracking import (
    _update_tensor_family_links,
    _locate_parent_tensors_in_args,
    _get_ancestors_from_parents,
    _add_backward_hook,
    _process_parent_param_passes,
    _make_raw_param_group_barcode,
    _get_operation_equivalence_type,
    _get_hash_from_args,
    _update_tensor_containing_modules,
)
from ..data_classes.internal_types import FuncExecutionContext
from .salient_args import extract_salient_args
from .source_tensors import _get_input_module_info

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def log_function_output_tensors(
    self,
    func: Callable,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
):
    """Dispatch to exhaustive or fast logging based on current logging mode.

    Called by every decorated torch function wrapper after executing the
    original function.  The mode was set in ``save_new_activations`` (fast)
    or ``log_forward_pass`` (exhaustive).
    """
    if self.logging_mode == "exhaustive":
        log_function_output_tensors_exhaustive(
            self,
            func,
            func_name,
            args,
            kwargs,
            arg_copies,
            kwarg_copies,
            out_orig,
            exec_ctx,
            is_bottom_level_func,
        )
    elif self.logging_mode == "fast":
        log_function_output_tensors_fast(
            self,
            func_name,
            args,
            kwargs,
            arg_copies,
            kwarg_copies,
            out_orig,
            exec_ctx,
            is_bottom_level_func,
        )


def _build_graph_relationship_fields(
    self,
    fields_dict: Dict[str, Any],
    parent_layer_labels: List[str],
    parent_layer_entries: List,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    out_orig: Any,
) -> None:
    """Populate graph structure fields: parents, children, ancestors, buffer/IO flags."""
    parent_layer_arg_locs = _locate_parent_tensors_in_args(self, parent_layer_entries, args, kwargs)
    input_ancestors, internally_initialized_ancestors = _get_ancestors_from_parents(
        parent_layer_entries
    )
    internal_parent_layer_labels = [
        label for label in parent_layer_labels if self[label].has_internally_initialized_ancestor
    ]

    fields_dict["parent_layers"] = parent_layer_labels
    fields_dict["parent_layer_arg_locs"] = parent_layer_arg_locs
    fields_dict["orig_ancestors"] = input_ancestors.union(internally_initialized_ancestors)
    fields_dict["child_layers"] = []
    fields_dict["has_children"] = False
    fields_dict["is_input_layer"] = False
    fields_dict["has_input_ancestor"] = len(input_ancestors) > 0
    fields_dict["input_ancestors"] = input_ancestors
    fields_dict["min_distance_from_input"] = None
    fields_dict["max_distance_from_input"] = None
    fields_dict["is_output_layer"] = False
    fields_dict["is_output_parent"] = False
    fields_dict["is_last_output_layer"] = False
    fields_dict["is_output_ancestor"] = False
    fields_dict["output_descendents"] = set()
    fields_dict["min_distance_from_output"] = None
    fields_dict["max_distance_from_output"] = None
    fields_dict["input_output_address"] = None
    fields_dict["is_buffer_layer"] = False
    fields_dict["buffer_address"] = None
    fields_dict["buffer_pass"] = None
    fields_dict["buffer_parent"] = None
    fields_dict["initialized_inside_model"] = len(parent_layer_labels) == 0
    fields_dict["has_internally_initialized_ancestor"] = len(internally_initialized_ancestors) > 0
    fields_dict["internally_initialized_parents"] = internal_parent_layer_labels
    fields_dict["internally_initialized_ancestors"] = internally_initialized_ancestors
    fields_dict["terminated_inside_model"] = False
    fields_dict["is_terminal_bool_layer"] = False
    fields_dict["in_cond_branch"] = False
    fields_dict["cond_branch_start_children"] = []

    is_part_of_iterable_output = any(
        issubclass(type(out_orig), cls) for cls in [list, tuple, dict, set]
    )
    fields_dict["is_part_of_iterable_output"] = is_part_of_iterable_output


def _extract_arg_tensors_and_params(
    normalized_name: str,
    args: tuple,
    kwargs: dict,
) -> Tuple[List, List]:
    """O(1) tensor/param extraction via lookup table, with BFS fallback."""
    spec = FUNC_ARG_SPECS.get(normalized_name) or _st._dynamic_arg_specs.get(normalized_name)
    if spec is not None:
        return extract_tensors_and_params(spec, args, kwargs)  # type: ignore[arg-type]

    # Tier 3 fallback: BFS crawl once, then cache for subsequent calls
    all_args = list(args) + list(kwargs.values())
    arg_tensors = get_vars_of_type_from_obj(all_args, torch.Tensor, [torch.nn.Parameter])
    arg_parameters = get_vars_of_type_from_obj(all_args, torch.nn.parameter.Parameter)
    _cache_dynamic_spec(normalized_name, args, kwargs, arg_tensors, arg_parameters)
    return arg_tensors, arg_parameters


def _build_param_fields(
    self,
    fields_dict: Dict[str, Any],
    arg_parameters: List,
) -> Dict:
    """Populate parameter-involvement fields. Returns parent_param_passes dict."""
    parent_param_passes = _process_parent_param_passes(arg_parameters)
    indiv_param_barcodes = list(parent_param_passes.keys())

    parent_param_logs = []
    for param in arg_parameters:
        addr = getattr(param, "tl_param_address", None)
        if addr is not None and addr in self.param_logs:
            parent_param_logs.append(self.param_logs[addr])

    fields_dict["parent_params"] = arg_parameters
    fields_dict["parent_param_barcodes"] = indiv_param_barcodes
    fields_dict["parent_param_passes"] = parent_param_passes
    fields_dict["parent_param_logs"] = parent_param_logs
    fields_dict["parent_param_shapes"] = [tuple(param.shape) for param in arg_parameters]
    fields_dict["num_params_total"] = sum(
        prod(shape) for shape in fields_dict["parent_param_shapes"]
    )
    fields_dict["num_params_trainable"] = sum(
        pl.num_params for pl in parent_param_logs if pl.trainable
    )
    fields_dict["num_params_frozen"] = sum(
        pl.num_params for pl in parent_param_logs if not pl.trainable
    )
    with pause_logging():
        fields_dict["parent_params_fsize"] = sum(
            p.nelement() * p.element_size() for p in arg_parameters
        )
    return parent_param_passes


def _build_module_context_fields(
    self,
    fields_dict: Dict[str, Any],
    arg_tensors: List,
    parent_layer_entries: List,
) -> None:
    """Populate module nesting, address, and input/output status fields."""
    containing_modules_origin_nested = _get_input_module_info(self, arg_tensors)
    containing_module_origin = (
        containing_modules_origin_nested[-1] if containing_modules_origin_nested else None
    )

    fields_dict["containing_module_origin"] = containing_module_origin
    fields_dict["containing_modules_origin_nested"] = containing_modules_origin_nested
    fields_dict["modules_entered"] = []
    fields_dict["modules_entered_argnames"] = defaultdict(list)
    fields_dict["module_passes_entered"] = []
    fields_dict["modules_exited"] = []
    fields_dict["module_passes_exited"] = []
    fields_dict["is_submodule_output"] = False
    fields_dict["is_bottom_level_submodule_output"] = False
    fields_dict["bottom_level_submodule_pass_exited"] = None
    fields_dict["module_entry_exit_threads_inputs"] = {
        p.tensor_label_raw: p.module_entry_exit_thread_output[:] for p in parent_layer_entries
    }
    fields_dict["module_entry_exit_thread_output"] = []


def _build_shared_fields_dict(
    self,
    func: Callable,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
) -> Tuple[Dict[str, Any], List, List, Dict]:
    """Build the fields_dict shared by all output tensors of a single function call.

    When a function produces multiple output tensors (e.g. ``torch.split``),
    many metadata fields are identical across outputs (function info, parent
    relationships, module context).  This function computes those shared fields
    once; per-tensor fields (shape, label, equivalence type) are added later
    in ``_log_output_tensor_info``.

    Returns:
        (fields_dict, parent_layer_entries, arg_tensors, parent_param_passes)
    """
    # Canonical layer_type: lowercase with underscores stripped (e.g. "conv2d").
    layer_type = func_name.lower().replace("_", "")

    # O(1) tensor/param extraction via lookup table (replaces BFS crawl)
    arg_tensors, arg_parameters = _extract_arg_tensors_and_params(layer_type, args, kwargs)

    # Separate tensor args (which define graph edges) from non-tensor args
    # (which become metadata and feed into operation_equivalence_type hashing).
    non_tensor_args = [arg for arg in args if not _check_if_tensor_arg(arg)]
    non_tensor_kwargs = {key: val for key, val in kwargs.items() if not _check_if_tensor_arg(val)}
    parent_layer_labels = get_attr_values_from_tensor_list(arg_tensors, "tl_tensor_label_raw")
    parent_layer_entries = [self[label] for label in parent_layer_labels]

    fields_dict = {}

    # General info
    fields_dict["layer_type"] = layer_type
    fields_dict["detach_saved_tensor"] = self.detach_saved_tensors
    fields_dict["output_device"] = self.output_device

    # Grad info
    fields_dict["grad_contents"] = None  # type: ignore[assignment]
    fields_dict["save_gradients"] = self.save_gradients
    fields_dict["has_saved_grad"] = False  # type: ignore[assignment]
    fields_dict["grad_shape"] = None  # type: ignore[assignment]
    fields_dict["grad_dtype"] = None  # type: ignore[assignment]
    fields_dict["grad_fsize"] = 0  # type: ignore[assignment]

    # Function call info
    fields_dict["func_applied"] = func  # type: ignore[assignment]
    fields_dict["func_applied_name"] = func_name
    fields_dict["func_call_stack"] = (
        _get_func_call_stack(self.num_context_lines) if self.save_source_context else []  # type: ignore[assignment]
    )
    fields_dict["func_time_elapsed"] = exec_ctx.time_elapsed  # type: ignore[assignment]
    fields_dict["func_rng_states"] = exec_ctx.rng_states  # type: ignore[assignment]
    fields_dict["func_autocast_state"] = exec_ctx.autocast_state  # type: ignore[assignment]
    fields_dict["func_argnames"] = _st._func_argnames.get(func_name.strip("_"), ())  # type: ignore[assignment]
    fields_dict["num_func_args_total"] = len(args) + len(kwargs)  # type: ignore[assignment]
    fields_dict["num_position_args"] = len(args)  # type: ignore[assignment]
    fields_dict["num_keyword_args"] = len(kwargs)  # type: ignore[assignment]
    fields_dict["func_position_args_non_tensor"] = non_tensor_args  # type: ignore[assignment]
    fields_dict["func_keyword_args_non_tensor"] = non_tensor_kwargs  # type: ignore[assignment]
    fields_dict["func_all_args_non_tensor"] = non_tensor_args + list(non_tensor_kwargs.values())  # type: ignore[assignment]

    _build_graph_relationship_fields(
        self, fields_dict, parent_layer_labels, parent_layer_entries, args, kwargs, out_orig
    )
    parent_param_passes = _build_param_fields(self, fields_dict, arg_parameters)
    _build_module_context_fields(self, fields_dict, arg_tensors, parent_layer_entries)

    # Function config — lightweight hyperparameter extraction, always on.
    fields_dict["func_config"] = extract_salient_args(
        layer_type,
        func_name,
        args,
        kwargs,
        fields_dict.get("parent_param_shapes", []),
    )

    return fields_dict, parent_layer_entries, arg_tensors, parent_param_passes


def _classify_new_tensor_in_model_log(
    self,
    fields_dict: Dict[str, Any],
    fields_dict_onetensor: Dict[str, Any],
    new_tensor_label: str,
) -> None:
    """Update ModelLog categories (internally_initialized, merge points) for a new tensor."""
    if fields_dict["initialized_inside_model"]:
        self.internally_initialized_layers.append(new_tensor_label)
    if fields_dict["has_input_ancestor"] and any(
        (
            self[parent_layer].has_internally_initialized_ancestor
            and not self[parent_layer].has_input_ancestor
        )
        for parent_layer in fields_dict_onetensor["parent_layers"]
    ):
        self._layers_where_internal_branches_merge_with_input.append(new_tensor_label)


def _tag_tensor_and_track_variations(
    self,
    out: torch.Tensor,
    new_layer_entry,
    fields_dict_onetensor: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
) -> None:
    """Tag the output tensor with its label, add backward hook, and track parent content variations.

    Parent content variation tracking detects in-place mutations: if a parent
    tensor's value at function-call time (from arg_copies) differs from its
    saved activation, the pre-mutation value is recorded in
    ``children_tensor_versions``.  This is critical for validation replay,
    which needs the actual input values each child operation saw.
    """
    out.tl_tensor_label_raw = fields_dict_onetensor["tensor_label_raw"]  # type: ignore[attr-defined]
    if self.save_gradients:
        _add_backward_hook(self, out, out.tl_tensor_label_raw)  # type: ignore[attr-defined]

    for parent_label in new_layer_entry.parent_layers:
        parent = self[parent_label]
        if parent.has_saved_activations and self.save_function_args:
            parent_tensor_contents = _get_parent_contents(
                parent_label,
                arg_copies,
                kwarg_copies,
                new_layer_entry.parent_layer_arg_locs,
            )
            if not tensor_nanequal(parent_tensor_contents, parent.tensor_contents):
                parent.children_tensor_versions[new_layer_entry.tensor_label_raw] = (
                    parent_tensor_contents
                )
                parent.has_child_tensor_variations = True


def log_function_output_tensors_exhaustive(
    self,
    func: Callable,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
):
    """Full metadata logging for each output tensor of a function call.

    For each loggable output tensor:
      1. Build per-tensor fields (label, shape, equivalence type, FLOPs).
      2. Create a LayerPassLog entry and optionally save activation data.
      3. Update bidirectional family links (parent→child, sibling, spouse).
      4. Tag the output tensor with ``tl_tensor_label_raw`` so downstream
         operations can identify it as a parent.
      5. Track parent content variations (for in-place mutation detection).

    Args:
        func: The original (unwrapped) function that was called.
        args: Positional arguments to the function.
        kwargs: Keyword arguments to the function.
        arg_copies: Pre-call copies of args (for child tensor variation tracking).
        kwarg_copies: Pre-call copies of kwargs.
        out_orig: Original output from the function (may be tensor, tuple, etc.).
        exec_ctx: Timing, RNG, and autocast state captured around the function call.
        is_bottom_level_func: True if this function was not called by another
            decorated function (i.e., it's a leaf in the decoration nesting).
    """
    fields_dict, parent_layer_entries, arg_tensors, parent_param_passes = _build_shared_fields_dict(
        self,
        func,
        func_name,
        args,
        kwargs,
        out_orig,
        exec_ctx,
    )

    out_iter = ensure_iterable(out_orig)

    for i, out in enumerate(out_iter):
        if not _output_should_be_logged(out, is_bottom_level_func):
            continue

        # Each output tensor gets its own fields_dict to avoid shared-mutation bugs.
        # Shallow-copy mutable containers (list, dict, set); immutable values
        # (str, int, bool, None, tuple, torch.dtype) are safely shared.
        fields_dict_onetensor = {}
        for key, value in fields_dict.items():
            if isinstance(value, (list, dict, set)):
                fields_dict_onetensor[key] = copy.copy(value)
            else:
                fields_dict_onetensor[key] = value
        # These nested structures need deep copies because they contain mutable
        # sub-containers that could be mutated independently per output tensor.
        for field in (
            "parent_layer_arg_locs",
            "containing_modules_origin_nested",
            "parent_param_passes",
        ):
            fields_dict_onetensor[field] = copy.deepcopy(fields_dict[field])
        _log_output_tensor_info(
            self, out, i, args, kwargs, parent_param_passes, fields_dict_onetensor
        )
        _make_layer_log_entry(
            self,
            out,
            fields_dict=fields_dict_onetensor,
            t_args=arg_copies,
            t_kwargs=kwarg_copies,
            activation_postfunc=self.activation_postfunc,
        )
        new_layer_entry = self[fields_dict_onetensor["tensor_label_raw"]]
        new_tensor_label = new_layer_entry.tensor_label_raw
        _update_tensor_family_links(self, new_layer_entry)

        _classify_new_tensor_in_model_log(
            self, fields_dict, fields_dict_onetensor, new_tensor_label
        )
        _tag_tensor_and_track_variations(
            self,
            out,
            new_layer_entry,
            fields_dict_onetensor,
            arg_copies,
            kwarg_copies,
        )


def _get_parent_contents(parent_label, arg_copies, kwarg_copies, parent_layer_arg_locs) -> Any:
    """Retrieve a parent tensor's pre-call value from the saved argument copies.

    Used for child tensor variation tracking: if a parent's value in arg_copies
    differs from its currently saved tensor_contents, the parent was mutated
    in-place between operations, and the variation is recorded.
    """
    for pos, label in parent_layer_arg_locs["args"].items():
        if label == parent_label:
            return index_nested(arg_copies, pos)
    for argname, label in parent_layer_arg_locs["kwargs"].items():
        if label == parent_label:
            return index_nested(kwarg_copies, argname)
    raise ValueError("Parent layer not found in function arguments.")


def log_function_output_tensors_fast(
    self,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
):
    """Fast-path logging: save new activation values into existing graph entries.

    Skips all metadata collection.  Instead:
      1. Increment counters identically to the exhaustive pass (counter alignment).
      2. Reconstruct the raw label from counters and verify it maps to the same
         final label as the exhaustive pass (graph-change detection).
      3. Save activation data and update shape/dtype/timing metadata.

    If any counter, label, or parent mismatch is detected, raises ValueError
    telling the user to re-run ``log_forward_pass``.
    """
    # Minimal info collection — only what's needed for counter alignment and saving.
    layer_type = func_name.lower().replace("_", "")
    non_tensor_args = [arg for arg in args if not _check_if_tensor_arg(arg)]
    non_tensor_kwargs = {key: val for key, val in kwargs.items() if not _check_if_tensor_arg(val)}

    # O(1) tensor extraction via lookup table (replaces BFS crawl)
    arg_tensors, _ = _extract_arg_tensors_and_params(layer_type, args, kwargs)
    out_iter = ensure_iterable(out_orig)

    for i, out in enumerate(out_iter):
        if not _output_should_be_logged(out, is_bottom_level_func):
            continue
        # Mirror the exhaustive pass's counter increments exactly.
        # The raw label reconstructed here MUST match the exhaustive pass's label.
        self._layer_counter += 1
        self._raw_layer_type_counter[layer_type] += 1
        realtime_tensor_num = self._layer_counter
        layer_type_num = self._raw_layer_type_counter[layer_type]
        tensor_label_raw = f"{layer_type}_{layer_type_num}_{realtime_tensor_num}_raw"
        # Skip orphans — these were pruned from the graph during postprocessing.
        if tensor_label_raw in self.orphan_layers:
            continue
        # Map parent raw labels → final labels for graph-change verification.
        parent_layer_labels_raw = get_attr_values_from_tensor_list(
            arg_tensors, "tl_tensor_label_raw"
        )
        parent_layer_labels_orig = []
        for raw_label in parent_layer_labels_raw:
            if raw_label in self._raw_to_final_layer_labels:
                parent_layer_labels_orig.append(self._raw_to_final_layer_labels[raw_label])
            elif raw_label not in self.orphan_layers:
                raise ValueError(
                    f"Fast-path parent {raw_label} not found in raw→final label map "
                    f"and not in orphan_layers. The computational graph may have changed."
                )
        # Tag tensor so downstream ops can find this tensor's label.
        out.tl_tensor_label_raw = tensor_label_raw
        if tensor_label_raw not in self._raw_to_final_layer_labels:
            raise ValueError(
                "The computational graph changed for this forward pass compared to the original "
                "call to log_forward_pass (either due to different inputs or a different "
                "random seed), so save_new_activations failed. Please re-run "
                "log_forward_pass with the desired inputs."
            )
        orig_tensor_label = self._raw_to_final_layer_labels[tensor_label_raw]
        if orig_tensor_label in self.unlogged_layers:
            continue
        orig_layer_entry = self.layer_dict_main_keys[orig_tensor_label]

        if self.save_gradients:
            _add_backward_hook(self, out, tensor_label_raw)  # Must pass RAW label (#86)

        # Structural integrity check: verify counter, type, label, and parents
        # all match the exhaustive pass.  Any mismatch means dynamic control flow
        # changed the graph and the fast pass cannot proceed.
        if (
            orig_layer_entry.realtime_tensor_num != self._layer_counter
            or orig_layer_entry.layer_type != layer_type
            or orig_layer_entry.tensor_label_raw != tensor_label_raw
            or set(orig_layer_entry.parent_layers) != set(parent_layer_labels_orig)
        ):
            raise ValueError(
                "The computational graph changed for this forward pass compared to the original "
                "call to log_forward_pass (either due to different inputs or a different "
                "random seed), so save_new_activations failed. Please re-run "
                "log_forward_pass with the desired inputs."
            )

        # Save activation data if this layer is in the save list.
        if (self._layer_nums_to_save == "all") or (
            orig_layer_entry.realtime_tensor_num in self._layer_nums_to_save
        ):
            self.layers_with_saved_activations.append(orig_layer_entry.layer_label)
            orig_layer_entry.save_tensor_data(
                out,
                arg_copies,
                kwarg_copies,
                self.save_function_args,
                self.activation_postfunc,
            )
            # Output layers are identity wrappers whose tensor_contents come
            # from their parent.  Propagate the parent's saved activation to
            # any child that is an output layer so postprocess_fast can find it.
            for child_layer in orig_layer_entry.child_layers:
                if child_layer in self.output_layers:
                    child_output = self.layer_dict_main_keys[child_layer]
                    if (
                        orig_layer_entry.has_child_tensor_variations
                        and child_layer in orig_layer_entry.children_tensor_versions
                    ):
                        # children_tensor_versions already has postfunc applied.
                        tensor_to_save = orig_layer_entry.children_tensor_versions[child_layer]
                        child_output.tensor_contents = safe_copy(tensor_to_save)
                    else:
                        child_output.tensor_contents = safe_copy(out)
                        if self.activation_postfunc is not None:
                            # pause_logging prevents activation_postfunc from
                            # triggering decorated torch ops that would be logged.
                            with pause_logging():
                                child_output.tensor_contents = self.activation_postfunc(
                                    child_output.tensor_contents
                                )
                    child_output.has_saved_activations = True
                    child_output.tensor_fsize = get_tensor_memory_amount(
                        child_output.tensor_contents
                    )

        # Update lightweight metadata that may vary across inputs
        # (shape can differ for dynamic-shape models that still share graph structure).
        orig_layer_entry.tensor_shape = tuple(out.shape)
        orig_layer_entry.tensor_dtype = out.dtype
        orig_layer_entry.tensor_fsize = get_tensor_memory_amount(out)
        orig_layer_entry.func_time_elapsed = exec_ctx.time_elapsed
        orig_layer_entry.func_rng_states = exec_ctx.rng_states
        orig_layer_entry.func_autocast_state = exec_ctx.autocast_state
        orig_layer_entry.func_position_args_non_tensor = non_tensor_args
        orig_layer_entry.func_keyword_args_non_tensor = non_tensor_kwargs

        # Update func_config — some may be input-dependent (e.g. interpolate size).
        orig_layer_entry.func_config = extract_salient_args(
            layer_type,
            func_name,
            args,
            kwargs,
            orig_layer_entry.parent_param_shapes,
        )


def _output_should_be_logged(out: Any, is_bottom_level_func: bool) -> bool:
    """Determine whether an output value should be logged as a new graph node.

    Two conditions must hold:
      1. ``out`` must be a torch.Tensor (non-tensor outputs like ints are skipped).
      2. Either the tensor is genuinely new (no ``tl_tensor_label_raw`` attribute),
         OR this is a bottom-level function.  Bottom-level functions are leaf
         operations in the decoration nesting — even if they return an already-
         labeled tensor (in-place ops), we log them to capture the operation.
         Non-bottom-level functions returning an already-labeled tensor are
         higher-level wrappers whose sub-operations were already logged.

    Returns:
        True if the output should be logged, False otherwise.
    """
    if type(out) is not torch.Tensor:
        return False

    if (not hasattr(out, "tl_tensor_label_raw")) or is_bottom_level_func:
        return True
    else:
        return False


def _check_if_tensor_arg(arg: Any) -> bool:
    """Helper function to check if an argument either is a tensor or is a list/tuple containing a tensor.

    Args:
        arg: argument

    Returns:
        True if arg is or contains a tensor, false otherwise
    """
    if issubclass(type(arg), torch.Tensor):
        return True
    elif type(arg) in [list, tuple]:
        for elt in arg:
            if issubclass(type(elt), torch.Tensor):
                return True
        return False
    elif type(arg) == dict:
        for val in arg.values():
            if issubclass(type(val), torch.Tensor):
                return True
        return False
    else:
        return False


def _log_output_tensor_info(
    self,
    t: torch.Tensor,
    i: int,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    parent_param_passes: Dict[str, int],
    fields_dict: Dict[str, Any],
) -> None:
    """Populate per-tensor fields that differ across outputs of a single function call.

    This includes:
      - Counter-based label generation (``tensor_label_raw``).
      - Operation equivalence type assignment (used by loop detection to
        identify structurally identical operations across forward-pass iterations).
      - FLOPs computation.
      - Shape, dtype, and memory size.

    Label format: ``"{layer_type}_{type_num}_{realtime_num}_raw"``
      - ``layer_type``: normalized function name (e.g. "conv2d")
      - ``type_num``: how many times this layer_type has been seen (monotonic)
      - ``realtime_num``: global operation counter across all types (monotonic)

    Args:
        t: The output tensor.
        i: Index of this tensor in a multi-output function call (0 for single outputs).
        args: Positional args to the function that created the tensor.
        kwargs: Keyword args to the function.
        parent_param_passes: Dict mapping param barcodes to their current pass number.
        fields_dict: Per-tensor fields dict to populate (mutated in place).
    """
    layer_type = fields_dict["layer_type"]
    indiv_param_barcodes = list(parent_param_passes.keys())
    self._layer_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    realtime_tensor_num = self._layer_counter
    layer_type_num = self._raw_layer_type_counter[layer_type]
    tensor_label_raw = f"{layer_type}_{layer_type_num}_{realtime_tensor_num}_raw"

    # Determine operation equivalence type — the fingerprint used by loop detection
    # to group structurally identical operations (same layer across passes).
    if len(parent_param_passes) > 0:
        # Parameterized ops: equivalence is defined by the exact set of parameters
        # used, combined with the operation type.  E.g., two conv2d calls using the
        # same weight+bias tensors are the same layer on different passes.
        operation_equivalence_type = _make_raw_param_group_barcode(indiv_param_barcodes, layer_type)
        fields_dict["operation_equivalence_type"] = operation_equivalence_type
        self.layers_computed_with_params[operation_equivalence_type].append(tensor_label_raw)
        fields_dict["pass_num"] = len(self.layers_computed_with_params[operation_equivalence_type])
    else:
        # Non-parameterized ops: equivalence is a hash of the operation type,
        # non-tensor args, output index, and containing module.  Each unique
        # non-param operation is seen only once (pass_num=1).
        operation_equivalence_type = _get_operation_equivalence_type(
            args, kwargs, i, layer_type, fields_dict
        )
        fields_dict["operation_equivalence_type"] = operation_equivalence_type
        fields_dict["pass_num"] = 1

    # equivalent_operations is a DIRECT reference to the ModelLog-level set —
    # all entries sharing this equivalence type point to the same set object.
    self.equivalent_operations[operation_equivalence_type].add(tensor_label_raw)
    fields_dict["equivalent_operations"] = self.equivalent_operations[operation_equivalence_type]

    # In-place ops return the same tensor object, which already has tl_tensor_label_raw.
    fields_dict["function_is_inplace"] = hasattr(t, "tl_tensor_label_raw")
    fields_dict["gradfunc"] = type(t.grad_fn).__name__

    if fields_dict["is_part_of_iterable_output"]:
        fields_dict["iterable_output_index"] = i
    else:
        fields_dict["iterable_output_index"] = None

    if (t.dtype == torch.bool) and (t.dim()) == 0:
        fields_dict["is_atomic_bool_layer"] = True
        try:
            fields_dict["atomic_bool_val"] = t.item()
        except RuntimeError:
            # .item() forbidden inside torch.vmap context
            fields_dict["atomic_bool_val"] = None
    else:
        fields_dict["is_atomic_bool_layer"] = False
        fields_dict["atomic_bool_val"] = None

    # General info
    fields_dict["tensor_label_raw"] = tensor_label_raw
    fields_dict["layer_total_num"] = None
    fields_dict["same_layer_operations"] = []
    fields_dict["realtime_tensor_num"] = realtime_tensor_num
    fields_dict["operation_num"] = None
    fields_dict["source_model_log"] = self
    fields_dict["_pass_finished"] = False

    # Other labeling info
    fields_dict["layer_label"] = None
    fields_dict["layer_label_short"] = None
    fields_dict["layer_label_w_pass"] = None
    fields_dict["layer_label_w_pass_short"] = None
    fields_dict["layer_label_no_pass"] = None
    fields_dict["layer_label_no_pass_short"] = None
    fields_dict["layer_type"] = layer_type
    fields_dict["layer_label_raw"] = tensor_label_raw
    fields_dict["layer_type_num"] = layer_type_num
    fields_dict["layer_passes_total"] = 1
    fields_dict["lookup_keys"] = []

    # Saved tensor info
    fields_dict["tensor_contents"] = None
    fields_dict["has_saved_activations"] = False
    fields_dict["activation_postfunc"] = self.activation_postfunc
    fields_dict["function_args_saved"] = False
    fields_dict["creation_args"] = None
    fields_dict["creation_kwargs"] = None
    fields_dict["tensor_shape"] = tuple(t.shape)
    fields_dict["tensor_dtype"] = t.dtype
    with pause_logging():
        fields_dict["tensor_fsize"] = t.nelement() * t.element_size()

    # FLOPs computation
    fields_dict["flops_forward"] = compute_forward_flops(
        fields_dict.get("func_applied_name"),  # type: ignore[arg-type]
        fields_dict["tensor_shape"],
        fields_dict.get("parent_param_shapes", []),
        args,
        kwargs,
    )
    fields_dict["flops_backward"] = compute_backward_flops(
        fields_dict.get("func_applied_name"),  # type: ignore[arg-type]
        fields_dict["flops_forward"],
    )

    # Child tensor variation tracking
    fields_dict["has_child_tensor_variations"] = False
    fields_dict["children_tensor_versions"] = {}

    # If internally initialized, fix this information:
    if len(fields_dict["parent_layers"]) == 0:
        fields_dict["initialized_inside_model"] = True
        fields_dict["has_internally_initialized_ancestor"] = True
        fields_dict["internally_initialized_parents"] = []
        fields_dict["internally_initialized_ancestors"] = {tensor_label_raw}


def _make_layer_log_entry(
    self,
    t: torch.Tensor,
    fields_dict: Dict,
    t_args: Optional[Tuple] = None,
    t_kwargs: Optional[Dict] = None,
    activation_postfunc: Optional[Callable] = None,
):
    """Create a LayerPassLog (or BufferLog) entry and register it in ModelLog.

    Instantiates the appropriate log class from ``fields_dict``, conditionally
    saves activation data (if this layer is in ``_layer_nums_to_save``), and
    appends the entry to ``_raw_layer_dict`` and ``_raw_layer_labels_list``.

    Args:
        t: The tensor to log.
        fields_dict: Complete field dictionary (~80 fields) for the log entry.
        t_args: Positional arguments to the function that created the tensor.
        t_kwargs: Keyword arguments to the function that created the tensor.
        activation_postfunc: Optional transform applied to activations before saving.
    """
    if t_args is None:
        t_args = []  # type: ignore[assignment]
    if t_kwargs is None:
        t_kwargs = {}

    if fields_dict.get("is_buffer_layer"):
        new_entry = BufferLog(fields_dict)
    else:
        new_entry = LayerPassLog(fields_dict)  # type: ignore[assignment]
    if (self._layer_nums_to_save == "all") or (
        new_entry.realtime_tensor_num in self._layer_nums_to_save
    ):
        new_entry.save_tensor_data(
            t,
            t_args,  # type: ignore[arg-type]
            t_kwargs,
            self.save_function_args,
            activation_postfunc,
        )
        self.layers_with_saved_activations.append(new_entry.tensor_label_raw)
    self._raw_layer_dict[new_entry.tensor_label_raw] = new_entry
    self._raw_layer_labels_list.append(new_entry.tensor_label_raw)

    return new_entry
