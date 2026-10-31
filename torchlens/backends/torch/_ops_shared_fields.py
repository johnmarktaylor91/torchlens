"""Shared operation fields and tensor variation tracking."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from ... import _state as _st
from ...capture.arg_positions import (
    _normalize_func_name,
)
from ...capture.projections import LiveOpView
from ...capture.salient_args import extract_salient_args
from ...data_classes.internal_types import FuncExecutionContext
from ...data_classes.op import (
    Op,
)
from ...ir.events import (
    OutputVersionEvent,
)
from ...utils.introspection import (
    _get_code_context,
    get_arg_tensors_for_resolution,
)
from ...utils.tensor_utils import (
    tensor_nanequal,
)
from ._tl import (
    get_live_label_list,
    get_tensor_label,
    set_tensor_label,
)
from .aliasing import (
    get_parent_contents_for_contract_position,
    parent_label_has_alias_contract,
)
from .buffer_writes import resolve_registered_buffer_address, session_validated_buffer_address
from .completeness_witness import internal_scalar_read
from .sources import log_source_tensor
from .tensor_tracking import (
    _add_tensor_backward_hook,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

if TYPE_CHECKING:
    from .ops import (
        TRANSFORM_FUNC_NAMES,
        _bind_writeback_receiver_label,
        _build_args_template,
        _build_graph_relationship_fields,
        _build_module_context_fields,
        _build_param_fields,
        _check_if_tensor_arg,
        _extract_arg_tensors_and_params,
        _record_label_version_snapshot,
        _unattributed_tensor_arg_positions,
    )

__all__ = (
    "_build_shared_fields_dict",
    "_classify_new_tensor_in_trace",
    "_tag_tensor_and_track_variations",
    "_get_parent_output_version_snapshot",
)


def _build_shared_fields_dict(
    self: "Trace",
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    func_call_id: int,
) -> tuple[dict[str, Any], list[Op], list[torch.Tensor], dict[str, int]]:
    """Build the fields_dict shared by all output tensors of a single function call.

    When a function produces multiple output tensors (e.g. ``torch.split``),
    many metadata fields are identical across outputs (function info, parent
    relationships, module context).  This function computes those shared fields
    once; per-tensor fields (shape, label, equivalence type) are added later
    in ``_log_output_tensor_info``.

    Returns:
        (fields_dict, parent_layer_entries, arg_tensors, parent_param_ops)
    """
    _bind_writeback_receiver_label(func_name, args)
    # Canonical layer_type: lowercase with underscores stripped (e.g. "conv2d").
    layer_type = _normalize_func_name(func_name)

    # O(1) tensor/param extraction via lookup table (replaces BFS crawl)
    arg_tensors, arg_parameters = _extract_arg_tensors_and_params(layer_type, args, kwargs)
    tensors_to_resolve = get_arg_tensors_for_resolution(args, kwargs)
    for tensor in tensors_to_resolve:
        if isinstance(tensor, torch.nn.Parameter) or get_tensor_label(tensor) is not None:
            continue
        # r81: never source-log a "buffer" off the raw static stamp -- a stale
        # cross-capture stamp (F1) or a legitimately stamped receiver whose
        # storage was rebound to input data mid-forward (F2) would re-root an
        # input-derived value as internal state. The session belt validates
        # object + storage identity; the tracker fallback is storage-anchored.
        buffer_address = session_validated_buffer_address(self, tensor)
        if buffer_address is None:
            buffer_address = resolve_registered_buffer_address(self, tensor)
        if buffer_address is not None:
            log_source_tensor(self, tensor, "buffer", buffer_address)

    # Separate tensor args (which define graph edges) from non-tensor args
    # (which become metadata and feed into equivalence_class hashing).
    non_tensor_args = [arg for arg in args if not _check_if_tensor_arg(arg)]
    non_tensor_kwargs = {key: val for key, val in kwargs.items() if not _check_if_tensor_arg(val)}
    parent_layer_labels = get_live_label_list(
        arg_tensors, self.capture_events.live_index.by_raw_label
    )
    parent_layer_entries = [
        cast(Op, LiveOpView(self, self.capture_events.live_index.require_event(label)))
        for label in parent_layer_labels
    ]

    fields_dict: dict[str, Any] = {}

    # General info
    fields_dict["type"] = layer_type
    fields_dict["detach_saved_activations"] = self.detach_saved_activations
    fields_dict["output_device"] = self.output_device
    fields_dict["_construction_done"] = False
    fields_dict["interventions"] = []
    should_capture_template = bool(
        getattr(self, "intervention_ready", False) or getattr(self, "save_arg_templates", False)
    )
    if should_capture_template:
        captured_template = _build_args_template(func, args, kwargs, self, func_name=func_name)
        fields_dict["args_template"] = captured_template
        fields_dict["kwargs_template"] = captured_template if kwargs else None
        fields_dict["func_id"] = captured_template.func_id
    else:
        fields_dict["args_template"] = None
        fields_dict["kwargs_template"] = None
        fields_dict["func_id"] = None
    fields_dict["container_path"] = ()
    fields_dict["container_spec"] = None
    fields_dict["multi_output_name"] = None

    # Grad info
    fields_dict["grad"] = None
    fields_dict["transformed_grad"] = None
    fields_dict["save_grads"] = getattr(self, "save_grads", None) not in (None, False)
    fields_dict["has_grad"] = False
    fields_dict["grad_shape"] = None
    fields_dict["transformed_grad_shape"] = None
    fields_dict["grad_dtype"] = None
    fields_dict["transformed_grad_dtype"] = None
    fields_dict["gradient_memory"] = 0
    fields_dict["transformed_gradient_memory"] = None

    # Function call info
    fields_dict["func"] = func
    fields_dict["func_call_id"] = func_call_id
    fields_dict["func_name"] = func_name
    fields_dict["func_qualname"] = getattr(func, "__qualname__", None)
    code_context_cache = getattr(self, "_code_context_cache", None)
    if code_context_cache is None:
        code_context_cache = {}
        self._code_context_cache = code_context_cache
    call_site_out: list[Any] = []
    fields_dict["code_context"] = _get_code_context(
        self.num_context_lines,
        source_loading_enabled=self.save_code_context,
        disable_col_offset=False,
        context_cache=code_context_cache,
        call_site_out=call_site_out,
    )
    self._raw_graph_ws._pending_call_site = (func_call_id, call_site_out[0])
    fields_dict["var_names"] = []
    fields_dict["func_duration"] = exec_ctx.time_elapsed
    fields_dict["func_rng_states"] = exec_ctx.rng_states
    fields_dict["func_autocast_state"] = exec_ctx.autocast_state
    # Exact-name lookup first (add / add_ / __add__ have distinct signatures --
    # W3 F9). The underscore-stripped key remains a best-effort fallback for
    # NON-dunder names whose own introspection stored nothing; a dunder whose
    # signature is opaque stays honestly unknown (empty) rather than borrowing
    # the namesake torch function's different signature.
    _op_arg_names = _st._arg_names.get(func_name)
    if _op_arg_names is None:
        if func_name.startswith("__"):
            _op_arg_names = ()
        else:
            _op_arg_names = _st._arg_names.get(func_name.strip("_"), ())
    fields_dict["arg_names"] = _op_arg_names
    fields_dict["num_args_total"] = len(args) + len(kwargs)
    fields_dict["num_pos_args"] = len(args)
    fields_dict["num_kwargs"] = len(kwargs)
    fields_dict["non_tensor_pos_args"] = non_tensor_args
    fields_dict["non_tensor_kwargs"] = non_tensor_kwargs
    fields_dict["func_non_tensor_args"] = non_tensor_args + list(non_tensor_kwargs.values())

    _build_graph_relationship_fields(
        self, fields_dict, parent_layer_labels, parent_layer_entries, args, kwargs, out_orig
    )
    parent_param_ops = _build_param_fields(self, fields_dict, arg_parameters)
    _build_module_context_fields(self, fields_dict, arg_tensors, parent_layer_entries)
    is_transform = bool(getattr(func, "__tl_is_transform_boundary__", False)) or (
        func_name in TRANSFORM_FUNC_NAMES
    )
    fields_dict["is_transform"] = is_transform
    fields_dict["transform_kind"] = getattr(func, "__tl_transform_kind__", None) or (
        func_name if is_transform else None
    )
    fields_dict["transform_chain"] = tuple(getattr(func, "__tl_transform_tags__", ()))
    fields_dict["transform_config"] = dict(getattr(func, "__tl_transform_config__", {}))
    fields_dict["transform_fn_name"] = getattr(func, "__tl_transform_fn_name__", None)
    fields_dict["transform_fn_qualname"] = getattr(func, "__tl_transform_fn_qualname__", None)
    fields_dict["transform_fn_source"] = getattr(func, "__tl_transform_fn_source__", None)
    (
        fields_dict["unattributed_tensor_args"],
        fields_dict["dropped_edge_tensor_args"],
    ) = _unattributed_tensor_arg_positions(
        self,
        args,
        kwargs,
        func_name,
        fields_dict["parent_arg_positions"],
        fields_dict["parent_params"],
    )

    # Function config — lightweight hyperparameter extraction, always on.
    param_shapes = cast(list[tuple[int, ...]] | None, fields_dict.get("param_shapes"))
    fields_dict["func_config"] = extract_salient_args(
        layer_type,
        func_name,
        args,
        kwargs,
        param_shapes,
    )

    return fields_dict, parent_layer_entries, arg_tensors, parent_param_ops


def _classify_new_tensor_in_trace(
    self: "Trace",
    fields_dict: dict[str, Any],
    new_tensor_label: str,
) -> None:
    """Update Trace categories for a new tensor.

    Args:
        self: Trace object being populated.
        fields_dict: Shared field values for this wrapped output.
        new_tensor_label: Raw label for the new tensor op.
    """
    if fields_dict["is_internal_source"]:
        self.internal_source_ops.append(new_tensor_label)


def _tag_tensor_and_track_variations(
    self: "Trace",
    out: torch.Tensor,
    new_layer_entry: Op,
    fields_dict_onetensor: dict[str, Any],
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
) -> None:
    """Tag the output tensor with its label, add backward hook, and track parent content variations.

    Parent content variation tracking detects in-place mutations: if a parent
    tensor's value at function-call time (from arg_copies) differs from its
    saved out, the pre-mutation value is recorded in
    ``out_versions_by_child``.  This is critical for validation replay,
    which needs the actual input values each child operation saw.
    """
    out_label = fields_dict_onetensor["_label_raw"]
    set_tensor_label(out, out_label)
    # Record the output's version at label time so a later op consuming this tensor can
    # distinguish a genuine in-place mutation (version bumped since it was labeled) from a
    # non-mutating identity return that reuses the same object (see is_inplace at capture).
    _record_label_version_snapshot(out)
    _add_tensor_backward_hook(self, out, out_label)

    child_event = self.capture_events.live_index.require_event(new_layer_entry._label_raw)
    for parent_label in new_layer_entry.parents:
        parent_event = self.capture_events.live_index.require_event(parent_label)
        contract = child_event.backend_semantics
        parent_tensor_contents = _get_parent_output_version_snapshot(
            self,
            parent_label,
            new_layer_entry.parent_arg_positions,
            contract.mutated_input_positions,
            contract.aliased_output_inputs,
            parent_event.output.has_saved_activation,
            parent_event.output.tensor.payload,
            arg_copies,
            kwarg_copies,
        )
        if parent_tensor_contents is not None:
            self.capture_events.append_output_version(
                OutputVersionEvent(
                    parent_raw_label=parent_label,
                    child_raw_label=new_layer_entry._label_raw,
                    child_output_path=tuple(fields_dict_onetensor["container_path"]),
                    payload=parent_tensor_contents,
                    transform_state=fields_dict_onetensor["activation_transform"],
                    detach_grad_policy=fields_dict_onetensor["detach_saved_activations"],
                )
            )


def _get_parent_output_version_snapshot(
    self: "Trace",
    parent_label: str,
    parent_arg_positions: dict[str, dict[Any, str]],
    mutated_input_positions: tuple[object, ...],
    aliased_output_inputs: tuple[object, ...],
    parent_has_saved_activation: bool,
    parent_saved_output: Any,
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
) -> Any | None:
    """Return the pre-call parent snapshot needed for child-version replay.

    Parameters
    ----------
    self
        Active trace.
    parent_label
        Parent label in the same label space as ``parent_arg_positions``.
    parent_arg_positions
        Mapping from argument positions to parent labels.
    mutated_input_positions
        Input positions the backend contract says may be mutated.
    aliased_output_inputs
        Input positions the backend contract says may alias the output.
    parent_has_saved_activation
        Whether the parent currently has a saved activation payload.
    parent_saved_output
        Parent's currently saved activation payload, if any.
    arg_copies
        Pre-call positional argument copies.
    kwarg_copies
        Pre-call keyword argument copies.

    Returns
    -------
    Any | None
        Pre-call parent value when replay needs a child-specific version;
        otherwise ``None``.
    """

    # Cheap gate first: without save_arg_values this function always returns
    # None, so the per-parent contract scan below was pure waste on the
    # default capture path.
    if not self.save_arg_values:
        return None
    contract_positions = tuple(mutated_input_positions) + tuple(aliased_output_inputs)
    should_snapshot_by_contract = parent_label_has_alias_contract(
        parent_label,
        parent_arg_positions,
        contract_positions,
    )
    should_snapshot_by_value = parent_has_saved_activation
    if not (should_snapshot_by_contract or should_snapshot_by_value):
        return None

    parent_tensor_contents = get_parent_contents_for_contract_position(
        parent_label,
        arg_copies,
        kwarg_copies,
        parent_arg_positions,
    )
    # ``tensor_nanequal`` compares captured content via ``torch.equal`` /
    # ``torch.allclose`` (Python ``bool`` output); mark it as a capture-internal read so
    # the completeness witness never records it as a user host escape. The
    # ``should_snapshot_by_contract`` short-circuit is preserved so ``tensor_nanequal`` is
    # never called with a missing ``parent_saved_output``.
    with internal_scalar_read():
        should_snapshot = should_snapshot_by_contract or not tensor_nanequal(
            parent_tensor_contents,
            parent_saved_output,
        )
    if should_snapshot:
        return parent_tensor_contents
    return None
