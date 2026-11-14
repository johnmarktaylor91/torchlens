"""Shared operation fields and tensor variation tracking."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ..._data_classes.internal_types import FuncExecutionContext
from ..._ir.events import (
    OutputVersionEvent,
)
from ..._ir.predicate import RecordContext
from ..._ir.refs import ParamRef
from ..._utils._tensor_label import set_tensor_label
from ..._utils.tensor_utils import (
    tensor_nanequal,
)
from ._call_layout import planned_slot_call, split_call_arguments
from ._completeness_origins import internal_scalar_read
from ._ops_function_family import NAME_FAMILY, name_family
from ._storage_exports import note_alias_rebound_snapshot, note_foreign_written_snapshot
from ._thin_defer import set_function_config
from ._thin_settle import SHARED_CALL_DEFAULTS, settle_non_tensor_args
from .aliasing import (
    get_parent_contents_for_contract_position,
    parent_label_has_alias_contract,
)
from .tensor_tracking import (
    _add_tensor_backward_hook,
    _append_module_suffix_to_equivalence_class,
    _get_equivalence_class,
    _make_raw_param_group_barcode,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


__all__ = (
    "_build_shared_fields_dict",
    "_classify_new_tensor_in_trace",
    "_tag_tensor_and_track_variations",
    "_get_parent_output_version_snapshot",
    "_predicate_equivalence_class",
    "_prepared_param_refs",
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
    defer: bool = False,
) -> tuple[dict[str, Any], list[Any], list[torch.Tensor], dict[str, int]]:
    """Build the fields shared by every output tensor of one call (function, parents, module).

    Per-tensor fields are added in ``_log_output_tensor_info``; each family has its own module.
    Returns ``(fields_dict, parent_records, arg_tensors, parent_param_ops)``. ``defer`` (the
    thin emit) leaves the settle families out: the defaults, the callable's descriptors and
    ``func_non_tensor_args``, which ``_thin_record`` fills at settle with the same builders.
    """
    # Canonical layer_type: lowercase with underscores stripped (e.g. "conv2d").
    layer_type = (NAME_FAMILY.get(func_name) or name_family(func_name))[0]
    # Non-tensor args (metadata, equivalence hashing) are op-time snapshots.
    call = planned_slot_call(self, func, func_name, args, kwargs)
    non_tensor_args, non_tensor_kwargs = split_call_arguments(call, _check_if_tensor_arg)
    bound_aliases = _bind_call_alias_labels(args, kwargs) if _VIEW_MUTATION_RELABELS else ()
    try:
        arg_tensors, arg_parameters, parent_layer_labels = _scan_call_arguments(
            self, call, layer_type
        )
        require_event = self.capture_events.live_index.require_event
        parent_records = [require_event(label) for label in parent_layer_labels]

        # General info; the call-wide constants come from the defaults table.
        fields_dict: dict[str, Any] = {"type": layer_type}
        fields_dict["detach_saved_activations"] = self._options_detach_saved_activations
        fields_dict["output_device"] = self._options_output_device
        if not defer:
            fields_dict.update(SHARED_CALL_DEFAULTS)
        _set_arg_template_fields(self, fields_dict, call, defer)
        # Whether this op saves gradients; the per-gradient placeholders come from the op
        # defaults (the freeze never reads them).
        fields_dict["save_grads"] = self._options_save_grads not in (None, False)

        # Function call info
        fields_dict["func"] = func
        fields_dict["func_call_id"] = func_call_id
        fields_dict["func_name"] = func_name
        _set_call_site_fields(self, fields_dict, func_call_id)
        fields_dict["func_duration"] = exec_ctx.time_elapsed
        fields_dict["func_rng_states"] = exec_ctx.rng_states
        fields_dict["func_autocast_state"] = exec_ctx.autocast_state
        num_pos_args = len(args)
        num_kwargs = len(kwargs)
        fields_dict["num_args_total"] = num_pos_args + num_kwargs
        fields_dict["num_pos_args"] = num_pos_args
        fields_dict["num_kwargs"] = num_kwargs
        fields_dict["non_tensor_pos_args"] = non_tensor_args
        fields_dict["non_tensor_kwargs"] = non_tensor_kwargs
        if not defer:
            fields_dict["func_non_tensor_args"] = settle_non_tensor_args(fields_dict)

        _build_graph_relationship_fields(
            self, fields_dict, parent_layer_labels, parent_records, args, kwargs, out_orig
        )
        parent_param_ops = _build_param_fields(self, fields_dict, arg_parameters)
        # Module nesting; the module-I/O placeholders are not written: the freeze never
        # reads them and postprocess derives them from the records.
        modules = _snapshot_exhaustive_module_stack(self)
        fields_dict["module"] = modules[-1] if modules else None
        fields_dict["modules"] = modules
        if not defer:
            _set_descriptive_fields(fields_dict, func, func_name)
        positions, params = fields_dict["parent_arg_positions"], fields_dict["parent_params"]
        unattributed, dropped = witness_from_slots(self, call, positions, params)
        fields_dict["unattributed_tensor_args"] = unattributed
        fields_dict["dropped_edge_tensor_args"] = dropped
        # Function config: lightweight hyperparameter extraction, always on.
        set_function_config(fields_dict, (layer_type, func_name, args, kwargs), defer)

        return fields_dict, parent_records, arg_tensors, parent_param_ops
    finally:
        if bound_aliases:
            _restore_call_alias_labels(bound_aliases)


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
    child_record: Any,
    fields_dict_onetensor: dict[str, Any],
    arg_copies: tuple[Any, ...],
    kwarg_copies: dict[str, Any],
    live_args: tuple[Any, ...],
    live_kwargs: dict[str, Any],
) -> None:
    """Tag the output tensor with its label, add backward hook, and track parent content variations.

    Parent content variation tracking detects in-place mutations: if a parent tensor's
    value at call time (from arg_copies) differs from its saved out, the pre-mutation
    value is recorded in ``out_versions_by_child`` for validation replay; ``live_args``
    and ``live_kwargs`` feed its foreign-write marks. ``child_record`` is the op's
    committed journal record, read with the expressions the live view's getters use.
    """
    out_label = fields_dict_onetensor["_label_raw"]
    set_tensor_label(out, out_label)
    # Record the output's version at label time so a later op consuming this tensor can
    # distinguish a genuine in-place mutation (version bumped since it was labeled) from a
    # non-mutating identity return that reuses the same object (see is_inplace at capture).
    _record_label_version_snapshot(out)
    _add_tensor_backward_hook(self, out, out_label)
    if not self._options_save_args:
        # Every parent's version snapshot is None without saved arguments (its first
        # test), so the loop appends nothing; its record reads are skipped.
        return

    require_event = self.capture_events.live_index.require_event
    child_event = require_event(child_record.label_raw)
    for parent_label in [edge.parent_label_raw for edge in child_record.parents]:
        parent_event = require_event(parent_label)
        contract = child_event.backend_semantics
        parent_tensor_contents = _get_parent_output_version_snapshot(
            self,
            parent_label,
            child_record.parent_arg_positions,
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
                    child_raw_label=child_record.label_raw,
                    child_output_path=tuple(fields_dict_onetensor["container_path"]),
                    payload=parent_tensor_contents,
                    transform_state=fields_dict_onetensor["activation_transform"],
                    detach_grad_policy=fields_dict_onetensor["detach_saved_activations"],
                )
            )
            _mark_foreign_written_snapshot(
                self, parent_label, child_record, contract, (live_args, live_kwargs)
            )


def _mark_foreign_written_snapshot(
    self: "Trace",
    parent_label: str,
    child_record: Any,
    contract: Any,
    live_call: tuple[tuple[Any, ...], dict[str, Any]],
) -> None:
    """Mark a stored parent snapshot as foreign-written unless the child's alias contract asked for it.

    Runs only after a snapshot was stored, so the live-call tuple is never built on the
    ordinary per-op path (the per-op pins in tests/test_capture_row_commit_cost.py).
    """
    positions = child_record.parent_arg_positions
    contract_positions = (*contract.mutated_input_positions, *contract.aliased_output_inputs)
    if not parent_label_has_alias_contract(parent_label, positions, contract_positions):
        note_foreign_written_snapshot(
            self, parent_label, child_record.label_raw, positions, live_call
        )
        if _VIEW_MUTATION_RELABELS:
            note_alias_rebound_snapshot(
                self,
                (parent_label, child_record.label_raw),
                positions,
                live_call,
                _VIEW_MUTATION_RELABELS,
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
    if not self._options_save_args:
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


def _prepared_param_refs(
    state: Any,
    arg_parameters: list[torch.nn.Parameter],
) -> tuple[ParamRef, ...]:
    """Resolve the Parameters an op read to their prep-time identities.

    Parameters
    ----------
    state
        Active recording state holding ``prepared_param_refs``.
    arg_parameters
        Parameters found among the call's arguments.

    Returns
    -------
    tuple[ParamRef, ...]
        One ref per prepared parameter occurrence, in argument order. A
        Parameter the prep never saw (created inside the forward) has no
        identity and is skipped, as exhaustive capture skips unstamped ones.
    """

    if not arg_parameters:
        return ()
    registry = state.prepared_param_refs
    if not registry:
        return ()
    return tuple(ref for param in arg_parameters if (ref := registry.get(id(param))) is not None)


def _predicate_equivalence_class(
    state: Any,
    ctx: RecordContext,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    param_refs: tuple[ParamRef, ...],
    *,
    func_name: str,
    layer_type: str,
    output_index: int,
    in_multi_output: bool,
    is_inplace_dunder: Callable[[str], bool],
) -> str:
    """Build the loop-grouping fingerprint of one predicate-mode op output.

    The same two fingerprints exhaustive capture mints (``_ops_activations``):
    parameterized ops key on ``layer_type`` plus the sorted parameter barcodes,
    parameter-free ops on ``layer_type`` plus a hash of the non-tensor
    arguments, both suffixed with the output slot of a multi-output call and
    the module stack.
    """

    _frames, modules = state.module_fields_for(ctx)
    if param_refs:
        barcodes = [ref.barcode for ref in param_refs]
        base = _make_raw_param_group_barcode(
            barcodes, layer_type, output_index=output_index if in_multi_output else None
        )
        return _append_module_suffix_to_equivalence_class(base, modules)
    is_inplace_output = is_inplace_dunder(func_name) or (
        func_name.endswith("_") and not func_name.startswith("__")
    )
    equivalence_layer_type = f"{layer_type}_inplace" if is_inplace_output else layer_type
    base = _get_equivalence_class(
        args,
        kwargs,
        output_index,
        equivalence_layer_type,
        {"in_multi_output": in_multi_output, "module": modules[-1] if modules else None},
    )
    return _append_module_suffix_to_equivalence_class(base, modules)


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._ops_argument_fields import (  # noqa: E402
    _scan_call_arguments,
    _set_arg_template_fields,
)
from ._ops_call_site_fields import (  # noqa: E402
    _set_call_site_fields,
)
from ._ops_capture_records import (  # noqa: E402
    _bind_call_alias_labels,
    _record_label_version_snapshot,
    _restore_call_alias_labels,
    _snapshot_exhaustive_module_stack,
)
from ._ops_descriptive_fields import (  # noqa: E402
    _set_descriptive_fields,
)
from ._ops_emission import (  # noqa: E402
    witness_from_slots,
)
from ._ops_exhaustive import (  # noqa: E402
    _check_if_tensor_arg,
)
from ._ops_predicate_events import (  # noqa: E402
    _build_graph_relationship_fields,
    _build_param_fields,
)
from .ops import (  # noqa: E402
    _VIEW_MUTATION_RELABELS,
)
