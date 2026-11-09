"""Shared operation fields and tensor variation tracking."""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from ... import _state as _st
from ..._capture._arg_snapshot import snapshot_mutable_arg, split_non_tensor_args
from ..._capture.arg_positions import (
    _normalize_func_name,
)
from ..._capture.projections import LiveOpView
from ..._capture.salient_args import extract_salient_args
from ..._data_classes.internal_types import FuncExecutionContext
from ..._data_classes.op import (
    Op,
)
from ..._ir.events import (
    OutputVersionEvent,
)
from ..._ir.predicate import RecordContext
from ..._ir.refs import ParamRef
from ..._utils._tensor_label import (
    clear_tensor_label,
    get_tensor_label,
    get_tensor_meta,
    mutated_parameter_label,
    raw_tensor_label,
    session_meta_is_anchored,
    set_tensor_label,
)
from ..._utils._tl_collective import _async_collective_elem
from ..._utils.introspection import (
    _get_code_context,
    get_arg_tensors_for_resolution,
)
from ..._utils.tensor_utils import (
    tensor_nanequal,
)
from ._completeness_origins import internal_scalar_read
from ._ops_arg_slots import (
    _ArgSlot,
    _LabelRead,
    _out_destination_ids,
    _peek_raw_graph_workspace,
    _session_parent_labels,
    _SlotCall,
    _SlotContext,
    _SnapshotScope,
    _storage_intact,
    _tensor_items,
    parents_from_slots,
    templates_requested,
)
from ._storage_exports import note_alias_rebound_snapshot, note_foreign_written_snapshot
from .aliasing import (
    get_parent_contents_for_contract_position,
    parent_label_has_alias_contract,
)
from .buffer_writes import (
    _RECEIVER_REBIND_NAMES,
    resolve_or_late_index_buffer_address,
    session_validated_buffer_address,
)
from .sources import log_closure_constant_if_known, log_source_tensor
from .tensor_tracking import (
    _add_tensor_backward_hook,
    _append_module_suffix_to_equivalence_class,
    _get_equivalence_class,
    _make_raw_param_group_barcode,
)

# Transform-boundary config attribute; its builder kwargs are snapshotted per op.
_TRANSFORM_CONFIG = "__tl_transform_config__"
# Marks "no transform config on this function"; most ops have none, and the fresh
# empty dict they record needs no snapshot.
_NO_CONFIG = object()

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


__all__ = (
    "_build_shared_fields_dict",
    "_classify_new_tensor_in_trace",
    "_tag_tensor_and_track_variations",
    "_get_parent_output_version_snapshot",
    "_predicate_call_site_chain",
    "_predicate_equivalence_class",
    "_prepared_param_refs",
)


def _set_grad_info_fields(self: "Trace", fields_dict: dict[str, Any]) -> None:
    """Record whether this op saves gradients.

    The per-gradient placeholders (``grad``, ``grad_shape``, ``gradient_memory`` and
    their ``transformed_`` forms) are not written: the freeze never reads them and
    the op defaults supply them.
    """
    fields_dict["save_grads"] = self._options_save_grads not in (None, False)


def _unvalidated_provenance(
    trace: "Trace", value: Any, read: _LabelRead, live: Mapping[str, Any]
) -> bool:
    """Return ``_tensor_has_known_provenance`` past its prepared-Parameter rung."""

    if _async_collective_elem(value) is not None:
        return _tensor_has_known_provenance(trace, value)
    meta = read.meta
    if meta is None:
        return False
    if read.storage_intact and read.live_label is not None:
        return True
    if meta.address is not None and session_validated_buffer_address(trace, value) is not None:
        return True
    return bool(
        read.storage_intact and meta.buffer_source is not None and meta.buffer_source in live
    )


def _object_facts(trace: "Trace", value: Any, scope: _SnapshotScope) -> tuple[Any, ...]:
    """Read one argument tensor's label and witness facts, applying any stale clear.

    Returns the trailing :class:`_ArgSlot` fields, from ``label`` on.
    """

    is_param = isinstance(value, torch.nn.Parameter)
    meta = get_tensor_meta(value)
    gate_label = mutated_parameter_label(value) if is_param else get_tensor_label(value)
    live_label = gate_label if gate_label is not None and gate_label in scope.live else None
    read = _LabelRead(meta, live_label, _storage_intact(meta, value, gate_label))
    in_clear_set = id(value) in scope.clear_ids or (
        id(value) in scope.visited_ids and (not is_param or gate_label is not None)
    )
    data_alias = bool(meta is not None and meta.data_alias and gate_label is not None)
    if in_clear_set and live_label is None and meta is not None and meta.label_raw is not None:
        clear_tensor_label(value)
        gate_label, data_alias = None, False
    param_unmutated = is_param and gate_label is None
    validated = is_param and _session_validated_parameter(trace, value)
    return (
        live_label,
        validated or _unvalidated_provenance(trace, value, read, scope.live),
        () if param_unmutated else _session_parent_labels(read, scope.live),
        data_alias,
        param_unmutated,
        validated,
    )


def snapshot_arg_slots(
    trace: "Trace", call: _SlotCall, arg_tensors: list[torch.Tensor]
) -> tuple[_ArgSlot, ...]:
    """Walk the call's arguments once and record every tensor slot's derivation reads.

    Records the slots as the current op's context (read by the parent-position
    recorder and the witness) and returns them in walk order.
    """

    items = list(_tensor_items(call))
    workspace = _peek_raw_graph_workspace(trace)
    if not items or workspace is None:
        return ()
    templates_on = templates_requested(trace)
    scope = _SnapshotScope(
        live=trace.capture_events.live_index.by_raw_label,
        clear_ids=frozenset({id(t) for t in arg_tensors} | _out_destination_ids(call.kwargs)),
        visited_ids=frozenset(id(v) for _, _, v, seen in items if seen and templates_on),
    )
    facts: dict[int, tuple[Any, ...]] = {}
    slots = []
    for kind, key, value, _visited in items:
        if id(value) not in facts:
            facts[id(value)] = _object_facts(trace, value, scope)
        slots.append(_ArgSlot(kind, key, id(value), *facts[id(value)]))
    by_key = {(slot.kind, slot.key): slot for slot in slots}
    workspace._pending_arg_slots = _SlotContext(call.args, call.kwargs, tuple(slots), by_key)
    return tuple(slots)


def _set_arg_template_fields(
    self: "Trace",
    fields_dict: dict[str, Any],
    call: _SlotCall,
) -> None:
    """Seed the replay argument templates and the portable function identity of one call."""
    if templates_requested(self):
        captured_template = _build_args_template(
            call.func, call.args, call.kwargs, self, func_name=call.func_name
        )
        fields_dict["args_template"] = captured_template
        fields_dict["kwargs_template"] = captured_template if call.kwargs else None
        # The portable function identity is what lets a LOADED bundle resolve its
        # callables for validation; a default capture records templates only for
        # in-process replay, so it keeps the default bundle's identity-free contract.
        fields_dict["func_id"] = (
            captured_template.func_id
            if getattr(self, "is_replayable", False) or self._options_save_args
            else None
        )
    else:
        fields_dict["args_template"] = None
        fields_dict["kwargs_template"] = None
        # A capture that saves arguments is replay-ready; its portable function
        # identity lets replay validation resolve the callable on a loaded
        # bundle (``validation/_loaded_callables.py``), which carries no callables.
        fields_dict["func_id"] = (
            _function_registry_key(call.func, call.func_name, self)
            if self._options_save_args
            else None
        )


def _repin_rebound_receiver(self: "Trace", args: tuple[Any, ...], tensor: torch.Tensor) -> bool:
    """Re-pin a rebound registered buffer receiver to its pre-call label; return whether it did.

    The caller checks the call is a rebind (``set_``, ``share_memory_``). Such a call moves
    the receiver onto another storage, so after the call the session storage gate refuses the
    label the receiver carried INTO the call. That label is still the right parent of the
    receiver slot: the call consumed the pre-call value, which its buffer source saved
    before the call. Re-logging the receiver as a new buffer source here would bind the
    slot to the post-call value. Only the receiver qualifies, only with a current-session
    stamp, and only when it still resolves to a registered buffer; the output relabel that
    follows re-stamps it with the call's own label. The stamp is read raw only after the
    session-anchor check, so a foreign stamp is never trusted.
    """

    if not args or tensor is not args[0]:
        return False
    label = raw_tensor_label(tensor)
    if label is None or not session_meta_is_anchored(get_tensor_meta(tensor)):
        return False
    if resolve_or_late_index_buffer_address(self, tensor) is None:
        return False
    set_tensor_label(tensor, label)
    return True


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
    """Build the fields shared by every output tensor of one call (function, parents, module).

    Per-tensor fields (shape, label, equivalence type) are added in ``_log_output_tensor_info``.
    Returns ``(fields_dict, parent_layer_entries, arg_tensors, parent_param_ops)``.
    """
    # Canonical layer_type: lowercase with underscores stripped (e.g. "conv2d").
    layer_type = _normalize_func_name(func_name)
    # Non-tensor args (metadata, equivalence hashing) are op-time snapshots.
    non_tensor_args, non_tensor_kwargs = split_non_tensor_args(args, kwargs, _check_if_tensor_arg)
    bound_aliases = _bind_call_alias_labels(args, kwargs) if _VIEW_MUTATION_RELABELS else ()
    try:
        # O(1) tensor/param extraction via lookup table (replaces BFS crawl)
        arg_tensors, arg_parameters = _extract_arg_tensors_and_params(layer_type, args, kwargs)
        tensors_to_resolve = get_arg_tensors_for_resolution(args, kwargs)
        # Inline, not a helper: once per op (tests/test_capture_row_commit_cost.py pins calls).
        for tensor in tensors_to_resolve:
            if isinstance(tensor, torch.nn.Parameter) or get_tensor_label(tensor) is not None:
                continue
            if func_name in _RECEIVER_REBIND_NAMES and _repin_rebound_receiver(self, args, tensor):
                continue
            # Never source-log a "buffer" off a raw static stamp (stale cross-capture, or
            # storage rebound to input data); the session belt checks object and storage.
            buffer_address = resolve_or_late_index_buffer_address(self, tensor)
            if buffer_address is not None:
                log_source_tensor(self, tensor, "buffer", buffer_address)
            else:
                log_closure_constant_if_known(self, tensor)

        call = _SlotCall(func, func_name, args, kwargs)
        slots = snapshot_arg_slots(self, call, arg_tensors)
        # One entry per distinct parent: a repeat's multiplicity is its parent_arg_positions.
        parent_layer_labels = list(dict.fromkeys(parents_from_slots(self, slots, arg_tensors)))
        parent_layer_entries = [
            cast(Op, LiveOpView(self, self.capture_events.live_index.require_event(label)))
            for label in parent_layer_labels
        ]

        fields_dict: dict[str, Any] = {}

        # General info
        fields_dict["type"] = layer_type
        fields_dict["detach_saved_activations"] = self._options_detach_saved_activations
        fields_dict["output_device"] = self._options_output_device
        fields_dict["_construction_done"] = False
        _set_arg_template_fields(self, fields_dict, call)
        fields_dict["container_path"] = ()
        fields_dict["container_spec"] = None

        _set_grad_info_fields(self, fields_dict)

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
            source_loading_enabled=self._options_save_source_context,
            disable_col_offset=False,
            context_cache=code_context_cache,
            call_site_out=call_site_out,
        )
        self._raw_graph_ws._pending_call_site = (func_call_id, call_site_out[0])
        fields_dict["func_duration"] = exec_ctx.time_elapsed
        fields_dict["func_rng_states"] = exec_ctx.rng_states
        fields_dict["func_autocast_state"] = exec_ctx.autocast_state
        fields_dict["arg_names"] = _call_arg_names(func_name)
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
        config = getattr(func, _TRANSFORM_CONFIG, _NO_CONFIG)
        fields_dict["transform_config"] = (
            {} if config is _NO_CONFIG else snapshot_mutable_arg(config)
        )
        fields_dict["transform_fn_name"] = getattr(func, "__tl_transform_fn_name__", None)
        fields_dict["transform_fn_qualname"] = getattr(func, "__tl_transform_fn_qualname__", None)
        fields_dict["transform_fn_source"] = getattr(func, "__tl_transform_fn_source__", None)
        (
            fields_dict["unattributed_tensor_args"],
            fields_dict["dropped_edge_tensor_args"],
        ) = witness_from_slots(
            self, call, fields_dict["parent_arg_positions"], fields_dict["parent_params"]
        )

        # Function config — lightweight hyperparameter extraction, always on.
        param_shapes = cast(list[tuple[int, ...]] | None, fields_dict.get("param_shapes"))
        fields_dict["func_config"] = extract_salient_args(
            layer_type, func_name, args, kwargs, param_shapes
        )

        return fields_dict, parent_layer_entries, arg_tensors, parent_param_ops
    finally:
        if bound_aliases:
            _restore_call_alias_labels(bound_aliases)


def _call_arg_names(func_name: str) -> tuple[str, ...]:
    """Return the recorded argument names of the wrapped callable ``func_name``.

    Exact-name lookup first (add / add_ / __add__ have distinct signatures).
    The underscore-stripped key remains a best-effort fallback for
    NON-dunder names whose own introspection stored nothing; a dunder whose
    signature is opaque stays honestly unknown (empty) rather than borrowing
    the namesake torch function's different signature.
    """

    names = _st._arg_names.get(func_name)
    if names is not None:
        return names
    if func_name.startswith("__"):
        return ()
    return _st._arg_names.get(func_name.strip("_"), ())


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
    live_args: tuple[Any, ...],
    live_kwargs: dict[str, Any],
) -> None:
    """Tag the output tensor with its label, add backward hook, and track parent content variations.

    Parent content variation tracking detects in-place mutations: if a parent
    tensor's value at function-call time (from arg_copies) differs from its
    saved out, the pre-mutation value is recorded in
    ``out_versions_by_child``.  This is critical for validation replay,
    which needs the actual input values each child operation saw; ``live_args`` and ``live_kwargs`` feed its foreign-write marks.
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
            new_layer_entry._call_parent_arg_positions,
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
            _mark_foreign_written_snapshot(
                self, parent_label, new_layer_entry, contract, (live_args, live_kwargs)
            )


def _mark_foreign_written_snapshot(
    self: "Trace",
    parent_label: str,
    child_entry: Op,
    contract: Any,
    live_call: tuple[tuple[Any, ...], dict[str, Any]],
) -> None:
    """Mark a stored parent snapshot as foreign-written unless the child's alias contract asked for it.

    Runs only after a snapshot was stored, so the live-call tuple is never built on the
    ordinary per-op path (the per-op pins in tests/test_capture_row_commit_cost.py).
    """
    positions = child_entry._call_parent_arg_positions
    contract_positions = (*contract.mutated_input_positions, *contract.aliased_output_inputs)
    if not parent_label_has_alias_contract(parent_label, positions, contract_positions):
        note_foreign_written_snapshot(
            self, parent_label, child_entry._label_raw, positions, live_call
        )
        if _VIEW_MUTATION_RELABELS:
            note_alias_rebound_snapshot(
                self,
                (parent_label, child_entry._label_raw),
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


def _predicate_call_site_chain(trace: "Trace") -> tuple[Any, ...]:
    """Return the call-site chain of the torch call being recorded.

    The same frame walk exhaustive capture runs for its code context
    (``_ops_shared_fields``), asked only for the chain: no source lines, no
    column offsets, and the per-capture anchor in ``trace._code_context_cache``
    stops the walk at the capture call. Empty when no Python frame lies
    inside the innermost stateful module call. The key and class string it
    feeds are the ones exhaustive capture mints (``_ops_activations``), so the
    cooked Trace groups by call site (``callsite_v2``) with the same result.
    """

    context_cache = trace._code_context_cache
    call_site_out: list[Any] = []
    _get_code_context(
        0,
        source_loading_enabled=False,
        disable_col_offset=True,
        context_cache=context_cache,
        call_site_out=call_site_out,
    )
    return call_site_out[0] if call_site_out else ()


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


def _build_module_context_fields(
    self: "Trace",
    fields_dict: dict[str, Any],
    arg_tensors: list[torch.Tensor],
    parent_layer_entries: list[Op],
) -> None:
    """Populate the module nesting fields (``module``, ``modules``)."""
    modules = _snapshot_exhaustive_module_stack(self)
    module = modules[-1] if modules else None

    fields_dict["module"] = module
    fields_dict["modules"] = modules
    # The module-I/O placeholders (``module_call_stack``, ``input_to_module_calls``,
    # ``output_of_modules`` and the rest) are not written: the freeze never reads
    # them and postprocess derives them from the records.


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._ops_arguments import (  # noqa: E402
    _build_args_template,
    _function_registry_key,
    _session_validated_parameter,
    _tensor_has_known_provenance,
)
from ._ops_capture_records import (  # noqa: E402
    _bind_call_alias_labels,
    _record_label_version_snapshot,
    _restore_call_alias_labels,
    _snapshot_exhaustive_module_stack,
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
    _extract_arg_tensors_and_params,
)
from .ops import (  # noqa: E402
    _VIEW_MUTATION_RELABELS,
    TRANSFORM_FUNC_NAMES,
)
