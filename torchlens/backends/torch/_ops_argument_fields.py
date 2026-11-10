"""Argument family of the shared op fields: argument scan, slot snapshot and templates.

One of the per-family modules split out of ``_ops_shared_fields`` along the
families of ``torchlens/_trace_core/field_table.py`` (here ``args``), so a lane
changing how a call's arguments are scanned edits this module alone.
:func:`_scan_call_arguments` runs inside the shared-fields builder's alias binding.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

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
from ..._utils.introspection import get_arg_tensors_for_resolution
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
from .buffer_writes import (
    _RECEIVER_REBIND_NAMES,
    resolve_or_late_index_buffer_address,
    session_validated_buffer_address,
)
from .sources import log_closure_constant_if_known, log_source_tensor

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


__all__ = (
    "_scan_call_arguments",
    "_set_arg_template_fields",
    "snapshot_arg_slots",
)


def _scan_call_arguments(
    self: "Trace", call: _SlotCall, layer_type: str
) -> tuple[list[torch.Tensor], list[torch.nn.Parameter], list[str]]:
    """Find the call's argument tensors, label their sources and snapshot their slots.

    Returns ``(arg_tensors, arg_parameters, parent_layer_labels)``, one parent label
    per distinct parent: a repeat's multiplicity is its ``parent_arg_positions``.
    """
    func_name, args, kwargs = call.func_name, call.args, call.kwargs
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

    slots = snapshot_arg_slots(self, call, arg_tensors)
    parent_layer_labels = list(dict.fromkeys(parents_from_slots(self, slots, arg_tensors)))
    return arg_tensors, arg_parameters, parent_layer_labels


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


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._ops_arguments import (  # noqa: E402
    _build_args_template,
    _function_registry_key,
    _session_validated_parameter,
    _tensor_has_known_provenance,
)
from ._ops_predicate_events import (  # noqa: E402
    _extract_arg_tensors_and_params,
)
