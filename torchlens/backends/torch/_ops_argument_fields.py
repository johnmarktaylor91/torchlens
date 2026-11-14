"""Argument family of the shared op fields: argument scan, slot snapshot and templates.

One of the per-family modules split out of ``_ops_shared_fields`` along the
families of ``torchlens/_trace_core/field_table.py`` (here ``args``), so a lane
changing how a call's arguments are scanned edits this module alone.
:func:`_scan_call_arguments` runs inside the shared-fields builder's alias binding.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from ... import _state as _st
from ..._capture.arg_positions import (
    FUNC_ARG_SPECS,
    VARIADIC_TENSOR_ARG_FUNCS,
    ArgSpec,
    dynamic_spec_covers_call,
)
from ..._utils._tensor_label import (
    clear_tensor_label,
    get_tensor_label,
    get_tensor_meta,
    mutated_parameter_label,
    promote_mutated_parameters,
    raw_tensor_label,
    session_meta_is_anchored,
    set_tensor_label,
)
from ..._utils._tl_collective import _async_collective_elem
from ..._utils.introspection import get_arg_tensors_for_resolution
from ._arg_observation import FlatChecks, slot_parent_labels, snapshot_flat_slots
from ._arg_template_resolve import resolve_arg_template_fields
from ._args_template_cache import observe_template
from ._call_layout import planned_tensor_items
from ._ops_arg_slots import (
    _ArgSlot,
    _LabelRead,
    _out_destination_ids,
    _peek_raw_graph_workspace,
    _session_parent_labels,
    _SlotCall,
    _SnapshotScope,
    _storage_intact,
    _tensor_items,
    arg_slot_labels,
    new_arg_slot,
    new_slot_context,
    templates_requested,
)
from ._thin_defer import defer_arg_template
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

# ``isinstance(value, torch.nn.Parameter)`` runs the Parameter metaclass's Python
# ``__instancecheck__``: true for a Parameter, or for any tensor with a truthy
# ``_is_param``, else the ordinary type test. The per-tensor sites below decide the
# two exact types the scan meets (an exact Tensor subclasses nothing, so only its
# flag counts) and keep ``isinstance`` for every other type: the same answer.
_PARAMETER = torch.nn.Parameter
_TENSOR = torch.Tensor
#: Exact types that are never an ``AsyncCollectiveTensor`` (a strict Tensor subclass).
_PLAIN_TENSOR_TYPES = frozenset({_TENSOR, _PARAMETER})


def _scan_call_arguments(
    self: "Trace", call: _SlotCall, layer_type: str
) -> tuple[list[torch.Tensor], list[torch.nn.Parameter], list[str]]:
    """Find the call's argument tensors, label their sources and snapshot their slots.

    Returns ``(arg_tensors, arg_parameters, parent_layer_labels)``, one parent label
    per distinct parent: a repeat's multiplicity is its ``parent_arg_positions``.
    """
    func_name, args, kwargs = call.func_name, call.args, call.kwargs
    # Taken once and cleared, so the trace does not hold the call's tensors
    # while the rest of the emit (and the op commit) runs.
    runtime = self._wrapper_runtime_ws
    fast, runtime.fast_scan = runtime.fast_scan, None
    scanned = (
        _fast_lane_arg_tensors(layer_type, args, kwargs, fast[1], call.layout)
        if fast is not None and fast[0] is args
        else None
    )
    if scanned is None:
        # O(1) tensor/param extraction via lookup table (replaces BFS crawl)
        arg_tensors, arg_parameters = _extract_arg_tensors_and_params(layer_type, args, kwargs)
        tensors_to_resolve = get_arg_tensors_for_resolution(args, kwargs)
    else:
        arg_tensors, arg_parameters, tensors_to_resolve = scanned
    # The gated labels read here, handed to the snapshot while no source is logged.
    reads: dict[int, str] | None = {}
    # Inline, not a helper: once per op (tests/test_capture_row_commit_cost.py pins calls).
    for tensor in tensors_to_resolve:
        kind = type(tensor)
        if kind is _PARAMETER or (
            getattr(tensor, "_is_param", False)
            if kind is _TENSOR
            else isinstance(tensor, _PARAMETER)
        ):
            continue
        label = get_tensor_label(tensor)
        if label is not None:
            if reads is not None:
                reads[id(tensor)] = label
            continue
        reads = None
        if func_name in _RECEIVER_REBIND_NAMES and _repin_rebound_receiver(self, args, tensor):
            continue
        # Never source-log a "buffer" off a raw static stamp (stale cross-capture, or
        # storage rebound to input data); the session belt checks object and storage.
        buffer_address = resolve_or_late_index_buffer_address(self, tensor)
        if buffer_address is not None:
            log_source_tensor(self, tensor, "buffer", buffer_address)
        else:
            log_closure_constant_if_known(self, tensor)

    runtime.arg_reads = (id(call), reads)  # read by the snapshot of this very call
    slots = snapshot_arg_slots(self, call, arg_tensors)
    parents = slot_parent_labels(self, slots, arg_tensors)
    return (
        arg_tensors,
        arg_parameters,
        parents if len(parents) < 2 else list(dict.fromkeys(parents)),
    )


#: ``(normalized name, argument types, keyword names, keyword types) -> spec`` of the
#: fast-lane call shapes whose per-slot spec extraction was shown, on first sight, to
#: equal the admitted tensors in argument order. The value is the spec the slow path
#: extracted through; a dynamic spec that the name-keyed cache has since replaced
#: invalidates the entry.
_FAST_SHAPES: dict[tuple[Any, ...], tuple[bool, ArgSpec]] = {}


def _fast_lane_arg_tensors(
    layer_type: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    admitted: list[torch.Tensor],
    layout: Any = None,
) -> tuple[list[torch.Tensor], list[torch.nn.Parameter], list[torch.Tensor]] | None:
    """Return a fast-lane call's ``(arg tensors, parameters, tensors to resolve)``.

    ``admitted`` is the tensor list ``admit_call`` collected: every tensor of an
    admitted call is a top-level positional or keyword value, in argument order.
    The spec lookup is replaced by one check of the call's shape (argument and
    keyword types) against shapes whose spec extraction matched this list once;
    a new shape runs the spec path, compares and remembers the shape when equal.
    ``layout`` is the call's flat layout plan, whose guard key holds the same types.
    Returns ``None`` when the caller must run the spec path (a new or unproven
    shape, or a variadic transform).
    """

    # A call's flat layout plan carries these types, under the same guard key.
    shape = (
        (layer_type, *layout.types)
        if layout is not None
        else (
            layer_type,
            tuple(map(type, args)),
            tuple(kwargs) if kwargs else (),
            tuple(map(type, kwargs.values())) if kwargs else (),
        )
    )
    known = _FAST_SHAPES.get(shape)
    if known is not None and (known[0] or _st._dynamic_arg_specs.get(layer_type) is known[1]):
        return _split_admitted(admitted)
    if layer_type in VARIADIC_TENSOR_ARG_FUNCS:
        return None
    arg_tensors, arg_parameters = _extract_arg_tensors_and_params(layer_type, args, kwargs)
    fast = _split_admitted(admitted)
    spec = _per_slot_spec(layer_type, args, kwargs)
    if (
        spec is not None
        and _same_objects(fast[0], arg_tensors)
        and _same_objects(fast[1], arg_parameters)
    ):
        _FAST_SHAPES[shape] = spec
    else:
        _FAST_SHAPES.pop(shape, None)
    return arg_tensors, arg_parameters, fast[2]


def _split_admitted(
    admitted: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[Any], list[torch.Tensor]]:
    """Split admitted tensors into per-slot tensors, parameters and resolution targets.

    The resolution targets are the non-Parameter tensors once each, in argument
    order, as ``get_arg_tensors_for_resolution`` returns them for a flat call.
    """

    tensors: list[torch.Tensor] = []
    params: list[Any] = []
    for value in admitted:
        if type(value) is torch.nn.Parameter:
            params.append(value)
        else:
            tensors.append(value)
    to_resolve = tensors[:] if len(tensors) < 2 else list({id(t): t for t in tensors}.values())
    if params:
        tensors, params = promote_mutated_parameters(tensors, params)
    return tensors, params, to_resolve


def _per_slot_spec(
    layer_type: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[bool, ArgSpec] | None:
    """Return ``(is static, spec)`` the spec path extracted this call through, per slot.

    ``None`` when the call took the deduplicating crawl, whose result the
    admitted list does not reproduce in general.
    """

    spec = FUNC_ARG_SPECS.get(layer_type)
    if spec is not None:
        return (True, spec) if dynamic_spec_covers_call(spec, args, kwargs) else None
    cached = _st._dynamic_arg_specs.get(layer_type)
    if isinstance(cached, ArgSpec) and dynamic_spec_covers_call(cached, args, kwargs):
        return False, cached
    return None


def _same_objects(first: list[Any], second: list[Any]) -> bool:
    """Return whether two lists hold the same objects in the same order."""

    return len(first) == len(second) and all(a is b for a, b in zip(first, second, strict=True))


def _unvalidated_provenance(
    trace: "Trace", value: Any, read: _LabelRead, live: Mapping[str, Any]
) -> bool:
    """Return ``_tensor_has_known_provenance`` past its prepared-Parameter rung."""

    # An exact Tensor or Parameter is never an AsyncCollectiveTensor (a strict subclass).
    if type(value) not in _PLAIN_TENSOR_TYPES and _async_collective_elem(value) is not None:
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

    kind = type(value)
    is_param = kind is _PARAMETER or (
        bool(getattr(value, "_is_param", False))
        if kind is _TENSOR
        else isinstance(value, _PARAMETER)
    )
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
    recorder and the witness) and returns them in walk order. A flat call reuses
    the gated labels the resolve loop handed over for it (the runtime's ``arg_reads``).
    """

    runtime = trace._wrapper_runtime_ws
    handed, runtime.arg_reads = runtime.arg_reads, None

    layout = call.layout
    if layout is not None and layout.tensor_slots:
        workspace = _peek_raw_graph_workspace(trace)
        if workspace is None:
            return ()
        reads = handed[1] if handed is not None and handed[0] == id(call) else None
        flat = snapshot_flat_slots(trace, workspace, call, (arg_tensors, reads), _FLAT_CHECKS)
        if flat is not None:
            return flat
    items = list(_tensor_items(call)) if layout is None else planned_tensor_items(call, layout)
    workspace = _peek_raw_graph_workspace(trace)
    if not items or workspace is None:
        return ()
    # Built with C-level set and map calls, not comprehensions: once per op.
    clear_ids = set(map(id, arg_tensors)) | _out_destination_ids(call.kwargs)
    visited_ids = (
        {id(value) for _, _, value, seen in items if seen} if templates_requested(trace) else set()
    )
    scope = _SnapshotScope(trace.capture_events.live_index.by_raw_label, clear_ids, visited_ids)
    facts: dict[int, tuple[Any, ...]] = {}
    slot_list = []
    by_key = {}
    labels: dict[int, str | None] = {}
    for kind, key, value, _visited in items:
        value_facts = facts.get(id(value))
        if value_facts is None:
            value_facts = facts[id(value)] = _object_facts(trace, value, scope)
            labels[id(value)] = value_facts[0]
        slot = new_arg_slot(kind, key, id(value), value_facts)
        slot_list.append(slot)
        by_key[(kind, key)] = slot
    slots = tuple(slot_list)
    workspace._pending_arg_slots = new_slot_context(call.args, call.kwargs, slots, by_key, labels)
    return slots


def _set_arg_template_fields(
    self: "Trace",
    fields_dict: dict[str, Any],
    call: _SlotCall,
    defer: bool = False,
) -> None:
    """Seed the replay argument templates and the portable function identity of one call.

    Observes here every fact the stored values carry (labels, layouts, literals, copies
    and the function key, read at the call), then resolves them with
    ``resolve_arg_template_fields``, a pure function of those facts and the layout plan:
    now, or (``defer``, the thin path) when the row fills.
    """
    # The portable function identity is what lets a LOADED bundle resolve its
    # callables for validation; a default capture records templates only for
    # in-process replay, so it keeps the default bundle's identity-free contract.
    # A capture that saves arguments is replay-ready either way: its identity lets
    # replay validation resolve the callable on a loaded bundle
    # (``validation/_loaded_callables.py``), which carries no callables.
    # The parts are read at the call: ``_ops_arguments`` defines them after its own late imports.
    parts = _ops_arguments._TEMPLATE_PARTS
    replayable = getattr(self, "is_replayable", False)
    observed: tuple[Any, bool, bool, Any]
    if replayable or self._options_save_arg_templates:
        args, kwargs = call.args, call.kwargs
        facts = observe_template(
            parts,
            (call.func, call.func_name, args, kwargs),
            self,
            arg_slot_labels(self, args, kwargs),
            call.layout,
        )
        observed = (facts, bool(kwargs), bool(replayable or self._options_save_args), None)
    else:
        key = (
            _function_registry_key(call.func, call.func_name, self)
            if self._options_save_args
            else None
        )
        observed = (None, False, False, key)
    if defer:
        defer_arg_template(fields_dict, observed, parts)
    else:
        resolve_arg_template_fields(fields_dict, observed, parts)


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
from ..._data_classes.param import ParamAccessor  # noqa: E402
from . import _ops_arguments  # noqa: E402
from ._ops_arguments import (  # noqa: E402
    _function_registry_key,
    _session_validated_parameter,
    _tensor_has_known_provenance,
)
from ._ops_predicate_events import (  # noqa: E402
    _extract_arg_tensors_and_params,
)

#: The flat snapshot's reads from this package's cyclic modules.
_FLAT_CHECKS = FlatChecks(
    _object_facts, _session_validated_parameter, session_validated_buffer_address, ParamAccessor
)
