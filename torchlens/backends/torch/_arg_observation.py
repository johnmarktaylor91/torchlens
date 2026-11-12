"""Planned argument observation for flat calls: the slot facts and the witness.

A call with a flat layout plan (:mod:`._call_layout`) holds its tensors as
top-level values at the plan's slots. For such a call the slot snapshot reads
each tensor's facts here, for the two exact types nearly every argument has (a
plain ``torch.Tensor`` stamped by this capture, a prepared ``nn.Parameter``),
with the same reads and the same results as the general per-object read
(``_ops_argument_fields._object_facts``); any other tensor is left to that read.
Only the layout is cached (guarded by the callable and the exact argument
types); every per-tensor fact is read live, once per op.

:func:`_planned_witness` is the witness of a flat call whose every slot is
attributed, read from the same snapshot.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch

from ..._data_classes.param import ParamAccessor
from ..._utils import _tensor_label as _labels
from ..._utils._tensor_label import (
    TensorMeta,
    clear_tensor_label,
    get_live_tensor_label,
    get_param_meta,
    get_tensor_label,
)
from ._call_layout import planned_tensor_items
from ._ops_arg_slots import (
    _ArgSlot,
    _current_context,
    _out_destination_ids,
    _SlotCall,
    _SlotContext,
    _SnapshotScope,
    new_arg_slot,
    new_slot_context,
    parents_from_slots,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = (
    "READS_HANDOFF",
    "plain_facts",
    "slot_parent_labels",
    "snapshot_flat_slots",
    "validated_parameter",
)

_PARAMETER = torch.nn.Parameter
#: The tensor of a ``(kind, key, tensor, template_visited)`` slot item.
_ITEM_VALUE = operator.itemgetter(2)
_TENSOR = torch.Tensor
#: The facts of a plain tensor without TorchLens metadata (``_object_facts`` result).
_UNSTAMPED: tuple[Any, ...] = (None, False, (), False, False, False)
#: ``(id(call), gated labels)`` the builder's resolve loop read for the call it is about
#: to snapshot, with no source logged since; the snapshot takes and clears it. The
#: labels map tensor ids to strings, so a left-over entry holds no tensor.
READS_HANDOFF: list[tuple[int, dict[int, str] | None] | None] = [None]
#: The witness's slot facts when the op has no snapshot (read only).
_NO_FACTS: Mapping[Any, Any] = {}


#: ``(call, arg tensors, live labels)`` of the flat call being snapshotted.
_FlatFrame = tuple[_SlotCall, list[torch.Tensor], Mapping[str, Any]]


def validated_parameter(trace: Trace, value: torch.Tensor) -> bool:
    """Return ``_session_validated_parameter(trace, value)``, with a cheaper first sight.

    A prepared Parameter's first read in a capture resolves its prep address in the
    session's ``ParamAccessor``. An exact address key is the accessor's first lookup
    rung, so ``addr in param_logs`` and ``param_logs[addr]`` are the plain dict read
    here; every other outcome (no stamp, no exact key, another object, the offload
    rebind rung) runs the general check.
    """

    workspace = trace._raw_graph_ws
    validated = None if workspace is None else workspace._validated_params
    if validated is not None and validated.get(id(value)) is value:
        return True
    param_meta = get_param_meta(value)
    param_logs = trace._param_logs
    if param_meta is not None and validated is not None and type(param_logs) is ParamAccessor:
        addr = param_meta.param_address
        entry = param_logs._dict.get(addr) if isinstance(addr, str) and addr else None
        if entry is not None and getattr(entry, "_param_ref", None) is value:
            validated[id(value)] = value
            return True
    return _session_validated_parameter(trace, value)


def _stale_stamped_facts(
    trace: Trace, value: Any, meta: TensorMeta, label: str | None, frame: _FlatFrame
) -> tuple[Any, ...]:
    """Return the facts of a stamped plain tensor whose label is withheld or not live.

    ``label`` is the gated label read; the stamp is anchored to the active session,
    so its storage read is ``label is not None``. Same steps as ``_object_facts``.
    """

    call, arg_tensors, live = frame
    source = meta.buffer_source
    intact = label is not None
    data_alias = bool(meta.data_alias and intact)
    # ``_object_facts``'s clear set: every tensor of a flat call is template-visited.
    clears = (
        _templates_requested(trace)
        or any(tensor is value for tensor in arg_tensors)
        or (call.layout.has_out and id(value) in _out_destination_ids(call.kwargs))
    )
    if clears:
        clear_tensor_label(value)
        data_alias = False
    known = (
        meta.address is not None and session_validated_buffer_address(trace, value) is not None
    ) or bool(intact and source is not None and source in live)
    parents = (source,) if intact and source is not None and source in live else ()
    return (None, known, parents, data_alias, False, False)


def plain_facts(
    trace: Trace, value: Any, token: int | None, reads: dict[int, str] | None, frame: _FlatFrame
) -> tuple[Any, ...] | None:
    """Return a plain tensor's or prepared Parameter's slot facts, else ``None``.

    ``None`` leaves the tensor to the general read (``_object_facts``). ``token`` is
    the active label session's; ``reads`` holds the gated labels the builder's resolve
    loop just read (no source was logged since), so a tensor's label is read once.
    """

    kind = value.__class__
    meta = getattr(value, "_tl", None)
    if kind is _PARAMETER:
        if meta is not None:
            return None
        ok = validated_parameter(trace, value)
        return (None, ok, (), False, True, ok)
    if kind is not _TENSOR or getattr(value, "_is_param", False):
        return None
    if meta is None:
        return _UNSTAMPED
    if (
        meta.__class__ is not TensorMeta
        or token is None
        or meta.label_session != token
        or meta.label_raw is None
    ):
        return None
    label = reads.get(id(value)) if reads is not None else None
    if label is None:
        label = get_tensor_label(value)
    live = frame[2]
    if label is None or label not in live:
        return _stale_stamped_facts(trace, value, meta, label, frame)
    source = meta.buffer_source
    if source is None or source == label or source not in live:
        return (label, True, (label,), bool(meta.data_alias), False, False)
    return (label, True, (label, source), bool(meta.data_alias), False, False)


def snapshot_flat_slots(
    trace: Trace,
    workspace: Any,
    call: _SlotCall,
    inputs: tuple[list[torch.Tensor], dict[int, str] | None],
    object_facts: Callable[[Trace, Any, _SnapshotScope], tuple[Any, ...]],
) -> tuple[_ArgSlot, ...] | None:
    """``snapshot_arg_slots`` for a flat call: planned facts, the general read for the rest.

    Walks the layout's tensor slots (``planned_tensor_items`` order). ``inputs`` is
    ``(arg tensors, resolve-loop label reads)``; ``object_facts`` is the general
    per-object read, given the general scope. ``None`` sends the call to the general
    walk (the reference tests patch this function so).
    """

    arg_tensors, reads = inputs
    args, kwargs = call.args, call.kwargs
    live = trace.capture_events.live_index.by_raw_label
    session = _labels._ACTIVE_LABEL_SESSION
    token = None if session is None else session.token
    frame = (call, arg_tensors, live)
    facts: dict[int, tuple[Any, ...]] = {}
    labels: dict[int, str | None] = {}
    scope: _SnapshotScope | None = None
    slot_list = []
    by_key = {}
    for slot_key in call.layout.tensor_slots:
        kind, key = slot_key
        value = args[key] if kind == "args" else kwargs[key]
        value_id = id(value)
        value_facts = facts.get(value_id)
        if value_facts is None:
            value_facts = plain_facts(trace, value, token, reads, frame)
            if value_facts is None:
                if scope is None:
                    scope = _general_scope(trace, call, arg_tensors, live)
                value_facts = object_facts(trace, value, scope)
            facts[value_id] = value_facts
            labels[value_id] = value_facts[0]
        slot = new_arg_slot(kind, key, value_id, value_facts)
        slot_list.append(slot)
        by_key[slot_key] = slot
    slots = tuple(slot_list)
    workspace._pending_arg_slots = new_slot_context(args, kwargs, slots, by_key, labels)
    return slots


def _general_scope(
    trace: Trace, call: _SlotCall, arg_tensors: list[torch.Tensor], live: Mapping[str, Any]
) -> _SnapshotScope:
    """Return the scope ``snapshot_arg_slots`` builds for a flat call's general reads."""

    clear_ids = set(map(id, arg_tensors))
    if call.layout.has_out:
        clear_ids |= _out_destination_ids(call.kwargs)
    visited = (
        set(map(id, map(_ITEM_VALUE, planned_tensor_items(call, call.layout))))
        if _templates_requested(trace)
        else set()
    )
    return _SnapshotScope(live, clear_ids, visited)


def slot_parent_labels(
    trace: Trace, slots: tuple[_ArgSlot, ...], arg_tensors: list[torch.Tensor]
) -> list[str]:
    """Return ``parents_from_slots(trace, slots, arg_tensors)`` from the recorded labels."""

    context = _current_context_of(trace) if slots else None
    if context is None or context.slots is not slots:
        return parents_from_slots(trace, slots, arg_tensors)
    labels = context.labels
    parents: list[str] = []
    for tensor in arg_tensors:
        if id(tensor) in labels:
            label = labels[id(tensor)]
        else:  # nested deeper than the snapshot: read live
            label = get_live_tensor_label(tensor, trace.capture_events.live_index.by_raw_label)
        if label is not None:
            parents.append(label)
    return parents


def _current_context_of(trace: Trace) -> _SlotContext | None:
    """Return the op's recorded slot context, whatever arguments it was taken for."""

    try:
        return trace._raw_graph_ws._pending_arg_slots
    except AttributeError:
        return None


def _planned_witness(
    trace: Trace,
    call: _SlotCall,
    parent_arg_positions: dict[str, dict[Any, str]],
    recorded_parent_params: list[torch.nn.Parameter],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return the witness of a flat call whose every slot is attributed, else ``None``.

    On a flat layout the witness's tree walk visits exactly the planned tensor
    slots, so each slot gets the checks ``visit_tensor`` applies, read from the
    same snapshot facts. The verdict is ``((), ())`` only when no slot would be
    appended; any slot that would be (or has no snapshot fact) returns ``None`` and
    the caller runs the full witness. A call that mutates its receiver always runs
    the full witness (its receiver check reads more than the slot facts).
    """

    layout = call.layout
    if layout.mutates_receiver is None:
        func_name = call.func_name
        layout.mutates_receiver = (
            _is_inplace_augmented_assignment_dunder(func_name)
            or func_name in _SETTER_MUTATION_FUNC_NAMES
            or (func_name.endswith("_") and not func_name.startswith("__"))
        )
    if layout.mutates_receiver:
        return None
    args, kwargs = call.args, call.kwargs
    context = _current_context(trace, args, kwargs)
    facts = _NO_FACTS if context is None else context.by_key
    recorded_parent_labels: set[str] | None = None
    for slot in layout.tensor_slots:
        value = args[slot[1]] if slot[0] == "args" else kwargs[slot[1]]
        fact = facts.get(slot)  # the snapshot slot of this very value
        if fact is None or fact.value_id != id(value) or not fact.known_provenance:
            return None
        if fact.param_unmutated:
            if fact.param_validated and id(value) not in map(id, recorded_parent_params):
                return None
            continue
        provenance_labels = fact.session_parent_labels
        if not provenance_labels:
            continue
        # ``_dropped_edge_at_slot`` inline: a stamp at the slot must be the tensor's own.
        recorded_at_slot = parent_arg_positions[slot[0]].get(slot[1])
        if recorded_at_slot is not None:
            if recorded_at_slot not in provenance_labels:
                return None
            continue
        if recorded_parent_labels is None:
            recorded_parent_labels = {
                *parent_arg_positions["args"].values(),
                *parent_arg_positions["kwargs"].values(),
            }
        if recorded_parent_labels.isdisjoint(provenance_labels):
            return None
    return (), ()


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._ops_arg_slots import templates_requested as _templates_requested  # noqa: E402
from ._ops_arguments import _session_validated_parameter  # noqa: E402
from ._ops_capture_records import _is_inplace_augmented_assignment_dunder  # noqa: E402
from .buffer_writes import session_validated_buffer_address  # noqa: E402
from .ops import _SETTER_MUTATION_FUNC_NAMES  # noqa: E402
