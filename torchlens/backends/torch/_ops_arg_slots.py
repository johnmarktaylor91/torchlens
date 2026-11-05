"""One per-op snapshot of the tensor argument slots the shared-fields derivations read.

The shared-fields builder derives four things from an op's arguments: the parent
labels, ``parent_arg_positions``, the unattributed / dropped-edge witness and the
replay argument templates. Each used to walk the arguments and read every tensor's
label and provenance on its own. :func:`snapshot_arg_slots` walks the arguments once,
to the recorder's two nesting levels, and records per tensor slot the values those
derivations read. Tensors nested deeper keep the per-argument lookups.

Ordering: the snapshot is taken after the builder's source-logging loop (buffer and
closure-constant labels are assigned there) and before the parent labels are read.
It applies at snapshot time the stale-label clears that the derivations' live reads
(``get_live_tensor_label``) would apply later, so each recorded fact is the value the
derivation reads; nothing between the snapshot and the witness relabels an argument.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ._ops_arguments import (
    _capture_layout,
    _classify_arg_component,
    _function_registry_key,
    _literal_value_supported,
    _peek_raw_graph_workspace,
    _session_validated_parameter,
    _tensor_has_known_provenance,
    _unlabeled_tensor_literal,
    _unsupported_component,
)
from ._tl import (
    TensorMeta,
    clear_tensor_label,
    get_live_tensor_label,
    get_tensor_label,
    get_tensor_meta,
    is_tensor_data_alias,
    mutated_parameter_label,
    session_label_storage_intact,
    session_meta_is_anchored,
)
from ._tl_collective import _async_collective_elem
from .buffer_writes import session_validated_buffer_address

if TYPE_CHECKING:
    from ..._data_classes.op import Op
    from ..._data_classes.trace import Trace
    from ...intervention.types import ArgComponent, CapturedArgTemplate

SlotKey = tuple[str, Any]


@dataclass(frozen=True, slots=True)
class _SlotCall:
    """The wrapped call whose arguments the shared-fields builder derives from."""

    func: Callable[..., Any]
    func_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ArgSlot:
    """What the shared-fields derivations read about one tensor argument slot.

    ``key`` is the recorder position key (``0``, ``"mask"``, ``(1, 0)``). ``label`` is
    the tensor's live current-session label after any stale-label clear, else ``None``.
    The remaining fields are the witness reads: the known-provenance verdict, the
    session parent labels, the data-alias flag, whether the tensor is an unmutated
    Parameter, and whether it is a session-validated prepared Parameter.
    """

    kind: str
    key: Any
    value_id: int
    label: str | None
    known_provenance: bool
    session_parent_labels: tuple[str, ...]
    data_alias: bool
    param_unmutated: bool
    param_validated: bool


@dataclass(frozen=True, slots=True)
class _SlotContext:
    """The current op's slots, keyed for the derivations that cannot take them directly."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    slots: tuple[_ArgSlot, ...]
    by_key: dict[SlotKey, _ArgSlot]


@dataclass(frozen=True, slots=True)
class _SnapshotScope:
    """Per-op inputs of the snapshot: live labels and which tensors get cleared."""

    live: Mapping[str, Any]
    clear_ids: frozenset[int]
    visited_ids: frozenset[int]


def templates_requested(trace: Trace) -> bool:
    """Return whether this capture records replay argument templates."""

    return bool(getattr(trace, "is_replayable", False) or trace._options_save_arg_templates)


def _tensor_items(call: _SlotCall) -> Iterator[tuple[str, Any, Any, bool]]:
    """Yield ``(kind, key, tensor, template_visited)`` for each tensor in two levels.

    The walk is the parent-position recorder's: positional args then kwargs, and
    one level inside a mapping, tuple or list. ``template_visited`` is whether the
    argument-template walk reaches the slot (it descends into lists, tuples and dicts).
    """

    for kind, items in (("args", enumerate(call.args)), ("kwargs", call.kwargs.items())):
        for key, value in items:
            if isinstance(value, torch.Tensor):
                yield kind, key, value, True
            elif isinstance(value, (Mapping, tuple, list)):
                visited = isinstance(value, (list, tuple, dict))
                inner = value.items() if isinstance(value, Mapping) else enumerate(value)
                for sub_key, sub_value in inner:
                    if isinstance(sub_value, torch.Tensor):
                        yield kind, (key, sub_key), sub_value, visited


def _out_destination_ids(kwargs: dict[str, Any]) -> set[int]:
    """Return the object ids of the call's ``out=`` destination tensors."""

    out = kwargs.get("out")
    if isinstance(out, torch.Tensor):
        return {id(out)}
    if isinstance(out, (list, tuple)):
        return {id(item) for item in out if isinstance(item, torch.Tensor)}
    return set()


@dataclass(frozen=True, slots=True)
class _LabelRead:
    """One argument tensor's metadata and label reads, taken once per op."""

    meta: TensorMeta | None
    live_label: str | None
    storage_intact: bool


def _storage_intact(meta: TensorMeta | None, value: Any, gate_label: str | None) -> bool:
    """Return the label storage-integrity read the provenance rungs use.

    For an anchored stamped label the label gate already decided it: an anchored
    label is withheld only when its storage was rebound.
    """

    if meta is None or not session_meta_is_anchored(meta):
        return False
    if meta.label_raw is not None:
        return gate_label is not None
    return meta.buffer_source is not None and session_label_storage_intact(meta, value)


def _session_parent_labels(read: _LabelRead, live: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the live op and buffer-source labels the tensor should appear under."""

    if read.meta is None or not read.storage_intact:
        return ()
    labels = [] if read.live_label is None else [read.live_label]
    buffer_source = read.meta.buffer_source
    if buffer_source is not None and buffer_source in live and buffer_source not in labels:
        labels.append(buffer_source)
    return tuple(labels)


def _unvalidated_provenance(
    trace: Trace, value: Any, read: _LabelRead, live: Mapping[str, Any]
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


def _object_facts(trace: Trace, value: Any, scope: _SnapshotScope) -> tuple[Any, ...]:
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
    trace: Trace, call: _SlotCall, arg_tensors: list[torch.Tensor]
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


def _current_context(trace: Trace, args: Any, kwargs: Any) -> _SlotContext | None:
    """Return the slot context recorded for exactly this call's arguments, if any."""

    workspace = _peek_raw_graph_workspace(trace)
    context = None if workspace is None else workspace._pending_arg_slots
    if context is None or context.args is not args or context.kwargs is not kwargs:
        return None
    return context


def current_arg_slots(trace: Trace, args: Any, kwargs: Any) -> tuple[_ArgSlot, ...] | None:
    """Return the current op's slots when they were snapshotted for these arguments."""

    context = _current_context(trace, args, kwargs)
    return None if context is None else context.slots


def arg_slot_facts(trace: Trace, args: Any, kwargs: Any) -> dict[SlotKey, _ArgSlot]:
    """Return the current op's slots by recorder key (empty when none were taken)."""

    context = _current_context(trace, args, kwargs)
    return {} if context is None else context.by_key


def slot_fact(facts: dict[SlotKey, _ArgSlot], slot: SlotKey | None, value: Any) -> _ArgSlot | None:
    """Return the snapshot slot for ``value`` at ``slot``, or ``None`` to read live."""

    found = facts.get(slot) if slot is not None else None
    return found if found is not None and found.value_id == id(value) else None


def receiver_data_alias(facts: dict[SlotKey, _ArgSlot], receiver: torch.Tensor) -> bool:
    """Return ``is_tensor_data_alias`` for the call's receiver, from the snapshot if taken."""

    fact = slot_fact(facts, ("args", 0), receiver)
    return fact.data_alias if fact is not None else is_tensor_data_alias(receiver)


def parents_from_slots(
    trace: Trace, slots: tuple[_ArgSlot, ...], arg_tensors: list[torch.Tensor]
) -> list[str]:
    """Return the live parent labels of ``arg_tensors`` in extraction order.

    Same values as ``get_live_label_list``: tensors outside the snapshot (nested
    deeper than two levels) are read live.
    """

    live = trace.capture_events.live_index.by_raw_label
    by_id = {slot.value_id: slot.label for slot in slots}
    labels: list[str] = []
    for tensor in arg_tensors:
        label = by_id[id(tensor)] if id(tensor) in by_id else get_live_tensor_label(tensor, live)
        if label is not None:
            labels.append(label)
    return labels


def parent_arg_positions_from_slots(
    slots: tuple[_ArgSlot, ...], parent_entries: list[Op]
) -> dict[str, dict[Any, str]]:
    """Map each parent label to its argument positions, emitted parent by parent."""

    positions: dict[str, dict[Any, str]] = {"args": {}, "kwargs": {}}
    slots_by_label: dict[str, list[_ArgSlot]] = {}
    for slot in slots:
        if slot.label is not None:
            slots_by_label.setdefault(slot.label, []).append(slot)
    for entry in parent_entries:
        label = entry._label_raw
        for slot in slots_by_label.get(label, ()):
            positions[slot.kind][slot.key] = label
    return positions


def _child_key(slot_key: SlotKey | None, inner_key: Any) -> SlotKey | None:
    """Return the recorder key of a container member, or ``None`` past two levels."""

    if slot_key is None or isinstance(slot_key[1], tuple):
        return None
    return (slot_key[0], (slot_key[1], inner_key))


@dataclass(frozen=True, slots=True)
class _TemplateWalk:
    """State of one argument-template walk over a call."""

    trace: Trace
    by_key: dict[SlotKey, _ArgSlot]
    notes: list[str]

    def component(self, value: Any, slot_key: SlotKey | None) -> ArgComponent:
        """Classify one argument value, reading tensor labels from the snapshot."""

        # Deferred: intervention vocabulary sits above the backend layer.
        from ...intervention.types import LiteralValue, ParentRef

        if slot_key is None:
            return _classify_arg_component(value, self.notes, self.trace)
        if isinstance(value, torch.Tensor):
            slot = slot_fact(self.by_key, slot_key, value)
            if slot is None:
                return _classify_arg_component(value, self.notes, self.trace)
            if slot.label is not None:
                return ParentRef(slot.label, _capture_layout=_capture_layout(value))
            return _unlabeled_tensor_literal(value)
        if _literal_value_supported(value):
            return LiteralValue(value)
        if isinstance(value, (list, tuple)):
            return tuple(
                self.component(item, _child_key(slot_key, index))
                for index, item in enumerate(value)
            )
        if isinstance(value, dict):
            return tuple(
                (key, self.component(item, _child_key(slot_key, key)))
                for key, item in value.items()
            )
        return _unsupported_component(value, self.notes)


def templates_from_slots(
    trace: Trace, call: _SlotCall, slots: tuple[_ArgSlot, ...]
) -> CapturedArgTemplate:
    """Build the call's replay template; same values as ``_build_args_template``."""

    from ...intervention.types import CapturedArgTemplate

    walk = _TemplateWalk(trace, {(slot.kind, slot.key): slot for slot in slots}, [])
    args = tuple(walk.component(arg, ("args", index)) for index, arg in enumerate(call.args))
    kwargs = tuple(
        (str(key), walk.component(value, ("kwargs", key))) for key, value in call.kwargs.items()
    )
    return CapturedArgTemplate(
        args=args,
        kwargs=kwargs,
        func_id=_function_registry_key(call.func, call.func_name, trace),
        notes=tuple(walk.notes),
    )


def witness_from_slots(
    trace: Trace,
    call: _SlotCall,
    parent_arg_positions: dict[str, dict[Any, str]],
    recorded_parent_params: list[torch.nn.Parameter],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Run the unattributed / dropped-edge witness on the snapshot, then retire it."""

    from ._ops_emission import _unattributed_tensor_arg_positions

    try:
        return _unattributed_tensor_arg_positions(
            trace,
            call.args,
            call.kwargs,
            call.func_name,
            parent_arg_positions,
            recorded_parent_params,
        )
    finally:
        workspace = _peek_raw_graph_workspace(trace)
        if workspace is not None:
            workspace._pending_arg_slots = None
