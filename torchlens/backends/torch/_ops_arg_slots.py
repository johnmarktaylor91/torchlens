"""One per-op snapshot of the tensor argument slots the shared-fields derivations read.

The shared-fields builder derives four things from an op's arguments: the parent
labels, ``parent_arg_positions``, the unattributed / dropped-edge witness and the
replay argument templates. Each used to walk the arguments and read every tensor's
label and provenance on its own. The builder's ``snapshot_arg_slots`` walks the
arguments once, to the recorder's two nesting levels, and records per tensor slot
the values those derivations read; this module holds the records and the readers.
Tensors nested deeper keep the per-argument lookups.

The module imports only the tensor-label layer, so the capture modules that read
the snapshot (shared fields, the position recorder, the template classifier and
the witness) import it at module top without joining their import cycle.

Ordering: the snapshot is taken after the builder's source-logging loop (buffer and
closure-constant labels are assigned there) and before the parent labels are read.
It applies at snapshot time the stale-label clears that the derivations' live reads
(``get_live_tensor_label``) would apply later, so each recorded fact is the value the
derivation reads; nothing between the snapshot and the witness relabels an argument.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..._utils._tensor_label import (
    TensorMeta,
    get_live_tensor_label,
    is_tensor_data_alias,
    session_label_storage_intact,
    session_meta_is_anchored,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ..._ir.workspaces import RawGraphWorkspace

SlotKey = tuple[str, Any]


@dataclass(frozen=True, slots=True)
class _SlotCall:
    """The wrapped call whose arguments the shared-fields builder derives from."""

    func: Callable[..., Any]
    func_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    #: The call's flat argument layout plan (``_call_layout.CallLayout``), or ``None``
    #: when the call is not flat or the capture keeps no plans.
    layout: Any = None


@dataclass(frozen=True)
class _ArgSlot:
    """What the shared-fields derivations read about one tensor argument slot.

    No ``__slots__``: :func:`new_arg_slot` writes a record's fields into its instance
    dict in one step (the generated frozen ``__init__`` sets them one by one).

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


@dataclass(frozen=True)
class _SlotContext:
    """The current op's slots, keyed for the derivations that cannot take them directly.

    Built by :func:`new_slot_context` (no ``__slots__``, as :class:`_ArgSlot`).
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    slots: tuple[_ArgSlot, ...]
    by_key: dict[SlotKey, _ArgSlot]
    #: Each slot tensor's ``label`` by object id (read-only for its readers).
    labels: dict[int, str | None]


_NEW = object.__new__


def new_arg_slot(kind: str, key: Any, value_id: int, facts: tuple[Any, ...]) -> _ArgSlot:
    """Return ``_ArgSlot(kind, key, value_id, *facts)``, its fields written at once."""

    slot = _NEW(_ArgSlot)
    slot.__dict__.update(
        {
            "kind": kind,
            "key": key,
            "value_id": value_id,
            "label": facts[0],
            "known_provenance": facts[1],
            "session_parent_labels": facts[2],
            "data_alias": facts[3],
            "param_unmutated": facts[4],
            "param_validated": facts[5],
        }
    )
    return slot


def new_slot_context(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    slots: tuple[_ArgSlot, ...],
    by_key: dict[SlotKey, _ArgSlot],
    labels: dict[int, str | None],
) -> _SlotContext:
    """Return ``_SlotContext(args, kwargs, slots, by_key, labels)``, written at once."""

    context = _NEW(_SlotContext)
    context.__dict__.update(
        {"args": args, "kwargs": kwargs, "slots": slots, "by_key": by_key, "labels": labels}
    )
    return context


@dataclass(frozen=True, slots=True)
class _SnapshotScope:
    """Per-op inputs of the snapshot: live labels and which tensors get cleared."""

    live: Mapping[str, Any]
    clear_ids: AbstractSet[int]
    visited_ids: AbstractSet[int]


def _peek_raw_graph_workspace(trace: Trace | None) -> RawGraphWorkspace | None:
    """Read the trace's raw-graph workspace, if this capture still holds one.

    ``_raw_graph_ws`` is a declared ``FieldPolicy.DROP`` Trace field that only
    lives for one capture: postprocess pops it at its terminal consume and a
    loaded trace never had it, and ``trace`` itself may be ``None``. This is a
    direct private read guarded by ``AttributeError`` rather than a
    string-literal ``getattr`` default, so a rename breaks loudly.
    """

    if trace is None:
        return None
    try:
        return trace._raw_graph_ws
    except AttributeError:
        return None


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


def _current_context(trace: Trace | None, args: Any, kwargs: Any) -> _SlotContext | None:
    """Return the slot context recorded for exactly this call's arguments, if any."""

    # ``_peek_raw_graph_workspace`` inline (three reads per op): no trace, no
    # workspace or a workspace already popped all read as "no context".
    try:
        context = trace._raw_graph_ws._pending_arg_slots  # type: ignore[union-attr]
    except AttributeError:
        return None
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
    slots: tuple[_ArgSlot, ...], parent_records: list[Any]
) -> dict[str, dict[Any, str]]:
    """Map each parent label to its argument positions, emitted parent by parent.

    ``parent_records`` are the parents' indexed journal records (read by ``label_raw``).
    """

    return slot_positions_by_label(slots, [record.label_raw for record in parent_records])


def slot_positions_by_label(
    slots: tuple[_ArgSlot, ...], parent_labels: list[str]
) -> dict[str, dict[Any, str]]:
    """Map each parent label to its argument positions, emitted label by label."""

    positions: dict[str, dict[Any, str]] = {"args": {}, "kwargs": {}}
    slots_by_label: dict[str, list[_ArgSlot]] = {}
    for slot in slots:
        if slot.label is not None:
            slots_by_label.setdefault(slot.label, []).append(slot)
    for label in parent_labels:
        for slot in slots_by_label.get(label, ()):
            positions[slot.kind][slot.key] = label
    return positions


def arg_slot_labels(trace: Trace | None, args: Any, kwargs: Any) -> dict[int, str | None] | None:
    """Return the current op's snapshot labels by tensor id, or ``None`` to read live.

    The dict is the context's own: callers read it and never change it.
    """

    context = _current_context(trace, args, kwargs)
    return None if context is None else context.labels
