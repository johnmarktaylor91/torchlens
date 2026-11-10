"""Row keys behind the trace-level ops door.

``TraceOpAccessor`` holds one integer key per member instead of the ``Op``
record. A key is the op's row in the trace core's op store. A record with no
row there (a detached single-row copy, a stand-in row, any record of a trace
without a core) gets the next free negative key. Until records are handed out
weakly, the record behind a key is cached strongly in one side table on the
core: ``TraceCore._strong_facades["op"]`` is a list indexed by store row
(``None`` where no record was cached) and
``TraceCore._strong_facades["op.detached"]`` is a dictionary from negative key
to record. A door whose records reach no core keeps the same two tables itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import repeat
from typing import Any

from .._utils._record_group_table import record_member

OP_KIND = "op"
DETACHED_KIND = "op.detached"

# Read a record's slot without importing ``Op`` (this module sits below it).
_slot = object.__getattribute__


class OpTable:
    """The two side tables one ops door resolves its keys through.

    ``plain`` is true while every key this door holds is a store row, so a hot
    read can index ``rows`` directly (the door's lookups do, to add no call).
    """

    def __init__(self, rows: list[Any], detached: dict[int, Any]) -> None:
        """Wrap the row list and the detached dictionary; both are held strongly."""

        self.rows = rows
        self.detached = detached
        self.plain = not detached

    def record(self, key: int) -> Any:
        """Return the record behind one key."""

        return self.rows[key] if key >= 0 else self.detached[key]

    def records(self, keys: Iterable[int]) -> Iterator[Any]:
        """Return an iterator over the records behind ``keys``, in order."""

        rows, detached = self.rows, self.detached
        if self.plain:
            return map(rows.__getitem__, keys)
        return iter([rows[key] if key >= 0 else detached[key] for key in keys])

    def hold(self, records: Sequence[Any], store: Any) -> list[int]:
        """Cache every record and return its key, in order.

        A record of ``store`` takes its row as the key. Any other record, and a
        second record claiming a row another record already holds, takes a
        negative key; the same record always gets the same key back.
        """

        rows, detached = self.rows, self.detached
        size = len(rows)
        keys = [0] * len(records)
        ids: dict[int, int] | None = None
        for position, record in enumerate(records):
            try:
                row = _slot(record, "_row")
                if 0 <= row < size:
                    held = rows[row]
                    if held is None and _slot(record, "_core") is store:
                        rows[row] = held = record
                    if held is record:
                        keys[position] = row
                        continue
            except (TypeError, AttributeError):  # a stand-in row or a bare shell
                pass
            if ids is None:
                ids = {id(value): key for key, value in detached.items()}
            key = ids.get(id(record))
            if key is None:
                key = -1 - len(detached)
                while key in detached:
                    key -= 1
                detached[key] = record
                ids[id(record)] = key
            keys[position] = key
        self.plain = ids is None
        return keys

    def key_of(self, record: Any) -> int | None:
        """Return the key ``record`` is cached under, or ``None``."""

        try:
            row = _slot(record, "_row")
            if 0 <= row < len(self.rows) and self.rows[row] is record:
                return row
        except (TypeError, AttributeError):
            pass
        for key, held in self.detached.items():
            if held is record:
                return key
        return None

    def only(self, keys: Iterable[int]) -> OpTable:
        """Return a private table holding just the records behind ``keys``."""

        rows: list[Any] = [None] * len(self.rows)
        detached: dict[int, Any] = {}
        for key in keys:
            if key >= 0:
                rows[key] = self.rows[key]
            else:
                detached[key] = self.detached[key]
        return OpTable(rows, detached)


def _side_tables(trace: Any) -> tuple[list[Any], dict[int, Any], Any]:
    """Return the row list, detached dictionary and op store of ``trace`` (``None`` for no trace)."""

    core = getattr(trace, "__dict__", {}).get("_trace_core")
    facades = getattr(core, "_strong_facades", None)
    if core is None or not isinstance(facades, dict):
        return [], {}, None
    rows = facades.get(OP_KIND)
    if rows is None:
        rows = facades[OP_KIND] = []
    detached = facades.setdefault(DETACHED_KIND, {})
    store = core.ops
    if store is not None and len(rows) < len(store):
        rows.extend(repeat(None, len(store) - len(rows)))
    return rows, detached, store


def hold_ops(records: Sequence[Any], trace: Any) -> tuple[OpTable, list[int]]:
    """Cache ``records`` in the side tables of ``trace``; return the tables and one key per record."""

    rows, detached, store = _side_tables(trace)
    table = OpTable(rows, detached)
    return table, table.hold(records, store)


class OpRecordList(Sequence[Any]):
    """The records behind an ordered key list, read like the list the door used to hold."""

    __slots__ = ("_keys", "_table")

    def __init__(self, table: OpTable, keys: Sequence[int]) -> None:
        """Read ``keys`` through ``table``."""

        self._table = table
        self._keys = keys

    def __getitem__(self, index: Any) -> Any:
        """Return one record, or a plain list of records for a slice."""

        if isinstance(index, slice):
            return list(self._table.records(self._keys[index]))
        return self._table.record(self._keys[index])

    def __len__(self) -> int:
        """Return the number of records."""

        return len(self._keys)

    def __iter__(self) -> Iterator[Any]:
        """Iterate the records in order."""

        return self._table.records(self._keys)


class OpRecordMap(Mapping[Any, Any]):
    """The records behind a label-to-key dictionary, read like the dictionary the door used to hold."""

    __slots__ = ("_keys", "_table")

    def __init__(self, table: OpTable, keys: Mapping[Any, int]) -> None:
        """Read the label-to-key mapping ``keys`` through ``table``."""

        self._table = table
        self._keys = keys

    def __getitem__(self, label: Any) -> Any:
        """Return the record a label names."""

        return self._table.record(self._keys[label])

    def __contains__(self, label: object) -> bool:
        """Return whether ``label`` names a record."""

        return label in self._keys

    def __iter__(self) -> Iterator[Any]:
        """Iterate the labels in order."""

        return iter(self._keys)

    def __len__(self) -> int:
        """Return the number of labels."""

        return len(self._keys)


def _group_by_layer_aliases(keys: Sequence[int], ops: Iterable[Any]) -> dict[Any, list[int]]:
    """Group Op row keys under their layer label and short layer label, in list order.

    An Op lands once under each distinct alias, so ``groups[key]`` lists exactly
    the Ops for which ``key in {op.layer_label, op.short_layer_label}``, in the
    order a scan of ``ops`` would meet them. ``keys`` is the row key of each Op.
    """

    groups: dict[Any, list[int]] = {}
    for held, op in zip(keys, ops, strict=True):
        for alias in {op.layer_label, op.short_layer_label}:
            groups.setdefault(alias, []).append(held)
    return groups


@dataclass(frozen=True)
class _AliasIndexes:
    """Reverse indexes behind ``TraceOpAccessor._resolve_substring``.

    ``final`` maps each final-tier alias to its Op, ``collisions`` the final-tier
    aliases two Ops claim, ``raw`` each unsuffixed raw spelling to its Op, and
    ``layer`` each bare layer label to its Ops in list order. Every Op is held
    as its row key (``_op_door_keys``), never as the record.
    """

    final: dict[Any, int]
    collisions: dict[Any, list[int]]
    raw: dict[Any, int]
    layer: dict[Any, list[int]]


def _claim_alias(
    index: dict[Any, int], collisions: dict[Any, list[int]], alias: Any, held: int
) -> None:
    """Record ``alias -> row key``; a second, different Op makes the alias a collision."""

    existing = index.setdefault(alias, held)
    if existing == held or not isinstance(alias, str):
        return
    claimants = collisions.setdefault(alias, [existing])
    if held not in claimants:
        claimants.append(held)


def _is_counted_op(op: Any) -> bool:
    """Whether an op row counts toward the counted op total.

    The count excludes the input, output and buffer rows, the definition in
    ``backends/_finalize.py``; those rows are aliases of a tensor, not ops.
    """

    return not (
        getattr(op, "is_input", False)
        or getattr(op, "is_output", False)
        or record_member(op, "is_buffer", False)
    )
