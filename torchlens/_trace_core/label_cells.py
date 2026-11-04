"""Label-structure indexes shared by a trace and its forks, validated exactly.

Every replay builds the same structural indexes over a trace's ops: the
selector label index, the replay label maps and the function-call groups.
Forks of one sealed trace share its root ``OpRowStore``, and a clean fork's
label cells equal the parent's, so the indexes can be built once per root.

The validity rule is exact comparison, never a dirty bit: each use takes a
snapshot of the effective label cells (:data:`LABEL_FIELDS`) of the op
universe, read straight from the store layers in comprehensions (a fork
view's overlay, its base snapshot, sealed rows or frozen columns), and reuses
an entry only when the snapshot equals the one the entry was built from.
``lookup_keys`` lists are copied into the snapshot, never aliased, so an
in-place ``lookup_keys.append`` on any trace changes its next snapshot. Only
exact ``str``/``int``/``None`` values (and lists or tuples of them) take part,
so ``==`` on the snapshot is identity of everything the builders read; any
other value, store kind or facade class returns ``None`` and the caller builds
as before. Entries hold row positions and label strings, never facades: a hit
materializes each answer from the caller's own ops.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from sys import modules as _loaded_modules
from typing import Any

from .op_store import _MISSING, OpRowStore, OpStoreView, PooledCell, _SealedRowMajorOpRowStore

#: Every op field an index builder reads (directly or through
#: ``_string_labels`` and the pass tokens). The drift guard in
#: ``tests/test_fork_shared_label_index.py`` records the builders' reads on a
#: real trace and fails when this set and those reads differ.
LABEL_FIELDS: tuple[str, ...] = (
    "layer_label",
    "label",
    "short_layer_label",
    "short_label",
    "_layer_label_raw",
    "pass_number",
    "num_passes",
    "lookup_keys",
    "func_call_id",
)

#: The one list-valued label field (snapshotted as copies).
_LIST_FIELD = "lookup_keys"

#: Value classes whose ``==`` is identity of what the builders read.
_PLAIN = frozenset({str, int, type(None)})

#: Snapshot entries kept per root store (a dirty fork must not evict the
#: clean entry its siblings use).
_MAX_ENTRIES = 4

_ROOT_KINDS = (OpRowStore, _SealedRowMajorOpRowStore)

#: Module of the ``Op`` facade. Read from ``sys.modules`` at use time: this
#: module sits below ``_data_classes`` and must not import it, and any trace
#: with ops has already imported it. ``Op``'s label fields are plain cell
#: descriptors (pinned by the drift guard test), so its reads are the cells.
_OP_MODULE = "torchlens._data_classes.op"


def _op_class() -> type | None:
    """Return the ``Op`` facade class, or ``None`` before it is imported."""

    module = _loaded_modules.get(_OP_MODULE)
    return getattr(module, "Op", None)


def _sealed_cells(store: OpRowStore, rows: list[int], fid: int) -> list[Any]:
    """Return one field's effective cells for ``rows`` of a sealed root store."""

    base_rows = store._rows
    if base_rows is not None:
        return [base_rows[row][fid] for row in rows]
    n = store.layout.n_fields
    overlay = store._overlay
    column = store._columns[fid]  # type: ignore[index]
    values = column.values.tolist() if column.packed else column.values
    present = column.present
    return [
        overlay[key]
        if (key := row * n + fid) in overlay
        else (values[row] if present is None or present[row] else _MISSING)
        for row in rows
    ]


def _view_cells(view: OpStoreView, rows: list[int], fid: int) -> list[Any]:
    """Return one field's effective cells for ``rows`` of a fork view."""

    n = view.base.layout.n_fields
    overlay = view._overlay
    base_overlay = view._base_overlay
    base_rows = view._base_rows
    if base_rows is not None:
        return [
            overlay[key]
            if (key := row * n + fid) in overlay
            else (base_overlay[key] if key in base_overlay else base_rows[row][fid])
            for row in rows
        ]
    column = view.base._columns[fid]  # type: ignore[index]
    values = column.values.tolist() if column.packed else column.values
    present = column.present
    return [
        overlay[key]
        if (key := row * n + fid) in overlay
        else (
            base_overlay[key]
            if key in base_overlay
            else (values[row] if present is None or present[row] else _MISSING)
        )
        for row in rows
    ]


def _root_and_reader(store: Any) -> tuple[OpRowStore, Callable[..., list[Any]]] | None:
    """Return the root store and cell reader for a supported sealed store."""

    if store.__class__ is OpStoreView:
        root = store.base
        reader: Callable[..., list[Any]] = _view_cells
    elif store.__class__ in _ROOT_KINDS and store._sealed:
        root = store
        reader = _sealed_cells
    else:
        return None
    if root.__class__ not in _ROOT_KINDS or root._compacted_singletons:
        return None
    return root, reader


def _plain_column(cells: list[Any]) -> list[Any] | None:
    """Return a scalar column with pooled cells unwrapped, or ``None``."""

    if any(cell.__class__ is PooledCell for cell in cells):
        cells = [cell.prototype if cell.__class__ is PooledCell else cell for cell in cells]
    odd = {cell.__class__ for cell in cells} - _PLAIN
    if odd and (
        odd != {object} or {cell for cell in cells if cell.__class__ is object} != {_MISSING}
    ):
        return None
    return cells


def _list_column(cells: list[Any]) -> list[Any] | None:
    """Return the ``lookup_keys`` column as private copies, or ``None``."""

    copied = [
        cell.prototype[:]
        if cell.__class__ is PooledCell
        else (cell[:] if cell.__class__ is list or cell.__class__ is tuple else cell)
        for cell in cells
    ]
    kinds = {cell.__class__ for cell in copied}
    if not kinds <= {list, tuple, type(None), object}:
        return None
    if object in kinds and {cell for cell in copied if cell.__class__ is object} != {_MISSING}:
        return None
    elements = {
        element.__class__
        for cell in copied
        if cell.__class__ is list or cell.__class__ is tuple
        for element in cell
    }
    return copied if elements <= _PLAIN else None


def label_snapshot(ops: Sequence[Any]) -> tuple[OpRowStore, tuple[Any, ...]] | None:
    """Return ``(root store, exact label-cell snapshot)`` for ``ops``, or ``None``.

    ``None`` means "do not share": the ops are empty, not all exact ``Op``
    facades on one sealed store, or hold a label value outside the plain
    classes.
    """

    if not ops or {op.__class__ for op in ops} != {_op_class()}:
        return None
    store = ops[0]._core
    if {op._core for op in ops} != {store}:
        return None
    found = _root_and_reader(store)
    if found is None:
        return None
    root, reader = found
    rows = [op._row for op in ops]
    fid_by_name = root.layout.fid_by_name
    columns: list[Any] = [rows]
    for name in LABEL_FIELDS:
        cells = reader(store, rows, fid_by_name[name])
        column = _list_column(cells) if name == _LIST_FIELD else _plain_column(cells)
        if column is None:
            return None
        columns.append(column)
    return root, tuple(columns)


def shared_entry(ops: Sequence[Any]) -> dict[str, Any] | None:
    """Return the result table shared by every trace whose label cells equal ``ops``'.

    The table maps a builder kind to its position-form result; callers fill
    a missing kind once and every later trace with an equal snapshot reads
    it. ``None`` means "build privately" (see :func:`label_snapshot`).
    """

    found = label_snapshot(ops)
    if found is None:
        return None
    root, snapshot = found
    entries: list[tuple[tuple[Any, ...], dict[str, Any]]] | None = getattr(
        root, "_label_share", None
    )
    if entries is None:
        entries = root._label_share = []
    for position, (entry_snapshot, results) in enumerate(entries):
        if entry_snapshot == snapshot:
            if position:
                entries.insert(0, entries.pop(position))  # most recently used first
            return results
    results = {}
    entries.insert(0, (snapshot, results))
    del entries[_MAX_ENTRIES:]
    return results


def site_positions(ops: Sequence[Any]) -> dict[int, int]:
    """Map each op's identity to its position in ``ops`` (miss path only)."""

    return {id(op): position for position, op in enumerate(ops)}


class PositionGroups(Mapping[Any, tuple[Any, ...]]):
    """Read-only mapping answering position-form groups with one trace's ops.

    Lookups materialize only the group asked for, so a hit costs nothing per
    group that the caller never reads.
    """

    __slots__ = ("_ops", "_positions")

    def __init__(self, positions: Mapping[Any, tuple[int, ...]], ops: Sequence[Any]) -> None:
        """Bind shared positions to this trace's own ops."""

        self._positions = positions
        self._ops = ops

    def __getitem__(self, key: Any) -> tuple[Any, ...]:
        """Return the group for ``key`` as this trace's ops."""

        return tuple(map(self._ops.__getitem__, self._positions[key]))

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the group for ``key``, or ``default`` when absent."""

        positions = self._positions.get(key)
        if positions is None:
            return default
        return tuple(map(self._ops.__getitem__, positions))

    def __iter__(self) -> Iterator[Any]:
        """Iterate the group keys in build order."""

        return iter(self._positions)

    def __len__(self) -> int:
        """Return the number of groups."""

        return len(self._positions)


def positions_of(
    groups: Mapping[Any, Sequence[Any]], ops: Sequence[Any]
) -> dict[Any, tuple[int, ...]]:
    """Convert a facade-valued group table to positions in ``ops`` (miss path)."""

    where = site_positions(ops)
    return {key: tuple([where[id(op)] for op in members]) for key, members in groups.items()}


def shared_site_index(
    sites: Sequence[Any], build: Callable[[Sequence[Any]], Mapping[str, Sequence[Any]] | None]
) -> Mapping[str, Sequence[Any]] | None:
    """Return the selector label index for ``sites``, built once per equal snapshot.

    ``build`` is the selector evaluator's label-index builder; its ``None``
    ("cannot index exactly") is shared too. The answer maps each label to
    ``sites``' own ops, materialized per lookup.
    """

    results = shared_entry(sites)
    if results is None:
        return build(sites)
    if "site_index" not in results:
        index = build(sites)
        results["site_index"] = None if index is None else positions_of(index, sites)
    positions = results["site_index"]
    return None if positions is None else PositionGroups(positions, sites)
