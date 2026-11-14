"""Alias counts for container-cell pooling: the registry read and the census oracle.

Container pooling may replace a mutable container cell with a shared pooled
cell only when no other swept cell reaches the same container object;
replacing an aliased appearance would break the alias for later in-place
mutation. :func:`_registry_container_cells` counts container ids over the
pooling candidates and the cells the write-time alias registry recorded
(``_trace_core/alias_registry.py``) and collects the pool and
singleton-compaction sites; ``op.py``'s ``_pool_container_cells`` then keys
and pools them. :func:`_census_container_cells`, the former scan over every
swept cell, is kept as the test oracle: with :data:`ALIAS_ORACLE` set (the
test suite sets it to ``_op_alias_oracle.assert_registry_matches_census``),
every registry read runs the census beside it and raises on any difference in
the alias decisions. Standard library only.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Callable
from typing import Any

#: ``(row_cells, fid, value, precomputed_key_or_None)`` pool candidate site.
_CellSite = tuple[Any, int, Any, Any]
#: ``(store, row_idx, fid, row_cells, value)`` kind-table list cell site.
_SingletonSite = tuple[Any, int, int, Any, Any]


def _count_container_ids(
    value: Any,
    id_counts: dict[int, int],
    expanded_ids: set[int],
    depth: int,
) -> None:
    """Count every exact builtin MUTABLE container id reachable from one cell.

    The alias census behind the pooling guard: a container whose id is seen
    more than once across ALL swept cells is aliased in-store, and replacing
    any of its cell appearances would break the alias for later in-place
    mutation, so such cells never pool.

    Only ``dict``/``defaultdict``/``list``/``set`` ids matter (the pool key
    visits exactly those). Hashability bounds the walk: dict KEYS and
    ``set``/``frozenset`` members must be hashable, so no exact builtin
    mutable container can hide below them — dict values, list members, and
    tuple members are the only recursion edges.
    """

    cls = value.__class__
    if cls is dict or cls is defaultdict or cls is list:
        oid = id(value)
        id_counts[oid] = id_counts.get(oid, 0) + 1
        if depth >= 6 or oid in expanded_ids:
            return
        expanded_ids.add(oid)
        members = value if cls is list else value.values()
    elif cls is set:
        oid = id(value)
        id_counts[oid] = id_counts.get(oid, 0) + 1
        return
    elif cls is tuple and depth < 6 and id(value) not in expanded_ids:
        expanded_ids.add(id(value))
        members = value
    else:
        return
    # Recurse only into members this function acts on (an identity test on the
    # class, as above); every other member would return at once.
    walked = _WALKED_CLASS_IDS
    for member in members:
        if id(member.__class__) in walked:
            _count_container_ids(member, id_counts, expanded_ids, depth + 1)


#: ``id`` of each class ``_count_container_ids`` counts or expands.
_WALKED_CLASS_IDS = frozenset(map(id, (dict, defaultdict, list, set, tuple)))


#: Cell classes the pooling sweep acts on (everything else is skipped).
_SWEEP_CELL_CLASSES = frozenset({tuple, dict, list, set, defaultdict})
_GET_CLASS = operator.attrgetter("__class__")


def _sweep_columns(rows: list[list[Any]], scan_classes: frozenset[type], get_class: Any) -> Any:
    """Return the field ids whose column holds at least one swept-class cell.

    A ``__class__`` that is not hashable (only a pathological proxy can
    report one) cannot be tested against the set; every column is swept then.
    """

    try:
        return [
            fid
            for fid, column in enumerate(zip(*rows, strict=False))
            if not scan_classes.isdisjoint(map(get_class, column))
        ]
    except TypeError:
        return range(len(rows[0])) if rows else ()


#: ``_container_pool_key`` of an empty exact ``list``/``dict``/``set``.
_EMPTY_CONTAINER_POOL_KEYS: dict[type, Any] = {
    list: ("l", ()),
    dict: ("d", ()),
    set: ("s", frozenset()),
}


def _census_container_cells(
    stores: list[tuple[Any, Any]],
) -> tuple[dict[int, int], list[_CellSite], list[_SingletonSite]] | None:
    """Run container pooling's single scan over every swept cell.

    The alias census recurses only into cells whose class can hold a
    container; the same scan collects the pool candidates (container cells at
    sanctioned fids, carrying the constant key of an empty exact
    ``dict``/``list``/``set``, else ``None``) and the singleton-compaction
    sites (kind-table list cells), so no later pass re-walks rows x fields.
    An empty exact container counts its own id only (nothing to expand); an
    empty tuple holds no container. Columns with no tuple or swept container
    cell are skipped (same cells, same order).

    Returns
    -------
    tuple | None
        ``(id_counts, cell_sites, singleton_sites)``, or ``None`` when no
        store is in its building phase.
    """

    id_counts: dict[int, int] = {}
    expanded_ids: set[int] = set()
    cell_sites: list[_CellSite] = []
    singleton_sites: list[_SingletonSite] = []
    any_swept = False
    for store, fids in stores:
        rows = store.rows_building()
        if rows is None:
            continue
        any_swept = True
        _census_store(store, rows, fids, (id_counts, expanded_ids, cell_sites, singleton_sites))
    return (id_counts, cell_sites, singleton_sites) if any_swept else None


def _census_store(store: Any, rows: list[list[Any]], fids: Any, acc: tuple[Any, ...]) -> None:
    """Census every swept cell of one building store into the shared accumulators.

    ``acc`` is ``(id_counts, expanded_ids, cell_sites, singleton_sites)``.
    """

    id_counts, expanded_ids, cell_sites, singleton_sites = acc
    empty_keys = _EMPTY_CONTAINER_POOL_KEYS
    fid_set = None if fids is None else frozenset(fids)
    sweep_fids = _sweep_columns(rows, _SWEEP_CELL_CLASSES, _GET_CLASS)
    for row_idx, row_cells in enumerate(rows):
        for fid in sweep_fids:
            value = row_cells[fid]
            cls = value.__class__
            if cls is tuple:
                if value:
                    _count_container_ids(value, id_counts, expanded_ids, 0)
                continue
            if not (cls is dict or cls is list or cls is set or cls is defaultdict):
                continue
            if not value and cls is not defaultdict:
                oid = id(value)
                id_counts[oid] = id_counts.get(oid, 0) + 1
                if fid_set is None or fid in fid_set:
                    cell_sites.append((row_cells, fid, value, empty_keys[cls]))
                continue
            _count_container_ids(value, id_counts, expanded_ids, 0)
            if fid_set is None or fid in fid_set:
                cell_sites.append((row_cells, fid, value, None))
            if fid_set is None and cls is list:
                singleton_sites.append((store, row_idx, fid, row_cells, value))


#: Test-only check called with ``(stores, result)`` after every registry read.
ALIAS_ORACLE: Callable[[list[tuple[Any, Any]], Any], None] | None = None


def _count_cell(value: Any, id_counts: dict[int, int], expanded_ids: set[int]) -> None:
    """Count one swept cell exactly as the census does (empty containers inline)."""

    cls = value.__class__
    if cls is tuple:
        if value:
            _count_container_ids(value, id_counts, expanded_ids, 0)
    elif not value and cls is not defaultdict:
        oid = id(value)
        id_counts[oid] = id_counts.get(oid, 0) + 1
    else:
        _count_container_ids(value, id_counts, expanded_ids, 0)


def _registry_container_cells(
    stores: list[tuple[Any, Any]],
) -> tuple[dict[int, int], list[_CellSite], list[_SingletonSite]] | None:
    """Collect the pooling sites and the alias counts that decide them.

    A store with sanctioned fids and a write-time registry counts only its
    candidate cells and the registered cells outside them; every other store
    (kind tables, stores without a registry) is scanned in full as before.
    Same return shape and site order as :func:`_census_container_cells`.
    """

    id_counts: dict[int, int] = {}
    expanded_ids: set[int] = set()
    cell_sites: list[_CellSite] = []
    singleton_sites: list[_SingletonSite] = []
    any_swept = False
    for store, fids in stores:
        rows = store.rows_building()
        if rows is None:
            continue
        any_swept = True
        sites = None if fids is None else getattr(store, "_alias_sites", None)
        acc = (id_counts, expanded_ids, cell_sites, singleton_sites)
        if sites is None:
            _census_store(store, rows, fids, acc)
            continue
        _registry_store_sites(rows, fids, sites, acc)
    result = (id_counts, cell_sites, singleton_sites) if any_swept else None
    if ALIAS_ORACLE is not None:
        ALIAS_ORACLE(stores, result)
    return result


#: Cell classes the op-store census visits (tuples count ids; the rest are pool sites).
_CENSUS_CELL_CLASSES = frozenset({tuple, dict, list, set, defaultdict})

#: Fewest rows at which the container-free column scan pays for itself.
_CENSUS_SKIP_MIN_ROWS = 32


def _registry_store_sites(
    rows: list[list[Any]], fids: Any, sites: dict[tuple[int, int], Any], acc: tuple[Any, ...]
) -> None:
    """Count one op store's candidate cells and registered cells; collect its sites.

    ``acc`` is the census accumulator tuple (see :func:`_census_store`); this
    store adds no singleton sites.
    """

    id_counts, expanded_ids, cell_sites, _singleton_sites = acc

    empty_keys = _EMPTY_CONTAINER_POOL_KEYS
    sanctioned = frozenset(fids)
    ordered = sorted(sanctioned)
    if len(rows) >= _CENSUS_SKIP_MIN_ROWS:
        # Columns holding no container cell add nothing (every cell would take a
        # ``continue``); dropping them keeps the row-major visit order of the rest.
        ordered = [
            fid
            for fid in ordered
            if not _CENSUS_CELL_CLASSES.isdisjoint({row_cells[fid].__class__ for row_cells in rows})
        ]
    for row_cells in rows:
        for fid in ordered:
            value = row_cells[fid]
            cls = value.__class__
            if cls is tuple:
                if value:
                    _count_container_ids(value, id_counts, expanded_ids, 0)
                continue
            if not (cls is dict or cls is list or cls is set or cls is defaultdict):
                continue
            if not value and cls is not defaultdict:
                oid = id(value)
                id_counts[oid] = id_counts.get(oid, 0) + 1
                cell_sites.append((row_cells, fid, value, empty_keys[cls]))
                continue
            _count_container_ids(value, id_counts, expanded_ids, 0)
            cell_sites.append((row_cells, fid, value, None))
    for (row, fid), value in sites.items():
        if fid not in sanctioned and rows[row][fid] is value:
            _count_cell(value, id_counts, expanded_ids)
